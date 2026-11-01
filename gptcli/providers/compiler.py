"""
Unified message validation and request compilation boundary.

Every remote provider receives the same canonical conversation: a list of
`Message` dicts made of the assistant's preset messages (leading system
instructions and optional few-shot turns) followed by the live conversation.
Providers differ in which of those shapes their APIs can express:

- OpenAI's Responses API accepts ``system`` messages inline, in any position,
  and allows consecutive same-role messages.
- Anthropic, Cohere and Google only have a single leading instruction slot
  (``system`` / ``preamble`` / ``system_instruction``); all other turns must
  be user/assistant messages that alternate, start with a user turn and end
  with the user turn that requests the completion.

`compile_messages` is the single place that enforces those rules. It never
mutates the session's messages: instruction text is copied and every returned
turn is a fresh dict. When a conversation cannot be translated into an
equivalent provider request, `MessageCompileError` is raised *before* any
client is created or network request is made, with a diagnostic that points at
the offending message.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

from gptcli.completion import BadRequestError, Message

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
CANONICAL_ROLES = frozenset({SYSTEM_ROLE, USER_ROLE, ASSISTANT_ROLE})

#: System messages either stay inline in the sequence ("inline") or may only
#: appear as a leading prefix that is merged into the instruction slot
#: ("leading").
SystemPlacement = Literal["inline", "leading"]

INSTRUCTION_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ProviderCapabilities:
    """What a provider API can express in a completion request."""

    #: Human readable provider name, used in diagnostics.
    name: str
    #: How system messages are represented.
    system_placement: SystemPlacement
    #: Whether user/assistant turns must strictly alternate.
    alternating_roles: bool
    #: Whether empty message content is rejected by the API.
    non_empty_content: bool


@dataclass(frozen=True)
class CompiledMessages:
    """
    Result of compiling canonical messages for one request.

    ``instruction`` is the ordered merge of all leading system messages (or
    ``None`` if there are none). ``messages`` are fresh copies of the
    conversation turns, in their original order.
    """

    instruction: Optional[str]
    messages: List[Message]


class MessageCompileError(BadRequestError):
    """
    Raised before a request is made when the canonical conversation cannot be
    equivalently expressed for a provider.
    """

    def __init__(
        self,
        provider: str,
        index: int,
        role: Optional[str],
        reason: str,
        suggestion: Optional[str] = None,
    ):
        self.provider = provider
        #: 0-based index of the offending message in the original list, -1
        #: when the diagnostic concerns the whole conversation.
        self.index = index
        self.role = role
        self.reason = reason
        self.suggestion = suggestion

        location = (
            f"message #{index + 1}"
            if index is not None and index >= 0
            else "conversation"
        )
        role_part = f" (role={role})" if role else ""
        text = f"[{provider}] Cannot compile {location}{role_part}: {reason}."
        if suggestion:
            text += f" {suggestion}"
        super().__init__(text)


def _display_role(role: str, capabilities: ProviderCapabilities) -> str:
    if capabilities.name == "gemini" and role == ASSISTANT_ROLE:
        return "model"
    if capabilities.name == "cohere":
        return {"system": "SYSTEM", "user": "USER", "assistant": "CHATBOT"}.get(
            role, role
        )
    return role


def _validate_roles(
    messages: List[Message], capabilities: ProviderCapabilities
) -> None:
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in CANONICAL_ROLES:
            raise MessageCompileError(
                capabilities.name,
                index,
                role,
                f"unsupported role {role!r}; {capabilities.name} requests can only "
                "contain system, user and assistant messages",
                "Remove the message or rewrite it with a supported role.",
            )


def _validate_non_empty(
    messages: List[Message], capabilities: ProviderCapabilities
) -> None:
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, str) or content == "":
            raise MessageCompileError(
                capabilities.name,
                index,
                message.get("role"),
                "message content is empty; empty turns are rejected by the "
                f"{capabilities.name} API",
                "Clear the conversation (:clear) or re-run (:r) the last turn "
                "to remove the empty response.",
            )


def compile_messages(
    messages: List[Message], capabilities: ProviderCapabilities
) -> CompiledMessages:
    """
    Validate and compile canonical session ``messages`` for a provider.

    Raises `MessageCompileError` before any request is made when the messages
    cannot be represented for ``capabilities``. The input list and its dicts
    are never modified.
    """
    if len(messages) == 0:
        raise MessageCompileError(
            capabilities.name,
            -1,
            None,
            "the conversation is empty, but a completion request needs at "
            "least one user message",
            "Send a user message first.",
        )

    _validate_roles(messages, capabilities)

    instruction_texts: List[str] = []
    turn_start = 0

    if capabilities.system_placement == "leading":
        # Leading system messages are extracted, in order, into the
        # instruction slot.
        while (
            turn_start < len(messages) and messages[turn_start]["role"] == SYSTEM_ROLE
        ):
            instruction_texts.append(messages[turn_start]["content"])
            turn_start += 1

        for index in range(turn_start, len(messages)):
            if messages[index]["role"] == SYSTEM_ROLE:
                raise MessageCompileError(
                    capabilities.name,
                    index,
                    SYSTEM_ROLE,
                    "a system message appears after the conversation started; "
                    f"the {capabilities.name} API only accepts system "
                    "instructions before the first user turn, and moving this "
                    "message into the instruction slot would reorder the "
                    "preset conversation, so it cannot be translated "
                    "equivalently",
                    "Move this message to the top of the assistant's preset "
                    "messages (before any user/assistant turn), or remove it.",
                )

    if capabilities.non_empty_content:
        _validate_non_empty(messages, capabilities)

    turns = messages[turn_start:]

    if len(turns) == 0:
        raise MessageCompileError(
            capabilities.name,
            -1,
            None,
            "the conversation contains system instructions but no user "
            "message; a completion request must contain at least one user "
            "turn",
            "Send a user message first.",
        )

    if capabilities.system_placement == "leading" and turns[0]["role"] != USER_ROLE:
        raise MessageCompileError(
            capabilities.name,
            turn_start,
            turns[0]["role"],
            f"the first conversation turn must be a user message, but it is "
            f"{_display_role(turns[0]['role'], capabilities)}",
            "Move the assistant message into a user/assistant few-shot pair "
            "or remove it.",
        )

    if turns[-1]["role"] != USER_ROLE:
        raise MessageCompileError(
            capabilities.name,
            len(messages) - 1,
            turns[-1]["role"],
            "the last message of a completion request must be a user "
            f"message, but it is {_display_role(turns[-1]['role'], capabilities)}",
            "Wait for the previous response to finish or clear the "
            "conversation (:clear).",
        )

    if capabilities.alternating_roles:
        previous_role: Optional[str] = None
        for offset, message in enumerate(turns):
            role = message["role"]
            if role == previous_role:
                raise MessageCompileError(
                    capabilities.name,
                    turn_start + offset,
                    role,
                    f"consecutive {_display_role(role, capabilities)} turns are "
                    f"not allowed; the {capabilities.name} API requires user "
                    "and assistant turns to alternate",
                    "Remove or merge the duplicate turn, or clear the "
                    "conversation (:clear).",
                )
            previous_role = role

    instruction = (
        INSTRUCTION_SEPARATOR.join(instruction_texts) if instruction_texts else None
    )
    compiled_turns = [
        {"role": message["role"], "content": message["content"]}
        for message in messages[turn_start:]
    ]

    return CompiledMessages(instruction=instruction, messages=compiled_turns)


OPENAI_CAPABILITIES = ProviderCapabilities(
    name="openai",
    system_placement="inline",
    alternating_roles=False,
    non_empty_content=False,
)

AZURE_OPENAI_CAPABILITIES = ProviderCapabilities(
    name="azure",
    system_placement="inline",
    alternating_roles=False,
    non_empty_content=False,
)

ANTHROPIC_CAPABILITIES = ProviderCapabilities(
    name="anthropic",
    system_placement="leading",
    alternating_roles=True,
    non_empty_content=True,
)

COHERE_CAPABILITIES = ProviderCapabilities(
    name="cohere",
    system_placement="leading",
    alternating_roles=True,
    non_empty_content=True,
)

GOOGLE_CAPABILITIES = ProviderCapabilities(
    name="gemini",
    system_placement="leading",
    alternating_roles=True,
    non_empty_content=True,
)
