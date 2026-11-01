import copy

import pytest

from gptcli.completion import BadRequestError
from gptcli.providers.compiler import (
    ANTHROPIC_CAPABILITIES,
    AZURE_OPENAI_CAPABILITIES,
    COHERE_CAPABILITIES,
    GOOGLE_CAPABILITIES,
    OPENAI_CAPABILITIES,
    MessageCompileError,
    compile_messages,
)

LEADING_CAPABILITIES = [
    ANTHROPIC_CAPABILITIES,
    COHERE_CAPABILITIES,
    GOOGLE_CAPABILITIES,
]
ALL_CAPABILITIES = LEADING_CAPABILITIES + [
    OPENAI_CAPABILITIES,
    AZURE_OPENAI_CAPABILITIES,
]

SYSTEM = {"role": "system", "content": "You are an expert."}
SYSTEM_2 = {"role": "system", "content": "Always answer in Chinese."}
FEW_SHOT_USER = {"role": "user", "content": "Be brief."}
FEW_SHOT_ASSISTANT = {"role": "assistant", "content": "Understood."}
LIVE_USER = {"role": "user", "content": "Hello?"}
LIVE_ASSISTANT = {"role": "assistant", "content": "Hi there."}


def test_dev_preset_compiles_to_leading_instruction_and_ordered_turns():
    messages = [SYSTEM, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER]

    for capabilities in LEADING_CAPABILITIES:
        compiled = compile_messages(copy.deepcopy(messages), capabilities)

        assert compiled.instruction == SYSTEM["content"]
        assert compiled.messages == [
            FEW_SHOT_USER,
            FEW_SHOT_ASSISTANT,
            LIVE_USER,
        ]
        assert [m["role"] for m in compiled.messages] == [
            "user",
            "assistant",
            "user",
        ]


def test_openai_keeps_system_messages_inline_in_order():
    messages = [SYSTEM, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER]

    compiled = compile_messages(messages, OPENAI_CAPABILITIES)

    assert compiled.instruction is None
    assert compiled.messages == messages


def test_multiple_leading_systems_merge_in_order_for_leading_providers():
    messages = [SYSTEM, SYSTEM_2, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER]

    for capabilities in LEADING_CAPABILITIES:
        compiled = compile_messages(messages, capabilities)

        assert compiled.instruction == "You are an expert.\n\nAlways answer in Chinese."
        assert compiled.messages == [
            FEW_SHOT_USER,
            FEW_SHOT_ASSISTANT,
            LIVE_USER,
        ]


def test_multiple_systems_stay_inline_for_openai_family():
    messages = [SYSTEM, SYSTEM_2, LIVE_USER]

    for capabilities in [OPENAI_CAPABILITIES, AZURE_OPENAI_CAPABILITIES]:
        compiled = compile_messages(messages, capabilities)

        assert compiled.instruction is None
        assert compiled.messages == messages


def test_interspersed_system_is_diagnosed_for_leading_providers():
    # Second system message after the conversation has started cannot be moved
    # into the instruction slot without reordering the preset conversation.
    messages = [SYSTEM, FEW_SHOT_USER, SYSTEM_2, FEW_SHOT_ASSISTANT, LIVE_USER]

    for capabilities in LEADING_CAPABILITIES:
        with pytest.raises(MessageCompileError) as exc_info:
            compile_messages(messages, capabilities)

        error = exc_info.value
        assert error.provider == capabilities.name
        assert error.index == 2
        assert error.role == "system"
        assert "after the conversation started" in str(error)
        assert capabilities.name in str(error)


def test_interspersed_system_is_preserved_for_openai_family():
    messages = [SYSTEM, FEW_SHOT_USER, SYSTEM_2, LIVE_USER]

    for capabilities in [OPENAI_CAPABILITIES, AZURE_OPENAI_CAPABILITIES]:
        compiled = compile_messages(messages, capabilities)
        assert compiled.messages == messages
        assert compiled.messages[2] == SYSTEM_2


def test_consecutive_same_role_diagnosed_for_leading_providers():
    messages = [
        SYSTEM,
        LIVE_USER,
        FEW_SHOT_USER,
        FEW_SHOT_ASSISTANT,
        LIVE_USER,
    ]

    for capabilities in LEADING_CAPABILITIES:
        with pytest.raises(MessageCompileError) as exc_info:
            compile_messages(messages, capabilities)
        assert exc_info.value.index == 2
        assert exc_info.value.role == "user"
        assert "alternate" in str(exc_info.value)


def test_consecutive_same_role_allowed_for_openai_family():
    messages = [LIVE_USER, FEW_SHOT_USER, LIVE_USER]

    for capabilities in [OPENAI_CAPABILITIES, AZURE_OPENAI_CAPABILITIES]:
        compiled = compile_messages(messages, capabilities)
        assert compiled.messages == messages


def test_first_turn_must_be_user_for_leading_providers():
    messages = [SYSTEM, FEW_SHOT_ASSISTANT, LIVE_USER]

    for capabilities in LEADING_CAPABILITIES:
        with pytest.raises(MessageCompileError) as exc_info:
            compile_messages(messages, capabilities)
        assert exc_info.value.index == 1
        assert exc_info.value.role == "assistant"
        assert "first conversation turn must be a user" in str(exc_info.value)


def test_first_turn_inline_system_then_user_is_fine_for_openai():
    compiled = compile_messages(
        [SYSTEM, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER],
        OPENAI_CAPABILITIES,
    )
    assert compiled.messages[0]["role"] == "system"


@pytest.mark.parametrize("capabilities", ALL_CAPABILITIES)
def test_last_message_must_be_user(capabilities):
    messages = [SYSTEM, FEW_SHOT_USER, FEW_SHOT_ASSISTANT]

    with pytest.raises(MessageCompileError) as exc_info:
        compile_messages(messages, capabilities)
    assert exc_info.value.index == 2
    assert exc_info.value.role == "assistant"
    assert "last message of a completion request must be a user" in str(exc_info.value)


@pytest.mark.parametrize("capabilities", LEADING_CAPABILITIES)
def test_empty_content_diagnosed_for_strict_providers(capabilities):
    messages = [SYSTEM, FEW_SHOT_USER, {"role": "assistant", "content": ""}, LIVE_USER]

    with pytest.raises(MessageCompileError) as exc_info:
        compile_messages(messages, capabilities)
    assert exc_info.value.index == 2
    assert "empty" in str(exc_info.value)


def test_empty_content_allowed_for_openai_family():
    messages = [LIVE_USER, {"role": "assistant", "content": ""}, LIVE_USER]
    compiled = compile_messages(messages, OPENAI_CAPABILITIES)
    assert compiled.messages == messages


@pytest.mark.parametrize("capabilities", ALL_CAPABILITIES)
def test_unknown_role_is_diagnosed(capabilities):
    messages = [
        {"role": "tool", "content": "tool output"},
        LIVE_USER,
    ]

    with pytest.raises(MessageCompileError) as exc_info:
        compile_messages(messages, capabilities)
    assert exc_info.value.index == 0
    assert exc_info.value.role == "tool"
    assert "unsupported role" in str(exc_info.value)


@pytest.mark.parametrize("capabilities", ALL_CAPABILITIES)
def test_empty_conversation_is_diagnosed(capabilities):
    with pytest.raises(MessageCompileError) as exc_info:
        compile_messages([], capabilities)
    assert "empty" in str(exc_info.value)


@pytest.mark.parametrize("capabilities", LEADING_CAPABILITIES)
def test_only_system_messages_is_diagnosed(capabilities):
    with pytest.raises(MessageCompileError) as exc_info:
        compile_messages([SYSTEM, SYSTEM_2], capabilities)
    assert "no user" in str(exc_info.value)


def test_compilation_does_not_mutate_input():
    messages = [SYSTEM, SYSTEM_2, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER]
    snapshot = copy.deepcopy(messages)
    message_objects = list(messages)

    compiled = compile_messages(messages, GOOGLE_CAPABILITIES)

    assert messages == snapshot
    assert list(messages) == message_objects
    # Returned turns are copies, never the session's own dicts.
    assert all(
        compiled is not original
        for compiled, original in zip(compiled.messages, messages[2:])
    )


def test_compile_error_is_a_bad_request_error():
    with pytest.raises(BadRequestError):
        compile_messages(
            [SYSTEM, FEW_SHOT_USER, SYSTEM_2, LIVE_USER], GOOGLE_CAPABILITIES
        )
