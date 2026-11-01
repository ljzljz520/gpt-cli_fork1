import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gptcli.assistant import DEFAULT_ASSISTANTS
from gptcli.completion import MessageDeltaEvent, UsageEvent
from gptcli.providers.anthropic import AnthropicCompletionProvider
from gptcli.providers.azure_openai import AzureOpenAICompletionProvider
from gptcli.providers.cohere import CohereCompletionProvider
from gptcli.providers.compiler import MessageCompileError
from gptcli.providers.google import GoogleCompletionProvider
from gptcli.providers.openai import OpenAICompletionProvider

SYSTEM = {"role": "system", "content": "You are an expert."}
SYSTEM_2 = {"role": "system", "content": "Always answer in Chinese."}
FEW_SHOT_USER = {"role": "user", "content": "Be brief."}
FEW_SHOT_ASSISTANT = {"role": "assistant", "content": "Understood."}
LIVE_USER = {"role": "user", "content": "Hello?"}

PRESET_MESSAGES = [SYSTEM, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER]


def drain(iterator):
    return list(iterator)


def make_openai_provider():
    provider = OpenAICompletionProvider(api_key="sk-test")
    provider.client = MagicMock()
    return provider


def make_azure_provider():
    # Bypass __init__ which requires Azure environment configuration.
    provider = AzureOpenAICompletionProvider.__new__(AzureOpenAICompletionProvider)
    provider.client = MagicMock()
    return provider


def make_anthropic_provider():
    return AnthropicCompletionProvider(client=MagicMock())


def make_cohere_provider():
    return CohereCompletionProvider(client=MagicMock())


def make_google_provider():
    return GoogleCompletionProvider(client=MagicMock())


# ---------------------------------------------------------------------------
# OpenAI (Responses API): system messages stay inline, in original order.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_openai_request_presets_preserved_inline(stream):
    provider = make_openai_provider()
    messages = copy.deepcopy(PRESET_MESSAGES)

    if stream:
        provider.client.responses.create.return_value = iter(
            [
                SimpleNamespace(type="response.output_text.delta", delta="an"),
                SimpleNamespace(type="response.output_text.delta", delta="swer"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        usage=SimpleNamespace(input_tokens=3, output_tokens=2)
                    ),
                ),
            ]
        )
    else:
        provider.client.responses.create.return_value = SimpleNamespace(
            output_text="answer",
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        )

    args = {"model": "gpt-4o-mini", "temperature": 0.7, "top_p": 1.0}
    events = drain(provider.complete(messages, args, stream=stream))

    kwargs = provider.client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["stream"] is stream
    assert kwargs["store"] is False
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 1.0

    # The full preset conversation, system message inline and in order.
    assert kwargs["input"] == messages
    assert [m["role"] for m in kwargs["input"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    # Compiled dicts are copies, not the session's own message objects.
    assert kwargs["input"][0] is not messages[0]

    deltas = [e for e in events if isinstance(e, MessageDeltaEvent)]
    assert "".join(e.text for e in deltas) == "answer"
    assert any(isinstance(e, UsageEvent) for e in events)


def test_openai_multiple_systems_stay_inline():
    provider = make_openai_provider()
    messages = [SYSTEM, SYSTEM_2, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER]
    provider.client.responses.create.return_value = SimpleNamespace(
        output_text="answer", usage=None
    )

    drain(provider.complete(messages, {"model": "gpt-4o-mini"}, stream=False))

    sent = provider.client.responses.create.call_args.kwargs["input"]
    assert [m["role"] for m in sent] == [
        "system",
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert sent[1]["content"] == SYSTEM_2["content"]


def test_openai_rejects_before_request():
    provider = make_openai_provider()
    messages = [{"role": "tool", "content": "x"}, LIVE_USER]

    with pytest.raises(MessageCompileError):
        drain(provider.complete(messages, {"model": "gpt-4o-mini"}, stream=False))

    provider.client.responses.create.assert_not_called()


def test_azure_uses_same_inline_compilation():
    provider = make_azure_provider()
    messages = copy.deepcopy(PRESET_MESSAGES)
    provider.client.responses.create.return_value = SimpleNamespace(
        output_text="answer", usage=None
    )

    drain(provider.complete(messages, {"model": "gpt-4o-mini"}, stream=False))

    sent = provider.client.responses.create.call_args.kwargs["input"]
    assert sent[0]["role"] == "system"
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]


def test_azure_diagnostics_mention_azure():
    provider = make_azure_provider()
    with pytest.raises(MessageCompileError) as exc_info:
        drain(
            provider.complete(
                [{"role": "tool", "content": "x"}, LIVE_USER],
                {"model": "gpt-4o-mini"},
                stream=False,
            )
        )
    assert exc_info.value.provider == "azure"


# ---------------------------------------------------------------------------
# Anthropic: leading system(s) -> system slot, rest as alternating messages.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_request_leading_system_and_order(stream):
    provider = make_anthropic_provider()
    messages = copy.deepcopy(PRESET_MESSAGES)
    model = "claude-3-haiku-20240307"

    if stream:
        context = MagicMock()
        context.__enter__.return_value = iter(
            [
                SimpleNamespace(
                    type="message_start",
                    message=SimpleNamespace(usage=SimpleNamespace(input_tokens=3)),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="answer"),
                ),
                SimpleNamespace(
                    type="message_delta",
                    usage=SimpleNamespace(output_tokens=2),
                ),
            ]
        )
        provider.client.messages.stream.return_value = context
    else:
        provider.client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="answer")],
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        )

    args = {"model": model, "temperature": 0.7, "top_p": 1.0}
    events = drain(provider.complete(messages, args, stream=stream))

    call = (
        provider.client.messages.stream.call_args
        if stream
        else provider.client.messages.create.call_args
    )
    kwargs = call.kwargs
    assert kwargs["model"] == model
    assert kwargs["system"] == SYSTEM["content"]
    assert kwargs["messages"] == [
        FEW_SHOT_USER,
        FEW_SHOT_ASSISTANT,
        LIVE_USER,
    ]
    assert all(m["role"] != "system" for m in kwargs["messages"])
    # Compiled dicts are copies, not the session's own message objects.
    assert kwargs["messages"][0] is not messages[1]

    deltas = [e for e in events if isinstance(e, MessageDeltaEvent)]
    assert "".join(e.text for e in deltas) == "answer"


def test_anthropic_multiple_leading_systems_merged_into_system_slot():
    provider = make_anthropic_provider()
    messages = [SYSTEM, SYSTEM_2, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER]
    provider.client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="answer")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )

    drain(
        provider.complete(messages, {"model": "claude-3-haiku-20240307"}, stream=False)
    )

    kwargs = provider.client.messages.create.call_args.kwargs
    assert kwargs["system"] == "You are an expert.\n\nAlways answer in Chinese."
    assert all(m["role"] != "system" for m in kwargs["messages"])


def test_anthropic_interspersed_system_rejected_before_request():
    provider = make_anthropic_provider()
    messages = [SYSTEM, FEW_SHOT_USER, SYSTEM_2, FEW_SHOT_ASSISTANT, LIVE_USER]

    with pytest.raises(MessageCompileError) as exc_info:
        drain(
            provider.complete(
                messages, {"model": "claude-3-haiku-20240307"}, stream=False
            )
        )

    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.index == 2
    provider.client.messages.create.assert_not_called()
    provider.client.messages.stream.assert_not_called()


# ---------------------------------------------------------------------------
# Cohere: leading system(s) -> preamble, history USER/CHATBOT, last message
# split out as `message`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_cohere_request_preamble_history_and_message(stream):
    provider = make_cohere_provider()
    messages = copy.deepcopy(PRESET_MESSAGES)
    model = "command-r"

    if stream:
        provider.client.chat_stream.return_value = iter(
            [
                SimpleNamespace(event_type="text-generation", text="answer"),
                SimpleNamespace(
                    event_type="stream-end",
                    response=SimpleNamespace(
                        meta=SimpleNamespace(
                            tokens=SimpleNamespace(input_tokens=3, output_tokens=2)
                        )
                    ),
                ),
            ]
        )
    else:
        provider.client.chat.return_value = SimpleNamespace(
            text="answer",
            meta=SimpleNamespace(
                tokens=SimpleNamespace(input_tokens=3, output_tokens=2)
            ),
        )

    args = {"model": model, "temperature": 0.7, "top_p": 1.0}
    events = drain(provider.complete(messages, args, stream=stream))

    call = (
        provider.client.chat_stream.call_args
        if stream
        else provider.client.chat.call_args
    )
    kwargs = call.kwargs
    assert kwargs["model"] == model
    assert kwargs["preamble"] == SYSTEM["content"]
    assert kwargs["message"] == LIVE_USER["content"]

    history = kwargs["chat_history"]
    assert [(type(m).__name__, m.role, m.message) for m in history] == [
        ("UserMessage", "USER", FEW_SHOT_USER["content"]),
        ("ChatbotMessage", "CHATBOT", FEW_SHOT_ASSISTANT["content"]),
    ]

    deltas = [e for e in events if isinstance(e, MessageDeltaEvent)]
    assert "".join(e.text for e in deltas) == "answer"


def test_cohere_multiple_leading_systems_merged_into_preamble():
    provider = make_cohere_provider()
    messages = [SYSTEM, SYSTEM_2, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER]
    provider.client.chat.return_value = SimpleNamespace(text="answer", meta=None)

    drain(provider.complete(messages, {"model": "command-r"}, stream=False))

    kwargs = provider.client.chat.call_args.kwargs
    assert kwargs["preamble"] == "You are an expert.\n\nAlways answer in Chinese."
    assert all(
        not type(m).__name__.startswith("System") for m in kwargs["chat_history"]
    )


def test_cohere_interspersed_system_rejected_before_request():
    provider = make_cohere_provider()
    messages = [SYSTEM, FEW_SHOT_USER, SYSTEM_2, LIVE_USER]

    with pytest.raises(MessageCompileError) as exc_info:
        drain(provider.complete(messages, {"model": "command-r"}, stream=False))

    assert exc_info.value.provider == "cohere"
    assert exc_info.value.index == 2
    provider.client.chat.assert_not_called()
    provider.client.chat_stream.assert_not_called()


# ---------------------------------------------------------------------------
# Google Gemini: leading system(s) -> system_instruction, contents alternate
# user/model and preserve few-shot order.
# ---------------------------------------------------------------------------


def _gemini_usage():
    return SimpleNamespace(prompt_token_count=3, candidates_token_count=2)


@pytest.mark.parametrize("stream", [False, True])
def test_google_request_system_instruction_and_contents(stream):
    provider = make_google_provider()
    messages = copy.deepcopy(PRESET_MESSAGES)
    model = "gemini-2.0-flash"

    if stream:
        provider.client.models.generate_content_stream.return_value = iter(
            [SimpleNamespace(text="answer", usage_metadata=_gemini_usage())]
        )
    else:
        provider.client.models.generate_content.return_value = SimpleNamespace(
            text="answer", usage_metadata=_gemini_usage()
        )

    args = {"model": model, "temperature": 0.7, "top_p": 1.0}
    events = drain(provider.complete(messages, args, stream=stream))

    call = (
        provider.client.models.generate_content_stream.call_args
        if stream
        else provider.client.models.generate_content.call_args
    )
    kwargs = call.kwargs
    assert kwargs["model"] == model
    assert kwargs["config"].system_instruction == SYSTEM["content"]

    contents = kwargs["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert [c.parts[0].text for c in contents] == [
        FEW_SHOT_USER["content"],
        FEW_SHOT_ASSISTANT["content"],
        LIVE_USER["content"],
    ]

    deltas = [e for e in events if isinstance(e, MessageDeltaEvent)]
    assert "".join(e.text for e in deltas) == "answer"


def test_google_multiple_leading_systems_merged_into_instruction():
    provider = make_google_provider()
    messages = [SYSTEM, SYSTEM_2, FEW_SHOT_USER, FEW_SHOT_ASSISTANT, LIVE_USER]
    provider.client.models.generate_content.return_value = SimpleNamespace(
        text="answer", usage_metadata=None
    )

    drain(provider.complete(messages, {"model": "gemini-2.0-flash"}, stream=False))

    config = provider.client.models.generate_content.call_args.kwargs["config"]
    assert (
        config.system_instruction == "You are an expert.\n\nAlways answer in Chinese."
    )
    contents = provider.client.models.generate_content.call_args.kwargs["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]


def test_google_interspersed_system_rejected_before_request():
    provider = make_google_provider()
    # The motivating failure: a second system message used to raise a local
    # KeyError; it must now produce a specific pre-request diagnostic.
    messages = [SYSTEM, FEW_SHOT_USER, SYSTEM_2, FEW_SHOT_ASSISTANT, LIVE_USER]

    with pytest.raises(MessageCompileError) as exc_info:
        drain(provider.complete(messages, {"model": "gemini-2.0-flash"}, stream=False))

    assert exc_info.value.provider == "gemini"
    assert exc_info.value.index == 2
    assert "system" in str(exc_info.value)
    provider.client.models.generate_content.assert_not_called()
    provider.client.models.generate_content_stream.assert_not_called()


# ---------------------------------------------------------------------------
# Default assistants and session originals must stay compatible / untouched.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_default_dev_assistant_compiles_for_every_provider(stream):
    factories = [
        make_openai_provider,
        make_azure_provider,
        make_anthropic_provider,
        make_cohere_provider,
        make_google_provider,
    ]
    for factory in factories:
        provider = factory()
        messages = DEFAULT_ASSISTANTS["dev"]["messages"][:] + [LIVE_USER]

        if isinstance(provider, GoogleCompletionProvider):
            model = "gemini-2.0-flash"
            target = provider.client.models
            target.generate_content.return_value = SimpleNamespace(
                text="answer", usage_metadata=None
            )
            target.generate_content_stream.return_value = iter(
                [SimpleNamespace(text="answer", usage_metadata=None)]
            )
        elif isinstance(provider, CohereCompletionProvider):
            model = "command-r"
            provider.client.chat.return_value = SimpleNamespace(
                text="answer", meta=None
            )
            provider.client.chat_stream.return_value = iter(
                [SimpleNamespace(event_type="text-generation", text="answer")]
            )
        elif isinstance(provider, AnthropicCompletionProvider):
            model = "claude-3-haiku-20240307"
            provider.client.messages.create.return_value = SimpleNamespace(
                content=[SimpleNamespace(type="text", text="answer")],
                usage=SimpleNamespace(input_tokens=3, output_tokens=2),
            )
            context = MagicMock()
            context.__enter__.return_value = iter(
                [
                    SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="text_delta", text="answer"),
                    )
                ]
            )
            provider.client.messages.stream.return_value = context
        else:
            model = "gpt-4o-mini"
            if stream:
                provider.client.responses.create.return_value = iter(
                    [SimpleNamespace(type="response.output_text.delta", delta="answer")]
                )
            else:
                provider.client.responses.create.return_value = SimpleNamespace(
                    output_text="answer", usage=None
                )

        events = drain(
            provider.complete(
                messages,
                {"model": model, "temperature": 0.7, "top_p": 1.0},
                stream=stream,
            )
        )
        deltas = [e for e in events if isinstance(e, MessageDeltaEvent)]
        assert (
            "".join(e.text for e in deltas) == "answer"
        ), f"{factory.__name__} stream={stream}"


def test_compilation_via_providers_does_not_modify_session_originals():
    provider = make_anthropic_provider()
    messages = copy.deepcopy(PRESET_MESSAGES)
    snapshot = copy.deepcopy(messages)
    provider.client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="answer")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )

    drain(
        provider.complete(messages, {"model": "claude-3-haiku-20240307"}, stream=False)
    )

    assert messages == snapshot
    assert [m["role"] for m in messages] == [r["role"] for r in snapshot]
