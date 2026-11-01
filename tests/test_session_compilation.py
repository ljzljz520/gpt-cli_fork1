from types import SimpleNamespace
from unittest import mock

import pytest

from gptcli.assistant import DEFAULT_ASSISTANTS, Assistant
from gptcli.providers.compiler import MessageCompileError
from gptcli.providers.google import GoogleCompletionProvider
from gptcli.session import ChatSession


def setup_listener():
    listener = mock.MagicMock()
    streamer = mock.MagicMock()
    streamer.__enter__.return_value = streamer
    listener.response_streamer.return_value = streamer
    return listener


def gemini_assistant(messages):
    config = {
        "messages": [dict(m) for m in messages],
        "model": "gemini-2.0-flash",
    }
    return Assistant(config)


def test_default_assistant_with_strict_provider_clear_and_rerun():
    provider = GoogleCompletionProvider(client=mock.MagicMock())
    provider.client.models.generate_content.return_value = SimpleNamespace(
        text="ok", usage_metadata=None
    )
    assistant = gemini_assistant(DEFAULT_ASSISTANTS["dev"]["messages"])
    session = ChatSession(assistant, setup_listener(), stream=False)

    with mock.patch("gptcli.assistant.get_completion_provider", return_value=provider):
        # First turn: dev preset (system + few-shot) plus the live user turn.
        assert session.process_input("hi")
        first_call = provider.client.models.generate_content.call_args
        assert [c.role for c in first_call.kwargs["contents"]] == [
            "user",
            "model",
            "user",
        ]
        assert first_call.kwargs["config"].system_instruction is not None

        # Clear restores the preset messages verbatim.
        assert session.process_input(":c")

        # Second turn after clear compiles to the same request shape.
        provider.client.models.generate_content.reset_mock()
        assert session.process_input("again")
        clear_call = provider.client.models.generate_content.call_args
        assert [c.role for c in clear_call.kwargs["contents"]] == [
            "user",
            "model",
            "user",
        ]
        assert clear_call.kwargs["contents"][-1].parts[0].text == "again"

        # Rerun strips the assistant response and recompiles the same turn.
        assert session.process_input(":r")
        assert provider.client.models.generate_content.call_count == 2
        rerun_call = provider.client.models.generate_content.call_args
        assert [c.role for c in rerun_call.kwargs["contents"]] == [
            "user",
            "model",
            "user",
        ]
        assert rerun_call.kwargs["contents"][-1].parts[0].text == "again"


def test_compile_failure_rolls_back_user_message_and_blocks_rerun():
    listener = setup_listener()
    provider = GoogleCompletionProvider(client=mock.MagicMock())
    # Preset with a second system message inside the few-shot conversation:
    # not representable for Gemini, diagnosed before any request is made.
    preset = [
        {"role": "system", "content": "You are an expert."},
        {"role": "user", "content": "Be brief."},
        {"role": "system", "content": "Extra rule."},
        {"role": "assistant", "content": "Understood."},
    ]
    session = ChatSession(gemini_assistant(preset), listener, stream=False)

    with mock.patch("gptcli.assistant.get_completion_provider", return_value=provider):
        assert session.process_input("hi")

        error = listener.on_error.call_args.args[0]
        assert isinstance(error, MessageCompileError)
        assert error.index == 2
        # No request reached the client.
        provider.client.models.generate_content.assert_not_called()
        # The rejected user prompt was rolled back: rerun has nothing to do.
        listener.on_chat_rerun.reset_mock()
        assert session.process_input(":r")
        listener.on_chat_rerun.assert_called_once_with(False)
        provider.client.models.generate_content.assert_not_called()
