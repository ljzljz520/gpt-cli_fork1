import os
import cohere
from cohere.types import ChatbotMessage, UserMessage
from typing import Iterator, List, Optional

from gptcli.completion import (
    CompletionEvent,
    CompletionProvider,
    Message,
    CompletionError,
    BadRequestError,
    MessageDeltaEvent,
    Pricing,
    UsageEvent,
)
from gptcli.providers.compiler import COHERE_CAPABILITIES, compile_messages

api_key = os.environ.get("COHERE_API_KEY")


def map_message(message: Message):
    if message["role"] == "assistant":
        return ChatbotMessage(message=message["content"])
    return UserMessage(message=message["content"])


class CohereCompletionProvider(CompletionProvider):
    def __init__(self, client: Optional[cohere.Client] = None):
        # Allows injecting a mock client in tests.
        self.client = client if client is not None else cohere.Client(api_key=api_key)

    def complete(
        self, messages: List[Message], args: dict, stream: bool = False
    ) -> Iterator[CompletionEvent]:
        # Compile and validate before making a request. The compiler ensures
        # the last turn is a user message and history only contains
        # user/assistant turns.
        compiled = compile_messages(messages, COHERE_CAPABILITIES)

        kwargs = {}
        if "temperature" in args:
            kwargs["temperature"] = args["temperature"]
        if "top_p" in args:
            kwargs["p"] = args["top_p"]

        model = args["model"]

        if compiled.instruction is not None:
            kwargs["preamble"] = compiled.instruction

        message = compiled.messages[-1]

        chat_history = [map_message(m) for m in compiled.messages[:-1]]

        try:
            if stream:
                response_iter = self.client.chat_stream(
                    chat_history=chat_history,
                    message=message["content"],
                    model=model,
                    **kwargs,
                )

                for response in response_iter:
                    if response.event_type == "text-generation":
                        yield MessageDeltaEvent(response.text)

                    if (
                        response.event_type == "stream-end"
                        and response.response.meta
                        and response.response.meta.tokens
                        and (pricing := COHERE_PRICING.get(args["model"]))
                    ):
                        input_tokens = int(
                            response.response.meta.tokens.input_tokens or 0
                        )
                        output_tokens = int(
                            response.response.meta.tokens.output_tokens or 0
                        )
                        total_tokens = input_tokens + output_tokens

                        yield UsageEvent.with_pricing(
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=total_tokens,
                            pricing=pricing,
                        )

            else:
                response = self.client.chat(
                    chat_history=chat_history,
                    message=message["content"],
                    model=model,
                    **kwargs,
                )
                yield MessageDeltaEvent(response.text)

                if (
                    response.meta
                    and response.meta.tokens
                    and (pricing := COHERE_PRICING.get(args["model"]))
                ):
                    input_tokens = int(response.meta.tokens.input_tokens or 0)
                    output_tokens = int(response.meta.tokens.output_tokens or 0)
                    total_tokens = input_tokens + output_tokens

                    yield UsageEvent.with_pricing(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=total_tokens,
                        pricing=pricing,
                    )

        except cohere.BadRequestError as e:
            raise BadRequestError(e.body) from e
        except (
            cohere.TooManyRequestsError,
            cohere.InternalServerError,
            cohere.core.api_error.ApiError,  # type: ignore
        ) as e:
            raise CompletionError(e.body) from e


COHERE_PRICING: dict[str, Pricing] = {
    "command-r": {
        "prompt": 0.5 / 1_000_000,
        "response": 1.5 / 1_000_000,
    },
    "command-r-plus": {
        "prompt": 3.0 / 1_000_000,
        "response": 15.0 / 1_000_000,
    },
}
