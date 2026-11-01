import os
from google import genai
from google.genai import types

from typing import Iterator, List, Optional

from gptcli.completion import (
    CompletionEvent,
    CompletionProvider,
    Message,
    MessageDeltaEvent,
    Pricing,
    UsageEvent,
)
from gptcli.providers.compiler import GOOGLE_CAPABILITIES, compile_messages

ROLE_MAP = {
    "user": "user",
    "assistant": "model",
}


api_key = os.environ.get("GEMINI_API_KEY")


class GoogleCompletionProvider(CompletionProvider):
    def __init__(self, client=None):
        # Allows injecting a mock client in tests. The real client is created
        # lazily so config overrides of the module-level ``api_key`` apply.
        self.client = client

    def _get_client(self):
        if self.client is not None:
            return self.client
        return genai.Client(api_key=api_key)

    def complete(
        self, messages: List[Message], args: dict, stream: bool = False
    ) -> Iterator[CompletionEvent]:
        # Compile and validate before creating a client or making a request.
        compiled = compile_messages(messages, GOOGLE_CAPABILITIES)
        client = self._get_client()
        model = args["model"]

        contents = [
            types.Content(
                role=ROLE_MAP[m["role"]],
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in compiled.messages
        ]

        generate_content_config = types.GenerateContentConfig(
            system_instruction=compiled.instruction,
            temperature=args.get("temperature"),
            top_p=args.get("top_p"),
            thinking_config=(
                types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=args.get("thinking_budget"),
                )
                if args.get("thinking_budget")
                else None
            ),
            response_mime_type="text/plain",
        )

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if stream:
            response = client.models.generate_content_stream(
                model=model,
                contents=list(contents),
                config=generate_content_config,
            )

            for chunk in response:
                if chunk.usage_metadata:
                    prompt_tokens = chunk.usage_metadata.prompt_token_count or 0
                    completion_tokens = chunk.usage_metadata.candidates_token_count or 0
                    total_tokens = prompt_tokens + completion_tokens
                yield MessageDeltaEvent(chunk.text or "")

        else:
            response = client.models.generate_content(
                model=model,
                contents=list(contents),
                config=generate_content_config,
            )
            yield MessageDeltaEvent(response.text or "")

            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count or 0
                completion_tokens = response.usage_metadata.candidates_token_count or 0
                total_tokens = prompt_tokens + completion_tokens

        pricing = get_gemini_pricing(model, prompt_tokens)
        if pricing:
            yield UsageEvent.with_pricing(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                pricing=pricing,
            )


def get_gemini_pricing(model: str, prompt_tokens: int) -> Optional[Pricing]:
    if model.startswith("gemini-1.5-flash-8b"):
        return {
            "prompt": (0.0375 if prompt_tokens < 128000 else 0.075) / 1_000_000,
            "response": (0.15 if prompt_tokens < 128000 else 0.30) / 1_000_000,
        }
    if model.startswith("gemini-1.5-flash"):
        return {
            "prompt": (0.075 if prompt_tokens < 128000 else 0.15) / 1_000_000,
            "response": (0.30 if prompt_tokens < 128000 else 0.60) / 1_000_000,
        }
    elif model.startswith("gemini-1.5-pro"):
        return {
            "prompt": (1.25 if prompt_tokens < 128000 else 2.50) / 1_000_000,
            "response": (5.0 if prompt_tokens < 128000 else 10.0) / 1_000_000,
        }
    elif model.startswith("gemini-2.0-flash-lite"):
        return {
            "prompt": 0.075 / 1_000_000,
            "response": 0.30 / 1_000_000,
        }
    elif model.startswith("gemini-2.0-flash"):
        return {
            "prompt": 0.10 / 1_000_000,
            "response": 0.40 / 1_000_000,
        }
    elif model.startswith("gemini-2.5-pro"):
        return {
            "prompt": (1.25 if prompt_tokens < 200000 else 2.50) / 1_000_000,
            "response": (10.0 if prompt_tokens < 200000 else 15.0) / 1_000_000,
        }
    elif model.startswith("gemini-pro"):
        return {
            "prompt": 0.50 / 1_000_000,
            "response": 1.50 / 1_000_000,
        }
    else:
        return None
