from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from config import Settings
from metrics import record_api_call
from utils import extract_text, openai_image_part, vertex_image_part


class LLMClient:
    """Provider-neutral text and image generation client."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.model_provider
        self.client = self._create_client()

    def _create_client(self) -> Any:
        if self.provider in {"openai", "qwen"}:
            from openai import OpenAI

            return OpenAI(
                api_key=self.settings.openai_api_key or "EMPTY",
                base_url=self.settings.openai_base_url or None,
                timeout=self.settings.timeout_seconds,
            )

        from google import genai
        from google.genai import types

        http_options = types.HttpOptions(
            timeout=self.settings.timeout_seconds * 1000
        )

        if self.provider == "vertex":
            return genai.Client(
                vertexai=True,
                project=self.settings.vertex_ai_project_id,
                location=self.settings.vertex_ai_location,
                http_options=http_options,
            )
        if self.provider == "gemini":
            return genai.Client(
                api_key=self.settings.google_api_key,
                http_options=http_options,
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        images: Sequence[Path] = (),
        component: str = "unknown",
        attempt: int = 1,
    ) -> str:
        started = time.perf_counter()
        try:
            if self.provider == "openai":
                text, usage = self._generate_openai(
                    model=model,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    images=images,
                )
            elif self.provider == "qwen":
                text, usage = self._generate_qwen(
                    model=model,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    images=images,
                )
            else:
                text, usage = self._generate_google(
                    model=model,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    images=images,
                )
        except Exception as exc:
            record_api_call(
                {
                    "provider": self.provider,
                    "model": model,
                    "component": component,
                    "attempt": attempt,
                    "is_retry": attempt > 1,
                    "success": False,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "input_tokens": None,
                    "output_tokens": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise

        record_api_call(
            {
                "provider": self.provider,
                "model": model,
                "component": component,
                "attempt": attempt,
                "is_retry": attempt > 1,
                "success": True,
                "latency_seconds": round(time.perf_counter() - started, 3),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "error": None,
            }
        )
        return text

    def _generate_openai(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        images: Sequence[Path],
    ) -> tuple[str, dict[str, int | None]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(openai_image_part(image) for image in images)
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=self.settings.temperature,
            max_completion_tokens=self.settings.max_output_tokens,
            timeout=self.settings.timeout_seconds,
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("OpenAI model returned empty text.")
        return str(text).strip(), self._openai_usage(response)

    def _generate_qwen(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        images: Sequence[Path],
    ) -> tuple[str, dict[str, int | None]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(openai_image_part(image) for image in images)
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_output_tokens,
            timeout=self.settings.timeout_seconds,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.settings.enable_thinking,
                }
            },
        )
        if not completion.choices:
            raise RuntimeError("Qwen returned no choices.")
        choice = completion.choices[0]
        content_text = choice.message.content
        if not content_text:
            raise RuntimeError("Qwen returned no text.")
        if choice.finish_reason == "length":
            raise RuntimeError("Qwen output was truncated at the output token limit.")
        return str(content_text).strip(), self._openai_usage(completion)

    def _generate_google(
        self,
        *,
        model: str,
        system_prompt: str,
        prompt: str,
        images: Sequence[Path],
    ) -> tuple[str, dict[str, int | None]]:
        from google.genai import types

        contents: list[Any] = [prompt]
        contents.extend(vertex_image_part(image) for image in images)
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.settings.temperature,
                max_output_tokens=self.settings.max_output_tokens,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        candidate_tokens = self._optional_int(
            getattr(usage, "candidates_token_count", None)
        )
        thought_tokens = self._optional_int(
            getattr(usage, "thoughts_token_count", None)
        )
        output_tokens = (
            None
            if candidate_tokens is None and thought_tokens is None
            else (candidate_tokens or 0) + (thought_tokens or 0)
        )
        return extract_text(response), {
            "input_tokens": self._optional_int(
                getattr(usage, "prompt_token_count", None)
            ),
            "output_tokens": output_tokens,
        }

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return int(value) if value is not None else None

    def _openai_usage(self, response: Any) -> dict[str, int | None]:
        usage = getattr(response, "usage", None)
        return {
            "input_tokens": self._optional_int(
                getattr(usage, "prompt_tokens", None)
            ),
            "output_tokens": self._optional_int(
                getattr(usage, "completion_tokens", None)
            ),
        }
