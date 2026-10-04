"""Provider-neutral text and image client with ordered text/image parts."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from .utils import preprocess_image
except ImportError:
    from utils import preprocess_image


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "gemini"
    model: str = "gemini-2.5-pro"
    temperature: float = 0.1
    max_output_tokens: int = 8192
    timeout_seconds: float = 600


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    usage: dict[str, int | None]
    raw: Any = None


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    path: Path


LLMPart = TextPart | ImagePart


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_config()
        self.provider = self.config.provider.lower()
        self.client = self._create_client()

    def _create_client(self) -> Any:
        if self.provider == "mock":
            return None

        if self.provider == "openai":
            from openai import OpenAI

            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                base_url=os.getenv("OPENAI_BASE_URL") or None,
                timeout=self.config.timeout_seconds,
            )

        from google import genai
        from google.genai import types

        http_options = types.HttpOptions(
            timeout=int(self.config.timeout_seconds * 1000)
        )

        if self.provider == "gemini":
            return genai.Client(
                api_key=os.getenv(
                    "GEMINI_API_KEY",
                    os.getenv("GOOGLE_API_KEY"),
                ),
                http_options=http_options,
            )

        if self.provider == "vertex":
            return genai.Client(
                vertexai=True,
                project=os.getenv("VERTEX_AI_PROJECT_ID"),
                location=os.getenv("VERTEX_AI_LOCATION", "global"),
                http_options=http_options,
            )

        raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")

    def generate(
        self,
        *,
        prompt: str = "",
        images: Sequence[Path] = (),
        parts: Sequence[LLMPart] | None = None,
        system_prompt: str = "",
        model: str | None = None,
    ) -> LLMResponse:
        """Generate text from plain images or ordered text/image parts."""
        ordered_parts = (
            list(parts)  # for style and layout benchmark
            if parts is not None
            else [ImagePart(path) for path in images]  # for halluc and rel benchmark
        )

        if prompt:
            ordered_parts.append(TextPart(prompt))

        selected_model = model or self.config.model

        if self.provider == "mock":
            return LLMResponse(
                text='{"ok": true, "provider": "mock"}',
                model=selected_model,
                provider="mock",
                usage={},
            )

        if self.provider == "openai":
            return self._generate_openai(
                selected_model,
                system_prompt,
                ordered_parts,
            )

        return self._generate_google(
            selected_model,
            system_prompt,
            ordered_parts,
        )

    def _generate_openai(
        self,
        model: str,
        system_prompt: str,
        parts: Sequence[LLMPart],
    ) -> LLMResponse:
        content: list[dict[str, Any]] = []

        for part in parts:
            # text part -> type: "text"
            if isinstance(part, TextPart):
                content.append({
                    "type": "text",
                    "text": part.text,
                })
                continue
            
            # image part -> type: "image_url" with base64-encoded data
            data, mime = preprocess_image(part.path)
            encoded = base64.b64encode(data).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{encoded}",
                },
            })

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout_seconds,
        )

        text = response.choices[0].message.content if response.choices else None
        if not text:
            raise RuntimeError("OpenAI returned empty text")

        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=str(text).strip(),
            model=model,
            provider="openai",
            usage={
                "input_tokens": _int(
                    getattr(usage, "prompt_tokens", None)
                ),
                "output_tokens": _int(
                    getattr(usage, "completion_tokens", None)
                ),
            },
            raw=response,
        )

    def _generate_google(
        self,
        model: str,
        system_prompt: str,
        parts: Sequence[LLMPart],
    ) -> LLMResponse:
        from google.genai import types

        contents: list[Any] = []

        for part in parts:
            # text part -> add as a string
            if isinstance(part, TextPart):
                contents.append(part.text)
                continue
            
            # image part -> convert to bytes and add as a Part
            data, mime = preprocess_image(part.path)
            contents.append(types.Part.from_bytes(
                data=data,
                mime_type=mime,
            ))
        
        # Call the Google API
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            ),
        )

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Google model returned empty text")

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=str(text).strip(),
            model=model,
            provider=self.provider,
            usage={
                "input_tokens": _int(
                    getattr(usage, "prompt_token_count", None)
                ),
                "output_tokens": _int(
                    getattr(usage, "candidates_token_count", None)
                ),
            },
            raw=response,
        )


def load_config() -> LLMConfig:
    _load_env_file()
    return LLMConfig(
        provider=os.getenv(
            "LLM_PROVIDER",
            os.getenv("MODEL_PROVIDER", "gemini"),
        ).lower(),
        model=os.getenv("LLM_MODEL", "gemini-2.5-pro"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_output_tokens=int(
            os.getenv("LLM_MAX_OUTPUT_TOKENS", "8192")
        ),
        timeout_seconds=float(
            os.getenv("LLM_TIMEOUT_SECONDS", "600")
        ),
    )


def _load_env_file() -> None:
    """Load benchmark/.env without requiring python-dotenv.

    Explicit process environment variables take precedence. This keeps the
    README setup instructions useful when the pipeline is launched directly.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()

        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
            for character in key
        ):
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _int(value: Any) -> int | None:
    return int(value) if value is not None else None
