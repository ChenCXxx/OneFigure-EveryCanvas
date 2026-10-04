from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PARSE_AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(PARSE_AGENT_DIR / ".env")

def get_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def get_int(name: str, default: int) -> int:
    value = get_str(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer: {value!r}") from exc


def get_float(name: str, default: float) -> float:
    value = get_str(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number: {value!r}") from exc


def get_bool(name: str, default: bool) -> bool:
    value = get_str(name)
    if not value:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean: {value!r}")


def get_model_pricing() -> dict[str, dict[str, float]]:
    raw = get_str("MODEL_PRICING_JSON", "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MODEL_PRICING_JSON must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("MODEL_PRICING_JSON must be a JSON object.")

    pricing: dict[str, dict[str, float]] = {}
    for model, values in parsed.items():
        if not isinstance(values, dict):
            raise ValueError(f"Pricing for model {model!r} must be an object.")
        try:
            input_price = float(values["input"])
            output_price = float(values["output"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Pricing for {model!r} requires numeric input/output values."
            ) from exc
        if input_price < 0 or output_price < 0:
            raise ValueError(f"Pricing for {model!r} cannot be negative.")
        pricing[str(model)] = {
            "input": input_price,
            "output": output_price,
        }
        optional_keys = (
            "threshold_tokens",
            "input_above_threshold",
            "output_above_threshold",
        )
        for key in optional_keys:
            if key not in values:
                continue
            try:
                number = float(values[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Pricing field {key!r} for {model!r} must be numeric."
                ) from exc
            if number < 0:
                raise ValueError(
                    f"Pricing field {key!r} for {model!r} cannot be negative."
                )
            pricing[str(model)][key] = number
    return pricing


@dataclass(frozen=True)
class Settings:
    # Model
    model_provider: str
    parser_model: str
    style_model: str
    critic_model: str
    layout_model: str

    # API
    openai_api_key: str
    openai_base_url: str
    google_api_key: str

    # Vertex AI
    vertex_ai_project_id: str
    vertex_ai_location: str
    google_application_credentials: str

    # Generation
    temperature: float
    max_output_tokens: int
    timeout_seconds: int
    enable_thinking: bool
    model_pricing: dict[str, dict[str, float]]

    # External tools
    drawio_bin: str


def load_settings() -> Settings:
    settings = Settings(
        model_provider=get_str("MODEL_PROVIDER", "openai").lower(),

        parser_model=get_str("PARSER_MODEL", "gpt-5.5"),
        style_model=get_str("STYLE_MODEL", get_str("PARSER_MODEL", "gpt-5.5")),
        critic_model=get_str("CRITIC_MODEL", "gpt-5.5"),
        layout_model=get_str("LAYOUT_MODEL", "gpt-5.5"),
        
        openai_api_key=get_str("OPENAI_API_KEY"),
        openai_base_url=get_str("OPENAI_BASE_URL"),
        google_api_key=get_str("GOOGLE_API_KEY"),

        vertex_ai_project_id=get_str("VERTEX_AI_PROJECT_ID"),
        vertex_ai_location=get_str("VERTEX_AI_LOCATION", "global"),
        google_application_credentials=get_str(
            "GOOGLE_APPLICATION_CREDENTIALS"
        ),

        temperature=get_float("MODEL_TEMPERATURE", 0.1),
        max_output_tokens=get_int(
            "MODEL_MAX_OUTPUT_TOKENS",
            8192,
        ),
        timeout_seconds=get_int(
            "MODEL_TIMEOUT_SECONDS",
            600,
        ),
        enable_thinking=get_bool("QWEN_ENABLE_THINKING", False),
        model_pricing=get_model_pricing(),

        drawio_bin=get_str("DRAWIO_BIN"),
    )

    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    if settings.model_provider not in {"openai", "qwen", "vertex", "gemini"}:
        raise ValueError(
            "MODEL_PROVIDER must be 'openai', 'qwen', 'vertex', or 'gemini'"
        )

    if settings.model_provider in {"openai", "qwen"}:
        if not settings.openai_api_key and not settings.openai_base_url:
            raise EnvironmentError(
                "Set OPENAI_API_KEY or OPENAI_BASE_URL "
                f"when MODEL_PROVIDER={settings.model_provider}."
            )

    if settings.model_provider == "vertex":
        if not settings.vertex_ai_project_id:
            raise EnvironmentError(
                "Set VERTEX_AI_PROJECT_ID "
                "when MODEL_PROVIDER=vertex."
            )

    if settings.model_provider == "gemini":
        if not settings.google_api_key:
            raise EnvironmentError(
                "Set GOOGLE_API_KEY when MODEL_PROVIDER=gemini."
            )

    if settings.temperature < 0:
        raise ValueError("MODEL_TEMPERATURE must be >= 0")

    if settings.max_output_tokens <= 0:
        raise ValueError(
            "MODEL_MAX_OUTPUT_TOKENS must be > 0"
        )

    if settings.timeout_seconds <= 0:
        raise ValueError(
            "MODEL_TIMEOUT_SECONDS must be > 0"
        )
