"""Benchmark package with lazy optional LLM/image imports."""

__all__ = [
    "ImagePart",
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "TextPart",
    "load_config",
]


def __getattr__(name: str):
    if name in __all__:
        from . import llm

        return getattr(llm, name)
    raise AttributeError(name)
