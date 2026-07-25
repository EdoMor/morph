"""Pluggable model providers (R-103).

Selecting a provider is configuration, never a code change::

    provider = get_provider("ollama", model="gemma3:12b")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import (
    Message,
    ModelResponse,
    Provider,
    ProviderError,
    ToolCall,
    parse_text_tool_calls,
    render_tools_for_text_protocol,
)
from .echo import EchoProvider
from .google import GoogleProvider
from .ollama import OllamaProvider

_REGISTRY: dict[str, Callable[..., Provider]] = {
    "ollama": OllamaProvider,
    "google": GoogleProvider,
    "gemma": OllamaProvider,  # alias: the default way to run Gemma
    "echo": EchoProvider,
}


def register_provider(name: str, factory: Callable[..., Provider]) -> None:
    """Register a provider factory. Third-party backends plug in here."""
    _REGISTRY[name] = factory


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def get_provider(name: str, **kwargs: Any) -> Provider:
    """Instantiate a provider by name.

    Unknown names raise :class:`ProviderError` listing what is available, so a
    typo in config is immediately diagnosable.
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ProviderError(
            f"Unknown provider {name!r}. Available: {', '.join(available_providers())}"
        )
    if factory is EchoProvider:
        kwargs = {k: v for k, v in kwargs.items() if k in {"script", "reflexes", "final_text"}}
    else:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return factory(**kwargs)


__all__ = [
    "EchoProvider",
    "GoogleProvider",
    "Message",
    "ModelResponse",
    "OllamaProvider",
    "Provider",
    "ProviderError",
    "ToolCall",
    "available_providers",
    "get_provider",
    "parse_text_tool_calls",
    "register_provider",
    "render_tools_for_text_protocol",
]
