from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Load .env from current working directory or project root
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Stable session ID per process run for OpenCode Go prompt caching & routing (`x-opencode-session`)
_SESSION_ID = os.getenv("KINESIS_SESSION_ID", f"ses_kinesis_{uuid.uuid4().hex[:16]}")

# OpenCode Go model families by endpoint type (from https://opencode.ai/v2/docs/console/go/)
_OPENCODE_GO_ANTHROPIC_PREFIXES = ("claude-", "minimax-", "qwen")
_OPENCODE_GO_RESPONSES_PREFIXES = ("gpt-", "grok-", "muse-spark-")
_OPENCODE_GO_GEMINI_PREFIXES = ("gemini-",)


@dataclass
class Settings:
    """Configuration settings for Kinesis (Hybrid Training Agent)."""

    intervals_api_key: str = field(
        default_factory=lambda: os.getenv("INTERVALS_ICU_API_KEY", "").strip()
    )
    intervals_athlete_id: str = field(
        default_factory=lambda: os.getenv("INTERVALS_ICU_ATHLETE_ID", "0").strip() or "0"
    )
    intervals_base_url: str = field(
        default_factory=lambda: os.getenv(
            "INTERVALS_ICU_BASE_URL", "https://intervals.icu/api/v1"
        ).rstrip("/")
    )

    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "auto").strip().lower()
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "").strip()
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )

    opencode_go_api_key: str = field(
        default_factory=lambda: (
            os.getenv("OPENCODE_GO_API_KEY", "").strip()
            or os.getenv("OPENCODE_API_KEY", "").strip()
        )
    )
    opencode_go_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENCODE_GO_BASE_URL", "https://opencode.ai/inference/go"
        ).rstrip("/")
    )

    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip()
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    google_api_key: str = field(
        default_factory=lambda: (
            os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
        )
    )

    ncbi_api_key: str = field(
        default_factory=lambda: os.getenv("NCBI_API_KEY", "").strip()
    )
    ncbi_email: str = field(
        default_factory=lambda: os.getenv(
            "NCBI_EMAIL", "kinesis-agent@example.com"
        ).strip()
    )

    @property
    def has_intervals_credentials(self) -> bool:
        return bool(self.intervals_api_key)

    def resolve_provider(self) -> str:
        if self.llm_provider and self.llm_provider != "auto":
            return self.llm_provider
        if self.opencode_go_api_key or self.llm_model.startswith("opencode-go/"):
            return "opencode-go"
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        if self.google_api_key:
            return "google"
        raise ValueError(
            "No LLM API key found. Set OPENCODE_GO_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY in your .env file."
        )


def get_settings() -> Settings:
    return Settings()


def _build_opencode_go_chat_model(
    settings: Settings,
    model: str | None,
    temperature: float,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a LangChain ChatModel routed to the appropriate OpenCode Go inference endpoint."""
    raw_model = model or settings.llm_model or "kimi-k3"
    # Strip optional "opencode-go/" prefix if provided
    clean_model = (
        raw_model.split("/", 1)[1]
        if raw_model.startswith("opencode-go/")
        else raw_model
    )
    api_key = settings.opencode_go_api_key or "opencode-go-free"
    base_url = settings.opencode_go_base_url

    # Required headers per https://opencode.ai/v2/docs/console/go/#where-can-i-use-it
    go_headers = {
        "User-Agent": "kinesis-agent/0.1.0",
        "x-opencode-session": _SESSION_ID,
    }

    if clean_model.startswith(_OPENCODE_GO_ANTHROPIC_PREFIXES):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=clean_model,
            temperature=temperature,
            api_key=api_key,
            anthropic_api_url=f"{base_url}/anthropic",
            default_headers=go_headers,
            **kwargs,
        )

    if clean_model.startswith(_OPENCODE_GO_RESPONSES_PREFIXES):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=clean_model,
            temperature=temperature,
            api_key=api_key,
            base_url=f"{base_url}/openai/v1",
            use_responses_api=True,
            default_headers=go_headers,
            **kwargs,
        )

    if clean_model.startswith(_OPENCODE_GO_GEMINI_PREFIXES):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=clean_model,
            temperature=temperature,
            google_api_key=api_key,
            additional_headers=go_headers,
            **kwargs,
        )

    # Default OpenCode Go endpoint: OpenAI-compatible Chat Completions (/inference/go/openai/v1/chat/completions)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=clean_model,
        temperature=temperature,
        api_key=api_key,
        base_url=f"{base_url}/openai/v1",
        default_headers=go_headers,
        **kwargs,
    )


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a LangChain ChatModel based on .env configuration."""
    settings = get_settings()
    resolved_provider = (provider or settings.resolve_provider()).lower()
    resolved_temp = temperature if temperature is not None else settings.llm_temperature

    if resolved_provider in ("opencode-go", "opencode_go", "opencode", "go"):
        return _build_opencode_go_chat_model(
            settings=settings,
            model=model,
            temperature=resolved_temp,
            **kwargs,
        )

    if resolved_provider in ("anthropic", "claude"):
        from langchain_anthropic import ChatAnthropic

        model_name = model or settings.llm_model or "claude-sonnet-4-20250514"
        return ChatAnthropic(
            model=model_name,
            temperature=resolved_temp,
            api_key=settings.anthropic_api_key or None,
            **kwargs,
        )

    if resolved_provider in ("openai", "gpt"):
        from langchain_openai import ChatOpenAI

        model_name = model or settings.llm_model or "gpt-4o"
        return ChatOpenAI(
            model=model_name,
            temperature=resolved_temp,
            api_key=settings.openai_api_key or None,
            **kwargs,
        )

    if resolved_provider in ("google", "gemini", "google-genai"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = model or settings.llm_model or "gemini-2.5-pro"
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=resolved_temp,
            google_api_key=settings.google_api_key or None,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{resolved_provider}'. Choose 'opencode-go', 'anthropic', 'openai', or 'google'."
    )
