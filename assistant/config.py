"""Application configuration and OpenRouter client setup."""

import os
from collections.abc import Mapping

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"

load_dotenv()


def is_free_model(model_name: str) -> bool:
    """Accept only OpenRouter's free router or explicitly free model IDs."""
    model_name = model_name.strip()
    return model_name == DEFAULT_MODEL or model_name.endswith(":free")


def get_model_name(secrets: Mapping[str, str] | None = None) -> str:
    model_name = os.getenv("OPENROUTER_MODEL", "").strip()
    if not model_name and secrets is not None:
        model_name = str(secrets.get("OPENROUTER_MODEL", "")).strip()
    if not model_name and secrets is None:
        try:
            import streamlit as st

            model_name = str(st.secrets.get("OPENROUTER_MODEL", "")).strip()
        except Exception:
            pass
    model_name = model_name or DEFAULT_MODEL
    if not is_free_model(model_name):
        raise ValueError(
            "OPENROUTER_MODEL must be 'openrouter/free' or an OpenRouter model ID ending in ':free'."
        )
    return model_name


def get_api_key(secrets: Mapping[str, str] | None = None) -> str | None:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    if secrets is not None:
        key = str(secrets.get("OPENROUTER_API_KEY", "")).strip()
        return key or None
    try:
        import streamlit as st

        key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
        return key or None
    except Exception:
        return None


def build_chat_model(api_key: str, model_name: str | None = None) -> ChatOpenAI:
    model_name = model_name or get_model_name()
    if not is_free_model(model_name):
        raise ValueError(
            "Only 'openrouter/free' or an explicitly free OpenRouter model can be used."
        )
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,
        timeout=45,
        max_retries=1,
        default_headers={"HTTP-Referer": "https://streamlit.io", "X-Title": "Interview Prep Assistant"},
    )