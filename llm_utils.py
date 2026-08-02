import json
import os
import urllib.request

import streamlit as st


def _probe(host: str, token: str | None = None) -> dict | None:
    """Probe an Ollama server's /api/tags. Returns {'models': [...]} or None."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(f"{host.rstrip('/')}/api/tags", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


LOCAL_OLLAMA_HOST = "http://localhost:11434"
LOCAL_MODEL_PRIORITY = ["deepseek-v4-flash:cloud", "gpt-oss:20b-cloud", "glm-4.6:cloud"]

OLLAMA_CLOUD_HOST = "https://ollama.com"
CLOUD_MODEL_PRIORITY = ["deepseek-v4-flash", "gpt-oss:20b", "gpt-oss:120b", "glm-5.1"]

GROQ_MODEL = "llama-3.1-8b-instant"
OPENROUTER_MODEL = "qwen/qwen3-4b:free"


def _build_chain(models: list[str], temperature: float, groq, *, base_url: str | None = None,
                 client_kwargs: dict | None = None):
    """Build an Ollama chain over `models` with Groq as the final fallback."""
    from langchain_ollama import ChatOllama

    chain = ChatOllama(model=models[0], base_url=base_url, temperature=temperature,
                       client_kwargs=client_kwargs or {}, validate_model_on_init=False)
    for model in models[1:]:
        chain = chain.with_fallbacks(
            [ChatOllama(model=model, base_url=base_url, temperature=temperature,
                        client_kwargs=client_kwargs or {}, validate_model_on_init=False)]
        )
    return chain.with_fallbacks([groq])


def _pick_models(status: dict, priority: list[str]) -> list[str]:
    available = {m.get("name") for m in status.get("models", [])}
    return [m for m in priority if m in available]


def build_llm(temperature: float = 0.1, tags: list[str] | None = None, seed: int = 42):
    """Build an LLM chain with the following preference order:

    1. Local Ollama server (if reachable and has a known model)
    2. Ollama Cloud on ollama.com (if OLLAMA_API_KEY is configured)
    3. Groq, with OpenRouter as a final fallback

    ChatOllama connects lazily at invoke time, so constructing it without
    checking availability crashes on servers without Ollama (e.g. Streamlit
    Cloud). This helper probes each endpoint first and only wires models
    that actually exist into the chain.
    """
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI

    groq = ChatGroq(model_name=GROQ_MODEL, temperature=temperature, seed=seed, tags=tags)

    local_status = _probe(LOCAL_OLLAMA_HOST)
    local_models = _pick_models(local_status, LOCAL_MODEL_PRIORITY) if local_status else []
    if local_models:
        return _build_chain(local_models, temperature, groq)

    cloud_key = st.secrets.get("OLLAMA_API_KEY") or os.getenv("OLLAMA_API_KEY")
    if cloud_key:
        cloud_status = _probe(OLLAMA_CLOUD_HOST, cloud_key)
        cloud_models = _pick_models(cloud_status, CLOUD_MODEL_PRIORITY) if cloud_status else []
        if cloud_models:
            return _build_chain(
                cloud_models, temperature, groq,
                base_url=OLLAMA_CLOUD_HOST,
                client_kwargs={"headers": {"Authorization": f"Bearer {cloud_key}"}},
            )

    openrouter_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        openrouter = ChatOpenAI(
            model=OPENROUTER_MODEL,
            temperature=temperature,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_key,
            seed=seed,
            tags=tags,
        )
        return groq.with_fallbacks([openrouter])

    return groq
