"""Cliente LLM multi-proveedor usando init_chat_model de LangChain.

Soporta de forma transparente:
- google_genai (Gemini)  → GEMINI_API_KEY / GOOGLE_API_KEY
- openai      (gpt-*)    → OPENAI_API_KEY
- ollama      (local)    → OLLAMA_BASE_URL

El proveedor se selecciona con la variable LLM_PROVIDER y el modelo con
MODEL_NAME (ambas leidas desde .env por src/config.py).

Para el Taller 3 se priorizo `init_chat_model` porque:
- Es la API recomendada por LangChain 1.x para abstraer proveedores.
- Permite cambiar de modelo cambiando solo el .env, sin tocar codigo.
- Es la pieza que pide la rubrica del Taller 3 (Ruta A).
"""

from __future__ import annotations

import os

# Aliases para tolerar nombres comunes del proveedor desde el .env.
_PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "gpt": "openai",
    "ollama": "ollama",
    "local": "ollama",
    "gemini": "google_genai",
    "google": "google_genai",
    "google_genai": "google_genai",
    "google-genai": "google_genai",
    "googleai": "google_genai",
}

# Modelos por defecto cuando MODEL_NAME no esta en .env.
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "ollama": "gemma3:latest",
    "google_genai": "gemini-2.5-flash",
}


def _normalize_provider(provider: str) -> str:
    key = (provider or "").strip().lower()
    if key in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[key]
    raise ValueError(
        "Proveedor de LLM no soportado: "
        f"'{provider}'. Use uno de: openai, ollama, gemini (google_genai)."
    )


def _resolve_model_name(provider: str, model_name: str | None) -> str:
    if model_name and model_name.strip():
        return model_name.strip()
    return _DEFAULT_MODELS[provider]


def _ensure_provider_credentials(provider: str) -> None:
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY no esta configurada. Definela en .env o cambia LLM_PROVIDER."
        )

    if provider == "google_genai":
        # Si el usuario definio GEMINI_API_KEY en el .env, tiene prioridad absoluta sobre
        # cualquier GOOGLE_API_KEY que pudiera estar exportada en el shell (p.ej. una key
        # vieja en ~/.zshrc). Esto evita el escenario donde el SDK usa una key invalida.
        gemini_key = os.getenv("GEMINI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            os.environ["GOOGLE_API_KEY"] = gemini_key
        elif not google_key:
            raise RuntimeError(
                "Para usar Gemini configura GEMINI_API_KEY (o GOOGLE_API_KEY) en .env. "
                "Obten una gratis en https://aistudio.google.com/apikey"
            )


def create_chat_model(provider: str, model_name: str, temperature: float = 0.1):
    """Crea un ChatModel de LangChain para el proveedor configurado.

    Mantiene la firma usada por el resto del proyecto (provider, model_name,
    temperature) para no romper a chains.py / agent.py / scripts.
    """

    normalized_provider = _normalize_provider(provider)
    resolved_model = _resolve_model_name(normalized_provider, model_name)
    _ensure_provider_credentials(normalized_provider)

    from langchain.chat_models import init_chat_model

    kwargs: dict[str, object] = {
        "model": resolved_model,
        "model_provider": normalized_provider,
        "temperature": temperature,
    }

    if normalized_provider == "ollama":
        kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return init_chat_model(**kwargs)


def describe_active_provider(provider: str, model_name: str) -> str:
    """Devuelve un texto humano-legible para mostrar en la UI/CLI."""

    try:
        normalized_provider = _normalize_provider(provider)
    except ValueError:
        return f"{provider} · {model_name}"
    resolved_model = _resolve_model_name(normalized_provider, model_name)
    return f"{normalized_provider} · {resolved_model}"
