from __future__ import annotations

import os


def create_chat_model(provider: str, model_name: str, temperature: float = 0.1):
    normalized_provider = provider.strip().lower()

    if normalized_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY no esta configurada. Define la variable en .env."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)

    if normalized_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    raise ValueError(f"Proveedor de LLM no soportado: {provider}")
