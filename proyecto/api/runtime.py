"""Capa de runtime de la API: carga unica de config, knowledge base y chunks.

Mantiene en memoria los recursos costosos (knowledge_base.txt, chunks.json, config)
para evitar recargarlos en cada peticion. El modelo LLM se construye por peticion
porque el cliente puede alternar proveedor/modelo en cada turno (es lo que pide la
Ruta A para soportar Ollama, Gemini y OpenAI sin reinicio).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.config import AppConfig, build_app_config
from src.processing import load_chunks, load_knowledge_base
from src.vector_store import load_vector_index


@dataclass(slots=True, frozen=True)
class ApiResources:
    config: AppConfig
    knowledge_text: str
    chunks: list[dict[str, str | int]]
    vector_count: int
    embedding_model: str


@lru_cache(maxsize=1)
def get_resources() -> ApiResources:
    """Devuelve los recursos compartidos cargados una sola vez por proceso."""

    config = build_app_config()

    knowledge_text = ""
    chunks: list[dict[str, str | int]] = []
    if config.paths.knowledge_base_path.exists():
        knowledge_text = load_knowledge_base(config.paths.knowledge_base_path)
    if config.paths.chunks_path.exists():
        chunks = load_chunks(config.paths.chunks_path)

    vector_count = 0
    embedding_model = ""
    try:
        payload = load_vector_index(config.paths.vector_index_path)
        vector_count = int(payload.get("vectors", 0) or 0)
        embedding_model = str(payload.get("embedding_model", "") or "")
    except Exception:
        # La ausencia del indice no debe tumbar la API; /health lo reportara.
        pass

    return ApiResources(
        config=config,
        knowledge_text=knowledge_text,
        chunks=chunks,
        vector_count=vector_count,
        embedding_model=embedding_model,
    )


def reset_resources_cache() -> None:
    """Util para tests: invalida la cache singleton."""

    get_resources.cache_clear()
