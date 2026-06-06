"""FastAPI app que expone el agente conversacional como servicio REST.

Endpoints:
- POST /chat   -> conversa con el agente; preserva memoria por `thread_id` en Postgres.
- GET  /health -> chequeo de vida (Streamlit/N8N/Twilio pueden monitorearlo).
- GET  /info   -> snapshot de la configuracion: empresa, vectores, tools, providers.

Diseno:
- Reutiliza `src.agent.run_agent`: misma logica que Streamlit, sin duplicar codigo.
- El modelo LLM se construye por peticion (permite alternar Ollama/Gemini/OpenAI por turno).
- `thread_id` recibido del cliente se inyecta como llave de memoria persistente.
- CORS permisivo en desarrollo; para produccion conviene restringir origenes.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Permitir importar `src.*` cuando se lance con `uvicorn api.main:app` desde proyecto/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import AGENT_TOOL_NAMES, HITL_SENSITIVE_TOOLS, run_agent  # noqa: E402
from src.checkpointer import ensure_persistence_ready  # noqa: E402
from src.llm import create_chat_model  # noqa: E402

from .runtime import get_resources  # noqa: E402
from .schemas import (  # noqa: E402
    ChatRequest,
    ChatResponse,
    HealthResponse,
    InfoResponse,
    SourceItem,
)

PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "ollama": "gemma3:latest",
    "google_genai": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "opencode_go": "kimi-k2.6",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa Postgres y precarga los recursos antes del primer request."""

    # Best-effort: si Postgres no esta arriba, levantamos un warning pero seguimos.
    try:
        ensure_persistence_ready()
    except Exception as exc:
        print(f"[startup] WARN: Postgres no esta listo: {exc}", flush=True)
    # Forzar la carga de knowledge base + chunks + vector count.
    get_resources()
    yield


app = FastAPI(
    title="Sandwich Qbano Agent API",
    description=(
        "API REST del agente conversacional para Sandwich Qbano. "
        "Implementacion de la Ruta A del Taller 3 (LangChain + create_agent + "
        "PostgresSaver + Function Calling con Pydantic + HumanInTheLoopMiddleware + dynamic_prompt)."
    ),
    version="0.5.0",
    lifespan=lifespan,
)

# CORS abierto para desarrollo. Para produccion limitar a los dominios de N8N/WhatsApp.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_provider_and_model(payload: ChatRequest) -> tuple[str, str, float]:
    """Resuelve provider/model/temperature efectivos: payload > .env > default por proveedor."""

    resources = get_resources()
    config = resources.config

    provider = (payload.provider or config.runtime.provider or "ollama").strip()
    model_name = (payload.model or "").strip()
    if not model_name:
        # Si el cliente solo especifica provider, usamos el default de ese provider, no el del .env.
        if payload.provider and payload.provider != config.runtime.provider:
            model_name = PROVIDER_DEFAULT_MODEL.get(provider, "")
        else:
            model_name = config.runtime.model_name or PROVIDER_DEFAULT_MODEL.get(provider, "")
    if not model_name:
        model_name = PROVIDER_DEFAULT_MODEL.get(provider, "")

    temperature = (
        float(payload.temperature)
        if payload.temperature is not None
        else float(config.runtime.temperature)
    )
    return provider, model_name, temperature


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
def chat(payload: ChatRequest) -> ChatResponse:
    """Conversa con el agente. Cada peticion preserva memoria via `thread_id`."""

    resources = get_resources()
    config = resources.config

    provider, model_name, temperature = _resolve_provider_and_model(payload)

    try:
        model = create_chat_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
        )
    except (RuntimeError, ValueError) as exc:
        # Faltan credenciales del proveedor (GEMINI_API_KEY, OPENAI_API_KEY) o nombre invalido.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = run_agent(
            model=model,
            company_name=config.company.company_name,
            question=payload.message,
            chat_history=[],  # se rehidrata automaticamente desde Postgres con thread_id
            knowledge_text=resources.knowledge_text,
            chunks=resources.chunks,
            max_context_chars=int(config.runtime.max_context_chars),
            structured_data_path=config.paths.structured_data_path,
            vector_index_path=config.paths.vector_index_path,
            thread_id=payload.thread_id,
            llm_provider=provider,
            llm_model=model_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fallo del agente: {exc}") from exc

    sources = [
        SourceItem(title=str(item.get("title", "")), url=str(item.get("url", "")))
        for item in (result.sources or [])
        if isinstance(item, dict)
    ]

    return ChatResponse(
        answer=result.answer,
        route=result.route,
        context_mode=result.context_mode,
        reasoning=result.reasoning,
        sources=sources,
        thread_id=payload.thread_id,
        llm_provider=provider,
        llm_model=model_name,
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Chequeo de vida con detalle de cada subsistema critico."""

    resources = get_resources()
    config = resources.config

    knowledge_ready = bool(resources.knowledge_text) and bool(resources.chunks)
    chroma_ready = resources.vector_count > 0

    postgres_ready = True
    try:
        ensure_persistence_ready()
    except Exception:
        postgres_ready = False

    status = "ok" if (knowledge_ready and chroma_ready and postgres_ready) else "degraded"

    return HealthResponse(
        status=status,
        knowledge_base_ready=knowledge_ready,
        chroma_ready=chroma_ready,
        postgres_ready=postgres_ready,
        default_provider=str(config.runtime.provider),
        default_model=str(config.runtime.model_name),
    )


@app.get("/info", response_model=InfoResponse, tags=["meta"])
def info() -> InfoResponse:
    """Snapshot del estado interno: empresa, vectores, tools, providers."""

    resources = get_resources()
    config = resources.config

    available_providers = ["ollama", "google_genai", "openai", "opencode_go"]
    return InfoResponse(
        company=config.company.company_name,
        description=config.company.company_description,
        knowledge_chunks=len(resources.chunks),
        vector_count=resources.vector_count,
        embedding_model=resources.embedding_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        available_providers=available_providers,
        default_provider=str(config.runtime.provider),
        default_model=str(config.runtime.model_name),
        agent_tools=sorted(AGENT_TOOL_NAMES),
        sensitive_tools=sorted(HITL_SENSITIVE_TOOLS),
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Redireccion humana hacia /docs."""

    return {
        "service": "Sandwich Qbano Agent API",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
        "chat": "POST /chat",
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
