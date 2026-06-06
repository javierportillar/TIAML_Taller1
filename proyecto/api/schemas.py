"""Schemas Pydantic del contrato HTTP de la API.

El uso de Pydantic aqui es consistente con el resto del proyecto (tools, structured outputs)
y permite a FastAPI generar la documentacion OpenAPI automatica en `/docs`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload aceptado por `POST /chat`."""

    thread_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description=(
            "Identificador de la sesion conversacional. En produccion sera el numero de telefono "
            "del usuario de WhatsApp (formato E.164, p.ej. '573001234567'). PostgresSaver y la tabla "
            "conversation_messages usan este campo para aislar memorias por usuario."
        ),
        examples=["573001234567"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Texto crudo enviado por el usuario.",
        examples=["¿Cual es el WhatsApp de la empresa?"],
    )
    provider: Literal["ollama", "google_genai", "openai", "opencode_go"] | None = Field(
        default=None,
        description=(
            "Proveedor LLM a utilizar en este turno. Si se omite, se toma del .env "
            "(LLM_PROVIDER). Permite alternar Ollama/Gemini/OpenAI/OpenCode Go desde el cliente."
        ),
        examples=["ollama", "google_genai", "opencode_go"],
    )
    model: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Nombre del modelo concreto del proveedor seleccionado (gemma3:latest, "
            "gemini-2.5-flash, gpt-4o-mini, kimi-k2.6, etc.). Si se omite, se toma el default del proveedor."
        ),
        examples=["gemma3:latest", "gemini-2.5-flash", "kimi-k2.6"],
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Temperatura del LLM para este turno. Si se omite, se usa el default del .env.",
        examples=[0.0, 0.3],
    )


class SourceItem(BaseModel):
    """Fuente documental citada en la respuesta del agente."""

    title: str = ""
    url: str = ""


class ChatResponse(BaseModel):
    """Respuesta del agente devuelta por `POST /chat`."""

    answer: str = Field(..., description="Respuesta natural en lenguaje humano.")
    route: str = Field(
        ...,
        description=(
            "Tool o ruta elegida por el agente: consultar_datos_contacto, "
            "buscar_catalogo_productos, consultar_informacion_corporativa, "
            "solicitar_supervisor_humano, memory o conversation."
        ),
    )
    context_mode: str = Field(
        ...,
        description=(
            "Modo en que se obtuvo la respuesta: structured_json, vector, deterministic, "
            "conversation_memory, human_in_the_loop, etc."
        ),
    )
    reasoning: str = Field(
        "",
        description="Razonamiento breve interno del agente. Util para debugging y auditoria.",
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Fuentes citadas (titulo + URL). Vacio si la ruta no usa documentos.",
    )
    thread_id: str = Field(..., description="Eco del thread_id recibido para confirmar la sesion.")
    llm_provider: str = Field(..., description="Proveedor LLM efectivo usado para responder.")
    llm_model: str = Field(..., description="Modelo LLM efectivo usado para responder.")


class HealthResponse(BaseModel):
    """Payload de `GET /health`."""

    status: Literal["ok", "degraded"]
    knowledge_base_ready: bool
    chroma_ready: bool
    postgres_ready: bool
    default_provider: str
    default_model: str


class InfoResponse(BaseModel):
    """Payload de `GET /info`. Resumen para clientes y dashboards."""

    company: str
    description: str
    knowledge_chunks: int
    vector_count: int
    embedding_model: str
    available_providers: list[str]
    default_provider: str
    default_model: str
    agent_tools: list[str]
    sensitive_tools: list[str]
