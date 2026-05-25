from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from .chains import answer_question, build_question_context
from .structured_tool import search_structured_data


class ConsultaContactoInput(BaseModel):
    categoria: str = Field(
        description=(
            "Dato puntual o categoria solicitada: whatsapp, correo, pbx, redes, "
            "direccion, horarios, cobertura, sostenibilidad, proveedores, privacidad, "
            "pet friendly, cifras corporativas, etc."
        )
    )


class BuscarCatalogoInput(BaseModel):
    query: str = Field(
        description=(
            "Pregunta completa del usuario sobre menu, productos, sandwiches, combos, "
            "promociones, precios, rankings o comparaciones de precios."
        )
    )


class ConsultaCorporativaInput(BaseModel):
    query: str = Field(
        description=(
            "Pregunta completa del usuario sobre historia, informacion institucional, "
            "documentos oficiales, sostenibilidad narrativa o contexto corporativo abierto."
        )
    )


@dataclass(slots=True)
class ToolRuntimeContext:
    model: Any
    company_name: str
    original_question: str
    knowledge_text: str
    chunks: list[dict[str, str | int]]
    max_context_chars: int
    structured_data_path: Path
    vector_index_path: Path
    conversation_history: str


def _json_payload(
    *,
    route: str,
    answer: str,
    reasoning: str,
    context_mode: str,
    sources: list[dict[str, str]] | None = None,
) -> str:
    return json.dumps(
        {
            "route": route,
            "answer": answer,
            "reasoning": reasoning,
            "context_mode": context_mode,
            "sources": sources or [],
        },
        ensure_ascii=False,
    )


def decode_tool_payload(content: object) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _retrieval_only_payload(
    *,
    context: ToolRuntimeContext,
    route: str,
    query: str,
    reason: str,
) -> str:
    retrieved_context, context_mode, sources = build_question_context(
        question=query,
        knowledge_text=context.knowledge_text,
        chunks=context.chunks,
        max_context_chars=min(context.max_context_chars, 1800),
        vector_index_path=context.vector_index_path,
    )
    excerpt = retrieved_context.replace("\n\n---\n\n", "\n\n").strip()
    if len(excerpt) > 1200:
        excerpt = f"{excerpt[:1200].rstrip()}..."
    return _json_payload(
        route=route,
        answer=(
            "No pude generar una síntesis con el LLM configurado, pero recuperé "
            "este contexto relevante desde la base vectorial:\n\n"
            f"{excerpt}"
        ),
        reasoning=reason,
        context_mode=f"{context_mode}_retrieval_only",
        sources=sources,
    )


def build_agent_tools(context: ToolRuntimeContext) -> list[BaseTool]:
    @tool(args_schema=ConsultaContactoInput, return_direct=True)
    def consultar_datos_contacto(categoria: str) -> str:
        """Consulta datos puntuales estructurados de contacto, canales, cobertura, limites operativos, cifras y sostenibilidad de Sándwich Qbano."""

        try:
            query = f"{context.original_question}\nCategoria solicitada: {categoria}".strip()
            result = search_structured_data(query, context.structured_data_path)
            if result is None:
                return _json_payload(
                    route="consultar_datos_contacto",
                    answer=(
                        "No encontré ese dato puntual en la base estructurada actual. "
                        "Puedo intentar responderlo desde la base documental si formulas la pregunta de forma más abierta."
                    ),
                    reasoning="La herramienta estructurada no encontro coincidencias suficientes.",
                    context_mode="structured_json_empty",
                )
            return _json_payload(
                route="consultar_datos_contacto",
                answer=result.answer,
                reasoning=(
                    "Se ejecuto una tool Pydantic para recuperar datos puntuales "
                    "desde la base estructurada local."
                ),
                context_mode="structured_json",
                sources=result.sources,
            )
        except Exception:
            return _json_payload(
                route="consultar_datos_contacto",
                answer=(
                    "En este momento no pude verificar ese dato estructurado. "
                    "Intenta de nuevo o pregunta por otro canal de contacto."
                ),
                reasoning="La tool consultar_datos_contacto fallo y devolvio un mensaje cortes.",
                context_mode="tool_error",
            )

    @tool(args_schema=BuscarCatalogoInput, return_direct=True)
    def buscar_catalogo_productos(query: str) -> str:
        """Busca en el catalogo de Sándwich Qbano productos, precios, combos, promociones, rankings y comparaciones comerciales."""

        try:
            effective_query = query.strip() or context.original_question
            qa_result = answer_question(
                model=context.model,
                company_name=context.company_name,
                question=effective_query,
                knowledge_text=context.knowledge_text,
                chunks=context.chunks,
                max_context_chars=context.max_context_chars,
                vector_index_path=context.vector_index_path,
                conversation_history=context.conversation_history,
            )
            return _json_payload(
                route="buscar_catalogo_productos",
                answer=qa_result.answer,
                reasoning=(
                    "Se ejecuto una tool Pydantic para resolver una consulta "
                    "comercial con RAG y atajos deterministicos de catalogo."
                ),
                context_mode=qa_result.context_mode,
                sources=qa_result.sources,
            )
        except Exception:
            return _retrieval_only_payload(
                context=context,
                route="buscar_catalogo_productos",
                query=context.original_question,
                reason=(
                    "La tool buscar_catalogo_productos no pudo usar el LLM configurado "
                    "y devolvio recuperacion vectorial como respaldo cortes."
                ),
            )

    @tool(args_schema=ConsultaCorporativaInput, return_direct=True)
    def consultar_informacion_corporativa(query: str) -> str:
        """Consulta la base documental y vectorial para informacion corporativa abierta, historia, documentos oficiales y sostenibilidad narrativa."""

        try:
            effective_query = query.strip() or context.original_question
            qa_result = answer_question(
                model=context.model,
                company_name=context.company_name,
                question=effective_query,
                knowledge_text=context.knowledge_text,
                chunks=context.chunks,
                max_context_chars=context.max_context_chars,
                vector_index_path=context.vector_index_path,
                conversation_history=context.conversation_history,
            )
            return _json_payload(
                route="consultar_informacion_corporativa",
                answer=qa_result.answer,
                reasoning=(
                    "Se ejecuto una tool Pydantic para responder una consulta "
                    "abierta contra la base documental y el indice vectorial."
                ),
                context_mode=qa_result.context_mode,
                sources=qa_result.sources,
            )
        except Exception:
            return _retrieval_only_payload(
                context=context,
                route="consultar_informacion_corporativa",
                query=context.original_question,
                reason=(
                    "La tool consultar_informacion_corporativa no pudo usar el LLM configurado "
                    "y devolvio recuperacion vectorial como respaldo cortes."
                ),
            )

    return [
        consultar_datos_contacto,
        buscar_catalogo_productos,
        consultar_informacion_corporativa,
    ]
