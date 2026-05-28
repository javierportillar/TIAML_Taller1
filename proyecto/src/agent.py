from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    dynamic_prompt,
)
from langgraph.types import Command

from .chains import QAResult, answer_question
from .checkpointer import get_checkpointer, load_thread_messages, save_thread_turn
from .memory import (
    acknowledge_personal_declaration,
    answer_follow_up_from_memory,
    answer_personal_question_from_memory,
    format_chat_history,
)
from .prompts import load_prompt_config
from .tools import ToolRuntimeContext, build_agent_tools, decode_tool_payload


@dataclass(slots=True)
class AgentResult:
    answer: str
    route: str
    reasoning: str
    context_mode: str
    sources: list[dict[str, str]]


STRUCTURED_TERMS = {
    "abastecimiento",
    "agua",
    "aliado",
    "aliados",
    "ambiental",
    "ambiente",
    "apertura",
    "aperturas",
    "arbol",
    "árbol",
    "arboles",
    "árboles",
    "auditoria",
    "auditoría",
    "auditorias",
    "auditorías",
    "biodiversidad",
    "calidad",
    "call center",
    "canal",
    "canales",
    "carbono",
    "ciudad",
    "ciudades",
    "cliente",
    "clientes",
    "contacto",
    "correo",
    "cobertura",
    "crecimiento",
    "datos personales",
    "disponibilidad",
    "disponible",
    "direccion",
    "dirección",
    "domicilio",
    "domicilios",
    "donde esta",
    "donde está",
    "donde hay",
    "dónde está",
    "dónde hay",
    "empleo",
    "empleos",
    "energia",
    "energía",
    "equidad",
    "etica",
    "ética",
    "entrega",
    "entregas",
    "factura",
    "factura electronica",
    "factura electrónica",
    "facturacion",
    "facturación",
    "email",
    "franquicia",
    "franquicias",
    "gobernanza",
    "historia",
    "hora",
    "horario",
    "horarios",
    "huella",
    "instagram",
    "inocuidad",
    "innovacion",
    "innovación",
    "linkedin",
    "marca",
    "materialidad",
    "medio ambiente",
    "menu",
    "menú",
    "mision",
    "misión",
    "municipio",
    "municipios",
    "ods",
    "pago",
    "pagos",
    "pbx",
    "pet friendly",
    "pet-friendly",
    "politica de datos",
    "política de datos",
    "privacidad",
    "presencia",
    "proveedor",
    "proveedores",
    "punto de venta",
    "puntos de venta",
    "qbano en cifras",
    "que es",
    "qué es",
    "quien es",
    "quién es",
    "quienes son",
    "quiénes son",
    "redes",
    "residuo",
    "residuos",
    "responsabilidad social",
    "restaurante",
    "restaurantes",
    "seguridad alimentaria",
    "servicio al cliente",
    "sede",
    "sedes",
    "sostenibilidad",
    "sostenible",
    "sucursal",
    "sucursales",
    "telefono",
    "teléfono",
    "tienda",
    "tiendas",
    "ubicacion",
    "ubicación",
    "ubicaciones",
    "whatsapp",
    "youtube",
}


LOCATION_TERMS = {
    "amazonas",
    "antioquia",
    "arauca",
    "atlantico",
    "atlántico",
    "barranquilla",
    "bogota",
    "bogotá",
    "bolivar",
    "bolívar",
    "boyaca",
    "boyacá",
    "bucaramanga",
    "caldas",
    "cali",
    "caqueta",
    "caquetá",
    "cartagena",
    "casanare",
    "cauca",
    "cesar",
    "choco",
    "chocó",
    "colombia",
    "cordoba",
    "córdoba",
    "cundinamarca",
    "departamento",
    "departamentos",
    "guainia",
    "guainía",
    "guaviare",
    "huila",
    "ipiales",
    "la guajira",
    "magdalena",
    "manizales",
    "medellin",
    "medellín",
    "meta",
    "municipio",
    "municipios",
    "narino",
    "nariño",
    "norte de santander",
    "pasto",
    "pereira",
    "putumayo",
    "quindio",
    "quindío",
    "risaralda",
    "san andres",
    "san andrés",
    "santander",
    "sincelejo",
    "sucre",
    "tolima",
    "valle",
    "valle del cauca",
    "vaupes",
    "vaupés",
    "vichada",
    "villavicencio",
    "yumbo",
}


BRAND_OR_PRODUCT_TERMS = {
    "qbano",
    "sandwich",
    "sandwiches",
    "sanwich",
    "sanwiches",
    "sanwdich",
    "sanwdiches",
    "sandiwch",
    "sandiwches",
    "sanduche",
    "sanduches",
}


def _normalize_text(text: str) -> str:
    replacements = str.maketrans(
        {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "ü": "u",
            "ñ": "n",
        }
    )
    normalized = text.lower().translate(replacements)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {_normalize_text(text)} "
    normalized_phrase = f" {_normalize_text(phrase)} "
    return normalized_phrase in normalized_text


def _answer_conversational_message(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not normalized:
        return "Puedes hacerme preguntas sobre Sándwich Qbano, productos, precios, combos, contacto, cobertura o sostenibilidad."

    has_greeting = any(
        normalized.startswith(greeting)
        for greeting in ["hola", "buenas", "buenos dias", "buenas tardes", "buenas noches"]
    )
    asks_to_start = any(
        phrase in normalized
        for phrase in ["voy a hacerte", "puedo hacerte", "te voy a hacer", "unas preguntas", "vale"]
    )
    name_match = re.search(
        r"\b(?:me llamo|mi nombre es)\s+([a-záéíóúüñ]+)",
        question,
        flags=re.IGNORECASE,
    )
    if has_greeting and (asks_to_start or name_match):
        name = f" {name_match.group(1).strip().capitalize()}" if name_match else ""
        return (
            f"Hola{name}. Claro, puedes hacerme preguntas sobre Sándwich Qbano. "
            "Puedo responder sobre productos, precios, combos, promociones, contacto, cobertura, sedes reportadas y sostenibilidad usando la base disponible."
        )
    if normalized in {"hola", "buenas", "buenos dias", "buenas tardes", "buenas noches"}:
        return "Hola. Puedes preguntarme sobre Sándwich Qbano, su menú, precios, combos, contacto, cobertura o sostenibilidad."
    return None


def _looks_like_structured_query(question: str) -> bool:
    normalized = question.lower()
    return any(term in normalized for term in STRUCTURED_TERMS)


COMMERCIAL_CATALOG_TERMS = {
    "barato",
    "caro",
    "cuanto",
    "cuanto cuesta",
    "cuanto vale",
    "cuestan",
    "combo",
    "combos",
    "ensalada",
    "ensaladas",
    "hamburguesa",
    "hamburguesas",
    "individual",
    "individuales",
    "menu",
    "ofertas",
    "precio",
    "precios",
    "producto",
    "productos",
    "promo",
    "promocion",
    "promociones",
    "qbanito",
    "qbanitos",
    "qbowl",
    "qbowls",
    "sabor",
    "sabores",
    "tipos",
    "variedades",
    "valen",
    "wrap",
    "wraps",
}


COMMERCIAL_CATALOG_PHRASES = {
    "que productos",
    "que sabores",
    "que tipos",
    "que variedades",
    "que combos",
    "que sandwich",
    "que opciones",
    "que ofrecen",
    "que venden",
    "lista de productos",
    "lista de sandwich",
    "lista de combos",
    "menu completo",
    "tipos de sandwich",
    "tipos de combo",
    "mas caro",
    "mas barato",
    "mas economico",
    "ranking de precios",
    "diferencia de precio",
    "diferencias de precio",
    "diferencia de precios",
    "diferencias de precios",
}


STRONG_STRUCTURED_PHRASES = {
    "whatsapp",
    "redes sociales",
    "red social",
    "instagram",
    "facebook",
    "tiktok",
    "linkedin",
    "youtube",
    "pbx",
    "call center",
    "correo electronico",
    "correo de servicio",
    "telefono de servicio",
    "factura electronica",
    "facturacion electronica",
    "pet friendly",
    "pet-friendly",
    "medios de pago",
    "metodos de pago",
    "horario de atencion",
    "horarios de atencion",
    "hora de atencion",
    "hora de apertura",
    "huella de carbono",
    "fsq group",
    "datos personales",
    "politica de datos",
    "domicilio en",
    "pagina de contacto",
    "punto de venta",
    "puntos de venta",
    "servicio al cliente",
}


def _has_strong_structured_signal(question: str) -> bool:
    normalized = _normalize_text(question)
    return any(phrase in normalized for phrase in STRONG_STRUCTURED_PHRASES)


def _looks_like_commercial_catalog_query(question: str) -> bool:
    """Detecta preguntas claramente comerciales sobre productos/precios/combos.

    Estas preguntas deben ir al RAG/atajo determinístico de catálogo,
    nunca a la herramienta estructurada aunque haya coincidencias parciales.

    Importante: el solo hecho de mencionar "sandwich" o "qbano" no convierte la
    pregunta en comercial. Hace falta una señal de precio, combo, oferta, sabor,
    producto o frase comercial concreta.
    """
    normalized = _normalize_text(question)
    if not normalized:
        return False
    tokens = set(normalized.split())
    if tokens & COMMERCIAL_CATALOG_TERMS:
        return True
    return any(phrase in normalized for phrase in COMMERCIAL_CATALOG_PHRASES)


def _looks_like_location_query(question: str) -> bool:
    normalized = _normalize_text(question)
    has_brand_or_product = any(term in normalized for term in BRAND_OR_PRODUCT_TERMS)
    has_place = any(_contains_normalized_phrase(normalized, term) for term in LOCATION_TERMS)
    has_location_intent = any(
        _contains_normalized_phrase(normalized, phrase)
        for phrase in [
            "hay",
            "queda",
            "quedan",
            "esta",
            "estan",
            "ubicado",
            "ubicada",
            "ubicados",
            "ubicadas",
            "sede",
            "sedes",
            "sucursal",
            "sucursales",
            "punto de venta",
            "puntos de venta",
            "disponible",
            "disponibilidad",
            "venden",
            "tienen",
            "domicilio",
            "domicilios",
        ]
    )
    return has_brand_or_product and has_place and has_location_intent


_HUMAN_ESCALATION_PHRASES = (
    "hablar con un humano",
    "hablar con humano",
    "hablar con una persona",
    "hablar con persona real",
    "hablar con un asesor",
    "hablar con un agente",
    "hablar con un supervisor",
    "asesor humano",
    "agente humano",
    "supervisor humano",
    "persona humana",
    "atencion humana",
    "atencion personal",
    "pasame con un humano",
    "pasame con alguien",
    "necesito un humano",
    "necesito una persona",
    "necesito un asesor",
    "necesito un agente",
    "necesito un supervisor",
    "necesito hablar con",
    "quiero un asesor",
    "quiero un humano",
    "quiero un supervisor",
    "quiero hablar con un",
    "quiero hablar con una",
    "transferir a un humano",
    "transferir a una persona",
    "transfiereme",
    "escalar a un humano",
    "escalar mi caso",
    "escalamiento",
    "tengo una queja",
    "quiero presentar una queja",
    "presentar reclamo",
    "presentar un reclamo",
    "poner una queja",
)


def _looks_like_human_escalation(question: str) -> bool:
    """Detecta si el usuario quiere hablar con un humano / escalar el caso."""

    normalized = _normalize_text(question)
    return any(_contains_normalized_phrase(normalized, phrase) for phrase in _HUMAN_ESCALATION_PHRASES)


def _rewrite_combo_difference_follow_up(
    question: str,
    chat_history: list[dict[str, str]],
) -> str:
    normalized = _normalize_text(question)
    asks_difference = any(term in normalized for term in ["diferencia", "diferencias", "comparar", "compara"])
    asks_price = any(term in normalized for term in ["precio", "precios", "cuanto", "cuanto vale", "cuanto cuesta"])
    mentions_combo = "combo" in normalized or "combos" in normalized
    if not asks_difference or mentions_combo or not asks_price:
        return question

    for message in reversed(chat_history[-6:]):
        if message.get("role") != "assistant":
            continue
        content = _normalize_text(str(message.get("content", "")))
        if "combo" in content or "combos" in content:
            return "cual es la diferencia de precios entre los productos en combo y sin combo"
    return question


AGENT_TOOL_NAMES = {
    "consultar_datos_contacto",
    "buscar_catalogo_productos",
    "consultar_informacion_corporativa",
    "solicitar_supervisor_humano",
}

# Tools que pasan obligatoriamente por HumanInTheLoopMiddleware antes de ejecutarse.
# Aqui solo va una tool "sensible" (escalamiento a humano); el resto se auto-aprueba.
HITL_SENSITIVE_TOOLS = {"solicitar_supervisor_humano"}


def _build_agent_system_prompt() -> str:
    """Version legacy del prompt estatico, conservada para compatibilidad / pruebas."""

    prompts = load_prompt_config()
    return prompts["agent_router_system"]


def _build_dynamic_prompt_middleware():
    """Middleware que genera el system prompt DINAMICAMENTE por cada turno del agente.

    Cumple el requisito de la Ruta A del Taller 3 que exige uso de `dynamic_prompt`. Inyecta
    en tiempo real:
    - El prompt base configurado en `data/config/prompts_config.json` (editable desde la UI).
    - El nombre del proveedor + modelo LLM activo (cambia si el usuario eligio Ollama/Gemini/OpenAI).
    - La fecha/hora actual (util para horarios, contexto temporal).
    - Una nota explicita de las tools disponibles, asi el modelo no las olvida ni alucina.
    """

    @dynamic_prompt
    def _dynamic_system_prompt(request) -> str:
        base_prompt = load_prompt_config()["agent_router_system"]
        provider_name = ""
        model_name = ""
        try:
            llm = getattr(request, "model", None)
            if llm is not None:
                provider_name = type(llm).__name__
                model_name = getattr(llm, "model", "") or getattr(llm, "model_name", "")
        except Exception:
            pass

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        tool_list = ", ".join(sorted(AGENT_TOOL_NAMES))
        sensitive_list = ", ".join(sorted(HITL_SENSITIVE_TOOLS))
        return (
            f"{base_prompt}\n\n"
            "[Contexto dinamico inyectado via dynamic_prompt middleware]\n"
            f"- Fecha/hora actual: {now_str}\n"
            f"- LLM activo: {provider_name} ({model_name})\n"
            f"- Tools disponibles: {tool_list}\n"
            f"- Tools sensibles que requieren confirmacion humana (HITL): {sensitive_list}\n"
            "Usa SIEMPRE una tool; nunca respondas en texto libre sin invocar una."
        )

    return _dynamic_system_prompt


def _build_hitl_middleware() -> HumanInTheLoopMiddleware:
    """Middleware que pausa el agente antes de ejecutar tools marcadas como sensibles.

    El usuario (o el wrapper en Streamlit/FastAPI) recibe el interrupt y puede aprobar,
    editar argumentos o rechazar antes de que la tool corra.
    """

    interrupt_config = {
        tool_name: InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            description=(
                "El agente quiere escalar tu consulta a un supervisor humano. "
                "Confirma para continuar o rechaza para que sigamos con el flujo automatizado."
            ),
        )
        for tool_name in HITL_SENSITIVE_TOOLS
    }
    return HumanInTheLoopMiddleware(
        interrupt_on=interrupt_config,
        description_prefix="Solicitud de confirmacion del usuario",
    )


def _build_agent_user_message(question: str, history: str) -> str:
    return (
        "Historial reciente:\n"
        f"{history}\n\n"
        "Pregunta actual:\n"
        f"{question}\n\n"
        "Elige exactamente una de las tools disponibles. No respondas directamente sin invocar una tool."
    )


def _extract_tool_payload(agent_output: Any) -> dict[str, Any] | None:
    if not isinstance(agent_output, dict):
        return None

    messages = agent_output.get("messages", [])
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        payload = decode_tool_payload(getattr(message, "content", None))
        if not payload:
            continue
        route = str(payload.get("route", ""))
        if route in AGENT_TOOL_NAMES:
            return payload
    return None


def _select_fallback_tool_name(question: str) -> str:
    is_commercial = _looks_like_commercial_catalog_query(question)
    has_strong_structured = _has_strong_structured_signal(question)
    if (has_strong_structured or not is_commercial) and (
        _looks_like_structured_query(question) or _looks_like_location_query(question)
    ):
        return "consultar_datos_contacto"
    if is_commercial:
        return "buscar_catalogo_productos"
    return "consultar_informacion_corporativa"


def _run_named_tool(tools: list[Any], tool_name: str, question: str) -> dict[str, Any] | None:
    tool_by_name = {tool.name: tool for tool in tools}
    selected_tool = tool_by_name.get(tool_name)
    if selected_tool is None:
        return None

    if tool_name == "consultar_datos_contacto":
        content = selected_tool.invoke({"categoria": question})
    elif tool_name == "solicitar_supervisor_humano":
        content = selected_tool.invoke(
            {"motivo": question, "canal_preferido": "whatsapp"}
        )
    else:
        content = selected_tool.invoke({"query": question})
    return decode_tool_payload(content)


def _agent_result_from_payload(payload: dict[str, Any]) -> AgentResult:
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    clean_sources: list[dict[str, str]] = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        clean_sources.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
            }
        )

    return AgentResult(
        answer=str(payload.get("answer", "")).strip(),
        route=str(payload.get("route", "")),
        reasoning=str(payload.get("reasoning", "")).strip(),
        context_mode=str(payload.get("context_mode", "")),
        sources=clean_sources,
    )


def _load_effective_chat_history(
    thread_id: str,
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    if chat_history:
        return chat_history
    try:
        return load_thread_messages(thread_id)
    except Exception:
        return []


def _save_result_for_thread(
    *,
    thread_id: str,
    question: str,
    result: AgentResult,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> AgentResult:
    try:
        save_thread_turn(
            thread_id=thread_id,
            question=question,
            answer=result.answer,
            route=result.route,
            context_mode=result.context_mode,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
    except Exception:
        pass
    return result


def _get_agent_checkpointer():
    try:
        return get_checkpointer()
    except Exception:
        return None


def run_agent(
    *,
    model,
    company_name: str,
    question: str,
    chat_history: list[dict[str, str]],
    knowledge_text: str,
    chunks: list[dict[str, str | int]],
    max_context_chars: int,
    structured_data_path: Path,
    vector_index_path: Path,
    thread_id: str = "streamlit_local",
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> AgentResult:
    effective_chat_history = _load_effective_chat_history(thread_id, chat_history)

    conversational_answer = _answer_conversational_message(question)
    if conversational_answer:
        result = AgentResult(
            answer=conversational_answer,
            route="conversation",
            reasoning="La entrada es conversacional y no requiere consultar la base de conocimiento.",
            context_mode="conversation",
            sources=[],
        )
        return _save_result_for_thread(
            thread_id=thread_id,
            question=question,
            result=result,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

    memory_answer = answer_follow_up_from_memory(question, effective_chat_history)
    if memory_answer:
        result = AgentResult(
            answer=memory_answer,
            route="memory",
            reasoning="La pregunta depende del turno anterior; se resolvio con el historial de conversacion.",
            context_mode="conversation_memory",
            sources=[],
        )
        return _save_result_for_thread(
            thread_id=thread_id,
            question=question,
            result=result,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

    # Declaraciones personales del usuario ("Mi papa se llama Francisco", "Yo soy Javier").
    # Se acusan recibo sin disparar tools ni RAG; el dato queda en el historial.
    personal_ack = acknowledge_personal_declaration(question)
    if personal_ack:
        result = AgentResult(
            answer=personal_ack,
            route="conversation",
            reasoning="Declaracion personal del usuario; se acuso recibo y se preserva en el historial.",
            context_mode="conversation",
            sources=[],
        )
        return _save_result_for_thread(
            thread_id=thread_id,
            question=question,
            result=result,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

    # Preguntas sobre informacion personal compartida en turnos previos
    # ("Como me llamo?", "Como se llama mi papa?", "Que te dije sobre X?").
    # Se resuelven mirando el historial con el LLM, no consultando RAG ni datos corporativos.
    personal_memory_answer = answer_personal_question_from_memory(
        question, effective_chat_history, model
    )
    if personal_memory_answer:
        result = AgentResult(
            answer=personal_memory_answer,
            route="memory",
            reasoning="Pregunta sobre informacion personal del usuario; se respondio con el historial.",
            context_mode="conversation_memory",
            sources=[],
        )
        return _save_result_for_thread(
            thread_id=thread_id,
            question=question,
            result=result,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

    effective_question = _rewrite_combo_difference_follow_up(question, effective_chat_history)
    history_context = format_chat_history(effective_chat_history)
    runtime_context = ToolRuntimeContext(
        model=model,
        company_name=company_name,
        original_question=effective_question,
        knowledge_text=knowledge_text,
        chunks=chunks,
        max_context_chars=max_context_chars,
        structured_data_path=structured_data_path,
        vector_index_path=vector_index_path,
        conversation_history=history_context,
    )
    tools = build_agent_tools(runtime_context)
    checkpointer = _get_agent_checkpointer()

    # Atajo deterministico: si el usuario pide explicitamente hablar con un humano,
    # invocamos la tool sensible directamente (que tambien pasa por HumanInTheLoopMiddleware
    # cuando se hace via create_agent). El LLM gemma3 a veces prefiere otras tools mas amplias.
    if _looks_like_human_escalation(effective_question):
        payload = _run_named_tool(tools, "solicitar_supervisor_humano", effective_question)
        if payload:
            payload["reasoning"] = (
                f"{payload.get('reasoning', '')} "
                "Atajo deterministico de escalamiento humano + tool sensible HITL."
            ).strip()
            result = _agent_result_from_payload(payload)
            return _save_result_for_thread(
                thread_id=thread_id,
                question=question,
                result=result,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )

    try:
        graph = create_agent(
            model=model,
            tools=tools,
            middleware=[
                _build_dynamic_prompt_middleware(),
                _build_hitl_middleware(),
            ],
            checkpointer=checkpointer,
        )
        invoke_config = {"configurable": {"thread_id": thread_id}}
        agent_output = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": _build_agent_user_message(effective_question, history_context),
                    }
                ]
            },
            config=invoke_config,
        )

        # HumanInTheLoopMiddleware puede haber pausado el grafo: si hay un __interrupt__
        # en la salida, auto-aprobamos la tool sensible para mantener el flujo conversacional
        # (Streamlit/FastAPI pueden sobreescribir esto pidiendo confirmacion manual).
        if isinstance(agent_output, dict) and agent_output.get("__interrupt__"):
            agent_output = graph.invoke(
                Command(resume=[{"type": "approve"}]),
                config=invoke_config,
            )

        payload = _extract_tool_payload(agent_output)
        if payload:
            result = _agent_result_from_payload(payload)
            return _save_result_for_thread(thread_id=thread_id, question=question, result=result, llm_provider=llm_provider, llm_model=llm_model)
    except Exception:
        payload = None

    fallback_tool_name = _select_fallback_tool_name(effective_question)
    payload = _run_named_tool(tools, fallback_tool_name, effective_question)
    if payload:
        payload["reasoning"] = (
            f"{payload.get('reasoning', '')} "
            "Se uso como respaldo porque el modelo no completo una invocacion de tool valida."
        ).strip()
        result = _agent_result_from_payload(payload)
        return _save_result_for_thread(thread_id=thread_id, question=question, result=result, llm_provider=llm_provider, llm_model=llm_model)

    qa_result: QAResult = answer_question(
        model=model,
        company_name=company_name,
        question=effective_question,
        knowledge_text=knowledge_text,
        chunks=chunks,
        max_context_chars=max_context_chars,
        vector_index_path=vector_index_path,
        conversation_history=history_context,
    )
    result = AgentResult(
        answer=qa_result.answer,
        route="consultar_informacion_corporativa",
        reasoning=(
            "No se obtuvo una tool valida; se respondio con RAG documental como respaldo final."
        ),
        context_mode=qa_result.context_mode,
        sources=qa_result.sources,
    )
    return _save_result_for_thread(thread_id=thread_id, question=question, result=result, llm_provider=llm_provider, llm_model=llm_model)
