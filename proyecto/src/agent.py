from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .chains import QAResult, answer_question
from .memory import answer_follow_up_from_memory, format_chat_history
from .structured_tool import search_structured_data


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
) -> AgentResult:
    conversational_answer = _answer_conversational_message(question)
    if conversational_answer:
        return AgentResult(
            answer=conversational_answer,
            route="conversation",
            reasoning="La entrada es conversacional y no requiere consultar la base de conocimiento.",
            context_mode="conversation",
            sources=[],
        )

    memory_answer = answer_follow_up_from_memory(question, chat_history)
    if memory_answer:
        return AgentResult(
            answer=memory_answer,
            route="memory",
            reasoning="La pregunta depende del turno anterior; se resolvio con el historial de conversacion.",
            context_mode="conversation_memory",
            sources=[],
        )

    is_commercial = _looks_like_commercial_catalog_query(question)
    has_strong_structured = _has_strong_structured_signal(question)
    if (has_strong_structured or not is_commercial) and (
        _looks_like_structured_query(question) or _looks_like_location_query(question)
    ):
        structured_result = search_structured_data(question, structured_data_path)
        if structured_result:
            return AgentResult(
                answer=structured_result.answer,
                route="structured_tool",
                reasoning=(
                    "La consulta pide un dato puntual estructurado. "
                    "Se uso la herramienta deterministica de datos estructurados."
                ),
                context_mode="structured_json",
                sources=structured_result.sources,
            )

    effective_question = _rewrite_combo_difference_follow_up(question, chat_history)
    history_context = format_chat_history(chat_history)
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
    return AgentResult(
        answer=qa_result.answer,
        route="rag_vector",
        reasoning=(
            "La consulta requiere informacion abierta de la base documental. "
            "Se enruto al RAG con indice vectorial persistido e historial conversacional."
        ),
        context_mode=qa_result.context_mode,
        sources=qa_result.sources,
    )
