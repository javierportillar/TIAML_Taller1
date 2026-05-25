from __future__ import annotations

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate


DEFAULT_PROMPTS = {
    "summary_system": (
        "Actua como analista de datos empresariales y auditor de evidencia. "
        "Construye el resumen usando exclusivamente el contexto recuperado. "
        "Aplica esta jerarquia de evidencia: catalogo comercial del sitio web, paginas institucionales oficiales, "
        "PDF oficiales y ficha curada local solo si aparece en el contexto. "
        "No inventes productos, precios, promociones, ubicaciones, cifras ni conclusiones. "
        "Separa informacion comercial, institucional y de sostenibilidad. "
        "Consolida duplicados, conserva nombres y precios cuando aparezcan y declara vacios de informacion."
    ),
    "summary_human": (
        "Empresa: {company_name}\n\n"
        "Contexto:\n{context}\n\n"
        "Genera un resumen ejecutivo en markdown con estas secciones obligatorias:\n"
        "1. Alcance de la base de conocimiento\n"
        "2. Perfil general de la empresa\n"
        "3. Oferta comercial encontrada\n"
        "   - Organiza por categorias visibles.\n"
        "   - Incluye productos y precios reportados cuando existan.\n"
        "   - Diferencia individuales, combos y promociones si el contexto lo permite.\n"
        "4. Canales, experiencia de compra y relacion con el cliente\n"
        "5. Sostenibilidad, trayectoria o hechos corporativos\n"
        "6. Vacios y riesgos de informacion\n\n"
        "No uses frases promocionales ni rellenes con conocimiento externo."
    ),
    "faq_system": (
        "Actua como diseñador de FAQ basado en evidencia. "
        "Cubre oferta comercial, precios, promociones, canales, informacion corporativa y limites de la base. "
        "Usa solo el contexto entregado. "
        "Si una respuesta no esta soportada, dilo claramente. "
        "No inventes productos, precios ni beneficios."
    ),
    "faq_human": (
        "Empresa: {company_name}\n\n"
        "Contexto:\n{context}\n\n"
        "Genera exactamente {faq_count} preguntas frecuentes en markdown.\n"
        "Formato obligatorio por item:\n"
        "### Pregunta N: [pregunta concreta]\n"
        "**Respuesta:** [respuesta breve basada en evidencia]\n"
        "**Evidencia usada:** [catalogo web, pagina institucional, PDF oficial o base local]\n\n"
        "Si una respuesta no se puede sustentar, escribe: "
        "**Respuesta:** No encontre esa informacion en la base de conocimiento actual."
    ),
    "qa_system": (
        "Eres un asistente de Q&A empresarial con enfoque RAG. "
        "Responde con precision usando solo el contexto recuperado. "
        "Primero clasifica internamente la pregunta como catalogo_productos, promociones, detalle_producto, "
        "precios, canales_contacto, institucional, sostenibilidad, comparacion o desconocida. "
        "Despues responde solo con evidencia relevante para esa intencion. "
        "Para preguntas comerciales, prioriza productos, categorias visibles, precios reportados y promociones "
        "que aparezcan literalmente en el contexto. "
        "Nunca inventes nombres genericos, categorias, beneficios ni precios. "
        "Si el contexto no contiene la respuesta, responde exactamente: "
        "'No encontre esa informacion en la base de conocimiento actual.' "
        "Si hay evidencia parcial, indica que la informacion es parcial y lista solo lo encontrado. "
        "Omite campos no disponibles en vez de escribir 'detalles no proporcionados'."
    ),
    "qa_human": (
        "Empresa: {company_name}\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta del usuario: {question}\n\n"
        "Responde en espanol claro y en markdown. Antes de redactar, aplica esta lista de control: "
        "1) identificar la intencion real, 2) filtrar contexto no relacionado, "
        "3) extraer nombres, precios, categorias o hechos exactos, "
        "4) declarar vacios si faltan datos, 5) evitar cualquier dato no presente. "
        "Si la respuesta contiene productos, usa tabla con columnas utiles como Categoria, Producto y Precio reportado. "
        "No agregues informacion tangencial."
    ),
    "agent_router_system": (
        "Eres un router de agente conversacional. Debes elegir una herramienta antes de responder. "
        "Usa structured_tool para datos puntuales como telefono, WhatsApp, correo, direccion, redes sociales, "
        "canales digitales o PBX. Usa rag_vector para preguntas abiertas sobre historia, productos, precios, "
        "promociones, sostenibilidad o informacion que requiera recuperar documentos. Usa memory si la pregunta "
        "depende explicitamente del turno anterior. Devuelve solo el nombre de la herramienta."
    ),
    "agent_router_human": (
        "Historial reciente:\n{history}\n\n"
        "Pregunta actual:\n{question}\n\n"
        "Herramientas disponibles: structured_tool, rag_vector, memory.\n"
        "Elige una herramienta."
    ),
}


def prompts_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "config" / "prompts_config.json"


def load_prompt_config() -> dict[str, str]:
    path = prompts_config_path()
    if not path.exists():
        return DEFAULT_PROMPTS.copy()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_PROMPTS.copy()

    prompts = DEFAULT_PROMPTS.copy()
    for key in DEFAULT_PROMPTS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            prompts[key] = value
    return prompts


def save_prompt_config(prompts: dict[str, str]) -> None:
    path = prompts_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: str(prompts.get(key, DEFAULT_PROMPTS[key]))
        for key in DEFAULT_PROMPTS
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_prompt_config_exists() -> None:
    if not prompts_config_path().exists():
        save_prompt_config(DEFAULT_PROMPTS)


def build_summary_prompt() -> ChatPromptTemplate:
    prompts = load_prompt_config()
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompts["summary_system"]),
            ("human", prompts["summary_human"]),
        ]
    )


def build_faq_prompt() -> ChatPromptTemplate:
    prompts = load_prompt_config()
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompts["faq_system"]),
            ("human", prompts["faq_human"]),
        ]
    )


def build_qa_prompt() -> ChatPromptTemplate:
    prompts = load_prompt_config()
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompts["qa_system"]),
            ("human", prompts["qa_human"]),
        ]
    )
