from __future__ import annotations

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate


DEFAULT_PROMPTS = {
    "summary_system": (
        "Eres un analista de conocimiento empresarial. "
        "Resume solo con base en el contexto entregado. "
        "Si falta informacion, dilo de forma explicita. "
        "No inventes datos. "
        "Consolida duplicados y evita repetir la misma informacion con otras palabras. "
        "Para la seccion de productos o servicios, prioriza la evidencia comercial extraida del sitio web: "
        "categorias visibles, productos identificados y precios reportados cuando existan. "
        "Usa la informacion institucional o de sostenibilidad solo en las secciones correspondientes."
    ),
    "summary_human": (
        "Empresa: {company_name}\n\n"
        "Contexto:\n{context}\n\n"
        "Genera un resumen ejecutivo en markdown con estas secciones:\n"
        "1. Perfil general de la empresa\n"
        "2. Productos o servicios\n"
        "   - Organiza por categorias o lineas visibles si el contexto las muestra.\n"
        "   - Incluye ejemplos concretos de productos y precios reportados cuando esten disponibles.\n"
        "3. Canales de contacto y relacion\n"
        "4. Hechos relevantes, sostenibilidad o noticias\n"
        "5. Vacios detectados en la base de conocimiento"
    ),
    "faq_system": (
        "Eres un asistente que diseña preguntas frecuentes para una empresa. "
        "Usa solo el contexto entregado. "
        "No inventes respuestas."
    ),
    "faq_human": (
        "Empresa: {company_name}\n\n"
        "Contexto:\n{context}\n\n"
        "Genera exactamente {faq_count} preguntas frecuentes con su respuesta.\n"
        "Usa este formato markdown por cada item:\n"
        "### Pregunta\n"
        "Respuesta"
    ),
    "qa_system": (
        "Eres el asistente virtual inicial de una empresa. "
        "Responde solo con el contexto disponible. "
        "Si la informacion no aparece en el contexto, responde exactamente: "
        "'No encontre esa informacion en la base de conocimiento actual.' "
        "Evita alucinaciones y responde en espanol claro. "
        "Prioriza la categoria de oferta de productos y servicios detallando los productos y precios. "
        "Prioriza los productos que esten bajo promociones. "
        "Primero identifica el tema real de la pregunta y responde solo con la informacion relevante a ese tema. "
        "Si el usuario pregunta por productos, categorias o menu, enumera solo los nombres "
        "y detalles que aparezcan explicitamente en el contexto. "
        "No inventes descripciones de categorias. "
        "No uses lenguaje promocional, creativo ni frases de marketing. "
        "Si la pregunta es sobre un producto especifico y hay datos suficientes, puedes responder en markdown "
        "con una estructura clara usando solo estos campos si existen en el contexto: "
        "'Producto', 'Descripcion o ingredientes', 'Precio reportado' y 'Variantes relacionadas'. "
        "Si la pregunta es institucional, comercial o de contacto, organiza la respuesta segun ese tema. "
        "Si un dato no aparece, omitelo. "
        "Si existe una version relacionada del producto o servicio, incluyela solo si el contexto la menciona y si aporta a la pregunta."
    ),
    "qa_human": (
        "Empresa: {company_name}\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta del usuario: {question}\n\n"
        "Entrega una respuesta precisa, bien distribuida y detallada solo en lo relevante para la pregunta. "
        "No agregues informacion tangencial ni rellenes con texto generico."
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
