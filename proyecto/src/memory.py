from __future__ import annotations

import re


# Relaciones / sujetos que aparecen en una declaracion personal del usuario.
_PERSONAL_RELATIONS = (
    r"papa|pap[aá]|mama|mam[aá]|hermano|hermana|hijo|hija|"
    r"esposa|esposo|novia|novio|amigo|amiga|jefe|jefa|"
    r"profesor|profesora|colega|primo|prima|tio|tia|t[ií]o|t[ií]a|"
    r"abuelo|abuela|nieto|nieta"
)


def _is_personal_declaration(question: str) -> bool:
    """Detecta declaraciones tipo 'mi papa se llama X', 'yo soy X', 'mi nombre es X'.

    Sirve para responder con un acuse de recibo simple y dejar el dato en el historial,
    sin disparar el RAG ni el router de tools.
    """

    normalized = question.lower()
    if re.search(rf"\bmi\s+(?:{_PERSONAL_RELATIONS})\s+(?:se llama|es)\s+\w+", normalized):
        return True
    if re.search(r"\b(?:me llamo|mi nombre es|yo soy)\s+\w+", normalized):
        # Lo capturamos solo si NO es parte de un saludo (eso ya lo maneja el modulo conversacional).
        if not re.match(r"^\s*(?:hola|buenas|buenos dias|buenas tardes|buenas noches)\b", normalized):
            return True
    return False


def _is_personal_question(question: str) -> bool:
    """Detecta preguntas que solo se pueden responder mirando el historial conversacional."""

    normalized = question.lower()
    patterns = [
        # "como me llamo" / "cual es mi nombre"
        r"\bc[oó]mo me llamo\b",
        r"\b(?:cu[aá]l|qu[eé])\s+es\s+mi\s+(?:nombre|edad|trabajo|profesion|profesi[oó]n)\b",
        # "como se llama mi papa" / "cual es el nombre de mi mama"
        rf"\bc[oó]mo se llama mi\s+(?:{_PERSONAL_RELATIONS})\b",
        rf"\b(?:cu[aá]l|qu[eé])\s+es\s+el\s+nombre\s+de\s+mi\s+(?:{_PERSONAL_RELATIONS})\b",
        # referencias al historial
        r"\b(?:qu[eé]|que|c[oó]mo)\s+te\s+(?:dije|coment[eé]|cont[eé])\b",
        r"\brecuerdas\s+(?:lo\s+que|que|mi)\b",
        r"\bya\s+te\s+(?:dije|hab[ií]a dicho)\b",
    ]
    return any(re.search(p, normalized) for p in patterns)


def acknowledge_personal_declaration(question: str) -> str | None:
    """Devuelve un acuse breve para declaraciones personales o None si no aplica."""

    if not _is_personal_declaration(question):
        return None
    return (
        "Entendido. Lo tendré en cuenta durante nuestra conversación. "
        "Si me preguntas más adelante, puedo recordarlo."
    )


def answer_personal_question_from_memory(
    question: str,
    messages: list[dict[str, str]],
    model,
) -> str | None:
    """Responde preguntas sobre informacion personal usando solo el historial conversacional.

    No consulta la base vectorial ni los datos estructurados de la empresa: esos no contienen
    informacion personal del usuario. Usa el LLM activo (Ollama, Gemini, OpenAI) para sintetizar
    la respuesta a partir del historial.
    """

    if not _is_personal_question(question):
        return None

    history = format_chat_history(messages, max_turns=20)
    if history == "Sin historial previo.":
        return (
            "No tengo registrado eso en nuestra conversacion. "
            "¿Me lo puedes recordar?"
        )

    prompt = (
        "Eres un asistente conversacional. Responde la pregunta usando UNICAMENTE el historial "
        "siguiente. No inventes informacion ni uses conocimiento externo.\n\n"
        f"Historial:\n{history}\n\n"
        f"Pregunta actual: {question}\n\n"
        "Reglas:\n"
        "- Si el historial contiene la respuesta, respondela en 1 o 2 frases breves.\n"
        "- Si NO esta en el historial, responde EXACTAMENTE: "
        '"No tengo registrado eso en nuestra conversacion. ¿Me lo puedes recordar?"'
    )
    try:
        response = model.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        text = str(text).strip()
        return text or None
    except Exception:
        return None


def format_chat_history(messages: list[dict[str, str]], max_turns: int = 6) -> str:
    if not messages:
        return "Sin historial previo."

    selected = messages[-max_turns * 2 :]
    lines: list[str] = []
    for message in selected:
        role = "Usuario" if message.get("role") == "user" else "Asistente"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "Sin historial previo."


def answer_follow_up_from_memory(question: str, messages: list[dict[str, str]]) -> str | None:
    normalized = question.lower()
    asks_difference = any(term in normalized for term in ["diferencia", "diferencias", "comparar", "compara"])
    if asks_difference and any(term in normalized for term in ["precio", "precios", "combo", "combos"]):
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content", "")).lower()
            if "combo" in content and ("individual" in content or "sin combo" in content or "sándwiches en combo" in content):
                return None

    if not any(term in normalized for term in ["primero", "primera", "anterior", "mencionaste", "dijiste"]):
        return None
    if not any(term in normalized for term in ["cuanto", "cuánto", "precio", "cuesta", "vale", "valen"]):
        return None

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        for line in content.splitlines():
            match = re.match(r"^-\s+\**(?P<name>[^:*]+)\**:\s+(?P<price>[\d.$,\sA-ZCOP]+)", line.strip())
            if match:
                name = match.group("name").strip()
                price = match.group("price").strip()
                return (
                    "Tomando como referencia el primer producto mencionado en la respuesta anterior:\n\n"
                    f"- **{name}:** {price}"
                )
    return None
