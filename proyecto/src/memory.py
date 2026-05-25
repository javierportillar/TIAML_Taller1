from __future__ import annotations

import re


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
