from __future__ import annotations

import json
import re
from pathlib import Path

from .config import PathConfig
from .scraper import ScrapedDocument

STOPWORDS = {
    "de",
    "la",
    "el",
    "los",
    "las",
    "y",
    "en",
    "para",
    "con",
    "por",
    "del",
    "al",
    "un",
    "una",
    "que",
    "su",
    "sus",
    "se",
    "es",
    "como",
    "mas",
    "mas",
}

MENU_HINTS = {
    "menu",
    "menú",
    "producto",
    "productos",
    "oferta",
    "ofertas",
    "promocion",
    "promociones",
    "promoción",
    "promociones",
    "promo",
    "descuento",
    "descuentos",
    "ingrediente",
    "ingredientes",
    "precio",
    "precios",
    "trae",
    "contiene",
    "categoria",
    "categorias",
    "categoría",
    "categorías",
    "sandwich",
    "sándwich",
    "ensalada",
    "ensaladas",
    "hamburguesa",
    "hamburguesas",
    "wrap",
    "wraps",
    "qbowl",
    "qbowls",
    "combo",
    "combos",
}

CORPORATE_HINTS = {
    "historia",
    "sostenibilidad",
    "franquicia",
    "franquicias",
    "cobertura",
    "municipios",
    "sedes",
    "empresa",
    "fundacion",
    "fundación",
}


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_knowledge_base(company_name: str, documents: list[ScrapedDocument]) -> str:
    sections = [
        f"Empresa: {company_name}",
        "Base de conocimiento consolidada para el taller 1.",
        "",
    ]

    for index, document in enumerate(documents, start=1):
        section = [
            f"Fuente {index}",
            f"Titulo: {document.title}",
            f"URL: {document.url}",
        ]
        if document.meta_description:
            section.append(f"Descripcion: {document.meta_description}")
        section.extend(["Contenido:", document.text, ""])
        sections.append("\n".join(section))

    return normalize_text("\n\n".join(sections))


def _chunk_paragraphs(paragraphs: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    def build_overlap(source: list[str]) -> list[str]:
        if overlap <= 0 or not source:
            return []

        selected: list[str] = []
        total = 0
        for paragraph in reversed(source):
            paragraph_length = len(paragraph)
            if selected and total + paragraph_length > overlap:
                break
            selected.insert(0, paragraph)
            total += paragraph_length
            if total >= overlap:
                break
        return selected

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_length = len(paragraph)
        if current and current_length + paragraph_length > chunk_size:
            chunks.append("\n".join(current).strip())

            overlap_paragraphs = build_overlap(current)
            current = overlap_paragraphs + [paragraph]
            current_length = len("\n".join(current))
            continue

        current.append(paragraph)
        current_length += paragraph_length

    if current:
        chunks.append("\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def build_chunks(
    documents: list[ScrapedDocument],
    chunk_size: int = 1400,
    overlap: int = 180,
) -> list[dict[str, str | int]]:
    chunks: list[dict[str, str | int]] = []

    for document_index, document in enumerate(documents, start=1):
        paragraphs = [item.strip() for item in document.text.split("\n") if item.strip()]
        for chunk_index, chunk in enumerate(
            _chunk_paragraphs(paragraphs, chunk_size=chunk_size, overlap=overlap),
            start=1,
        ):
            chunks.append(
                {
                    "id": f"doc-{document_index}-chunk-{chunk_index}",
                    "title": document.title,
                    "url": document.url,
                    "content": chunk,
                    "char_count": len(chunk),
                }
            )

    return chunks


def build_source_index(documents: list[ScrapedDocument]) -> str:
    lines = ["# Indice de fuentes", ""]
    for index, document in enumerate(documents, start=1):
        lines.append(f"{index}. {document.title}")
        lines.append(f"   - URL: {document.url}")
        if document.meta_description:
            lines.append(f"   - Descripcion: {document.meta_description}")
    return "\n".join(lines).strip()


def save_processed_artifacts(
    company_name: str,
    documents: list[ScrapedDocument],
    paths: PathConfig,
) -> dict[str, int]:
    knowledge_base = build_knowledge_base(company_name, documents)
    chunks = build_chunks(documents)
    source_index = build_source_index(documents)

    paths.knowledge_base_path.write_text(knowledge_base, encoding="utf-8")
    paths.chunks_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths.source_index_path.write_text(source_index, encoding="utf-8")

    return {
        "documents": len(documents),
        "knowledge_chars": len(knowledge_base),
        "chunks": len(chunks),
    }


def load_knowledge_base(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_chunks(path: Path) -> list[dict[str, str | int]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z0-9]{3,}", text.lower()))
    return {token for token in tokens if token not in STOPWORDS}


def _normalize_match_text(text: str) -> str:
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


def select_relevant_chunks(
    question: str,
    chunks: list[dict[str, str | int]],
    top_k: int = 6,
) -> list[dict[str, str | int]]:
    question_terms = _tokenize(question)
    if not question_terms:
        return chunks[:top_k]

    question_lower = question.lower()
    normalized_question = _normalize_match_text(question)
    menu_query = any(term in question_lower for term in MENU_HINTS)
    corporate_query = any(term in question_lower for term in CORPORATE_HINTS)
    offer_query = any(
        term in normalized_question
        for term in ["oferta", "ofertas", "promocion", "promociones", "promo", "descuento", "descuentos"]
    )

    scored_chunks: list[tuple[int, dict[str, str | int]]] = []
    for chunk in chunks:
        content = str(chunk.get("content", ""))
        title = str(chunk.get("title", ""))
        url = str(chunk.get("url", "")).lower()
        title_lower = title.lower()
        normalized_chunk = _normalize_match_text(f"{title} {url} {content}")
        terms = _tokenize(f"{title} {url} {content}")
        score = len(question_terms & terms)

        if normalized_question and len(normalized_question) >= 8:
            if normalized_question in normalized_chunk:
                score += 10

        significant_overlap = len([term for term in question_terms if len(term) >= 5 and term in terms])
        if significant_overlap >= 2:
            score += 3

        if menu_query:
            if any(
                path in url
                for path in [
                    "/sandwich/",
                    "/wraps",
                    "/saludable",
                    "/otras-delicias",
                    "/promociones",
                    "sandwichqbano.com/",
                ]
            ):
                score += 2
            if url.endswith(".pdf"):
                score = 0

        if offer_query:
            if "/promociones" in url or "bestdiscount" in url:
                score += 8
            if "promo" in normalized_chunk or "promocion" in normalized_chunk or "descuento" in normalized_chunk:
                score += 4

        if corporate_query:
            if url.endswith(".pdf"):
                score += 2

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    if scored_chunks:
        return [chunk for _, chunk in scored_chunks[:top_k]]

    return chunks[:top_k]
