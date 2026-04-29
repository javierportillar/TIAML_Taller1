from __future__ import annotations

from dataclasses import dataclass
import html
import re

from langchain_core.output_parsers import StrOutputParser

from .processing import select_relevant_chunks
from .prompts import build_faq_prompt, build_qa_prompt, build_summary_prompt


@dataclass(slots=True)
class QAResult:
    answer: str
    context_mode: str
    sources: list[dict[str, str]]


def _run_chain(prompt, model, payload: dict[str, object]) -> str:
    chain = prompt | model | StrOutputParser()
    return chain.invoke(payload).strip()


def _clean_field(value: str) -> str:
    return html.unescape(value).strip().strip(".").strip()


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


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _parse_categories(chunks: list[dict[str, str | int]]) -> list[str]:
    categories: list[str] = []
    for chunk in chunks:
        for line in str(chunk.get("content", "")).splitlines():
            if not line.startswith("Categorias visibles del sitio:"):
                continue
            values = line.split(":", maxsplit=1)[1]
            for item in values.split(","):
                category = _clean_field(item)
                if category and category not in categories:
                    categories.append(category)
    return categories


def _parse_products_by_category(
    chunks: list[dict[str, str | int]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {
        "Destacados y combos": [],
        "Sándwiches individuales": [],
        "Q-Bowls": [],
        "Wraps": [],
        "Ensaladas": [],
    }
    seen_names: set[str] = set()

    def resolve_category(url: str, name: str) -> str | None:
        url = url.lower()
        name_lower = name.lower()
        if "/sandwich/individual" in url:
            return "Sándwiches individuales"
        if "/saludable/q-bowls" in url:
            return "Q-Bowls"
        if "/wraps" in url:
            return "Wraps"
        if "/saludable" in url and "ensalada" in name_lower:
            return "Ensaladas"
        if url == "https://www.sandwichqbano.com/":
            return "Destacados y combos"
        return None

    for chunk in chunks:
        url = str(chunk.get("url", ""))
        content = str(chunk.get("content", ""))
        for line in content.splitlines():
            if not line.startswith("Producto: "):
                continue
            try:
                product_part = line[len("Producto: ") :]
                name_part, remainder = product_part.split(". Descripcion:", maxsplit=1)
                description_part, price_part = remainder.split(". Precio reportado:", maxsplit=1)
            except ValueError:
                continue

            name = _clean_field(name_part)
            if not name or name in seen_names:
                continue

            category = resolve_category(url, name)
            if not category:
                continue

            description = _clean_field(description_part)
            price = _clean_field(price_part)
            grouped[category].append(
                {
                    "name": name,
                    "description": description,
                    "price": price,
                }
            )
            seen_names.add(name)

    return grouped


def _render_bullet_lines(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items if item]


def _build_deterministic_summary(
    company_name: str,
    chunks: list[dict[str, str | int]],
) -> str:
    categories = _parse_categories(chunks)
    products = _parse_products_by_category(chunks)

    if not (categories or any(products.values())):
        return ""

    lines: list[str] = [f"## Resumen Ejecutivo de {company_name}", ""]

    lines.extend(["### 1. Perfil general de la empresa", ""])
    if profile_facts:
        lines.extend(_render_bullet_lines(profile_facts[:6]))
    else:
        lines.append("- No se encontraron suficientes hechos corporativos confirmados en la base actual.")
    if coverage_facts:
        lines.append("")
        lines.append("Cobertura y operación reportada:")
        lines.extend(_render_bullet_lines(coverage_facts[:6]))
    lines.append("")

    lines.extend(["### 2. Productos o servicios", ""])
    if categories:
        lines.append(f"- Categorías visibles del sitio: {', '.join(categories)}.")
    lines.append("")

    for category_name in [
        "Destacados y combos",
        "Sándwiches individuales",
        "Q-Bowls",
        "Wraps",
        "Ensaladas",
    ]:
        category_products = products.get(category_name, [])
        if not category_products:
            continue
        lines.append(f"**{category_name}:**")
        for product in category_products[:5]:
            lines.append(
                f"- {product['name']}: {product['price']}."
            )
        lines.append("")

    lines.extend(["### 3. Canales de contacto y relación", ""])
    lines.append("- No se identificaron canales de contacto estructurados en el catálogo comercial scrapeado.")
    lines.append("")

    lines.extend(["### 4. Hechos relevantes, sostenibilidad o noticias", ""])
    lines.append("- Esta sección debe completarse desde las fuentes institucionales scrapeadas cuando el modelo reciba el contexto completo.")
    lines.append("")

    lines.extend(["### 5. Vacíos detectados en la base de conocimiento", ""])
    lines.append("- No incluye inventario en tiempo real por sede ni disponibilidad dinámica del carrito.")

    return "\n".join(lines).strip()


def _chunk_order(chunk: dict[str, str | int]) -> int:
    chunk_id = str(chunk.get("id", ""))
    try:
        return int(chunk_id.rsplit("-", maxsplit=1)[-1])
    except ValueError:
        return 9999


def _render_summary_context(chunks: list[dict[str, str | int]], max_context_chars: int) -> str:
    def is_pdf(chunk: dict[str, str | int]) -> bool:
        return str(chunk.get("url", "")).lower().endswith(".pdf")

    def is_menu_source(chunk: dict[str, str | int]) -> bool:
        url = str(chunk.get("url", "")).lower()
        return any(
            token in url
            for token in [
                "sandwichqbano.com/",
                "/sandwich/",
                "/wraps",
                "/saludable",
                "/otras-delicias",
            ]
        ) and not is_pdf(chunk)

    def pdf_priority(chunk: dict[str, str | int]) -> int:
        content = str(chunk.get("content", "")).lower()
        score = 0
        if "quiénes somos" in content or "quienes somos" in content:
            score += 6
        if "municipios" in content or "franquicias" in content:
            score += 4
        if "sostenibilidad" in content or "ods" in content:
            score += 3
        if "contacto" in content:
            score += 2
        return score

    def add_chunks(
        selected: list[dict[str, str | int]],
        candidates: list[dict[str, str | int]],
        max_per_url: int,
        current_chars: int,
    ) -> int:
        url_counts: dict[str, int] = {}
        for chunk in selected:
            url = str(chunk.get("url", ""))
            url_counts[url] = url_counts.get(url, 0) + 1

        for chunk in candidates:
            url = str(chunk.get("url", ""))
            if url_counts.get(url, 0) >= max_per_url:
                continue
            block = _render_chunk_context([chunk])
            if selected and current_chars + len(block) > max_context_chars:
                continue
            selected.append(chunk)
            url_counts[url] = url_counts.get(url, 0) + 1
            current_chars += len(block) + 6
        return current_chars

    selected: list[dict[str, str | int]] = []
    current_chars = 0

    menu_chunks = sorted(
        [chunk for chunk in chunks if is_menu_source(chunk)],
        key=lambda chunk: (
            0 if str(chunk.get("url", "")).lower() == "https://www.sandwichqbano.com/" else 1,
            0 if "precio reportado" in str(chunk.get("content", "")).lower() else 1,
            _chunk_order(chunk),
        ),
    )
    current_chars = add_chunks(selected, menu_chunks, max_per_url=2, current_chars=current_chars)

    pdf_chunks = sorted(
        [chunk for chunk in chunks if is_pdf(chunk)],
        key=lambda chunk: (pdf_priority(chunk), -_chunk_order(chunk)),
        reverse=True,
    )
    current_chars = add_chunks(selected, pdf_chunks, max_per_url=2, current_chars=current_chars)

    if not selected:
        return ""

    return _render_chunk_context(selected)[:max_context_chars]


def generate_summary(
    model,
    company_name: str,
    knowledge_text: str,
    chunks: list[dict[str, str | int]],
    max_context_chars: int,
) -> str:
    deterministic_summary = _build_deterministic_summary(company_name, chunks)
    if deterministic_summary:
        if model is None:
            return deterministic_summary
        return _run_chain(
            build_summary_prompt(),
            model,
            {
                "company_name": company_name,
                "context": deterministic_summary[:max_context_chars],
            },
        )

    context = _render_summary_context(chunks, max_context_chars=max_context_chars)
    if not context:
        context = knowledge_text[:max_context_chars]

    return _run_chain(
        build_summary_prompt(),
        model,
        {"company_name": company_name, "context": context},
    )


def generate_faq(
    model,
    company_name: str,
    knowledge_text: str,
    faq_count: int = 8,
) -> str:
    return _run_chain(
        build_faq_prompt(),
        model,
        {
            "company_name": company_name,
            "context": knowledge_text,
            "faq_count": faq_count,
        },
    )


def _parse_product_records(chunks: list[dict[str, str | int]]) -> list[dict[str, str]]:
    best_by_name: dict[str, dict[str, str | int]] = {}

    def record_score(name: str, description: str, price: str, url: str) -> int:
        score = 0
        normalized_name = _normalize_match_text(name)
        normalized_price = _normalize_match_text(price)
        normalized_url = url.lower().rstrip("/")

        if "desde" not in normalized_price and "hasta" not in normalized_price:
            score += 5
        if normalized_url.endswith("/sandwich/individual") and "individual" in normalized_name:
            score += 4
        elif normalized_url.endswith("/sandwich"):
            score += 3
        elif normalized_url.endswith("/sandwich/individual"):
            score += 2
        elif normalized_url.endswith("/"):
            score -= 1
        if len(description) >= 20:
            score += 1
        return score

    for chunk in chunks:
        url = str(chunk.get("url", ""))
        for line in str(chunk.get("content", "")).splitlines():
            if not line.startswith("Producto: "):
                continue

            try:
                product_part = line[len("Producto: ") :]
                name_part, remainder = product_part.split(". Descripcion:", maxsplit=1)
                description_part, price_part = remainder.split(". Precio reportado:", maxsplit=1)
            except ValueError:
                continue

            name = _clean_field(name_part)
            description = _clean_field(description_part)
            price = _clean_field(price_part)
            normalized_name = _normalize_match_text(name)
            if not normalized_name or not price:
                continue

            candidate = {
                "name": name,
                "description": description,
                "price": price,
                "url": url,
                "_score": record_score(name, description, price, url),
            }
            current = best_by_name.get(normalized_name)
            if current is None or int(candidate["_score"]) > int(current["_score"]):
                best_by_name[normalized_name] = candidate

    records = list(best_by_name.values())
    records.sort(key=lambda item: _normalize_match_text(str(item["name"])))
    return [
        {
            "name": str(item["name"]),
            "description": str(item["description"]),
            "price": str(item["price"]),
            "url": str(item["url"]),
        }
        for item in records
    ]


def _is_exhaustive_product_question(question: str) -> bool:
    normalized_question = _normalize_match_text(question)
    has_list_intent = any(
        phrase in normalized_question
        for phrase in [
            "todos",
            "todas",
            "lista",
            "enlista",
            "enlistes",
            "listame",
            "nombres",
            "cuales son",
        ]
    )
    has_product_intent = any(
        term in normalized_question
        for term in [
            "sandwich",
            "qbanito",
            "wrap",
            "qbowl",
            "q bowl",
            "ensalada",
            "hamburguesa",
            "productos",
            "menu",
        ]
    )
    has_price_intent = any(
        term in normalized_question
        for term in [
            "precio",
            "precios",
            "cuanto",
            "cuestan",
            "valen",
        ]
    )
    return (has_list_intent and has_product_intent) or (has_product_intent and has_price_intent)


def _is_offer_question(question: str) -> bool:
    normalized_question = _normalize_match_text(question)
    return any(
        term in normalized_question
        for term in [
            "promocion",
            "promociones",
            "promo",
            "oferta",
            "ofertas",
            "descuento",
            "descuentos",
        ]
    )


def _resolve_product_scope(question: str) -> tuple[str | None, str | None]:
    normalized_question = _normalize_match_text(question)

    scope: str | None = None
    if "qbanito" in normalized_question:
        scope = "qbanito"
    elif "sandwich" in normalized_question:
        scope = "sandwich"
    elif "wrap" in normalized_question:
        scope = "wrap"
    elif "qbowl" in normalized_question or "q bowl" in normalized_question:
        scope = "qbowl"
    elif "ensalada" in normalized_question:
        scope = "ensalada"
    elif "hamburguesa" in normalized_question:
        scope = "hamburguesa"

    variant: str | None = None
    if "combo" in normalized_question and "individual" not in normalized_question:
        variant = "combo"
    elif "individual" in normalized_question and "combo" not in normalized_question:
        variant = "individual"

    return scope, variant


def _matches_product_scope(
    product: dict[str, str],
    scope: str | None,
    variant: str | None,
) -> bool:
    normalized_name = _normalize_match_text(product["name"])

    if scope == "sandwich" and "sandwich" not in normalized_name:
        return False
    if scope == "qbanito" and "qbanito" not in normalized_name:
        return False
    if scope == "wrap" and "wrap" not in normalized_name:
        return False
    if scope == "qbowl" and "qbowl" not in normalized_name and "q bowl" not in normalized_name:
        return False
    if scope == "ensalada" and "ensalada" not in normalized_name:
        return False
    if scope == "hamburguesa" and "hamburguesa" not in normalized_name:
        return False

    if variant == "combo" and "combo" not in normalized_name:
        return False
    if variant == "individual" and "individual" not in normalized_name:
        return False

    return True


def _group_product_label(product_name: str) -> tuple[int, str]:
    normalized_name = _normalize_match_text(product_name)
    if "combo" in normalized_name:
        return 0, "Combos"
    if "individual" in normalized_name:
        return 1, "Individuales"
    return 2, "Otros"


def _build_exhaustive_product_answer(
    question: str,
    chunks: list[dict[str, str | int]],
) -> tuple[str, list[dict[str, str]]] | None:
    if not _is_exhaustive_product_question(question):
        return None

    products = _parse_product_records(chunks)
    scope, variant = _resolve_product_scope(question)
    if not scope:
        return None

    filtered = [
        product for product in products if _matches_product_scope(product, scope=scope, variant=variant)
    ]
    if not filtered:
        return None

    filtered.sort(key=lambda item: (_group_product_label(item["name"])[0], _normalize_match_text(item["name"])))

    intro = f"Encontré {len(filtered)} productos con precio reportado en la base actual."
    if scope == "sandwich":
        qbanitos = [product for product in products if _matches_product_scope(product, scope="qbanito", variant=None)]
        total_sandwich_category = len(filtered) + len(qbanitos)
        if qbanitos:
            intro = (
                f"Encontré {len(filtered)} sándwiches con precio reportado en la base actual. "
                f"La categoría `/sandwich` reúne {total_sandwich_category} productos en total y "
                f"{len(qbanitos)} de ellos son qbanitos."
            )

    lines = [intro, ""]

    grouped: dict[str, list[dict[str, str]]] = {"Combos": [], "Individuales": [], "Otros": []}
    for product in filtered:
        grouped[_group_product_label(product["name"])[1]].append(product)

    for label in ["Combos", "Individuales", "Otros"]:
        items = grouped[label]
        if not items:
            continue
        lines.append(f"**{label}**")
        for item in items:
            lines.append(f"- {item['name']}: {item['price']}")
        lines.append("")

    sources = _dedupe_preserve_order([item["url"] for item in filtered])
    return (
        "\n".join(lines).strip(),
        [{"title": "Fuente de productos", "url": url} for url in sources[:6]],
    )


def _build_offer_answer(
    question: str,
    chunks: list[dict[str, str | int]],
) -> tuple[str, list[dict[str, str]]] | None:
    if not _is_offer_question(question):
        return None

    products = _parse_product_records(chunks)
    filtered = [
        product
        for product in products
        if "/promociones" in product["url"].lower()
    ]
    if not filtered:
        return None

    filtered.sort(key=lambda item: _normalize_match_text(item["name"]))
    lines = [f"Encontré {len(filtered)} promociones u ofertas con precio reportado en la base actual.", ""]
    for item in filtered:
        lines.append(f"- {item['name']}: {item['price']}")

    return (
        "\n".join(lines).strip(),
        [{"title": "Fuente de promociones", "url": "https://www.sandwichqbano.com/promociones?order=OrderByBestDiscountDESC"}],
    )


def _render_chunk_context(chunks: list[dict[str, str | int]]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        parts.append(
            "\n".join(
                [
                    f"Titulo: {chunk['title']}",
                    f"URL: {chunk['url']}",
                    "Contenido:",
                    str(chunk["content"]),
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def build_question_context(
    question: str,
    knowledge_text: str,
    chunks: list[dict[str, str | int]],
    max_context_chars: int,
) -> tuple[str, str, list[dict[str, str]]]:
    def dedupe_sources(source_items: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for item in source_items:
            key = item["url"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    if len(knowledge_text) <= max_context_chars:
        selected_chunks = select_relevant_chunks(question, chunks, top_k=4)
        sources = dedupe_sources([
            {"title": str(chunk["title"]), "url": str(chunk["url"])}
            for chunk in selected_chunks
        ])
        return knowledge_text, "full", sources

    selected_chunks = select_relevant_chunks(question, chunks, top_k=6)
    context = _render_chunk_context(selected_chunks)[:max_context_chars]
    sources = dedupe_sources([
        {"title": str(chunk["title"]), "url": str(chunk["url"])}
        for chunk in selected_chunks
    ])
    return context, "focused", sources


def answer_question(
    model,
    company_name: str,
    question: str,
    knowledge_text: str,
    chunks: list[dict[str, str | int]],
    max_context_chars: int,
) -> QAResult:
    offer_answer = _build_offer_answer(question=question, chunks=chunks)
    if offer_answer:
        answer, sources = offer_answer
        return QAResult(answer=answer, context_mode="deterministic", sources=sources)

    deterministic_answer = _build_exhaustive_product_answer(question=question, chunks=chunks)
    if deterministic_answer:
        answer, sources = deterministic_answer
        return QAResult(answer=answer, context_mode="deterministic", sources=sources)

    context, context_mode, sources = build_question_context(
        question=question,
        knowledge_text=knowledge_text,
        chunks=chunks,
        max_context_chars=max_context_chars,
    )
    answer = _run_chain(
        build_qa_prompt(),
        model,
        {
            "company_name": company_name,
            "context": context,
            "question": question,
        },
    )
    return QAResult(answer=answer, context_mode=context_mode, sources=sources)
