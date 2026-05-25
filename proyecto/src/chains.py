from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re

from langchain_core.output_parsers import StrOutputParser

from .processing import select_relevant_chunks
from .prompts import build_faq_prompt, build_qa_prompt, build_summary_prompt
from .vector_store import search_vector_index


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


def _parse_profile_facts(chunks: list[dict[str, str | int]]) -> tuple[list[str], list[str]]:
    profile_facts: list[str] = []
    coverage_facts: list[str] = []
    seen: set[str] = set()

    patterns = [
        r"Qbano es una empresa colombiana[^.]+(?:\.)",
        r"Qbano, con 44 años de historia[^.]+(?:\.)",
        r"Mantener el liderazgo[^.]+(?:\.)",
        r"Preparar y entregar un producto[^.]+(?:\.)",
    ]
    coverage_patterns = [
        r"Atendemos a nuestros clientes[^.]+(?:\.)",
        r"En ventas por domicilios[^.]+(?:\.)",
        r"LINEA WHATSAPP TELEFONOS EN PUNTO DE VENTA APLICACIONES[^.]+(?:\.)",
    ]

    for chunk in chunks:
        content = str(chunk.get("content", ""))
        if not str(chunk.get("url", "")).lower().endswith(".pdf"):
            continue
        for pattern in patterns:
            for match in re.findall(pattern, content, flags=re.IGNORECASE):
                fact = _clean_field(match)
                key = _normalize_match_text(fact)
                if fact and key not in seen:
                    profile_facts.append(fact)
                    seen.add(key)
        for pattern in coverage_patterns:
            for match in re.findall(pattern, content, flags=re.IGNORECASE):
                fact = _clean_field(match)
                key = _normalize_match_text(fact)
                if fact and key not in seen:
                    coverage_facts.append(fact)
                    seen.add(key)

    return profile_facts, coverage_facts


def _render_bullet_lines(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items if item]


def _build_deterministic_summary(
    company_name: str,
    chunks: list[dict[str, str | int]],
) -> str:
    categories = _parse_categories(chunks)
    products = _parse_products_by_category(chunks)
    profile_facts, coverage_facts = _parse_profile_facts(chunks)

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


def _has_sandwich_term(normalized_text: str) -> bool:
    aliases = [
        "sandwich",
        "sandwiches",
        "sanwich",
        "sanwiches",
        "sanwdich",
        "sanwdiches",
        "sandiwch",
        "sandiwches",
        "sandwiche",
        "sandwichs",
        "sanduche",
        "sanduches",
    ]
    return any(alias in normalized_text for alias in aliases)


def _has_price_intent(normalized_question: str) -> bool:
    return any(
        term in normalized_question
        for term in [
            "precio",
            "precios",
            "cuanto",
            "cuanto cuesta",
            "cuanto vale",
            "cuestan",
            "vale",
            "valen",
            "costo",
            "costos",
        ]
    )


def _is_exhaustive_product_question(question: str) -> bool:
    normalized_question = _normalize_match_text(question)
    non_catalog_intent = [
        "factura",
        "facturacion",
        "pago",
        "pagos",
        "horario",
        "horarios",
        "domicilio",
        "domicilios",
        "entrega",
        "entregas",
        "disponible",
        "disponibilidad",
        "inventario",
    ]
    if any(term in normalized_question for term in non_catalog_intent):
        return False

    commercial_phrases = [
        "que venden",
        "que productos venden",
        "que opciones hay",
        "opciones hay",
        "opciones tienen",
        "que ofrecen",
        "que hay para comer",
        "menu completo",
        "catalogo",
        "carta",
        "oferta comercial",
        "lineas de producto",
        "linea de productos",
        "portafolio",
        "dime el menu",
        "muestrame el menu",
        "muestrame productos",
        "ver productos",
        "que sabores",
        "cuales sabores",
        "que tipos",
        "que variedades",
        "que variantes",
        "puedo pedir",
        "puedo comer",
        "puedo elegir",
        "para pedir",
        "para comer",
        "en combo",
        "en combos",
        "productos en combo",
        "productos que sean en combo",
        "sandwiches en combo",
        "sanwiches en combo",
        "sandiwches en combo",
    ]
    has_broad_commercial_intent = any(
        phrase in normalized_question for phrase in commercial_phrases
    )
    has_list_intent = any(
        phrase in normalized_question
        for phrase in [
            "todos",
            "todas",
            "lista",
            "enlista",
            "enlistes",
            "listame",
            "muestrame",
            "mostrar",
            "indicame",
            "indica",
            "nombres",
            "cuales son",
            "hay",
            "existen",
            "que hay",
            "que tienen",
            "que tienes",
            "tienen",
            "tienes",
            "hay disponible",
            "disponibles",
            "tipos de",
            "sabores",
            "variedades",
            "variantes",
        ]
    )
    has_product_intent = any(
        term in normalized_question
        for term in [
            "sandwich",
            "sandwiches",
            "qbanito",
            "wrap",
            "qbowl",
            "q bowl",
            "ensalada",
            "hamburguesa",
            "productos",
            "menu",
            "catalogo",
            "carta",
            "opciones",
            "venden",
            "ofrecen",
            "pedir",
            "comer",
            "elegir",
            "combo",
            "combos",
            "sabor",
            "sabores",
            "tipo",
            "tipos",
            "variedad",
            "variedades",
            "variante",
            "variantes",
        ]
    ) or _has_sandwich_term(normalized_question)
    has_price_intent = _has_price_intent(normalized_question)
    has_combo_intent = "combo" in normalized_question or "combos" in normalized_question
    return (
        has_broad_commercial_intent
        or (has_list_intent and has_product_intent)
        or (has_product_intent and has_price_intent)
        or (has_combo_intent and has_product_intent)
    )


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
    elif _has_sandwich_term(normalized_question):
        scope = "sandwich"
    elif "wrap" in normalized_question:
        scope = "wrap"
    elif "qbowl" in normalized_question or "q bowl" in normalized_question:
        scope = "qbowl"
    elif "ensalada" in normalized_question:
        scope = "ensalada"
    elif "hamburguesa" in normalized_question:
        scope = "hamburguesa"
    elif any(
        term in normalized_question
        for term in [
            "producto",
            "productos",
            "menu",
            "catalogo",
            "carta",
            "opciones",
            "venden",
            "ofrecen",
            "portafolio",
            "combo",
            "combos",
            "pedir",
            "comer",
            "elegir",
            "sabor",
            "sabores",
            "tipo",
            "tipos",
            "variedad",
            "variedades",
            "variante",
            "variantes",
        ]
    ):
        scope = "all"

    variant: str | None = None
    if "combo" in normalized_question and "individual" not in normalized_question:
        variant = "combo"
    elif "individual" in normalized_question and "combo" not in normalized_question:
        variant = "individual"

    if scope is None and variant == "combo":
        scope = "all"

    return scope, variant


def _matches_product_scope(
    product: dict[str, str],
    scope: str | None,
    variant: str | None,
) -> bool:
    normalized_name = _normalize_match_text(product["name"])

    if scope == "all":
        pass
    elif scope is None:
        return False
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


PRODUCT_QUERY_STOPWORDS = {
    "a",
    "al",
    "con",
    "compara",
    "comparacion",
    "comparar",
    "cual",
    "cuales",
    "cuanto",
    "cuesta",
    "cuestan",
    "de",
    "del",
    "diferencia",
    "diferencias",
    "dime",
    "el",
    "en",
    "entre",
    "hay",
    "la",
    "las",
    "los",
    "me",
    "precio",
    "precios",
    "producto",
    "productos",
    "que",
    "sin",
    "tiene",
    "tienen",
    "un",
    "una",
    "vale",
    "valen",
    "y",
}

PRODUCT_GENERIC_TOKENS = {
    "combo",
    "combos",
    "individual",
    "personales",
    "personal",
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
    "wrap",
    "wraps",
    "qbowl",
    "qbanito",
    "hamburguesa",
    "ensalada",
    "producto",
    "productos",
}


def _distinctive_product_tokens(text: str) -> set[str]:
    tokens = set(_normalize_match_text(text).split())
    return {
        token
        for token in tokens
        if len(token) > 2
        and token not in PRODUCT_QUERY_STOPWORDS
        and token not in PRODUCT_GENERIC_TOKENS
    }


def _requested_product_variants(normalized_question: str) -> list[str | None]:
    wants_combo = "combo" in normalized_question or "combos" in normalized_question
    wants_individual = (
        "individual" in normalized_question
        or "sin combo" in normalized_question
        or "sin combos" in normalized_question
        or "no combo" in normalized_question
    )
    if wants_combo and wants_individual:
        return ["combo", "individual"]
    if wants_combo:
        return ["combo"]
    if wants_individual:
        return ["individual"]
    return [None]


def _build_specific_product_price_answer(
    question: str,
    chunks: list[dict[str, str | int]],
) -> tuple[str, list[dict[str, str]]] | None:
    normalized_question = _normalize_match_text(question)
    if not _has_price_intent(normalized_question):
        return None

    query_tokens = _distinctive_product_tokens(question)
    if not query_tokens:
        return None

    products = _parse_product_records(chunks)
    scope, _ = _resolve_product_scope(question)
    variants = _requested_product_variants(normalized_question)

    matches: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for variant in variants:
        scoped_matches: list[tuple[int, dict[str, str]]] = []
        for product in products:
            if scope and not _matches_product_scope(product, scope=scope, variant=variant):
                continue
            if not scope and variant and not _matches_product_scope(product, scope="all", variant=variant):
                continue
            product_tokens = _distinctive_product_tokens(product["name"])
            overlap = query_tokens & product_tokens
            if not overlap:
                continue
            score = len(overlap) * 10
            if query_tokens <= product_tokens:
                score += 8
            if scope:
                score += 3
            scoped_matches.append((score, product))

        scoped_matches.sort(
            key=lambda item: (
                -item[0],
                _group_product_label(item[1]["name"])[0],
                _normalize_match_text(item[1]["name"]),
            )
        )
        if scoped_matches:
            best_score = scoped_matches[0][0]
            for score, product in scoped_matches:
                if score < best_score:
                    break
                normalized_name = _normalize_match_text(product["name"])
                if normalized_name not in seen_names:
                    matches.append(product)
                    seen_names.add(normalized_name)

    if not matches:
        return None

    matches.sort(key=lambda item: (_group_product_label(item["name"])[0], _normalize_match_text(item["name"])))
    lines = ["Encontré estos precios para el producto solicitado:", ""]
    for product in matches:
        lines.append(f"- **{product['name']}:** {product['price']}")

    sources = _dedupe_preserve_order([item["url"] for item in matches])
    return (
        "\n".join(lines).strip(),
        [{"title": "Fuente de productos", "url": url} for url in sources[:6]],
    )


def _is_product_overview_question(question: str) -> bool:
    normalized_question = _normalize_match_text(question)
    asks_types = any(
        phrase in normalized_question
        for phrase in [
            "que tipo de productos",
            "que tipos de productos",
            "tipos de productos",
            "categorias de productos",
            "categorias tienen",
            "categorias tienes",
            "lineas de producto",
            "linea de productos",
            "que venden",
            "que ofrecen",
        ]
    )
    asks_exhaustive = any(
        phrase in normalized_question
        for phrase in [
            "todos",
            "todas",
            "enlista",
            "listame",
            "lista completa",
            "precios",
            "precio",
            "cuanto",
            "cuestan",
            "valen",
            "diferencia",
            "compara",
        ]
    )
    asks_combo_overview = (
        ("combo" in normalized_question or "combos" in normalized_question)
        and any(
            phrase in normalized_question
            for phrase in [
                "tienen productos",
                "tienes productos",
                "hay productos",
                "tienen combos",
                "tienes combos",
                "hay combos",
                "productos en combo",
            ]
        )
    )
    return (asks_types or asks_combo_overview) and not asks_exhaustive


def _build_product_overview_answer(
    question: str,
    chunks: list[dict[str, str | int]],
) -> tuple[str, list[dict[str, str]]] | None:
    if not _is_product_overview_question(question):
        return None

    products = _parse_product_records(chunks)
    if not products:
        return None

    grouped: dict[str, list[dict[str, str]]] = {}
    for product in products:
        label = _group_product_label(product["name"])[1]
        grouped.setdefault(label, []).append(product)

    ordered_labels = sorted(
        grouped,
        key=lambda label: _group_product_label(grouped[label][0]["name"])[0],
    )
    combo_labels = [
        label
        for label in ordered_labels
        if any("combo" in _normalize_match_text(product["name"]) for product in grouped[label])
    ]
    combo_count = sum(1 for product in products if "combo" in _normalize_match_text(product["name"]))
    normalized_question = _normalize_match_text(question)
    combo_only_summary = (
        ("combo" in normalized_question or "combos" in normalized_question)
        and not any(
            phrase in normalized_question
            for phrase in ["que tipo de productos", "que tipos de productos", "tipos de productos", "categorias"]
        )
    )

    if combo_only_summary:
        lines = [
            f"Sí. En la base actual encontré {combo_count} productos en combo con precio reportado.",
            "",
            "Categorías con combos:",
        ]
        for label in combo_labels:
            combo_items = [
                product for product in grouped[label] if "combo" in _normalize_match_text(product["name"])
            ]
            lines.append(f"- **{label}:** {len(combo_items)} productos en combo.")
        lines.extend(
            [
                "",
                "Si quieres, puedo listar una categoría concreta, por ejemplo: `sándwiches en combo`, o comparar combo vs individual.",
            ]
        )
        sources = _dedupe_preserve_order([item["url"] for item in products])
        return (
            "\n".join(lines).strip(),
            [{"title": "Fuente de productos", "url": url} for url in sources[:6]],
        )

    lines = [
        "La base actual muestra estas líneas de producto de Sándwich Qbano:",
        "",
    ]
    for label in ordered_labels:
        lines.append(f"- **{label}:** {len(grouped[label])} productos con precio reportado.")

    if "combo" in _normalize_match_text(question) or "combos" in _normalize_match_text(question):
        lines.extend(
            [
                "",
                f"Sí, hay productos en combo: encontré {combo_count} combos con precio reportado.",
                f"Las categorías con combos son: {', '.join(combo_labels)}.",
                "Si necesitas precios, puedes preguntar por una categoría específica, por ejemplo: `sándwiches en combo` o `wraps en combo`.",
            ]
        )

    sources = _dedupe_preserve_order([item["url"] for item in products])
    return (
        "\n".join(lines).strip(),
        [{"title": "Fuente de productos", "url": url} for url in sources[:6]],
    )


def _price_to_int(price: str) -> int | None:
    match = re.search(r"\d[\d.]*", price)
    if not match:
        return None
    try:
        return int(match.group(0).replace(".", ""))
    except ValueError:
        return None


def _canonical_combo_base_name(product_name: str) -> tuple[str | None, str | None]:
    normalized = _normalize_match_text(product_name)
    product_type: str | None = None
    if "sandwich" in normalized:
        product_type = "Sándwich"
        type_tokens = {"sandwich"}
    elif "wrap" in normalized:
        product_type = "Wrap"
        type_tokens = {"wrap"}
    elif "qbowl" in normalized or "q bowl" in normalized:
        product_type = "Qbowl"
        type_tokens = {"qbowl", "q", "bowl"}
    elif "ensalada" in normalized:
        product_type = "Ensalada"
        type_tokens = {"ensalada"}
    elif "hamburguesa" in normalized:
        product_type = "Hamburguesa"
        type_tokens = {"hamburguesa"}
    elif "perro" in normalized:
        product_type = "Perro"
        type_tokens = {"perro"}
    else:
        return None, None

    ignored_tokens = type_tokens | {
        "combo",
        "individual",
        "personal",
        "grande",
        "deliciosa",
        "delicioso",
        "de",
    }
    base = " ".join(token for token in normalized.split() if token not in ignored_tokens)
    if not base:
        return None, None
    return product_type, base


def _is_combo_difference_question(question: str) -> bool:
    normalized_question = _normalize_match_text(question)
    has_combo = "combo" in normalized_question or "combos" in normalized_question
    has_individual = any(
        phrase in normalized_question
        for phrase in ["sin combo", "individual", "sin combos", "no combo"]
    )
    asks_difference = any(
        term in normalized_question
        for term in [
            "diferencia",
            "diferencias",
            "comparar",
            "compara",
            "comparacion",
            "comparación",
            "entre",
            "vs",
            "versus",
        ]
    )
    asks_price = _has_price_intent(normalized_question) or "precio" in normalized_question or "precios" in normalized_question
    return has_combo and asks_difference and (has_individual or asks_price or "producto" in normalized_question)


def _build_combo_difference_answer(
    question: str,
    chunks: list[dict[str, str | int]],
) -> tuple[str, list[dict[str, str]]] | None:
    if not _is_combo_difference_question(question):
        return None

    products = _parse_product_records(chunks)
    pairs: list[dict[str, object]] = []
    by_key: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for product in products:
        product_type, base_name = _canonical_combo_base_name(product["name"])
        if not product_type or not base_name:
            continue
        normalized_name = _normalize_match_text(product["name"])
        key = (product_type, base_name)
        by_key.setdefault(key, {})
        if "combo" in normalized_name:
            by_key[key]["combo"] = product
        elif "individual" in normalized_name or product_type in {"Hamburguesa", "Perro"}:
            by_key[key]["individual"] = product

    scope, _ = _resolve_product_scope(question)
    query_tokens = _distinctive_product_tokens(question)
    for (product_type, base_name), variants in by_key.items():
        combo = variants.get("combo")
        individual = variants.get("individual")
        if not combo or not individual:
            continue
        if scope and not _matches_product_scope(combo, scope=scope, variant="combo"):
            continue
        if query_tokens:
            base_tokens = _distinctive_product_tokens(base_name)
            if not (query_tokens & base_tokens):
                continue
        combo_price = _price_to_int(combo["price"])
        individual_price = _price_to_int(individual["price"])
        if combo_price is None or individual_price is None:
            continue
        pairs.append(
            {
                "type": product_type,
                "base": base_name.title(),
                "combo": combo,
                "individual": individual,
                "difference": combo_price - individual_price,
            }
        )

    if not pairs:
        return None

    pairs.sort(key=lambda item: (str(item["type"]), str(item["base"])))
    if query_tokens:
        intro = "Esta es la diferencia de precio para el producto solicitado:"
    else:
        intro = "Estas son las diferencias de precio entre combo e individual cuando existe par comparable:"

    lines = [intro, ""]
    lines.append("| Producto | Individual | Combo | Diferencia |")
    lines.append("|---|---:|---:|---:|")
    for pair in pairs:
        combo = pair["combo"]
        individual = pair["individual"]
        assert isinstance(combo, dict)
        assert isinstance(individual, dict)
        difference = f"{int(pair['difference']):,}".replace(",", ".")
        lines.append(
            "| "
            f"{pair['type']} {pair['base']} | "
            f"{individual['price']} | "
            f"{combo['price']} | "
            f"{difference} COP |"
        )

    if not query_tokens:
        differences = [int(pair["difference"]) for pair in pairs]
        if differences:
            avg = round(sum(differences) / len(differences))
            lines.extend(
                [
                    "",
                    f"Promedio de incremento al pasar a combo: {avg:,}".replace(",", ".") + " COP.",
                    f"Menor diferencia: {min(differences):,}".replace(",", ".") + " COP.",
                    f"Mayor diferencia: {max(differences):,}".replace(",", ".") + " COP.",
                ]
            )

    sources = _dedupe_preserve_order(
        [
            str(pair[variant]["url"])
            for pair in pairs
            for variant in ["combo", "individual"]
            if isinstance(pair.get(variant), dict)
        ]
    )
    return (
        "\n".join(lines).strip(),
        [{"title": "Fuente de productos", "url": url} for url in sources[:6]],
    )


def _build_product_ranking_answer(
    question: str,
    chunks: list[dict[str, str | int]],
) -> tuple[str, list[dict[str, str]]] | None:
    normalized_question = _normalize_match_text(question)
    wants_cheapest = any(
        phrase in normalized_question
        for phrase in ["mas barato", "mas economico", "economico", "barato", "menor precio"]
    )
    wants_expensive = any(
        phrase in normalized_question
        for phrase in ["mas caro", "mayor precio", "costoso", "mas costoso"]
    )
    wants_recommendation = any(
        phrase in normalized_question
        for phrase in ["recomienda", "recomiendas", "recomendacion", "recomendación"]
    )
    if not (wants_cheapest or wants_expensive or wants_recommendation):
        return None

    products = _parse_product_records(chunks)
    scope, variant = _resolve_product_scope(question)
    if scope is None and not any(term in normalized_question for term in ["producto", "productos", "menu", "comer", "pedir"]):
        scope = "all"

    candidates = [
        product
        for product in products
        if _matches_product_scope(product, scope=scope, variant=variant)
        and _price_to_int(product["price"]) is not None
    ]
    if not candidates:
        return None

    reverse = wants_expensive and not (wants_cheapest or wants_recommendation)
    candidates.sort(
        key=lambda product: (
            _price_to_int(product["price"]) or 0,
            _group_product_label(product["name"])[0],
            _normalize_match_text(product["name"]),
        ),
        reverse=reverse,
    )
    selected = candidates[:5]
    if reverse:
        intro = "Estos son los productos de mayor precio reportado en la base actual:"
    elif wants_recommendation:
        intro = "Si buscas una opción económica, estos son los productos de menor precio reportado:"
    else:
        intro = "Estos son los productos de menor precio reportado en la base actual:"

    lines = [intro, ""]
    for product in selected:
        lines.append(f"- **{product['name']}:** {product['price']}")

    sources = _dedupe_preserve_order([item["url"] for item in selected])
    return (
        "\n".join(lines).strip(),
        [{"title": "Fuente de productos", "url": url} for url in sources[:6]],
    )


def _wants_product_descriptions(question: str) -> bool:
    normalized_question = _normalize_match_text(question)
    return any(
        term in normalized_question
        for term in [
            "sabor",
            "sabores",
            "ingrediente",
            "ingredientes",
            "trae",
            "contiene",
            "descripcion",
            "descripciones",
            "como es",
            "que tiene",
            "que llevan",
            "variedad",
            "variedades",
            "tipo",
            "tipos",
        ]
    )


def _group_product_label(product_name: str) -> tuple[int, str]:
    normalized_name = _normalize_match_text(product_name)
    if "promo" in normalized_name or "miercoles" in normalized_name or "qbanisima" in normalized_name:
        return 0, "Promociones"
    if "sandwich" in normalized_name and "combo" in normalized_name:
        return 1, "Sándwiches en combo"
    if "sandwich" in normalized_name and "individual" in normalized_name:
        return 2, "Sándwiches individuales"
    if "qbanito" in normalized_name:
        return 3, "Qbanitos"
    if "wrap" in normalized_name and "combo" in normalized_name:
        return 4, "Wraps en combo"
    if "wrap" in normalized_name:
        return 5, "Wraps individuales"
    if "qbowl" in normalized_name or "q bowl" in normalized_name:
        return 6, "Q-Bowls"
    if "ensalada" in normalized_name:
        return 7, "Ensaladas"
    if "hamburguesa" in normalized_name:
        return 8, "Hamburguesas"
    if "perro" in normalized_name:
        return 9, "Perros"
    if any(term in normalized_name for term in ["agua", "coca cola", "sprite", "limonada", "quatro"]):
        return 10, "Bebidas"
    if any(term in normalized_name for term in ["papa", "yuca", "salsa", "churros"]):
        return 11, "Acompañamientos y otros"
    if "combo" in normalized_name:
        return 12, "Otros combos"
    return 13, "Otros productos"


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
    include_descriptions = _wants_product_descriptions(question)

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
    elif scope == "all":
        intro = (
            f"Encontré {len(filtered)} productos con precio reportado en la base actual. "
            "Los agrupo por tipo según el nombre del producto y la fuente scrapeada."
        )

    lines = [intro, ""]

    grouped: dict[str, list[dict[str, str]]] = {}
    for product in filtered:
        label = _group_product_label(product["name"])[1]
        grouped.setdefault(label, []).append(product)

    for label in sorted(grouped, key=lambda group: _group_product_label(grouped[group][0]["name"])[0]):
        items = grouped[label]
        lines.append(f"**{label}**")
        for item in items:
            if include_descriptions and item.get("description"):
                lines.append(
                    f"- {item['name']}: {item['price']}. Ingredientes/descripción: {item['description']}"
                )
            else:
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
    vector_index_path: Path | None = None,
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

    if vector_index_path and vector_index_path.exists():
        selected_chunks = search_vector_index(question, vector_index_path, top_k=6)
        if selected_chunks:
            context = _render_chunk_context(selected_chunks)[:max_context_chars]
            sources = dedupe_sources([
                {"title": str(chunk["title"]), "url": str(chunk["url"])}
                for chunk in selected_chunks
            ])
            return context, "vector", sources

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
    vector_index_path: Path | None = None,
    conversation_history: str | None = None,
) -> QAResult:
    offer_answer = _build_offer_answer(question=question, chunks=chunks)
    if offer_answer:
        answer, sources = offer_answer
        return QAResult(answer=answer, context_mode="deterministic", sources=sources)

    specific_product_answer = _build_specific_product_price_answer(question=question, chunks=chunks)
    if specific_product_answer:
        answer, sources = specific_product_answer
        return QAResult(answer=answer, context_mode="deterministic", sources=sources)

    product_overview_answer = _build_product_overview_answer(question=question, chunks=chunks)
    if product_overview_answer:
        answer, sources = product_overview_answer
        return QAResult(answer=answer, context_mode="deterministic", sources=sources)

    combo_difference_answer = _build_combo_difference_answer(question=question, chunks=chunks)
    if combo_difference_answer:
        answer, sources = combo_difference_answer
        return QAResult(answer=answer, context_mode="deterministic", sources=sources)

    product_ranking_answer = _build_product_ranking_answer(question=question, chunks=chunks)
    if product_ranking_answer:
        answer, sources = product_ranking_answer
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
        vector_index_path=vector_index_path,
    )
    if conversation_history:
        context = (
            "Historial de conversacion reciente:\n"
            f"{conversation_history}\n\n"
            "Contexto recuperado de la base de conocimiento:\n"
            f"{context}"
        )[:max_context_chars]
    if model is None:
        return QAResult(
            answer="No encontré una respuesta determinística para esa pregunta y no hay modelo LLM disponible para consultar el contexto.",
            context_mode=context_mode,
            sources=sources,
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
