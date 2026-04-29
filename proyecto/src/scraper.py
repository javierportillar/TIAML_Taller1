from __future__ import annotations

import html
from io import BytesIO
import json
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import CompanyConfig

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


@dataclass(slots=True)
class ScrapedDocument:
    url: str
    title: str
    meta_description: str
    text: str


def _clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _normalize_url(base_url: str, href: str) -> str | None:
    if not href:
        return None
    absolute = urljoin(base_url, href)
    absolute, _ = urldefrag(absolute)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return absolute


def _is_allowed_url(url: str, company: CompanyConfig) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    if company.allowed_domains and domain not in {item.lower() for item in company.allowed_domains}:
        return False

    return not any(keyword.lower() in path for keyword in company.excluded_keywords)


def _looks_like_pdf(url: str, content_type: str = "") -> bool:
    return url.lower().endswith(".pdf") or "application/pdf" in content_type.lower()


def _append_unique(blocks: list[str], seen: set[str], value: str) -> None:
    cleaned = _clean_text(value)
    if not cleaned or cleaned in seen:
        return
    seen.add(cleaned)
    blocks.append(cleaned)


def _format_currency(value: Any, currency: str = "COP") -> str:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return ""
    formatted = f"{numeric:,}".replace(",", ".")
    return f"{formatted} {currency}".strip()


def _extract_offer_summary(offers: Any) -> str:
    if isinstance(offers, list):
        for item in offers:
            summary = _extract_offer_summary(item)
            if summary:
                return summary
        return ""

    if not isinstance(offers, dict):
        return ""

    currency = _clean_text(str(offers.get("priceCurrency", "COP"))) or "COP"
    low_price = offers.get("lowPrice")
    high_price = offers.get("highPrice")
    direct_price = offers.get("price")

    if low_price is not None and high_price is not None:
        low = _format_currency(low_price, currency)
        high = _format_currency(high_price, currency)
        if low and high:
            if low == high:
                return low
            return f"desde {low} hasta {high}"

    if low_price is not None:
        return _format_currency(low_price, currency)

    if direct_price is not None:
        return _format_currency(direct_price, currency)

    nested_offers = offers.get("offers")
    if nested_offers is not None:
        return _extract_offer_summary(nested_offers)

    return ""


def _format_product_block(product: dict[str, Any]) -> str:
    name = _clean_text(str(product.get("name", "")))
    if not name:
        return ""

    parts = [f"Producto: {name}."]

    description = _clean_text(str(product.get("description", "")))
    if description:
        parts.append(f"Descripcion: {description}.")

    price_summary = _extract_offer_summary(product.get("offers"))
    if price_summary:
        parts.append(f"Precio reportado: {price_summary}.")

    return " ".join(parts)


def _extract_catalog_price_summary(product: dict[str, Any]) -> str:
    for item in product.get("items", []):
        if not isinstance(item, dict):
            continue
        for seller in item.get("sellers", []):
            if not isinstance(seller, dict):
                continue
            offer = seller.get("commertialOffer", {})
            if not isinstance(offer, dict):
                continue
            price = offer.get("Price")
            if isinstance(price, (int, float)) and price > 0:
                return _format_currency(int(price), "COP")

    return ""


def _format_catalog_product_block(product: dict[str, Any]) -> str:
    name = _clean_text(str(product.get("productName", "")))
    if not name:
        return ""

    parts = [f"Producto: {name}."]

    ingredients = product.get("Ingredientes", [])
    description = ""
    if isinstance(ingredients, list) and ingredients:
        description = _clean_text(str(ingredients[0]))
    if not description:
        description = _clean_text(str(product.get("description", "")))
    if description:
        parts.append(f"Descripcion: {description}.")

    price_summary = _extract_catalog_price_summary(product)
    if price_summary:
        parts.append(f"Precio reportado: {price_summary}.")

    return " ".join(parts)


def _build_vtex_catalog_url(page_url: str) -> str | None:
    parsed = urlparse(page_url)
    path_segments = [segment.strip() for segment in parsed.path.split("/") if segment.strip()]

    if not path_segments:
        return None
    if any(segment.isdigit() for segment in path_segments):
        return None
    if parsed.query and "productclusterids" in parsed.query.lower():
        return None
    if path_segments[-1].lower() == "p":
        return None

    query_path = "/".join(quote(segment, safe="") for segment in path_segments)
    map_param = ",".join(["c"] * len(path_segments))
    return (
        f"{parsed.scheme}://{parsed.netloc}"
        f"/api/catalog_system/pub/products/search/{query_path}?map={map_param}&_from=0&_to=49"
    )


def enrich_document_with_vtex_catalog(
    session: requests.Session,
    document: ScrapedDocument,
    timeout: int = 20,
) -> ScrapedDocument:
    catalog_url = _build_vtex_catalog_url(document.url)
    if not catalog_url:
        return document

    try:
        response = session.get(catalog_url, timeout=timeout, headers=HEADERS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return document

    if not isinstance(payload, list) or not payload:
        return document

    blocks: list[str] = []
    seen: set[str] = set()
    product_names: list[str] = []

    for product in payload:
        if not isinstance(product, dict):
            continue
        block = _format_catalog_product_block(product)
        _append_unique(blocks, seen, block)

        product_name = _clean_text(str(product.get("productName", "")))
        if product_name and product_name not in product_names:
            product_names.append(product_name)

    if product_names:
        blocks.append(f"Productos visibles en la pagina: {', '.join(product_names)}.")

    if not blocks:
        return document

    enriched_text = "\n".join(blocks + [document.text]).strip()
    return ScrapedDocument(
        url=document.url,
        title=document.title,
        meta_description=document.meta_description,
        text=enriched_text,
    )


def _walk_jsonld_nodes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        nodes = [payload]
        for value in payload.values():
            if isinstance(value, (dict, list)):
                nodes.extend(_walk_jsonld_nodes(value))
        return nodes

    if isinstance(payload, list):
        nodes: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, (dict, list)):
                nodes.extend(_walk_jsonld_nodes(item))
        return nodes

    return []


def _extract_jsonld_blocks(soup: BeautifulSoup) -> list[str]:
    blocks: list[str] = []
    seen: set[str] = set()

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_payload = script.string or script.get_text(" ", strip=True)
        if not raw_payload:
            continue

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue

        for node in _walk_jsonld_nodes(payload):
            node_type = _clean_text(str(node.get("@type", ""))).lower()

            if node_type == "itemlist":
                item_list = node.get("itemListElement")
                if not isinstance(item_list, list):
                    continue

                for item in item_list:
                    if not isinstance(item, dict):
                        continue
                    product = item.get("item")
                    if isinstance(product, dict):
                        _append_unique(blocks, seen, _format_product_block(product))
                continue

            if node_type == "product":
                _append_unique(blocks, seen, _format_product_block(node))

    return blocks


def _extract_anchor_blocks(soup: BeautifulSoup) -> list[str]:
    product_names: list[str] = []
    category_names: list[str] = []
    seen_products: set[str] = set()
    seen_categories: set[str] = set()

    ignored_labels = {
        "logo",
        "menu",
        "menú",
        "entrar",
        "ver producto",
        "facebook",
        "youtube",
        "pinterest",
    }

    for anchor in soup.find_all("a", href=True):
        href = _clean_text(anchor.get("href", ""))
        if not href:
            continue

        candidate = _clean_text(anchor.get("aria-label", ""))
        if not candidate:
            candidate = _clean_text(anchor.get_text(" ", strip=True))
        if not candidate:
            image = anchor.find("img", alt=True)
            candidate = _clean_text(image.get("alt", "") if image else "")

        normalized = candidate.lower()
        if not candidate or normalized in ignored_labels:
            continue

        if href.endswith("/p") or "/p?" in href:
            if candidate not in seen_products:
                seen_products.add(candidate)
                product_names.append(candidate)
            continue

        if anchor.find("p") and 3 <= len(candidate) <= 40 and candidate not in seen_categories:
            seen_categories.add(candidate)
            category_names.append(candidate)

    blocks: list[str] = []
    if category_names:
        blocks.append(f"Categorias visibles del sitio: {', '.join(category_names[:15])}.")
    if product_names:
        blocks.append(f"Productos visibles en la pagina: {', '.join(product_names[:20])}.")

    return blocks


def _extract_blocks(soup: BeautifulSoup) -> list[str]:
    structured_blocks = _extract_jsonld_blocks(soup)
    anchor_blocks = _extract_anchor_blocks(soup)

    for tag in soup(["script", "style", "noscript", "svg", "form", "iframe"]):
        tag.decompose()

    candidate_containers = [
        soup.find("main"),
        soup.find("article"),
        soup.find(
            attrs={
                "class": lambda values: values
                and any("footer" in value.lower() for value in values)
            }
        ),
        soup.body,
        soup,
    ]

    def collect_from(container) -> list[str]:
        blocks: list[str] = []
        seen: set[str] = set()

        for node in container.select("h1, h2, h3, h4, p, li"):
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) < 20:
                continue
            if text in seen:
                continue
            seen.add(text)
            blocks.append(text)

        return blocks

    combined_blocks: list[str] = []
    combined_seen: set[str] = set()

    for block in structured_blocks + anchor_blocks:
        _append_unique(combined_blocks, combined_seen, block)

    for container in candidate_containers:
        if not container:
            continue
        for block in collect_from(container):
            _append_unique(combined_blocks, combined_seen, block)

    if combined_blocks:
        return combined_blocks

    return structured_blocks + anchor_blocks + collect_from(soup)


def fetch_document(session: requests.Session, url: str, timeout: int = 20) -> ScrapedDocument | None:
    response = session.get(url, timeout=timeout, headers=HEADERS)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    document = build_document_from_html(
        url=url,
        html=response.text,
        content_type=content_type,
    )
    if not document and "text/html" in content_type:
        document = _build_minimal_html_document(url=url, html=response.text)
    if not document:
        return None
    document = enrich_document_with_vtex_catalog(session, document, timeout=timeout)
    if len(document.text.strip()) < 60:
        return None
    return document


def build_document_from_html(url: str, html: str, content_type: str) -> ScrapedDocument | None:
    if "text/html" not in content_type:
        return None

    soup = BeautifulSoup(html, "lxml")
    blocks = _extract_blocks(soup)
    text = "\n".join(blocks).strip()
    if len(text) < 60:
        return None

    title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = _clean_text(meta_tag.get("content", "") if meta_tag else "")

    return ScrapedDocument(
        url=url,
        title=title or "Sin titulo",
        meta_description=meta_description,
        text=text,
    )


def _build_minimal_html_document(url: str, html: str) -> ScrapedDocument:
    soup = BeautifulSoup(html, "lxml")
    title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = _clean_text(meta_tag.get("content", "") if meta_tag else "")
    return ScrapedDocument(
        url=url,
        title=title or "Sin titulo",
        meta_description=meta_description,
        text="",
    )


def build_document_from_pdf(url: str, content: bytes) -> ScrapedDocument | None:
    reader = PdfReader(BytesIO(content))
    pages: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        cleaned = _clean_text(extracted)
        if cleaned:
            pages.append(cleaned)

    text = "\n".join(pages).strip()
    if len(text) < 80:
        return None

    title = url.split("/")[-1] or "Documento PDF"
    metadata = reader.metadata or {}
    if metadata.get("/Title"):
        title = _clean_text(str(metadata["/Title"]))

    return ScrapedDocument(
        url=url,
        title=title,
        meta_description="Documento PDF oficial",
        text=text,
    )


def discover_links(html: str, base_url: str, company: CompanyConfig, max_links: int = 20) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        normalized = _normalize_url(base_url, anchor["href"])
        if not normalized or normalized in seen:
            continue
        if not _is_allowed_url(normalized, company):
            continue
        seen.add(normalized)
        links.append(normalized)
        if len(links) >= max_links:
            break

    return links


def scrape_company_sources(
    company: CompanyConfig,
    max_pages: int = 25,
    sleep_seconds: float = 0.2,
) -> list[ScrapedDocument]:
    session = requests.Session()
    configured_urls = list(
        dict.fromkeys(company.seed_urls + company.additional_sources + company.document_sources)
    )
    target_pages = max(max_pages, len(configured_urls))
    queue = deque(configured_urls)
    visited: set[str] = set()
    documents: list[ScrapedDocument] = []

    while queue and len(documents) < target_pages:
        current_url = queue.popleft()
        if current_url in visited:
            continue
        visited.add(current_url)

        if not _is_allowed_url(current_url, company):
            continue

        try:
            response = session.get(current_url, timeout=20, headers=HEADERS)
            response.raise_for_status()
        except requests.RequestException:
            continue

        content_type = response.headers.get("Content-Type", "")
        if _looks_like_pdf(current_url, content_type):
            pdf_document = build_document_from_pdf(current_url, response.content)
            if pdf_document:
                documents.append(pdf_document)
        else:
            if "text/html" not in content_type:
                continue

            html_document = build_document_from_html(
                url=current_url,
                html=response.text,
                content_type=content_type,
            )
            if not html_document:
                html_document = _build_minimal_html_document(
                    url=current_url,
                    html=response.text,
                )
            if html_document:
                enriched_document = enrich_document_with_vtex_catalog(session, html_document)
                if len(enriched_document.text.strip()) >= 60:
                    documents.append(enriched_document)

            for link in discover_links(response.text, current_url, company):
                if link not in visited:
                    queue.append(link)

        time.sleep(sleep_seconds)

    return documents


def save_raw_documents(documents: list[ScrapedDocument], destination: Path) -> None:
    payload = [asdict(document) for document in documents]
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_raw_documents(path: Path) -> list[ScrapedDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ScrapedDocument(**item) for item in payload]
