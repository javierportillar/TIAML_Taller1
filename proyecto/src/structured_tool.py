from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path


@dataclass(slots=True)
class StructuredToolResult:
    answer: str
    matches: list[dict[str, str]]
    sources: list[dict[str, str]]
    score: int


def _normalize(text: str) -> str:
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
    normalized = re.sub(r"[^a-z0-9@.]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {_normalize(text)} "
    normalized_phrase = f" {_normalize(phrase)} "
    return normalized_phrase in normalized_text


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    return records if isinstance(records, list) else []


def _score_record(query: str, record: dict[str, object]) -> int:
    normalized_query = _normalize(query)
    query_terms = set(normalized_query.split())
    keywords = record.get("keywords", [])
    keyword_text = " ".join(str(keyword) for keyword in keywords if keyword)
    searchable = _normalize(
        " ".join(
            [
                str(record.get("label", "")),
                str(record.get("category", "")),
                str(record.get("value", "")),
                keyword_text,
            ]
        )
    )
    searchable_terms = set(searchable.split())
    score = len(query_terms & searchable_terms)

    for keyword in keywords if isinstance(keywords, list) else []:
        normalized_keyword = _normalize(str(keyword))
        if normalized_keyword and _contains_normalized_phrase(normalized_query, normalized_keyword):
            score += 4

    category = _normalize(str(record.get("category", "")))
    if category and category in normalized_query:
        score += 2
    limit_terms = [
        "horario",
        "horarios",
        "hora",
        "abre",
        "abren",
        "cierra",
        "cierran",
        "domicilio",
        "domicilios",
        "entrega",
        "entregas",
        "disponible",
        "disponibilidad",
        "inventario",
        "pago",
        "pagos",
        "factura",
        "facturacion",
    ]
    if category == "limite informacion" and any(
        _contains_normalized_phrase(normalized_query, term) for term in limit_terms
    ):
        score += 6
    return score


def search_structured_data(
    query: str,
    path: Path,
    min_score: int = 2,
    top_k: int = 4,
) -> StructuredToolResult | None:
    normalized_query = _normalize(query)
    scored: list[tuple[int, dict[str, object]]] = []
    for record in _load_records(path):
        if not isinstance(record, dict):
            continue
        score = _score_record(query, record)
        if score >= min_score:
            scored.append((score, record))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    limit_request_terms = [
        "horario",
        "horarios",
        "hora",
        "abre",
        "abren",
        "cierra",
        "cierran",
        "domicilio",
        "domicilios",
        "entrega",
        "entregas",
        "disponible",
        "disponibilidad",
        "inventario",
        "pago",
        "pagos",
        "factura",
        "facturacion",
    ]
    if any(_contains_normalized_phrase(normalized_query, term) for term in limit_request_terms):
        limit_scored = [
            (score, record)
            for score, record in scored
            if _normalize(str(record.get("category", ""))) == "limite informacion"
        ]
        if limit_scored:
            scored = limit_scored

    exact_terms = [
        "abastecimiento",
        "agua",
        "aliado",
        "aliados",
        "ambiental",
        "ambiente",
        "apertura",
        "aperturas",
        "arbol",
        "arboles",
        "auditoria",
        "auditorias",
        "biodiversidad",
        "calidad",
        "whatsapp",
        "call center",
        "carbono",
        "pbx",
        "correo",
        "email",
        "direccion",
        "colombia",
        "departamento",
        "departamentos",
        "narino",
        "nariño",
        "pasto",
        "ipiales",
        "hora",
        "horario",
        "horarios",
        "abre",
        "abren",
        "cierra",
        "cierran",
        "domicilio",
        "domicilios",
        "entrega",
        "entregas",
        "disponible",
        "disponibilidad",
        "inventario",
        "agotado",
        "pago",
        "pagos",
        "medio de pago",
        "medios de pago",
        "facturacion",
        "factura",
        "factura electronica",
        "crecimiento",
        "empleo",
        "empleos",
        "energia",
        "equidad",
        "etica",
        "facturacion",
        "franquicia",
        "franquicias",
        "gobernanza",
        "historia",
        "huella",
        "inocuidad",
        "innovacion",
        "marca",
        "materialidad",
        "medio ambiente",
        "menu",
        "mision",
        "ods",
        "pet friendly",
        "pet-friendly",
        "politica de datos",
        "privacidad",
        "proveedor",
        "proveedores",
        "qbano en cifras",
        "que es",
        "quien es",
        "quienes son",
        "residuo",
        "residuos",
        "responsabilidad social",
        "seguridad alimentaria",
        "sostenibilidad",
        "sostenible",
        "sucursal",
        "sucursales",
        "sede",
        "sedes",
        "ciudad",
        "ciudades",
        "municipio",
        "municipios",
        "cobertura",
        "presencia",
        "punto de venta",
        "puntos de venta",
        "restaurante",
        "restaurantes",
        "tienda",
        "tiendas",
        "ubicacion",
        "ubicaciones",
        "redes",
        "instagram",
        "facebook",
        "tiktok",
        "youtube",
        "linkedin",
    ]
    requested_terms = [term for term in exact_terms if _contains_normalized_phrase(normalized_query, term)]
    if requested_terms:
        exact_scored: list[tuple[int, dict[str, object]]] = []
        for score, record in scored:
            record_text = _normalize(
                " ".join(
                    [
                        str(record.get("id", "")),
                        str(record.get("category", "")),
                        " ".join(str(item) for item in record.get("keywords", [])),
                    ]
                )
            )
            if any(_contains_normalized_phrase(record_text, term) for term in requested_terms):
                exact_scored.append((score, record))
        if exact_scored:
            scored = exact_scored

    selected = [record for _, record in scored[:top_k]]
    lines = ["Encontré esta información estructurada:", ""]
    matches: list[dict[str, str]] = []
    source_labels: list[str] = []

    for record in selected:
        label = str(record.get("label", "")).strip()
        value = str(record.get("value", "")).strip()
        source = str(record.get("source", "")).strip()
        if not label or not value:
            continue
        lines.append(f"- **{label}:** {value}")
        matches.append(
            {
                "id": str(record.get("id", "")),
                "label": label,
                "value": value,
                "source": source,
            }
        )
        if source and source not in source_labels:
            source_labels.append(source)

    sources = [
        {"title": source, "url": "local://structured_company_data"}
        for source in source_labels
    ]
    return StructuredToolResult(
        answer="\n".join(lines).strip(),
        matches=matches,
        sources=sources,
        score=scored[0][0],
    )
