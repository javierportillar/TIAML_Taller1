from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(slots=True)
class CompanyConfig:
    company_name: str
    company_description: str
    seed_urls: list[str]
    additional_sources: list[str]
    document_sources: list[str]
    allowed_domains: list[str]
    priority_topics: list[str]
    excluded_keywords: list[str]


@dataclass(slots=True)
class RuntimeConfig:
    provider: str
    model_name: str
    temperature: float
    max_context_chars: int


@dataclass(slots=True)
class PathConfig:
    base_dir: Path
    company_config_path: Path
    raw_documents_path: Path
    knowledge_base_path: Path
    chunks_path: Path
    source_index_path: Path
    build_state_path: Path
    batch_results_path: Path

    def ensure_directories(self) -> None:
        for path in (
            self.company_config_path.parent,
            self.raw_documents_path.parent,
            self.knowledge_base_path.parent,
            self.build_state_path.parent,
            self.batch_results_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class AppConfig:
    company: CompanyConfig
    runtime: RuntimeConfig
    paths: PathConfig


def _derive_domains(urls: list[str]) -> list[str]:
    domains: set[str] = set()
    for url in urls:
        domain = urlparse(url).netloc.strip().lower()
        if domain:
            domains.add(domain)
    return sorted(domains)


def _as_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def load_company_config(config_path: Path) -> CompanyConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    seed_urls = _as_list(payload.get("seed_urls"))
    additional_sources = _as_list(payload.get("additional_sources"))
    document_sources = _as_list(payload.get("document_sources"))
    allowed_domains = _as_list(payload.get("allowed_domains"))

    if not allowed_domains:
        allowed_domains = _derive_domains(seed_urls + additional_sources + document_sources)

    priority_topics = _as_list(
        payload.get(
            "priority_topics",
            [
                "historia",
                "productos y servicios",
                "contacto",
                "sedes",
                "horarios",
                "sostenibilidad",
                "noticias",
            ],
        )
    )
    excluded_keywords = _as_list(
        payload.get(
            "excluded_keywords",
            [
                "login",
                "auth",
                "carrito",
                "checkout",
                "terminos",
                "privacidad",
                "politica",
            ],
        )
    )

    return CompanyConfig(
        company_name=str(payload.get("company_name", "EMPRESA_ASIGNADA")).strip(),
        company_description=str(
            payload.get(
                "company_description",
                "Describe aqui a la empresa, su sector y por que esta fue asignada al grupo.",
            )
        ).strip(),
        seed_urls=seed_urls,
        additional_sources=additional_sources,
        document_sources=document_sources,
        allowed_domains=allowed_domains,
        priority_topics=priority_topics,
        excluded_keywords=excluded_keywords,
    )


def build_app_config(company_config_path: Path | None = None) -> AppConfig:
    base_dir = Path(__file__).resolve().parents[1]
    load_dotenv(base_dir / ".env", override=True)

    paths = PathConfig(
        base_dir=base_dir,
        company_config_path=company_config_path
        or base_dir / "data" / "config" / "company_profile.json",
        raw_documents_path=base_dir / "data" / "raw" / "raw_documents.json",
        knowledge_base_path=base_dir / "data" / "processed" / "knowledge_base.txt",
        chunks_path=base_dir / "data" / "processed" / "chunks.json",
        source_index_path=base_dir / "data" / "processed" / "source_index.md",
        build_state_path=base_dir / "data" / "processed" / "build_state.json",
        batch_results_path=base_dir / "results" / "test_results.csv",
    )
    paths.ensure_directories()

    company = load_company_config(paths.company_config_path)
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    default_model = "gpt-4o-mini" if provider == "openai" else "llama3.1:8b"

    runtime = RuntimeConfig(
        provider=provider,
        model_name=os.getenv("MODEL_NAME", default_model).strip(),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "18000")),
    )

    return AppConfig(company=company, runtime=runtime, paths=paths)
