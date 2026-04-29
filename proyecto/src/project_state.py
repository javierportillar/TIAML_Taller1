from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .config import AppConfig
from .processing import save_processed_artifacts
from .scraper import save_raw_documents, scrape_company_sources


def _relative_to_base(base_dir: Path, target: Path) -> str:
    try:
        return str(target.relative_to(base_dir))
    except ValueError:
        return str(target)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_file(base_dir: Path, path: Path) -> dict[str, str | int | bool]:
    payload: dict[str, str | int | bool] = {
        "path": _relative_to_base(base_dir, path),
        "exists": path.exists(),
    }
    if not path.exists():
        return payload

    stat = path.stat()
    payload.update(
        {
            "size": stat.st_size,
            "sha256": _sha256_file(path),
        }
    )
    return payload


def _watched_local_files(config: AppConfig) -> list[Path]:
    base_dir = config.paths.base_dir
    return [
        config.paths.company_config_path,
        base_dir / "data" / "config" / "prompts_config.json",
        base_dir / "src" / "prompts.py",
        base_dir / "src" / "chains.py",
        base_dir / "src" / "scraper.py",
        base_dir / "src" / "processing.py",
        base_dir / "app.py",
    ]


def collect_local_state(config: AppConfig) -> dict[str, object]:
    base_dir = config.paths.base_dir
    watched = _watched_local_files(config)
    return {
        "watched_files": {
            _relative_to_base(base_dir, path): _fingerprint_file(base_dir, path)
            for path in watched
        }
    }


def load_local_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def detect_local_state_changes(config: AppConfig) -> tuple[bool, list[str], dict[str, object]]:
    required_outputs = [
        config.paths.raw_documents_path,
        config.paths.knowledge_base_path,
        config.paths.chunks_path,
        config.paths.source_index_path,
    ]
    missing_outputs = [path.name for path in required_outputs if not path.exists()]

    current_state = collect_local_state(config)
    previous_state = load_local_state(config.paths.build_state_path)
    previous_files = previous_state.get("watched_files", {})
    current_files = current_state.get("watched_files", {})

    changed_files: list[str] = []
    for relative_path, current_payload in current_files.items():
        previous_payload = previous_files.get(relative_path)
        if previous_payload != current_payload:
            changed_files.append(relative_path)

    if missing_outputs:
        changed_files = changed_files + [f"output:{name}" for name in missing_outputs]

    return bool(changed_files), changed_files, current_state


def rebuild_processed_state(config: AppConfig, max_pages: int = 25) -> dict[str, int]:
    documents = scrape_company_sources(config.company, max_pages=max_pages)

    if not documents:
        raise RuntimeError(
            "No se extrajeron documentos. Revisa las URLs configuradas y los dominios permitidos."
        )

    save_raw_documents(documents, config.paths.raw_documents_path)
    stats = save_processed_artifacts(
        company_name=config.company.company_name,
        documents=documents,
        paths=config.paths,
    )

    local_state = collect_local_state(config)
    payload = {
        **local_state,
        "generated_at": datetime.now(UTC).isoformat(),
        "stats": stats,
        "max_pages": max_pages,
    }
    config.paths.build_state_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats
