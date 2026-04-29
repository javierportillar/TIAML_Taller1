from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import build_app_config
from src.project_state import rebuild_processed_state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construye la base de conocimiento a partir del scraping."
    )
    parser.add_argument("--max-pages", type=int, default=25)
    args = parser.parse_args()

    config = build_app_config()
    try:
        stats = rebuild_processed_state(config, max_pages=args.max_pages)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Base de conocimiento construida con exito.")
    print(f"Documentos: {stats['documents']}")
    print(f"Caracteres de conocimiento: {stats['knowledge_chars']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Archivo principal: {config.paths.knowledge_base_path}")
    print(f"Estado local: {config.paths.build_state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
