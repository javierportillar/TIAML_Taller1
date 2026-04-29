from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.chains import answer_question
from src.config import build_app_config
from src.llm import create_chat_model
from src.processing import load_chunks, load_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta una bateria de preguntas sobre la base de conocimiento."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/test_questions.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/test_results.csv"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=0,
        help="Limite de contexto para esta corrida. Si es 0 usa el valor de .env.",
    )
    args = parser.parse_args()

    config = build_app_config()
    knowledge_text = load_knowledge_base(config.paths.knowledge_base_path)
    chunks = load_chunks(config.paths.chunks_path)
    model = create_chat_model(
        provider=config.runtime.provider,
        model_name=config.runtime.model_name,
        temperature=config.runtime.temperature,
    )

    dataframe = pd.read_csv(config.paths.base_dir / args.input_csv)
    if args.limit > 0:
        dataframe = dataframe.head(args.limit)

    records: list[dict[str, object]] = []
    total = len(dataframe.index)
    output_path = config.paths.base_dir / args.output_csv
    max_context_chars = (
        args.max_context_chars
        if args.max_context_chars > 0
        else config.runtime.max_context_chars
    )

    for index, row in enumerate(dataframe.itertuples(index=False), start=1):
        print(f"[{index}/{total}] Procesando pregunta...")
        question = str(getattr(row, "question", "")).strip()
        if not question:
            continue

        try:
            result = answer_question(
                model=model,
                company_name=config.company.company_name,
                question=question,
                knowledge_text=knowledge_text,
                chunks=chunks,
                max_context_chars=max_context_chars,
            )
            model_answer = result.answer
            context_mode = result.context_mode
            sources = " | ".join(source["url"] for source in result.sources)
            status = "OK"
        except Exception as exc:
            model_answer = f"ERROR: {exc}"
            context_mode = "error"
            sources = ""
            status = "ERROR"

        records.append(
            {
                "id": getattr(row, "id", ""),
                "question": question,
                "expected_answer": getattr(row, "expected_answer", ""),
                "model_answer": model_answer,
                "evaluation": getattr(row, "evaluation", ""),
                "notes": getattr(row, "notes", ""),
                "context_mode": context_mode,
                "sources": sources,
            }
        )
        pd.DataFrame(records).to_csv(output_path, index=False)
        print(f"[{index}/{total}] {status}")

    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"Resultados guardados en: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
