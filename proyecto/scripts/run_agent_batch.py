from __future__ import annotations

import csv
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import run_agent
from src.config import build_app_config
from src.llm import create_chat_model
from src.processing import load_chunks, load_knowledge_base


def main() -> None:
    config = build_app_config()
    questions_path = ROOT / "results" / "agent_test_questions.csv"
    output_path = ROOT / "results" / "agent_test_results.csv"

    knowledge_text = load_knowledge_base(config.paths.knowledge_base_path)
    chunks = load_chunks(config.paths.chunks_path)
    model = create_chat_model(
        provider=config.runtime.provider,
        model_name=config.runtime.model_name,
        temperature=config.runtime.temperature,
    )

    batch_thread_id = f"batch_agent_test_{uuid4().hex}"
    chat_history: list[dict[str, str]] = []
    rows: list[dict[str, str]] = []
    with questions_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row["question"]
            result = run_agent(
                model=model,
                company_name=config.company.company_name,
                question=question,
                chat_history=chat_history,
                knowledge_text=knowledge_text,
                chunks=chunks,
                max_context_chars=config.runtime.max_context_chars,
                structured_data_path=config.paths.structured_data_path,
                vector_index_path=config.paths.vector_index_path,
                thread_id=batch_thread_id,
                llm_provider=config.runtime.provider,
                llm_model=config.runtime.model_name,
            )
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": result.answer})
            rows.append(
                {
                    "case_id": row["case_id"],
                    "question": question,
                    "expected_route": row["expected_route"],
                    "actual_route": result.route,
                    "context_mode": result.context_mode,
                    "passed_route": str(row["expected_route"] == result.route),
                    "answer_preview": result.answer[:500].replace("\n", " "),
                    "reasoning": result.reasoning,
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resultados escritos en {output_path}")


if __name__ == "__main__":
    main()
