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

    # Reporte ejecutivo en stdout para que el revisor no tenga que abrir el CSV.
    total = len(rows)
    passed = sum(1 for r in rows if r["passed_route"] == "True")
    print()
    print("=" * 60)
    print(f"RESULTADO BATCH: {passed}/{total} rutas correctas "
          f"({passed / max(total, 1):.0%})")
    print("=" * 60)

    # Desglose por modo de contexto: confirma que la mayoria de turnos se
    # resuelven con atajos deterministicos sin pegarle al LLM.
    from collections import Counter

    by_context = Counter(r["context_mode"] for r in rows)
    print()
    print("Desglose por modo de contexto:")
    for mode, count in sorted(by_context.items(), key=lambda kv: -kv[1]):
        marker = " <-- determinista" if mode == "deterministic" else ""
        print(f"  {mode:32s} {count:3d}  ({count / total:.0%}){marker}")

    deterministic_count = by_context.get("deterministic", 0)
    print()
    print(f"Atajos deterministicos: {deterministic_count}/{total} "
          f"({deterministic_count / max(total, 1):.0%})")

    fails = [r for r in rows if r["passed_route"] != "True"]
    if fails:
        print()
        print("Casos fallidos:")
        for r in fails:
            print(f"  - {r['case_id']}: esperaba={r['expected_route']} "
                  f"obtuvo={r['actual_route']} | {r['question'][:60]}")


if __name__ == "__main__":
    main()
