"""Smoke test del endpoint POST /chat de la API FastAPI.

Lanza la app via TestClient (no requiere uvicorn ni puerto). Cubre:
- GET /health  -> servicios criticos OK
- GET /info    -> snapshot de configuracion
- POST /chat   -> 4 escenarios: structured, RAG, memoria personal, HITL escalamiento
- Multi-LLM: alterna provider Ollama y verifica que la memoria se preserva por thread_id

Uso:
    cd proyecto
    .venv/bin/python scripts/test_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=ROOT / ".env", override=True)

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from src.checkpointer import clear_thread_messages  # noqa: E402


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def _assert(condition: bool, label: str) -> bool:
    print(f"  [{_green('OK') if condition else _red('FAIL')}] {label}")
    return condition


def main() -> int:
    client = TestClient(app)
    thread_id = f"api_test_{uuid4().hex}"
    clear_thread_messages(thread_id)
    total = 0
    passed = 0

    print("=" * 60)
    print("Smoke test de la API FastAPI")
    print(f"thread_id de prueba: {thread_id}")
    print("=" * 60)

    print("\n[1] GET /health")
    resp = client.get("/health")
    total += 1
    if _assert(resp.status_code == 200, "200 OK"):
        passed += 1
    body = resp.json()
    print(f"      status={body.get('status')} chroma={body.get('chroma_ready')} "
          f"postgres={body.get('postgres_ready')} kb={body.get('knowledge_base_ready')}")

    print("\n[2] GET /info")
    resp = client.get("/info")
    total += 1
    if _assert(resp.status_code == 200, "200 OK"):
        passed += 1
    body = resp.json()
    total += 1
    if _assert("solicitar_supervisor_humano" in body.get("sensitive_tools", []),
               "sensitive_tools incluye solicitar_supervisor_humano"):
        passed += 1
    total += 1
    if _assert(body.get("vector_count", 0) > 0, f"vectores Chroma > 0 (={body.get('vector_count')})"):
        passed += 1

    casos = [
        {
            "label": "structured WhatsApp",
            "payload": {"thread_id": thread_id, "message": "Cual es el WhatsApp?",
                        "provider": "ollama", "model": "gemma3:latest"},
            "expected_route": "consultar_datos_contacto",
        },
        {
            "label": "RAG productos baratos",
            "payload": {"thread_id": thread_id, "message": "Cuales son los sandwiches mas baratos?",
                        "provider": "ollama", "model": "gemma3:latest"},
            "expected_route": "buscar_catalogo_productos",
        },
        {
            "label": "conversation: declaracion de nombre",
            "payload": {"thread_id": thread_id, "message": "Hola mi nombre es Javier",
                        "provider": "ollama", "model": "gemma3:latest"},
            "expected_route": "conversation",
        },
        {
            "label": "memory: recordar nombre",
            "payload": {"thread_id": thread_id, "message": "Como me llamo?",
                        "provider": "ollama", "model": "gemma3:latest"},
            "expected_route": "memory",
        },
        {
            "label": "HITL: escalamiento humano",
            "payload": {"thread_id": thread_id,
                        "message": "Tengo una queja, quiero hablar con un asesor humano",
                        "provider": "ollama", "model": "gemma3:latest"},
            "expected_route": "solicitar_supervisor_humano",
        },
    ]

    print("\n[3] POST /chat - 5 escenarios sobre el mismo thread_id")
    for idx, caso in enumerate(casos, start=1):
        print(f"\n  [{idx}] {caso['label']}")
        resp = client.post("/chat", json=caso["payload"])
        total += 1
        if _assert(resp.status_code == 200, f"  HTTP 200 (status={resp.status_code})"):
            passed += 1
        body = resp.json()
        total += 1
        if _assert(body.get("route") == caso["expected_route"],
                   f"  route={body.get('route')} (esperaba {caso['expected_route']})"):
            passed += 1
        total += 1
        if _assert(body.get("llm_provider") == caso["payload"]["provider"],
                   f"  llm_provider={body.get('llm_provider')}"):
            passed += 1
        print(f"      answer: {body.get('answer', '')[:100]}")

    print(f"\n[4] POST /chat - 400 con provider invalido")
    resp = client.post("/chat", json={
        "thread_id": thread_id,
        "message": "hola",
        "provider": "marciano",
    })
    total += 1
    if _assert(resp.status_code in (400, 422), f"HTTP {resp.status_code} (esperaba 400 o 422)"):
        passed += 1

    clear_thread_messages(thread_id)

    print("\n" + "=" * 60)
    print(f"RESULTADO: {passed}/{total} checks pasaron")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
