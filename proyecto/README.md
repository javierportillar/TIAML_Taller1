# `proyecto/` — código fuente del agente Sándwich Qbano

Esta carpeta contiene la implementación completa del agente conversacional
(Talleres 1, 2 y 3 — Ruta A) del curso **Técnicas Avanzadas de IA Aplicadas
en Modelos de Lenguaje** (Maestría en IA, UAO).

> 📖 **La documentación completa del proyecto vive en el [README de la raíz del repositorio](../README.md):**
> arquitectura end-to-end, stack tecnológico justificado, decisiones técnicas,
> pruebas y evidencias, checklist de la rúbrica Ruta A y guía de instalación.

## Bootstrap rápido

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # ajustar GEMINI_API_KEY si aplica
ollama pull gemma3:latest        # modelo default Taller 3
docker compose up -d             # Postgres 16 + n8n 2.x
python scripts/build_knowledge_base.py --max-pages 25
```

## Comandos clave

| Tarea | Comando |
|-------|---------|
| Lanzar UI Streamlit | `streamlit run app.py` |
| Lanzar API FastAPI | `bash scripts/run_api.sh` |
| Correr 33 casos batch del agente | `python scripts/run_agent_batch.py` |
| Smoke test API (20 checks) | `python scripts/test_api.py` |
| Importar workflow N8N | ver `n8n/README.md` |

## Estructura interna

```text
api/         FastAPI: endpoints, schemas Pydantic, runtime singleton
n8n/         Workflow JSON + guía de setup del bridge WhatsApp
src/         Núcleo del agente: tools, memoria, RAG, checkpointer
scripts/     CLI: build KB, batch tests, run_api, smoke tests
data/        Configuración + corpus + Chroma persistente
results/     CSV de preguntas y resultados
```
