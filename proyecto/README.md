# Taller 1 y 2 - Sistema Q&A Empresarial y Agente Conversacional

Proyecto del modulo 1 y su evolucion para el modulo 2. Primero construye una base de conocimiento empresarial mediante scraping, texto limpio y LangChain para resumen, FAQ y Q&A. Luego transforma el Q&A en un agente conversacional con memoria, herramienta estructurada, router y RAG vectorial.

La empresa configurada para esta entrega es **Sándwich Qbano**.

## Que incluye

- Web scraping de paginas HTML, catalogo VTEX y PDF oficial.
- Consolidacion en `data/processed/knowledge_base.txt`.
- Fragmentacion en `data/processed/chunks.json`.
- App en Streamlit con secciones: `Agente conversacional`, `Resumen`, `FAQ`, `Q&A clasico` y `Configuracion`.
- Memoria conversacional visible en interfaz de chat.
- Herramienta estructurada para datos puntuales de contacto y canales.
- Router/agente que decide entre memoria, herramienta estructurada y RAG.
- Base vectorial local persistida en `data/vector/vector_index.json`.
- Prompts editables desde la interfaz.
- Fuentes editables desde la interfaz.
- Reprocesamiento automatico cuando cambian fuentes, prompts o archivos clave del pipeline.

## Estructura

```text
proyecto/
  app.py
  requirements.txt
  .env.example
  README.md
  pipeline_explicacion.md
  pipeline_codigo_explicacion.md
  prompt_experimentacion.md
  src/
    config.py
    scraper.py
    processing.py
    prompts.py
    llm.py
    chains.py
    agent.py
    memory.py
    structured_tool.py
    vector_store.py
    project_state.py
  scripts/
    build_knowledge_base.py
    run_question_batch.py
    run_agent_batch.py
  data/
    config/
      company_profile.json
      prompts_config.json
    structured/
      company_structured_data.json
    vector/
      vector_index.json
    raw/
      raw_documents.json
    processed/
      knowledge_base.txt
      chunks.json
      source_index.md
      build_state.json
  results/
    test_questions.csv
    test_results.csv
```

## Instalacion

Desde la carpeta del proyecto:

```bash
cd taller1/proyecto
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuracion del modelo

Copia el archivo de ejemplo:

```bash
cp .env.example .env
```

Para Ollama local:

```env
LLM_PROVIDER=ollama
MODEL_NAME=gemma4:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
TEMPERATURE=0.0
MAX_CONTEXT_CHARS=200000
```

Para OpenAI:

```env
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=tu_clave
TEMPERATURE=0.1
MAX_CONTEXT_CHARS=18000
```

## Construir la base de conocimiento

```bash
python3 scripts/build_knowledge_base.py --max-pages 25
```

Este comando genera o actualiza:

- `data/raw/raw_documents.json`
- `data/processed/knowledge_base.txt`
- `data/processed/chunks.json`
- `data/processed/source_index.md`
- `data/processed/build_state.json`
- `data/vector/vector_index.json`

## Ejecutar la app

```bash
streamlit run app.py
```

La app permite:

- `Agente conversacional`: chat con memoria, router, herramienta estructurada y RAG vectorial.
- `Resumen`: genera un resumen ejecutivo basado en la base de conocimiento.
- `FAQ`: genera preguntas frecuentes y respuestas.
- `Q&A clasico`: responde preguntas del usuario usando el contexto procesado.
- `Configuracion`: permite editar parametros, fuentes y prompts.

## Donde se guardan los cambios visuales

- Fuentes, URLs, dominios y descripcion de empresa: `data/config/company_profile.json`.
- Prompts de resumen, FAQ y Q&A: `data/config/prompts_config.json`.
- Estado de la ultima construccion: `data/processed/build_state.json`.

Cuando se guardan cambios en fuentes o prompts, la app detecta el cambio y reconstruye automaticamente el scraping, la base de conocimiento, los chunks y el indice de fuentes.

## Prompt Engineering

La experimentacion de prompts esta documentada en:

- `prompt_experimentacion.md`: bitacora de diseño, versiones, problemas observados, ajustes y criterios de evaluacion.
- `results/prompt_experiments.csv`: matriz resumida de pruebas e hipotesis de mejora.
- `data/config/prompts_config.json`: prompts activos editables desde la interfaz.
- `src/prompts.py`: prompts por defecto usados si no existe configuracion local.

El diseño separa las tres tareas del proyecto: resumen, FAQ y Q&A. En Q&A, el prompt clasifica la intencion de la pregunta y exige responder solo con evidencia del contexto; para preguntas exhaustivas de productos, `src/chains.py` complementa el prompt con una ruta deterministica para evitar listados inventados.

## Preguntas de prueba

Editar:

```text
results/test_questions.csv
```

Ejecutar:

```bash
python3 scripts/run_question_batch.py
```

Salida:

```text
results/test_results.csv
```

## Pruebas del agente del Taller 2

Editar:

```text
results/agent_test_questions.csv
```

Ejecutar:

```bash
python3 scripts/run_agent_batch.py
```

Salida:

```text
results/agent_test_results.csv
```

## Archivos que no deben entregarse

No incluir:

- `.venv/`
- `.env`
- `__pycache__/`
- `.DS_Store`

Si se entrega por GitHub, estos archivos quedan ignorados por `.gitignore`. Si se entrega como ZIP manual, excluirlos al comprimir.

## Evidencias principales

- `data/processed/knowledge_base.txt`: conocimiento limpio consolidado.
- `data/processed/source_index.md`: fuentes usadas.
- `results/test_results.csv`: ejecucion de preguntas de prueba.
- `pipeline_explicacion.md`: explicacion tecnica del flujo completo.
- `pipeline_codigo_explicacion.md`: explicacion del pipeline con fragmentos de codigo.
- `prompt_experimentacion.md`: evidencia de experimentacion y optimizacion de prompts.
