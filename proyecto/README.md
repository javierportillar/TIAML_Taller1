# Taller 1 - Sistema Q&A Empresarial

Proyecto del modulo 1 para construir una base de conocimiento empresarial mediante scraping, procesarla en texto limpio y usar LangChain para tres tareas: resumen, FAQ y Q&A.

La empresa configurada para esta entrega es **Sándwich Qbano**.

## Que incluye

- Web scraping de paginas HTML, catalogo VTEX y PDF oficial.
- Consolidacion en `data/processed/knowledge_base.txt`.
- Fragmentacion en `data/processed/chunks.json`.
- App en Streamlit con cuatro secciones: `Resumen`, `FAQ`, `Q&A` y `Configuracion`.
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
  src/
    config.py
    scraper.py
    processing.py
    prompts.py
    llm.py
    chains.py
    project_state.py
  scripts/
    build_knowledge_base.py
    run_question_batch.py
  data/
    config/
      company_profile.json
      prompts_config.json
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

## Ejecutar la app

```bash
streamlit run app.py
```

La app permite:

- `Resumen`: genera un resumen ejecutivo basado en la base de conocimiento.
- `FAQ`: genera preguntas frecuentes y respuestas.
- `Q&A`: responde preguntas del usuario usando el contexto procesado.
- `Configuracion`: permite editar parametros, fuentes y prompts.

## Donde se guardan los cambios visuales

- Fuentes, URLs, dominios y descripcion de empresa: `data/config/company_profile.json`.
- Prompts de resumen, FAQ y Q&A: `data/config/prompts_config.json`.
- Estado de la ultima construccion: `data/processed/build_state.json`.

Cuando se guardan cambios en fuentes o prompts, la app detecta el cambio y reconstruye automaticamente el scraping, la base de conocimiento, los chunks y el indice de fuentes.

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
