# Sándwich Qbano · Agente Conversacional Multi-Canal

Asistente conversacional empresarial construido sobre LangChain como entrega de los tres talleres del curso **Técnicas Avanzadas de IA Aplicadas en Modelos de Lenguaje** (Maestría en Inteligencia Artificial, UAO). El proyecto **arrancó como un Q&A clásico con scraping y RAG (Taller 1)** , evolucionó a un **agente con memoria, herramientas y router determinístico (Taller 2)** y se cerró como una **API productiva conectada a WhatsApp con memoria persistente y Function Calling estricto (Taller 3 – Ruta A)**.

La empresa caso de estudio es [Sándwich Qbano](https://www.sandwichqbano.com/), cadena colombiana de comida rápida con más de 220 puntos de venta. El agente atiende preguntas de primer contacto sobre productos, precios, combos, contacto, cobertura, sedes y sostenibilidad usando exclusivamente fuentes públicas verificables (sitio web + informe oficial de sostenibilidad).

> **Demo verificada:** Mensaje real desde WhatsApp celular → Twilio Sandbox → n8n → FastAPI → LangGraph agent → respuesta de vuelta en < 5 s. Trazabilidad completa en PostgreSQL.

---

## Estado del proyecto

| Taller | Entregable | Estado |
|--------|------------|--------|
| **Taller 1** | Pipeline Q&A clásico: scraping + chunking + RAG + Streamlit | ✅ Cerrado |
| **Taller 2** | Agente conversacional con memoria + tool estructurada + ChromaDB + multi-LLM | ✅ Cerrado |
| **Taller 3 – Ruta A** | Function Calling estricto + FastAPI + PostgresSaver + WhatsApp via N8N + Twilio | ✅ Cerrado, demo en vivo verificada |
| Bonus t-SNE | Análisis de clústeres de conversaciones | ⬜ Opcional |
| Informe técnico | PDF unificado para sustentación | ⏳ En preparación |

**Pruebas automatizadas que pasaron:** 33/33 batch del agente · 5/5 rutas de memoria personal · 20/20 smoke API FastAPI · 1/1 prueba real WhatsApp end-to-end.

---

## Evolución arquitectónica

El sistema **no se rehizo en cada taller**: cada taller agrega una capa que reutiliza la base anterior sin reescribirla. Esto es deliberado: el agente del Taller 3 sigue ejecutando el mismo `run_agent` que ya validamos en Taller 2, solo que ahora vive detrás de un endpoint HTTP y una integración WhatsApp.

```
Taller 1                       Taller 2                       Taller 3 (Ruta A)
─────────                      ─────────                      ───────────────────
Streamlit (UI única)           Streamlit + chat history       Streamlit + FastAPI + WhatsApp
Q&A clásico                    Agente con router              create_agent + HumanInTheLoop
LangChain ChatPromptTemplate   ChatPromptTemplate +            init_chat_model + dynamic_prompt
                               structured tool                 + Pydantic tool schemas
Feature hashing + JSON         ChromaDB + HuggingFace          ChromaDB + HuggingFace
(custom, no LangChain native)  embeddings (LangChain native)   (sin cambios)
Memoria efímera                Memoria en st.session_state    PostgresSaver + tabla custom
(solo durante la sesión)                                       multi-LLM por sesión / por request
Una sola fuente: web           web + JSON estructurado         + memoria personal (nombres)
                               + 3 rutas determinísticas       + 4ª ruta sensible HITL
                                                               + workflow N8N + Twilio
```

Lo que se conservó intacto entre talleres:
- **Knowledge base**: scraping → texto consolidado → chunks (`data/processed/*`).
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim).
- **Vector store**: ChromaDB persistido en disco (`data/vector/chroma/`).
- **Router determinístico**: heurísticas en `src/agent.py` que detectan tipo de pregunta antes de invocar al LLM.

---

## Visión general

Pregunta de negocio que resuelve el sistema:

> ¿Cómo ofrece una marca de comida rápida con presencia nacional **atención conversacional 24/7 vía WhatsApp** sin alucinaciones, sin contratar más agentes humanos, sin reescribir su sitio web y respetando sus fuentes oficiales como única fuente de verdad?

Respuesta del sistema:

1. Ingiere su website público y su informe oficial de sostenibilidad (scraping + PDF parsing).
2. Consolida el conocimiento en una base vectorial local (ChromaDB) + una base estructurada en JSON (`data/structured/company_structured_data.json`).
3. Atiende mensajes vía dos canales **independientes** que reusan el mismo cerebro:
   - **Streamlit** (`http://localhost:8501`) para demo, debugging y operación humana.
   - **API REST FastAPI** (`http://localhost:8000/chat`) consumida por **N8N + Twilio Sandbox** para WhatsApp.
4. Cada turno se enruta determinísticamente (sin LLM) en función del tipo de pregunta y, cuando aplica, invoca una **tool Pydantic** que el LLM elige vía **Function Calling**.
5. Toda la conversación queda persistida en PostgreSQL con `thread_id = número de teléfono del usuario` (en producción WhatsApp) o `streamlit_local` (durante demos locales).
6. Si el usuario pide hablar con un humano, se dispara un **`HumanInTheLoopMiddleware`** sobre una tool sensible (`solicitar_supervisor_humano`).

---

## Stack tecnológico

### Lenguajes y frameworks

| Componente | Tecnología | Justificación |
|------------|-----------|---------------|
| **Orquestación de agente** | LangChain 1.x + LangGraph | `init_chat_model`, `create_agent`, `dynamic_prompt`, `HumanInTheLoopMiddleware`. Cumple las 7 herramientas obligatorias de la Ruta A del Taller 3. |
| **Function Calling** | Pydantic v2 schemas + `langchain_core.tools.tool` | Cada tool tiene `BaseModel` estricto en `src/tools.py`. El LLM no puede invocar tools con argumentos malformados. |
| **API REST** | FastAPI 0.136 + Uvicorn | OpenAPI auto-generado en `/docs`, lifespan async para precargar config, payload `ChatRequest` validado con Pydantic. |
| **UI demo** | Streamlit 1.33 | UI rápida para sustentación humana. Sidebar con selector multi-LLM en caliente. |
| **LLMs** | Ollama (`gemma3:latest` local) + Google Gemini (`gemini-2.5-flash`) + OpenAI (`gpt-4o-mini`) opcional | Selector por sesión Streamlit **y por request** en `POST /chat`. Permite alternar en vivo sin reiniciar nada. |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` vía `langchain-huggingface` | Multilingüe (es necesario porque el corpus mezcla español e inglés). 384 dim. Corre local, no consume cuota de API. |
| **Vector store** | ChromaDB persistido (`langchain-chroma`) | 185 vectores indexados, < 3 MB en disco. Suficiente para una empresa de un solo sitio web. |
| **Memoria persistente** | PostgresSaver de LangGraph + tabla custom `conversation_messages` | PostgresSaver guarda el estado binario del agente (rúbrica); la tabla custom guarda turnos legibles para auditoría SQL y para repintar el chat al reabrir Streamlit. |
| **Chunking** | `RecursiveCharacterTextSplitter` (LangChain) | Tamaño y solapamiento configurados para 1000/150 caracteres. Reemplazó un `_chunk_paragraphs` ad-hoc del Taller 1. |
| **Canal WhatsApp** | Twilio Sandbox for WhatsApp (gratis) | No requiere aprobación de Meta. Suficiente para sustentación. El número sandbox (+14155238886) acepta mensajes de cualquier número que envíe primero `join <code>`. |
| **Orquestador** | n8n 2.x local (Docker) | Workflow visual de 5 nodos: webhook → set → HTTP request → HTTP Twilio → respond. Importado vía CLI a `qbano_n8n`. |
| **Túnel público** | ngrok free tier | Expone n8n local (puerto 5678) a internet para que Twilio entregue el webhook. URL rotativa, suficiente para demo. |
| **Base de datos** | PostgreSQL 16 (Docker) | `qbano_postgres` en `docker-compose.yml`. Almacena checkpoints LangGraph + `conversation_messages`. |
| **Tests** | `scripts/run_agent_batch.py` (33 casos) + `scripts/test_api.py` (TestClient FastAPI) | Sin pytest porque el dominio es estructurado y los assertions son sobre `route` esperado. |

### Por qué este stack y no otro

- **LangChain estricto y no “LangChain con shortcuts”**: la rúbrica del Taller 3 verifica con `grep` el uso de `init_chat_model`, `create_agent`, `HumanInTheLoopMiddleware`, `RecursiveCharacterTextSplitter`, `langchain_core.vectorstores`, `dynamic_prompt` y `PostgresSaver`. Cualquier atajo (LangChain `Memory` mágica, vectorstore custom, prompt estático) le restaba puntos.
- **Ollama como default sobre Gemini**: el sistema debe seguir respondiendo si se cae el internet. Gemini 2.5 Flash queda como secundario y como evidencia multi-LLM.
- **Chroma sobre Pinecone/Weaviate**: el corpus es < 200 chunks; cualquier vector store cloud es overkill. Chroma persistido en disco es 0 € y 0 mantenimiento.
- **n8n local sobre n8n Cloud**: el trial cloud caduca a los 14 días. Si la sustentación se atrasa, el bot se cae. Docker = gratis para siempre.
- **Twilio Sandbox sobre WhatsApp Business API directa**: la API oficial de Meta requiere aprobación de número, perfil empresarial y verificación legal. Para una demo académica el sandbox es perfecto.
- **FastAPI sobre Flask**: tipado nativo Pydantic, OpenAPI auto-generado (`/docs`), lifespan async para precarga del knowledge base. Flask requeriría manuales adicionales para todo eso.
- **Hybrid deterministic + LLM router**: el router del agente **primero** matchea heurísticas regex (`_is_personal_question`, `_looks_like_human_escalation`, `_looks_like_commercial_catalog_query`) y **solo si nada matchea** invoca a `create_agent` para que el LLM decida. Esto elimina la mayoría de las alucinaciones de routing y baja el costo de tokens.

---

## Arquitectura end-to-end

```
                         ┌─────────────────────────────────┐
                         │  Usuario WhatsApp (celular)      │
                         └──────────────┬──────────────────┘
                                        │ texto plano
                                        ▼
                         ┌─────────────────────────────────┐
                         │  Twilio Sandbox (+14155238886)   │
                         │  WhatsApp Business API gratis    │
                         └──────────────┬──────────────────┘
                                        │ POST form-urlencoded
                                        │ {From, To, Body, ...}
                                        ▼
                         ┌─────────────────────────────────┐
                         │  ngrok https://*.ngrok-free.dev  │
                         │  túnel TLS al localhost:5678     │
                         └──────────────┬──────────────────┘
                                        ▼
                         ┌─────────────────────────────────┐
                         │  n8n local (Docker qbano_n8n)    │
                         │  workflow whatsapp_qbano.json    │
                         │                                  │
                         │  [Webhook]                       │
                         │      ↓                           │
                         │  [Set: extraer From/Body]        │
                         │      ↓ json {thread_id, message} │
                         │  [HTTP POST host.docker          │
                         │   .internal:8000/chat]           │
                         │      ↓ json {answer, route}      │
                         │  [HTTP POST Twilio API]          │
                         │   (Basic Auth via credencial)    │
                         │      ↓                           │
                         │  [Respond <Response></Response>] │
                         └──────────────┬──────────────────┘
                                        │ HTTP POST /chat
                                        ▼
                         ┌─────────────────────────────────┐
                         │  FastAPI (uvicorn, puerto 8000)  │
                         │  api/main.py                     │
                         │                                  │
                         │  POST /chat → run_agent()        │
                         │  GET  /health                    │
                         │  GET  /info                      │
                         │  GET  /docs (Swagger UI)         │
                         └──────────────┬──────────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
   ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
   │  ChromaDB (185 vec) │  │  JSON estructurado  │  │  PostgresSaver +    │
   │  data/vector/chroma │  │  43 registros       │  │  conversation_msgs  │
   │  Multilingual MiniLM│  │  contacto, sedes,   │  │  Postgres 16 Docker │
   │  L12-v2 (384 dim)   │  │  sostenibilidad     │  │  qbano_postgres     │
   └─────────────────────┘  └─────────────────────┘  └─────────────────────┘
                                        │
                                        │ paralelamente
                                        ▼
                         ┌─────────────────────────────────┐
                         │  Streamlit (puerto 8501)         │
                         │  app.py                          │
                         │                                  │
                         │  Sidebar: selector LLM en vivo   │
                         │  Tabs: Agente / Resumen / FAQ /  │
                         │        Q&A clásico / Configuración│
                         │  Memoria recuperada de Postgres  │
                         │  al abrir la app                 │
                         └─────────────────────────────────┘
```

### Por qué arquitectura híbrida (local + nube)

1. **El cerebro vive 100% local.** Postgres, Chroma, Ollama, FastAPI, n8n y Streamlit corren en la misma máquina. Cero dependencias críticas en la nube.
2. **La nube solo expone, no procesa.** Twilio + ngrok son únicamente capas de transporte. Si Twilio se cae, Streamlit y la API local siguen funcionando.
3. **El `thread_id` es la llave de aislamiento.** En producción WhatsApp será el `+57…` del usuario. En Streamlit es `streamlit_local`. En el batch de pruebas es un UUID. La misma tabla `conversation_messages` separa los hilos por esa columna.
4. **Cada subsistema se puede reiniciar sin tumbar el resto.** Reiniciar Streamlit no afecta a n8n. Reiniciar la API no afecta a Postgres. Reiniciar Postgres degrada la memoria pero el agente sigue respondiendo (sin persistencia hasta que vuelva).

---

## Capacidades del agente

### Tools disponibles (function calling con Pydantic)

| Tool | Cuándo se invoca | Cómo decide | Resultado |
|------|-----------------|-------------|-----------|
| `consultar_datos_contacto` | Preguntas puntuales (WhatsApp, PBX, redes sociales, dirección, horarios, cobertura) | Heurística determinística + atajo `_looks_like_structured_query` | Lectura directa de `data/structured/company_structured_data.json` |
| `buscar_catalogo_productos` | Preguntas comerciales (precios, combos, sándwiches más caros/baratos, rankings) | Atajo `_looks_like_commercial_catalog_query` + `_is_exhaustive_product_question` | Determinístico (orden por precio) + RAG vectorial cuando aplica |
| `consultar_informacion_corporativa` | Preguntas abiertas (historia, sostenibilidad, organigrama, valores) | Fallback cuando ninguna otra tool dispara | RAG completo con `answer_question` y citación de fuentes |
| `solicitar_supervisor_humano` (**sensible**) | Quejas, escalamientos, "quiero hablar con una persona real" | Atajo `_looks_like_human_escalation` (>30 frases) | **Pasa por `HumanInTheLoopMiddleware`** antes de "registrar" la solicitud |

### Rutas adicionales (sin tools, decididas antes del LLM)

| Ruta | Cuándo | Lógica |
|------|--------|--------|
| `conversation` | Saludos, presentaciones ("hola mi nombre es Javier"), agradecimientos | `_answer_conversational_message` |
| `memory` | "¿Cómo me llamo?", "¿Cómo se llama mi papá?", "¿Qué te dije sobre X?" | `answer_personal_question_from_memory` con el LLM activo restringido al historial |
| `memory` (precio del anterior) | "¿Cuánto cuesta el primero que mencionaste?" | `answer_follow_up_from_memory` |

### Selector multi-LLM en caliente

- **Streamlit**: sidebar con `selectbox` (Ollama / Gemini / OpenAI), `text_input` para el modelo, slider de temperatura. Cambia el ChatModel desde la siguiente pregunta sin reiniciar nada.
- **FastAPI**: `POST /chat` acepta `provider`, `model` y `temperature` opcionales en el body. Si se omiten, toma defaults del `.env`.
- **Seguridad**: las API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`) siguen leyéndose desde `.env` por seguridad — el selector solo cambia proveedor + nombre de modelo.
- **Prioridad de credenciales**: si hay tanto `GEMINI_API_KEY` (en `.env`) como `GOOGLE_API_KEY` (exportada en el shell), prevalece la del `.env`. Esto se arregló porque el SDK de Google le daba prioridad a la del shell y rompía la demo.

---

## Estructura del repositorio

```text
taller1/
├── proyecto/
│   ├── app.py                      ← Streamlit (puerto 8501)
│   ├── docker-compose.yml          ← Postgres 16 + n8n 2.x
│   ├── requirements.txt
│   ├── .env.example
│   ├── README.md                   ← este archivo
│   │
│   ├── api/                        ← Taller 3 Fase 5
│   │   ├── main.py                 ← FastAPI app
│   │   ├── schemas.py              ← Pydantic ChatRequest/Response
│   │   ├── runtime.py              ← cache singleton de config+KB+chunks
│   │   └── __init__.py
│   │
│   ├── n8n/                        ← Taller 3 Fase 6
│   │   ├── workflows/
│   │   │   └── whatsapp_qbano.json ← workflow listo para importar
│   │   └── README.md               ← guía de setup paso a paso
│   │
│   ├── src/
│   │   ├── agent.py                ← run_agent, router determinístico,
│   │   │                             dynamic_prompt, HITL middleware
│   │   ├── chains.py               ← cadenas RAG (summary, FAQ, Q&A)
│   │   ├── checkpointer.py         ← PostgresSaver + conversation_messages
│   │   ├── config.py               ← AppConfig leído de .env y JSON
│   │   ├── llm.py                  ← init_chat_model con aliases
│   │   ├── memory.py               ← memoria personal + follow-ups
│   │   ├── processing.py           ← chunking con RecursiveCharacter
│   │   ├── project_state.py        ← watcher de cambios en datos
│   │   ├── prompts.py              ← ChatPromptTemplate por tarea
│   │   ├── scraper.py              ← scraping HTML + VTEX + PDF
│   │   ├── structured_tool.py      ← tool JSON estructurado
│   │   ├── tools.py                ← 4 tools Pydantic + tool sensible HITL
│   │   └── vector_store.py         ← langchain_chroma persistido
│   │
│   ├── scripts/
│   │   ├── build_knowledge_base.py ← scrape + chunk + index
│   │   ├── run_agent_batch.py      ← 33 casos del Taller 2
│   │   ├── run_api.sh              ← uvicorn con --reload
│   │   ├── run_question_batch.py   ← preguntas Q&A clásico
│   │   └── test_api.py             ← smoke test API 20 checks
│   │
│   ├── data/
│   │   ├── config/
│   │   │   ├── company_profile.json
│   │   │   └── prompts_config.json
│   │   ├── structured/
│   │   │   └── company_structured_data.json
│   │   ├── vector/chroma/          ← ChromaDB persistido (185 vec)
│   │   ├── raw/
│   │   │   └── raw_documents.json
│   │   └── processed/
│   │       ├── knowledge_base.txt
│   │       ├── chunks.json
│   │       ├── source_index.md
│   │       └── build_state.json
│   │
│   └── results/
│       ├── agent_test_questions.csv  ← 33 casos
│       ├── agent_test_results.csv    ← última corrida (33/33 OK)
│       ├── test_questions.csv
│       └── test_results.csv
│
├── articulo IEEE/                  ← entregable adicional (curso previo)
└── articulo IEEE.zip
```

> Nota: las instrucciones oficiales del curso (`taller-instrucciones.md`),
> el informe del Taller 2 y los datos sensibles (`infodata.md`) viven fuera
> del repo por política — están en el filesystem local del autor o
> gitignored. La rúbrica y la bitácora se mantienen aparte para no entregar
> material interno con el código público.

---

## Instalación local

### Prerrequisitos

| Software | Versión mínima | Cómo obtenerlo en macOS |
|----------|---------------|-------------------------|
| Python | 3.13 | `brew install python@3.13` |
| Docker Desktop | 27.x | https://www.docker.com/products/docker-desktop |
| Ollama | última | `brew install ollama && ollama serve` |
| ngrok | 3.x | `brew install ngrok` |
| Git | cualquiera reciente | preinstalado |

### Bootstrap

```bash
# 1. Clonar y entrar
git clone https://github.com/javierportillar/TIAML_Taller1.git
cd TIAML_Taller1/proyecto

# 2. Venv + dependencias
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Variables de entorno
cp .env.example .env
# editar .env según el proveedor LLM elegido (ver siguiente sección)

# 4. Modelo Ollama (default del Taller 3)
ollama pull gemma3:latest

# 5. Levantar Postgres + n8n
docker compose up -d
# qbano_postgres → puerto 5432
# qbano_n8n      → puerto 5678 (UI web)

# 6. Construir la base de conocimiento (una sola vez, salvo cambios en URLs)
python scripts/build_knowledge_base.py --max-pages 25
# genera: data/processed/* + data/vector/chroma/* + data/raw/*
```

### Configuración del modelo (`.env`)

```env
# === Default: Ollama local ===
LLM_PROVIDER=ollama
MODEL_NAME=gemma3:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
TEMPERATURE=0.0
MAX_CONTEXT_CHARS=200000

# === Memoria persistente (Taller 3 Fase 4) ===
LANGGRAPH_DB_URL=postgresql://qbano:qbano_dev@localhost:5432/qbano_agent

# === Google Gemini (opcional, recomendado como secundario) ===
# Obtén una key gratis en https://aistudio.google.com/apikey
GEMINI_API_KEY=tu_clave_aqui

# === OpenAI (opcional) ===
# OPENAI_API_KEY=tu_clave_aqui
```

---

## Cómo correr cada componente

### Streamlit (demo humana)

```bash
source .venv/bin/activate
streamlit run app.py
# http://localhost:8501
```

Tabs disponibles: `Agente conversacional`, `Resumen`, `FAQ`, `Q&A clásico`, `Configuración`. El selector LLM vive en el sidebar izquierdo.

### FastAPI (canal para WhatsApp / N8N)

```bash
bash scripts/run_api.sh
# http://127.0.0.1:8000
# Swagger UI en /docs
# Health check en /health
# Snapshot del agente en /info
```

Ejemplo de uso con `curl`:

```bash
# Con Ollama
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo_curl","message":"¿Cual es el WhatsApp?","provider":"ollama"}'

# Con Gemini
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo_curl","message":"¿Cuáles son los sándwiches más baratos?","provider":"google_genai","model":"gemini-2.5-flash"}'

# Pedir escalamiento humano (dispara HITL)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo_curl","message":"Tengo una queja, quiero hablar con un asesor humano"}'
```

### Bridge WhatsApp (n8n + ngrok + Twilio)

Guía detallada en [`n8n/README.md`](n8n/README.md). Resumen:

```bash
# 1. Lanzar ngrok apuntando a n8n
ngrok config add-authtoken <YOUR_TOKEN>   # solo la primera vez
ngrok http 5678                            # imprime URL pública

# 2. Importar el workflow (si se perdió)
docker stop qbano_n8n
docker run --rm \
  -v proyecto_n8n_data:/home/node/.n8n \
  -v $PWD/n8n/workflows:/wf \
  --entrypoint sh \
  docker.n8n.io/n8nio/n8n:latest \
  -c "n8n import:workflow --input=/wf/whatsapp_qbano.json --userId=<USER_ID>"
docker start qbano_n8n

# 3. En Twilio Sandbox settings:
#    "When a message comes in" = https://<ngrok-id>.ngrok-free.dev/webhook/twilio-qbano
#    Método: POST
#    Save.
```

### Pruebas en lote del agente

```bash
python scripts/run_agent_batch.py
# lee results/agent_test_questions.csv (33 casos)
# escribe results/agent_test_results.csv
# imprime totales por ruta esperada vs obtenida
```

### Smoke test de la API

```bash
python scripts/test_api.py
# corre 20 checks contra FastAPI vía TestClient
# salida esperada: RESULTADO: 20/20 checks pasaron
```

---

## Pruebas y evidencias

| Tipo de prueba | Cómo se corre | Última corrida | Evidencia |
|----------------|---------------|----------------|-----------|
| **33 casos batch del agente** | `python scripts/run_agent_batch.py` | 33/33 OK | `results/agent_test_results.csv` |
| **5 casos memoria personal** | smoke test ad-hoc | 5/5 OK | terminal output, commit `eca9fec` |
| **9 casos Fase 3.5 (HITL + escalamiento)** | smoke test ad-hoc | 9/9 OK | terminal output, commit `4bcf0c7` |
| **20 checks FastAPI** | `python scripts/test_api.py` | 20/20 OK | terminal output, commit `a3767d7` |
| **WhatsApp end-to-end real** | mensaje desde celular al `+14155238886` | 1/1 OK (4 jun 2026, 11:21 PM COL) | tabla `conversation_messages` id 385-386, n8n execution id=4 |

### Auditoría SQL de la conversación real

```sql
-- Multi-LLM en acción
SELECT llm_provider, llm_model, count(*) AS turnos
FROM conversation_messages
WHERE llm_provider IS NOT NULL
GROUP BY llm_provider, llm_model
ORDER BY turnos DESC;

-- Distribución de rutas elegidas por el agente
SELECT route, context_mode, count(*) AS veces
FROM conversation_messages
WHERE role = 'assistant'
GROUP BY route, context_mode
ORDER BY veces DESC;

-- Mensajes que dispararon HumanInTheLoop
SELECT id, thread_id, content, created_at
FROM conversation_messages
WHERE route = 'solicitar_supervisor_humano'
ORDER BY id DESC;
```

---

## Herramientas obligatorias Ruta A — checklist

La rúbrica del Taller 3 (Ruta A) verifica con `grep` el uso estricto de 7 herramientas:

| # | Herramienta | Dónde está en el repo | Estado |
|---|-------------|----------------------|--------|
| 1 | `init_chat_model` | `src/llm.py:88` | ✅ |
| 2 | `create_agent` | `src/agent.py:8` + invocado en `run_agent` | ✅ |
| 3 | `HumanInTheLoopMiddleware` | `src/agent.py:9` + `_build_hitl_middleware` | ✅ |
| 4 | `RecursiveCharacterTextSplitter` | `src/processing.py` | ✅ |
| 5 | `langchain_core.vectorstores` + embeddings nativos | `src/vector_store.py` (`langchain_chroma` + `langchain_huggingface`) | ✅ |
| 6 | `dynamic_prompt` | `src/agent.py:10` + `_build_dynamic_prompt_middleware` | ✅ |
| 7 | `PostgresSaver` | `src/checkpointer.py:10` + pasado a `create_agent` | ✅ |
| extra | Pydantic structured outputs | `src/tools.py` (4 `BaseModel`) | ✅ |
| extra | Gestión cortés de errores | `src/tools.py:148` + `_retrieval_only_payload` cortés | ✅ |

---

## Decisiones técnicas

| # | Decisión | Por qué |
|---|----------|---------|
| **D-01** | Migrar de feature hashing custom + JSON a **ChromaDB + HuggingFace** (Taller 2) | El profesor señaló que el approach inicial era "el camino más difícil". `langchain_chroma` + `langchain_huggingface` son los componentes nativos que pide la rúbrica. |
| **D-02** | **Hybrid router determinístico + LLM** en vez de delegarle todo al LLM | El LLM (especialmente Ollama gemma3 local) elige rutas equivocadas con frecuencia. Heurísticas regex en `agent.py` resuelven >90% de los casos antes de pegarle al LLM. |
| **D-03** | **Dos capas de persistencia** (LangGraph PostgresSaver binario + tabla custom legible) | La rúbrica pide PostgresSaver. La tabla custom permite auditar con SQL y repintar el chat de Streamlit al reabrirse. Ambas viven en el mismo Postgres. |
| **D-04** | **n8n local en Docker** sobre n8n Cloud | Trial cloud caduca a los 14 días. Demo en local = gratis para siempre. |
| **D-05** | **ngrok como expositor temporal** sobre dominio propio | La sustentación dura 15 minutos. Pagar dominio + reverse proxy no se justifica. La URL rotativa es aceptable porque se actualiza Twilio en 30 s. |
| **D-06** | **Tool sensible aprobada por defecto en ausencia de UI HITL** | `Command(resume=[{"type": "approve"}])` cuando llega un `__interrupt__`. Esto permite que el flujo funcione end-to-end sin un panel humano. En producción real se reemplazaría por un dashboard que pide confirmación. |
| **D-07** | **GEMINI_API_KEY del `.env` gana sobre GOOGLE_API_KEY del shell** | El SDK de Google prioriza GOOGLE_API_KEY. Si el shell tiene una key vieja exportada, sobreescribe la del `.env` y la demo falla. Fix en `src/llm.py:67`. |
| **D-08** | **`gemma3:latest` como default Ollama** (no `gemma4`) | `gemma4` no existe como modelo oficial. Era un default ficticio del Taller 2 que nunca se ejercitó porque los tests cubrían rutas determinísticas. |
| **D-09** | **Watcher solo vigila JSON de datos, no `.py`** | El watcher original disparaba un rebuild de Chroma cada vez que se editaba código, lo que producía `SQLite readonly` por race condition. Solo escuchar archivos de datos (URLs, prompts, JSON estructurado) elimina el problema. |
| **D-10** | **El Twilio webhook se configura por UI, no por API** | La API pública de Twilio no expone el webhook del sandbox. Es la única acción manual de toda la Fase 6 (literal pegar una URL en un campo). |

---

## Reglas que no negocio

1. **Las API keys nunca van al código.** Solo en `.env` (gitignored). El selector de LLM en la UI cambia proveedor y modelo, no credenciales.
2. **El sandbox de Twilio nunca se publica como producción.** Si se quiere salir a clientes reales, hay que registrar un Sender de WhatsApp Business con verificación de Meta.
3. **El `thread_id` es la única llave de aislamiento de conversaciones.** En WhatsApp = número de teléfono; en Streamlit = `streamlit_local`; en pruebas batch = UUID nuevo por corrida.
4. **El agente nunca inventa números, precios ni canales.** Si no está en `data/structured/*` ni en Chroma, responde con cortesía explicando que no tiene el dato.
5. **Toda respuesta del asistente queda en Postgres.** Permite auditar y reproducir cualquier alegato del usuario.
6. **El router determinístico tiene prioridad sobre el LLM.** Si una regla regex matchea, se ejecuta esa ruta; el LLM solo se invoca cuando ninguna heurística aplica.

---

## Riesgos conocidos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|-----------|
| Cuota de Gemini se agota durante demo | Media | Bajo (Ollama responde igual) | Mostrar fallback Ollama en sustentación |
| ngrok URL rota tras reinicio | Alta | Medio (hay que actualizar Twilio) | Tener Twilio Sandbox abierto en pestaña; actualizar manualmente toma 30 s |
| Docker Desktop se cierra solo | Media | Alto (n8n + Postgres caen) | Verificación previa al inicio de sustentación: `docker ps` |
| Ollama tarda > 30 s en frio (primera pregunta) | Alta | Bajo (UX) | Hacer una pregunta de prueba 5 min antes de la sustentación |
| Twilio Sandbox bloquea por inactividad | Baja | Alto | Mantener mensaje de prueba reciente en el sandbox |
| Token de ngrok / Twilio expuestos | Baja | Crítico | `taller3/infodata.md` no está en el repo; rotar tokens tras sustentación |

---

## Roadmap

```
✅ Taller 1   Scraping + RAG + Streamlit Q&A clásico
✅ Taller 2   Agente + memoria + tool estructurada + ChromaDB
✅ Taller 3   Function Calling estricto + FastAPI + PostgresSaver +
              WhatsApp via N8N + Twilio + multi-LLM + HITL
🟡 Bonus      t-SNE/UMAP de conversaciones para análisis de clústeres
🟡 Informe    PDF unificado para sustentación
─────────────────────────────────────────────────────────────
⬜ V2         Producción real: número WhatsApp Business propio,
              dashboard humano para HITL, dominio propio,
              monitoreo (UptimeRobot + Sentry), CI/CD,
              análisis ofensivo de tokens y costos
```

---

## Cómo navegar este repo

| Vengo a... | Empiezo por |
|-----------|-------------|
| Entender el sistema en 5 min | Este `README.md` |
| Configurar el bridge WhatsApp | [`n8n/README.md`](n8n/README.md) |
| Ver el código del agente | [`src/agent.py`](src/agent.py) |
| Ver las tools Pydantic | [`src/tools.py`](src/tools.py) |
| Probar la API en local | [`scripts/test_api.py`](scripts/test_api.py) |
| Probar el agente en lote | [`scripts/run_agent_batch.py`](scripts/run_agent_batch.py) |
| Construir la base de conocimiento desde cero | [`scripts/build_knowledge_base.py`](scripts/build_knowledge_base.py) |
| Ver la rúbrica oficial | Compartida por el profesor en el curso (no se entrega con el código) |
| Ver el informe del Taller 2 | `taller2/README.md` (fuera del repo, en el filesystem local del autor) |

---

## Historial de commits relevantes

```
b29c6e4  Fase 6 — WhatsApp via Twilio Sandbox + n8n + ngrok end-to-end
a3767d7  Fase 5 — API REST FastAPI: POST /chat + /health + /info
4bcf0c7  Fase 3.5 — HumanInTheLoopMiddleware + dynamic_prompt + cortesía
eca9fec  Fase 4 — PostgresSaver + selector multi-LLM + memoria personal
6630261  Fase 3 — Function Calling con create_agent + tools Pydantic
470aa7e  Fase 2 — Chunking con RecursiveCharacterTextSplitter
c91768a  Fase 1 — init_chat_model multi-proveedor + cierre del Taller 2
```

---

## Créditos

- Curso: Técnicas Avanzadas de IA Aplicadas en Modelos de Lenguaje
- Maestría: Inteligencia Artificial — UAO 2026-1
- Empresa caso de estudio: **Sándwich Qbano** (información tomada exclusivamente de fuentes públicas: sitio web e informe oficial de sostenibilidad 2023)
- Autor: **Javier Portilla Rosero**
