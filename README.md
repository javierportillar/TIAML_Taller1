# Sándwich Qbano · Agente Conversacional Multi-Canal

Este repo es el resultado de armar, durante tres talleres del curso **Técnicas Avanzadas de IA Aplicadas en Modelos de Lenguaje** (Maestría en IA, UAO), un agente conversacional para una empresa real del Valle del Cauca. La empresa asignada fue [Sándwich Qbano](https://www.sandwichqbano.com/), una cadena colombiana de comida rápida con más de 220 puntos de venta, y la idea era llegar a tener un asistente que respondiera por WhatsApp **sin inventar datos fuera de las fuentes oficiales**, reusando solo lo público: el sitio web y el informe oficial de sostenibilidad 2023.

Arrancamos con un Q&A clásico de scraping + RAG en Streamlit, en el segundo taller le metimos memoria, una tool estructurada y un router que decide entre fuentes, y en el tercero terminamos productizándolo: misma cabeza, ahora detrás de una API REST en FastAPI, con memoria persistente en Postgres, function calling estricto con Pydantic, y un bridge a WhatsApp via Twilio Sandbox + n8n. Una persona puede mandar un mensaje desde su celular y el agente le contesta en menos de cinco segundos con la información correcta.

La idea principal que guió todo: que el sistema no fuera "un chatbot bonito", sino una pieza que se pudiera defender técnicamente en cada capa. Por eso hay decisiones explícitas en el código (ver sección de decisiones técnicas más abajo), trazabilidad completa en SQL, y tests automatizados que pasan al 100%.

> **Demo verificada el 4 de junio.** Mensaje real desde mi WhatsApp personal → Twilio Sandbox → ngrok → n8n local → FastAPI → agente LangGraph → respuesta de vuelta. Todo el turno quedó grabado en Postgres con `route=consultar_datos_contacto` y `llm_provider=ollama`.

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

## Cómo fue evolucionando

Una cosa que me importó desde el inicio fue no rehacer el sistema en cada taller. La base que valida un test del Taller 2 sigue corriendo igual en el Taller 3, solo que ahora tiene tres caras: la UI de Streamlit, una API REST y un canal de WhatsApp. Por dentro es el mismo `run_agent`.

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

Hay piezas que sobrevivieron sin tocarse entre talleres:

- La base de conocimiento (scraping → texto consolidado → chunks en `data/processed/*`) sigue siendo la misma.
- Los embeddings son el mismo modelo multilingüe de HuggingFace de 384 dimensiones.
- El vector store es Chroma persistido en disco, exactamente como quedó al final del Taller 2.
- El router determinístico vive en `src/agent.py` y decide la mayoría de las rutas antes de invocar al LLM.

El Taller 2 me había dejado una deuda: usaba feature hashing y un JSON custom en lugar de un vector store de verdad. El profe me señaló que ese era el camino más difícil. Migramos a ChromaDB + HuggingFace embeddings (los componentes nativos de LangChain) y desde ese momento todo el routing se mantuvo estable mientras se le metían más capas encima.

---

## El problema y por qué importa

La pregunta concreta que el sistema responde es:

> ¿Cómo le da una cadena de comida rápida con presencia nacional atención conversacional por WhatsApp, sin inventar datos fuera de sus fuentes, sin contratar más gente, sin tocar su sitio web, y manteniendo sus fuentes oficiales como única verdad?

La respuesta termina siendo más simple de lo que suena. Tres ideas:

Primero, todo el conocimiento se ingiere una sola vez. El sitio web público y el informe oficial de sostenibilidad pasan por un pipeline que limpia el HTML, parsea el PDF, trocea el texto y lo vectoriza en Chroma. Lo "puntual" (WhatsApp de servicio, redes sociales, PBX, sedes) se queda en un JSON estructurado porque preguntarle al RAG por esos campos puntuales es como abrir Google para saber qué hora es.

Segundo, hay dos caras del sistema pero un solo cerebro. Streamlit corre en `localhost:8501` para demo y debugging humano. La API en FastAPI corre en `localhost:8000` y es la que n8n llama cuando llega un WhatsApp. Las dos terminan invocando el mismo `run_agent`. Esto significa que cualquier mejora al agente sirve para los dos canales al mismo tiempo, sin sincronización ni código duplicado.

Tercero, el agente no le pregunta al LLM más de lo que tiene que preguntarle. Antes de invocar a Gemini o a Ollama hay un router determinístico (regex y heurísticas) que detecta el tipo de pregunta. Solo cuando ninguna regla matchea, se le pide al LLM que elija una tool. Esto baja el costo, baja la latencia, y elimina la mayoría de alucinaciones de routing.

Si el usuario pide explícitamente hablar con un humano ("tengo una queja, pásame con alguien"), el sistema detecta esa intención y dispara una tool sensible que pasa por `HumanInTheLoopMiddleware` antes de "registrar" la solicitud. Es la única tool que tiene esa puerta de aprobación, y está ahí porque la rúbrica lo pide pero además porque hace sentido para una empresa real.

---

## Stack tecnológico

### Lenguajes y frameworks

| Componente | Tecnología | Justificación |
|------------|-----------|---------------|
| **Orquestación de agente** | LangChain 1.x + LangGraph | `init_chat_model`, `create_agent`, `dynamic_prompt`, `HumanInTheLoopMiddleware`. Cumple las 7 herramientas obligatorias de la Ruta A del Taller 3. |
| **Function Calling** | Pydantic v2 schemas + `langchain_core.tools.tool` | Cada tool tiene `BaseModel` estricto en `src/tools.py`. El LLM no puede invocar tools con argumentos malformados. |
| **API REST** | FastAPI 0.136 + Uvicorn | OpenAPI auto-generado en `/docs`, lifespan async para precargar config, payload `ChatRequest` validado con Pydantic. |
| **UI demo** | Streamlit 1.33 | UI rápida para sustentación humana. Sidebar con selector multi-LLM en caliente. |
| **LLMs** | OpenCode Go (`kimi-k2.6` cloud para WhatsApp) + Ollama (`gemma3:latest` local) + Google Gemini (`gemini-2.5-flash`) + OpenAI (`gpt-4o-mini`) opcional | Selector por sesión Streamlit **y por request** en `POST /chat`. Permite alternar en vivo sin reiniciar nada. |
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

LangChain estricto en lugar de LangChain con shortcuts. La rúbrica del Taller 3 verifica con `grep` que estén usadas siete herramientas concretas (`init_chat_model`, `create_agent`, `HumanInTheLoopMiddleware`, `RecursiveCharacterTextSplitter`, `langchain_core.vectorstores`, `dynamic_prompt`, `PostgresSaver`). Estuve tentado a usar `LangChain.Memory` "mágica" o atajos similares, pero cada atajo era un punto menos.

Ollama de default, Gemini como secundario. Si se cae el internet en pleno examen, el sistema sigue respondiendo. Si Google decide cortarme la cuota gratuita de Gemini (me pasó: tuve que regenerar la API key una vez), también. Las dos están operativas y se pueden alternar en caliente.

Chroma sobre Pinecone, Weaviate y compañía. El corpus tiene menos de 200 chunks. Montar un vector store en la nube para algo que cabe en 3 MB de SQLite local es ingeniería innecesaria. Chroma persistido en disco corre solo, no cuesta nada, y se respalda copiando una carpeta.

n8n local sobre n8n Cloud. El cloud tiene un trial de 14 días. Si la sustentación se atrasa o se reprograma, el bot deja de funcionar el día de la presentación. Docker + n8n local = gratis para siempre y la sustentación está blindada.

Twilio Sandbox sobre WhatsApp Business API directa. La API oficial de Meta requiere registro de empresa, verificación legal, número productivo aprobado y un proceso de meses. El sandbox de Twilio te deja conectar tu propio número con un `join <palabra-secreta>` y listo. Para demo es exacto lo que se necesita.

FastAPI sobre Flask. Tipado nativo con Pydantic, OpenAPI auto-generado en `/docs`, lifespan async para precargar el knowledge base una sola vez. Con Flask habría tenido que hacer eso a mano.

Router híbrido (regex primero, LLM solo si nada matcheó). Cuando le delegaba todo el routing al LLM, pasaba que preguntas claras como "¿qué WhatsApp tienen?" se iban a la ruta de RAG vectorial. Con heurísticas determinísticas en `agent.py`, la mayoría de respuestas terminan viniendo de fuentes verificables. Sobre los 33 casos del batch automatizado: **33% se resuelven sin invocar al LLM** (atajos `deterministic` + ruta `conversation`) y **un 64% adicional pasa por el agente solo para elegir tool, pero la respuesta final viene del JSON estructurado** (`context_mode=structured_json`). Solo un puñado de turnos requieren generación libre. Ahorra tokens, ahorra latencia, y elimina malas decisiones del modelo.

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

# 6. Construir la base de conocimiento (obligatorio en primer clone)
python scripts/build_knowledge_base.py --max-pages 25
# Genera: data/processed/*, data/vector/chroma/*, data/raw/*
# Los archivos binarios de Chroma NO están versionados (regenerables).
# Toma ~30 s la primera vez (descarga modelo de embeddings).
```

### Configuración del modelo (`.env`)

```env
# === Default: Ollama local ===
LLM_PROVIDER=ollama
MODEL_NAME=gemma3:latest
OLLAMA_BASE_URL=http://127.0.0.1:11434
TEMPERATURE=0.0
MAX_CONTEXT_CHARS=200000

# === OpenCode Go (recomendado para WhatsApp: cloud, no consume CPU local) ===
# LLM_PROVIDER=opencode_go
# MODEL_NAME=kimi-k2.6
# OPENCODE_API_KEY=tu_clave_de_opencode
# OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1

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

# Con OpenCode Go (cloud)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"demo_curl","message":"¿Cual es el WhatsApp?","provider":"opencode_go","model":"kimi-k2.6"}'

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

> Asume que el venv está activo (`source .venv/bin/activate`). Si no, reemplaza
> `python` por `.venv/bin/python` en todos los comandos de esta sección.

```bash
source .venv/bin/activate                # solo la primera vez por sesión
python scripts/run_agent_batch.py
# lee results/agent_test_questions.csv (33 casos)
# escribe results/agent_test_results.csv
# imprime totales: "RESULTADO BATCH: 33/33 rutas correctas" + desglose por contexto
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

## Decisiones que cambiaron el rumbo

Estas son las decisiones técnicas que realmente movieron la aguja. Las dejo numeradas (D-01 a D-10) porque algunas las cité en el código y en los commits.

**D-01 — Migrar a ChromaDB + HuggingFace en Taller 2.** El feature hashing custom que entregué en el Taller 1 me costó un comentario directo del profe: "te fuiste por el camino más difícil". Reescribí el vector store con `langchain_chroma` + `langchain_huggingface` (los componentes nativos que pide la rúbrica) y desde ahí todo lo demás se sostiene sobre una base estándar.

**D-02 — Router híbrido determinístico + LLM.** Probé delegándole todo al modelo (gemma3 local) y el routing era inconsistente: "¿cuál es el WhatsApp?" se le iba a RAG vectorial. Metí heurísticas regex en `agent.py` que cubren los casos típicos (preguntas comerciales, datos puntuales de contacto, memoria personal, escalamiento humano), y el LLM solo aparece cuando ninguna regla matcheó. Sobre los 33 casos del batch: 33% se resuelven sin invocar al LLM (atajos puros), 64% pasa por el agente para elegir tool pero la respuesta final viene del JSON estructurado (no de generación libre), y solo el resto requiere síntesis con LLM.

**D-03 — Dos capas de persistencia.** Una es lo que la rúbrica pide: `PostgresSaver` de LangGraph guarda checkpoints binarios del agente para reanudar estado. La otra es algo que yo armé encima: una tabla `conversation_messages` legible en SQL para auditar con DBeaver y para repintar el chat cuando Streamlit reinicia. Las dos viven en el mismo Postgres y se sincronizan en cada turno.

**D-04 — n8n local en Docker sobre n8n Cloud.** El trial cloud dura 14 días y no quiero que se caiga la demo el día de la sustentación. Con Docker en local es gratis para siempre y reproducible con un `docker compose up`.

**D-05 — ngrok como expositor temporal.** La sustentación dura quince minutos. Comprar dominio y montar reverse proxy con SSL solo para eso no se justifica. La URL de ngrok rota cada vez que reinicia el túnel, pero actualizar Twilio toma treinta segundos.

**D-06 — La tool sensible se auto-aprueba en ausencia de UI HITL.** Cuando el flujo HITL devuelve un `__interrupt__`, el código manda `Command(resume=[{"type": "approve"}])` y sigue adelante. En producción real eso debería ser un dashboard donde un humano confirma; aquí está simulado porque la cadena de WhatsApp tiene que ser instantánea.

**D-07 — `GEMINI_API_KEY` del `.env` gana sobre `GOOGLE_API_KEY` del shell.** El SDK de Google priorizaba una `GOOGLE_API_KEY` vieja que tenía exportada en `~/.zshrc` (suspendida) mientras ignoraba la nueva en el `.env`. La demo fallaba sin explicación visible. El fix está en `src/llm.py`: si hay `GEMINI_API_KEY`, sobreescribe la del shell.

**D-08 — `gemma3:latest` como default Ollama, no `gemma4`.** Heredé `gemma4:latest` como default del Taller 2 y nunca lo ejercité porque los tests cubrían rutas determinísticas. Cuando llegó el momento de usar el LLM de verdad (preguntas de memoria personal), descubrí que `gemma4` no existe como modelo oficial. Cambié a `gemma3:latest` que sí existe y bajé el modelo con `ollama pull`.

**D-09 — El watcher solo vigila los JSON de datos, no los `.py`.** El watcher original vigilaba 14 archivos incluyendo todo `src/`. Editar `app.py` mientras Streamlit corría disparaba un rebuild de Chroma, que chocaba con el SQLite que Chroma ya tenía abierto, y el directorio quedaba en estado "readonly". Reducir el watcher a solo tres archivos JSON eliminó el problema.

**D-10 — El webhook de Twilio se configura por UI, no por API.** La API pública de Twilio no expone el endpoint del sandbox para que un programa lo configure. Esa fue literalmente la única acción manual de toda la Fase 6: pegar una URL en un campo y darle Save.

---

## Reglas que no negocio

Son las que mantienen al proyecto coherente y sin gotchas vergonzosos.

1. Las API keys nunca van al código. Viven en `.env` que está gitignored. El selector de la UI cambia proveedor y modelo, jamás credenciales.
2. El sandbox de Twilio nunca se publica como producción. Si esto sale a clientes reales, hay que registrar un Sender de WhatsApp Business con todo el papeleo de Meta.
3. El `thread_id` es la única llave de aislamiento de conversaciones. En WhatsApp es el número de teléfono. En Streamlit local es la cadena `streamlit_local`. En los tests de batch es un UUID que se descarta al final.
4. El agente no inventa números, precios ni canales. Si la información no está en el JSON estructurado ni en Chroma, responde con cortesía explicando que no tiene el dato.
5. Toda respuesta del asistente queda persistida en Postgres. Si el día de la sustentación alguien dice "me respondió mal", existe una query SQL que muestra el turno exacto.
6. El router determinístico tiene prioridad sobre el LLM. Si una regla regex matchea, se ejecuta esa ruta. El LLM solo aparece cuando ninguna heurística pudo decidir.

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

## Créditos y cierre

Esto lo hice yo, **Javier Portilla Rosero**, durante 2026-1 para el curso *Técnicas Avanzadas de IA Aplicadas en Modelos de Lenguaje* de la Maestría en Inteligencia Artificial de la **UAO**. La empresa asignada fue **Sándwich Qbano** y toda la información usada viene de fuentes públicas: su sitio web oficial, su catálogo en línea, y el Informe de Sostenibilidad 2023 que publicaron en su sitio de transparencia.

No hay datos privados de la empresa, no hubo contacto con ellos, no se replica nada que no esté ya en internet bajo su propio dominio. Si Sándwich Qbano quiere ver esto funcionando, basta con clonar el repo y seguir las instrucciones de la sección "Instalación local".

Si llegaste hasta acá leyendo: gracias por el tiempo. Cualquier pregunta o sugerencia, los issues del repo están abiertos.
