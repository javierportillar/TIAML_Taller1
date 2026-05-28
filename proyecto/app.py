from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from src.agent import run_agent
from src.chains import answer_question, generate_faq, generate_summary
from src.checkpointer import clear_thread_messages, load_thread_messages
from src.config import build_app_config
from src.llm import create_chat_model, describe_active_provider
from src.processing import load_chunks, load_knowledge_base
from src.vector_store import load_vector_index
from src.project_state import detect_local_state_changes, rebuild_processed_state
from src.prompts import DEFAULT_PROMPTS, load_prompt_config, save_prompt_config

STREAMLIT_THREAD_ID = "streamlit_local"

PROVIDER_OPTIONS = ["ollama", "google_genai", "openai"]
PROVIDER_LABELS = {
    "ollama": "Ollama (local)",
    "google_genai": "Google Gemini",
    "openai": "OpenAI",
}
PROVIDER_DEFAULT_MODEL = {
    "ollama": "gemma3:latest",
    "google_genai": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}


def _on_provider_change() -> None:
    """Cuando el usuario cambia el proveedor en el sidebar, sugerimos su modelo por defecto."""
    selected = st.session_state.get("provider_param", "")
    suggested = PROVIDER_DEFAULT_MODEL.get(selected)
    if suggested:
        st.session_state.model_name_param = suggested


def _check_provider_credentials(provider: str) -> tuple[bool, str]:
    """Verifica que existan credenciales validas para el proveedor seleccionado."""
    if provider == "google_genai":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            return False, (
                "Falta `GEMINI_API_KEY` en `.env`. Obten una gratis en "
                "https://aistudio.google.com/apikey y reinicia Streamlit."
            )
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return False, "Falta `OPENAI_API_KEY` en `.env`. Configurala y reinicia Streamlit."
    if provider == "ollama" and not os.getenv("OLLAMA_BASE_URL"):
        # Ollama puede usar default, no es bloqueante
        return True, ""
    return True, ""

st.set_page_config(page_title="Taller 3 - Agente Conversacional", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e7e9ee;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_project_state():
    config = build_app_config()
    knowledge_exists = config.paths.knowledge_base_path.exists()
    chunks_exists = config.paths.chunks_path.exists()
    return config, knowledge_exists, chunks_exists


def _load_knowledge(paths) -> tuple[str, list[dict[str, str | int]]]:
    knowledge_text = load_knowledge_base(paths.knowledge_base_path)
    chunks = load_chunks(paths.chunks_path)
    return knowledge_text, chunks


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _join_lines(values: list[str]) -> str:
    return "\n".join(values)


def _derive_domains(urls: list[str]) -> list[str]:
    domains: set[str] = set()
    for url in urls:
        domain = urlparse(url).netloc.strip().lower()
        if domain:
            domains.add(domain)
    return sorted(domains)


def _save_company_profile(config, payload: dict[str, object]) -> None:
    config.paths.company_config_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _init_session_state(config) -> None:
    defaults = {
        "summary_output": "",
        "summary_error": "",
        "faq_output": "",
        "faq_error": "",
        "qa_output": "",
        "qa_error": "",
        "qa_sources": [],
        "qa_context_mode": "",
        "rebuild_notice": "",
        "runtime_signature": "",
        "config_save_notice": "",
        "agent_messages": [],
        "agent_events": [],
        "temperature_param": float(config.runtime.temperature),
        "max_context_chars_param": int(config.runtime.max_context_chars),
        "provider_param": config.runtime.provider,
        "model_name_param": config.runtime.model_name,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.agent_messages:
        try:
            persisted_messages = load_thread_messages(STREAMLIT_THREAD_ID)
        except Exception:
            persisted_messages = []
        if persisted_messages:
            st.session_state.agent_messages = persisted_messages
            st.session_state.agent_events = [{} for _ in persisted_messages]


def _reset_generated_outputs() -> None:
    st.session_state.summary_output = ""
    st.session_state.summary_error = ""
    st.session_state.faq_output = ""
    st.session_state.faq_error = ""
    st.session_state.qa_output = ""
    st.session_state.qa_error = ""
    st.session_state.qa_sources = []
    st.session_state.qa_context_mode = ""


config, knowledge_exists, chunks_exists = _load_project_state()
_init_session_state(config)

runtime_signature = "|".join(
    [
        config.runtime.provider,
        config.runtime.model_name,
        str(config.runtime.temperature),
        str(config.runtime.max_context_chars),
    ]
)
if st.session_state.runtime_signature != runtime_signature:
    st.session_state.runtime_signature = runtime_signature
    st.session_state.temperature_param = float(config.runtime.temperature)
    st.session_state.max_context_chars_param = int(config.runtime.max_context_chars)
    # Resincronizar el selector con los valores del .env solo al detectar cambios externos.
    if config.runtime.provider in PROVIDER_OPTIONS:
        st.session_state.provider_param = config.runtime.provider
    st.session_state.model_name_param = config.runtime.model_name

# Si el provider guardado en sesion no esta en la lista (por ejemplo, valor raro de .env), lo normalizamos.
if st.session_state.provider_param not in PROVIDER_OPTIONS:
    st.session_state.provider_param = PROVIDER_OPTIONS[0]

# ---------------------------------------------------------------------------
# Sidebar: selector multi-LLM (Taller 3, cierre del pendiente del Taller 2)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Modelo LLM activo")
    st.caption(
        "Cambia de proveedor y modelo sin reiniciar Streamlit. "
        "Las API keys siguen leyendose desde `.env` por seguridad."
    )
    st.selectbox(
        "Proveedor",
        options=PROVIDER_OPTIONS,
        format_func=lambda value: PROVIDER_LABELS.get(value, value),
        key="provider_param",
        on_change=_on_provider_change,
    )
    st.text_input(
        "Modelo",
        key="model_name_param",
        help=(
            "Sugerencias: `gemma3:latest` para Ollama, `gemini-2.5-flash` para Gemini, "
            "`gpt-4o-mini` para OpenAI."
        ),
    )
    st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="temperature_param",
    )
    creds_ok, creds_msg = _check_provider_credentials(st.session_state.provider_param)
    if not creds_ok:
        st.error(creds_msg)
    else:
        st.success(
            f"Activo: {describe_active_provider(st.session_state.provider_param, st.session_state.model_name_param)}"
        )

rebuild_required, changed_files, _ = detect_local_state_changes(config)
if rebuild_required:
    with st.spinner("Se detectaron cambios locales. Reprocesando la base de conocimiento..."):
        try:
            stats = rebuild_processed_state(config, max_pages=25)
            _reset_generated_outputs()
            st.session_state.rebuild_notice = (
                "Base reprocesada por cambios en: "
                + ", ".join(changed_files)
                + f". Documentos: {stats['documents']}, Chunks: {stats['chunks']}."
            )
            knowledge_exists = config.paths.knowledge_base_path.exists()
            chunks_exists = config.paths.chunks_path.exists()
        except Exception as exc:
            st.error(f"No fue posible reprocesar automaticamente la base: {exc}")
            st.stop()

st.title("Sistema Q&A Empresarial")
st.caption("Taller 3: agente conversacional con Function Calling, memoria, RAG vectorial y tools Pydantic.")

if st.session_state.rebuild_notice:
    st.info(st.session_state.rebuild_notice)
if st.session_state.config_save_notice:
    st.success(st.session_state.config_save_notice)
    st.session_state.config_save_notice = ""

provider = st.session_state.provider_param
model_name = (st.session_state.model_name_param or "").strip() or PROVIDER_DEFAULT_MODEL.get(provider, "")
temperature = float(st.session_state.temperature_param)
max_context_chars = int(st.session_state.max_context_chars_param)

st.subheader(config.company.company_name)
st.write(config.company.company_description)

if config.company.company_name == "EMPRESA_ASIGNADA":
    st.warning(
        "Aun no has configurado la empresa real. Edita data/config/company_profile.json antes de la demo."
    )

if not (knowledge_exists and chunks_exists):
    st.error(
        "Todavia no existe la base de conocimiento procesada. Ejecuta primero: "
        "`python3 scripts/build_knowledge_base.py --max-pages 25`"
    )
    st.stop()

knowledge_text, chunks = _load_knowledge(config.paths)

col1, col2, col3 = st.columns(3)
col1.metric("Fuentes procesadas", len({str(chunk["url"]) for chunk in chunks}))
col2.metric("Chunks", len(chunks))
vector_payload = load_vector_index(config.paths.vector_index_path)
vector_count = int(vector_payload.get("vectors", 0) or 0)
if vector_count == 0:
    fallback_records = vector_payload.get("records", [])
    if isinstance(fallback_records, list):
        vector_count = len(fallback_records)
col3.metric("Vectores Chroma", vector_count)
st.caption(
    f"Vector store: Chroma · Embeddings: {vector_payload.get('embedding_model', 'sentence-transformers')}"
)

with st.expander("URLs configuradas"):
    for url in config.company.seed_urls + config.company.additional_sources:
        st.write(f"- {url}")

agent_tab, summary_tab, faq_tab, qa_tab, config_tab = st.tabs(
    ["Agente conversacional", "Resumen", "FAQ", "Q&A clasico", "Configuracion"]
)

with agent_tab:
    st.write(
        "Chat principal del Taller 3. El agente usa create_agent con tools Pydantic, "
        "memoria conversacional y RAG con indice vectorial persistido."
    )
    left, right = st.columns([0.75, 0.25])
    with right:
        if st.button("Limpiar conversacion", use_container_width=True):
            st.session_state.agent_messages = []
            st.session_state.agent_events = []
            try:
                clear_thread_messages(STREAMLIT_THREAD_ID)
            except Exception:
                pass
            st.rerun()
        st.markdown("**Herramientas disponibles**")
        st.write("- `memory`: seguimiento por historial")
        st.write("- `consultar_datos_contacto`: JSON de datos puntuales")
        st.write("- `buscar_catalogo_productos`: catalogo, precios, combos y promociones")
        st.write("- `consultar_informacion_corporativa`: base documental vectorial")
        st.caption(f"Indice vectorial: `{config.paths.vector_index_path.name}`")

    with left:
        for index, message in enumerate(st.session_state.agent_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                event = (
                    st.session_state.agent_events[index]
                    if index < len(st.session_state.agent_events)
                    else {}
                )
                if message["role"] == "assistant" and event:
                    st.caption(
                        f"Ruta: {event.get('route', 'n/a')} | "
                        f"Contexto: {event.get('context_mode', 'n/a')}"
                    )
                    if event.get("reasoning"):
                        with st.expander("Decision del agente"):
                            st.write(event["reasoning"])
                    if event.get("sources"):
                        with st.expander("Fuentes usadas"):
                            for source in event["sources"]:
                                st.write(f"- {source['title']} - {source['url']}")

        agent_question = st.chat_input(
            "Pregunta al agente. Ej: ¿Cuál es el WhatsApp? / Enlista productos / ¿Y cuánto cuesta el primero?"
        )
        if agent_question:
            st.session_state.agent_messages.append(
                {"role": "user", "content": agent_question}
            )
            st.session_state.agent_events.append({})
            try:
                model = create_chat_model(
                    provider=provider, model_name=model_name, temperature=temperature
                )
                result = run_agent(
                    model=model,
                    company_name=config.company.company_name,
                    question=agent_question,
                    chat_history=st.session_state.agent_messages[:-1],
                    knowledge_text=knowledge_text,
                    chunks=chunks,
                    max_context_chars=max_context_chars,
                    structured_data_path=config.paths.structured_data_path,
                    vector_index_path=config.paths.vector_index_path,
                    thread_id=STREAMLIT_THREAD_ID,
                    llm_provider=provider,
                    llm_model=model_name,
                )
                st.session_state.agent_messages.append(
                    {"role": "assistant", "content": result.answer}
                )
                st.session_state.agent_events.append(
                    {
                        "route": result.route,
                        "reasoning": result.reasoning,
                        "context_mode": result.context_mode,
                        "sources": result.sources,
                    }
                )
            except Exception as exc:
                st.session_state.agent_messages.append(
                    {"role": "assistant", "content": f"Error del agente: {exc}"}
                )
                st.session_state.agent_events.append(
                    {
                        "route": "error",
                        "reasoning": str(exc),
                        "context_mode": "error",
                        "sources": [],
                    }
                )
            st.rerun()

with summary_tab:
    st.write(
        "Genera un panorama ejecutivo de la empresa con el conocimiento consolidado."
    )
    if st.button("Generar resumen", use_container_width=True):
        try:
            model = create_chat_model(
                provider=provider, model_name=model_name, temperature=temperature
            )
            st.session_state.summary_output = generate_summary(
                model=model,
                company_name=config.company.company_name,
                knowledge_text=knowledge_text,
                chunks=chunks,
                max_context_chars=max_context_chars,
            )
            st.session_state.summary_error = ""
        except Exception as exc:
            st.session_state.summary_error = str(exc)

    if st.session_state.summary_error:
        st.error(st.session_state.summary_error)
    if st.session_state.summary_output:
        st.markdown(st.session_state.summary_output)

with faq_tab:
    faq_count = st.slider("Numero de preguntas frecuentes", 5, 15, 8)
    if st.button("Generar FAQ", use_container_width=True):
        try:
            model = create_chat_model(
                provider=provider, model_name=model_name, temperature=temperature
            )
            st.session_state.faq_output = generate_faq(
                model=model,
                company_name=config.company.company_name,
                knowledge_text=knowledge_text[:max_context_chars],
                faq_count=faq_count,
            )
            st.session_state.faq_error = ""
        except Exception as exc:
            st.session_state.faq_error = str(exc)

    if st.session_state.faq_error:
        st.error(st.session_state.faq_error)
    if st.session_state.faq_output:
        st.markdown(st.session_state.faq_output)

with qa_tab:
    question = st.text_area(
        "Pregunta del usuario",
        placeholder="Ejemplo: Cuales son los productos principales de la empresa?",
        height=120,
    )
    if st.button("Responder pregunta", use_container_width=True):
        if not question.strip():
            st.session_state.qa_error = "Escribe una pregunta antes de ejecutar."
        else:
            try:
                model = create_chat_model(
                    provider=provider, model_name=model_name, temperature=temperature
                )
                result = answer_question(
                    model=model,
                    company_name=config.company.company_name,
                    question=question,
                    knowledge_text=knowledge_text,
                    chunks=chunks,
                    max_context_chars=max_context_chars,
                    vector_index_path=config.paths.vector_index_path,
                )
                st.session_state.qa_output = result.answer
                st.session_state.qa_context_mode = result.context_mode
                st.session_state.qa_sources = result.sources
                st.session_state.qa_error = ""
            except Exception as exc:
                st.session_state.qa_error = str(exc)

    if st.session_state.qa_error:
        st.error(st.session_state.qa_error)
    if st.session_state.qa_output:
        st.markdown(st.session_state.qa_output)
        st.caption(f"Modo de contexto: {st.session_state.qa_context_mode}")
        if st.session_state.qa_sources:
            st.write("Fuentes sugeridas para esta respuesta:")
            for source in st.session_state.qa_sources:
                st.write(f"- {source['title']} - {source['url']}")

with config_tab:
    st.subheader("Fuentes y prompts")
    st.write(
        "Los cambios guardados aqui modifican archivos locales. Al recargar, la app detecta el cambio "
        "y reconstruye scraping, base de conocimiento, chunks e indice."
    )

    profile_payload = _read_json(config.paths.company_config_path)
    prompt_payload = load_prompt_config()

    st.markdown("#### Parametros de ejecucion")
    st.caption(
        "El proveedor, modelo y la temperatura ahora se controlan desde el sidebar izquierdo "
        "(seccion 'Modelo LLM activo'). Aqui solo configuras el limite de contexto y las fuentes."
    )
    st.number_input(
        "Maximo de caracteres de contexto",
        min_value=2000,
        max_value=200000,
        step=1000,
        key="max_context_chars_param",
    )
    with st.expander("Configuracion tecnica", expanded=False):
        st.write(f"Proveedor activo: `{provider}`")
        st.write(f"Modelo activo: `{model_name}`")
        st.write(f"Proveedor inicial (.env): `{config.runtime.provider}`")
        st.write(f"Modelo inicial (.env): `{config.runtime.model_name}`")
        st.caption(
            "Usa el sidebar para cambiar proveedor / modelo en caliente. "
            "Las API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`) se siguen leyendo desde `.env`."
        )

    with st.form("company_profile_form"):
        st.markdown("#### Empresa y fuentes")
        st.caption("Cuando guardas esta seccion se actualiza `data/config/company_profile.json`.")
        company_name_input = st.text_input(
            "Nombre de la empresa",
            value=str(profile_payload.get("company_name", "")),
        )
        company_description_input = st.text_area(
            "Descripcion de la empresa",
            value=str(profile_payload.get("company_description", "")),
            height=110,
        )
        seed_urls_input = st.text_area(
            "URLs semilla",
            value=_join_lines(config.company.seed_urls),
            height=90,
            help="Una URL por linea. Normalmente incluye la pagina principal.",
        )
        additional_sources_input = st.text_area(
            "URLs adicionales para scraping",
            value=_join_lines(config.company.additional_sources),
            height=220,
            help="Agrega o borra URLs. Cada cambio fuerza reprocesamiento.",
        )
        document_sources_input = st.text_area(
            "Documentos PDF u otras fuentes",
            value=_join_lines(config.company.document_sources),
            height=90,
        )
        allowed_domains_input = st.text_area(
            "Dominios permitidos",
            value=_join_lines(config.company.allowed_domains),
            height=90,
            help="La app tambien agregara automaticamente los dominios derivados de las URLs.",
        )
        priority_topics_input = st.text_area(
            "Temas prioritarios",
            value=_join_lines(config.company.priority_topics),
            height=100,
        )
        excluded_keywords_input = st.text_area(
            "Palabras excluidas",
            value=_join_lines(config.company.excluded_keywords),
            height=90,
        )
        save_profile = st.form_submit_button("Guardar fuentes y reprocesar", use_container_width=True)

    if save_profile:
        seed_urls = _split_lines(seed_urls_input)
        additional_sources = _split_lines(additional_sources_input)
        document_sources = _split_lines(document_sources_input)
        allowed_domains = sorted(
            set(_split_lines(allowed_domains_input))
            | set(_derive_domains(seed_urls + additional_sources + document_sources))
        )
        updated_profile = {
            **profile_payload,
            "company_name": company_name_input.strip(),
            "company_description": company_description_input.strip(),
            "seed_urls": seed_urls,
            "additional_sources": additional_sources,
            "document_sources": document_sources,
            "allowed_domains": allowed_domains,
            "priority_topics": _split_lines(priority_topics_input),
            "excluded_keywords": _split_lines(excluded_keywords_input),
        }
        _save_company_profile(config, updated_profile)
        _reset_generated_outputs()
        st.session_state.config_save_notice = (
            "Fuentes guardadas. Recargando para ejecutar el reprocesamiento."
        )
        st.rerun()

    with st.form("prompt_config_form"):
        st.markdown("#### Prompts editables")
        st.caption(
            "Cuando guardas esta seccion se actualiza `data/config/prompts_config.json`. "
            "Mantén las variables entre llaves como `{company_name}`, `{context}`, `{question}` y `{faq_count}` "
            "cuando correspondan; LangChain las necesita para inyectar datos."
        )
        edited_prompts: dict[str, str] = {}
        prompt_labels = {
            "summary_system": "Resumen - system",
            "summary_human": "Resumen - human",
            "faq_system": "FAQ - system",
            "faq_human": "FAQ - human",
            "qa_system": "Q&A - system",
            "qa_human": "Q&A - human",
            "agent_router_system": "Agente router - system",
            "agent_router_human": "Agente router - human",
        }
        for key, label in prompt_labels.items():
            edited_prompts[key] = st.text_area(
                label,
                value=prompt_payload.get(key, DEFAULT_PROMPTS[key]),
                height=180 if key.endswith("_system") else 220,
            )
        save_prompts = st.form_submit_button("Guardar prompts y reprocesar", use_container_width=True)

    if save_prompts:
        save_prompt_config(edited_prompts)
        _reset_generated_outputs()
        st.session_state.config_save_notice = (
            "Prompts guardados. Recargando para ejecutar el reprocesamiento."
        )
        st.rerun()

source_index_path = Path(config.paths.source_index_path)
if source_index_path.exists():
    with st.expander("Indice de fuentes"):
        st.markdown(source_index_path.read_text(encoding="utf-8"))
