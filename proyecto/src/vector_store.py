"""Vector store del Taller 2.

Implementacion basada en LangChain:
- Embeddings: HuggingFaceEmbeddings con sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dims).
- Vector store: ChromaDB persistido en disco (data/vector/chroma).

La API publica (build_vector_index, search_vector_index, load_vector_index)
se conserva intencionalmente para no romper el resto del pipeline (chains.py,
app.py, scripts/run_agent_batch.py).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


COLLECTION_NAME = "qbano_kb"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSIONS = 384


_embeddings_singleton: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Devuelve el modelo de embeddings reutilizable.

    Se cachea en memoria para evitar recargar el modelo (~480 MB) en cada llamada.
    """

    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_singleton


def _resolve_persist_dir(output_path: Path) -> Path:
    """Convierte la ruta heredada (vector_index.json) en el directorio Chroma.

    En el Taller 1 la persistencia era un solo JSON; aqui usamos un directorio
    porque Chroma persiste sqlite + binarios. Mantenemos la convencion de pasar
    la misma ruta para no romper el resto del pipeline.
    """

    path = Path(output_path)
    if path.suffix == ".json":
        return path.with_name("chroma")
    return path


def _doc_metadata(chunk: dict[str, Any]) -> dict[str, str | int]:
    return {
        "id": str(chunk.get("id", "")),
        "title": str(chunk.get("title", "")),
        "url": str(chunk.get("url", "")),
        "char_count": int(chunk.get("char_count", 0) or 0),
    }


def build_vector_index(
    chunks: list[dict[str, str | int]],
    output_path: Path,
) -> dict[str, int]:
    """Construye (o reconstruye) el indice Chroma con los chunks procesados.

    - Borra el directorio Chroma previo para garantizar idempotencia.
    - Embebe todos los chunks de una sola vez con sentence-transformers.
    - Persiste en disco en `<output_path>/chroma`.
    - Tambien deja un snapshot legible en `<output_path>.json` (resumen, no contiene los vectores).
    """

    persist_dir = _resolve_persist_dir(output_path)
    persist_dir.parent.mkdir(parents=True, exist_ok=True)

    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    for chunk in chunks:
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata=_doc_metadata(chunk),
            )
        )

    if not documents:
        # Aseguramos al menos un placeholder para que Chroma cree la coleccion vacia.
        return {"vectors": 0, "dimensions": EMBEDDING_DIMENSIONS}

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )

    try:
        total = vectorstore._collection.count()
    except Exception:
        total = len(documents)

    snapshot_path = Path(output_path)
    if snapshot_path.suffix == ".json":
        snapshot_payload = {
            "type": "chromadb",
            "embedding_provider": "langchain_huggingface.HuggingFaceEmbeddings",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "collection_name": COLLECTION_NAME,
            "dimensions": EMBEDDING_DIMENSIONS,
            "vectors": total,
            "persist_directory": str(persist_dir.resolve()),
        }
        snapshot_path.write_text(
            json.dumps(snapshot_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {"vectors": total, "dimensions": EMBEDDING_DIMENSIONS}


def load_vector_index(path: Path) -> dict[str, Any]:
    """Devuelve un resumen del indice persistido.

    Se mantiene por compatibilidad con app.py, que muestra el numero de vectores
    en el header. No expone los embeddings sin procesar.
    """

    persist_dir = _resolve_persist_dir(path)
    if not persist_dir.exists():
        return {"records": [], "vectors": 0}

    snapshot_path = Path(path)
    snapshot: dict[str, Any] = {}
    if snapshot_path.suffix == ".json" and snapshot_path.exists():
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            snapshot = {}

    try:
        embeddings = get_embeddings()
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )
        total = vectorstore._collection.count()
    except Exception:
        total = int(snapshot.get("vectors", 0) or 0)

    # El callsite de app.py usa len(payload.get("records", [])) como contador.
    # Producimos una lista de marcadores para que ese contador funcione sin
    # cargar los embeddings en memoria.
    return {
        "type": "chromadb",
        "embedding_provider": snapshot.get(
            "embedding_provider", "langchain_huggingface.HuggingFaceEmbeddings"
        ),
        "embedding_model": snapshot.get("embedding_model", EMBEDDING_MODEL_NAME),
        "dimensions": snapshot.get("dimensions", EMBEDDING_DIMENSIONS),
        "vectors": total,
        "records": [{"_": i} for i in range(total)],
    }


def _open_vectorstore(persist_dir: Path) -> Chroma | None:
    if not persist_dir.exists():
        return None
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def search_vector_index(
    query: str,
    path: Path,
    top_k: int = 6,
) -> list[dict[str, str | int | float]]:
    """Recupera los chunks mas similares a la pregunta.

    Devuelve la misma estructura que la implementacion anterior
    (id, title, url, content, char_count, score) para no romper a chains.py.
    """

    persist_dir = _resolve_persist_dir(path)
    vectorstore = _open_vectorstore(persist_dir)
    if vectorstore is None:
        return []

    try:
        matches = vectorstore.similarity_search_with_score(query, k=top_k)
    except Exception:
        return []

    results: list[dict[str, str | int | float]] = []
    for document, distance in matches:
        meta = document.metadata or {}
        # Chroma devuelve distancia (mientras menor, mas similar). Convertimos
        # a un puntaje aproximado de similitud para que la convencion siga siendo
        # "score mas alto = mas relevante".
        score = round(max(0.0, 1.0 - float(distance)), 4)
        results.append(
            {
                "id": str(meta.get("id", "")),
                "title": str(meta.get("title", "")),
                "url": str(meta.get("url", "")),
                "content": document.page_content,
                "char_count": int(meta.get("char_count", 0) or 0),
                "score": score,
            }
        )
    return results
