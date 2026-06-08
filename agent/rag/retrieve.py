"""
rag/retrieve.py — Recherche sémantique dans la base ISTQB.

Fournit :
  - retrieve(query)         : retourne les chunks ISTQB pertinents pour une query.
  - build_rag_context(docs) : formate les chunks en texte injectable dans un prompt.

Pipeline :
  1. ChromaDB bi-encoder  → top-K chunks par similarité cosinus
  2. Filtrage par seuil   → RAG_SCORE_THRESHOLD (défaut 0.10)
  3. Cross-encoder rerank → réordonnancement précis (query ↔ passage)

Langfuse :
  retrieve() est décorée avec @observe(name="rag-retrieve") pour mesurer
  la latence du pipeline RAG et tracer le nombre de chunks retenus.
"""

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from config import RAG_SCORE_THRESHOLD, RAG_TOP_K, RAG_RERANKER_ENABLED
from rag.vector_store import get_or_create_store, is_indexed
from tracing import observe

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Chunk ISTQB récupéré avec son score de similarité."""
    content: str
    score: float
    source: str
    page: int
    language: str


@observe(name="rag-retrieve")
def retrieve(query: str) -> list[RetrievedChunk]:
    """
    Recherche les chunks ISTQB les plus pertinents pour une requête.

    Args:
        query: texte de recherche (typiquement : description US + acceptance criteria)

    Returns:
        Liste de RetrievedChunk triés par score décroissant, filtrés par seuil.
        Liste vide si la base n'est pas indexée ou si aucun résultat pertinent.
    """
    if not is_indexed():
        logger.warning(
            "Base vectorielle vide — lancer scripts/ingest_docs.py pour indexer les PDFs ISTQB."
        )
        return []

    store = get_or_create_store()

    try:
        results: list[tuple[Document, float]] = (
            store.similarity_search_with_relevance_scores(query, k=RAG_TOP_K)
        )
    except Exception as e:
        logger.error("Erreur lors du retrieval ChromaDB : %s", e)
        return []

    chunks: list[RetrievedChunk] = []
    for doc, score in results:
        if score < RAG_SCORE_THRESHOLD:
            logger.debug(
                "Chunk ignoré (score %.2f < seuil %.2f) : %s…",
                score, RAG_SCORE_THRESHOLD, doc.page_content[:60],
            )
            continue

        chunks.append(RetrievedChunk(
            content=doc.page_content,
            score=round(score, 3),
            source=doc.metadata.get("source", "ISTQB"),
            page=doc.metadata.get("page", 0),
            language=doc.metadata.get("language", "?"),
        ))

    logger.info(
        "RAG — query: '%.60s…' → %d/%d chunks retenus (seuil=%.2f)",
        query, len(chunks), len(results), RAG_SCORE_THRESHOLD,
    )

    # Reranking cross-encoder (optionnel, activé via RAG_RERANKER_ENABLED)
    if RAG_RERANKER_ENABLED and len(chunks) > 1:
        from rag.reranker import rerank
        chunks = rerank(query, chunks)

    return chunks


def build_rag_context(chunks: list[RetrievedChunk]) -> str:
    """
    Formate les chunks récupérés en texte structuré pour injection dans le prompt.

    Args:
        chunks: liste de RetrievedChunk issus de retrieve()

    Returns:
        Texte formaté avec source, page et contenu de chaque chunk.
        Chaîne vide si aucun chunk.
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source ISTQB {i} — {chunk.source}, p.{chunk.page} "
            f"(score: {chunk.score:.2f})]\n{chunk.content}"
        )

    return "\n\n---\n\n".join(parts)
