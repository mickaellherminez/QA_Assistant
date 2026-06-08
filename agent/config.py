"""
config.py — Chargement centralisé de la configuration.

Toutes les variables d'environnement sont lues ici.
Aucun autre module ne doit appeler os.getenv directement.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Chargement du fichier .env situé dans le même répertoire
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

_BASE_DIR = Path(__file__).resolve().parent


def _require(key: str) -> str:
    """Lève une erreur explicite si une variable obligatoire est absente."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Variable d'environnement manquante : '{key}'. "
            f"Vérifiez votre fichier .env (voir .env.example)."
        )
    return value


def _resolve_path(value: str) -> str:
    """
    Résout un chemin de manière stable, indépendamment du répertoire courant.

    - Chemin absolu : conservé tel quel.
    - Chemin relatif : résolu par rapport au dossier agent/.
    """
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((_BASE_DIR / path).resolve())


# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = _require("OPENAI_API_KEY")
MODEL: str = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── RAG — documents ISTQB ────────────────────────────────────────────────────
ISTQB_DOCS_DIR: str = _resolve_path(os.getenv("ISTQB_DOCS_DIR", "../pdf"))
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "istqb_best_practices")

# ── User Stories API ──────────────────────────────────────────────────────────
US_API_ENDPOINT: str = os.getenv(
    "US_API_ENDPOINT",
    (
        "https://raw.githubusercontent.com/"
        "mickaellherminez/github-user-stories-fake-api/main/data/user-stories.json"
    ),
)

# ── RAG / ChromaDB ────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = _resolve_path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"))
RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.10"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
RAG_RERANKER_ENABLED: bool = os.getenv("RAG_RERANKER_ENABLED", "true").lower() == "true"

# ── Langfuse (observabilité LLM) — optionnel ─────────────────────────────────
# Laisser vide pour désactiver le tracing.
LANGFUSE_PUBLIC_KEY: str | None = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: str | None = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
