"""
tracing.py — Initialisation centralisée de Langfuse (observabilité LLM).

Exporte :
  - observe           : décorateur de span (no-op si Langfuse désactivé)
  - propagate_attributes : gestionnaire de contexte pour attacher session_id / tags
  - get_langfuse_handler : factory CallbackHandler LangChain liée au trace courant
  - flush             : vide la file d'envoi Langfuse (à appeler à l'arrêt)

Activation :
  Définir LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY dans le .env.
  Si ces variables sont absentes, le module se désactive silencieusement :
  l'application fonctionne normalement, sans tracing.

Ordre d'import (important) :
  Importer ce module AVANT d'importer les clients LLM (OpenAI, LangChain),
  afin que le contexte de trace soit établi avant toute invocation.
"""

import functools
import logging
import os

logger = logging.getLogger(__name__)

# ── Détection de la configuration ─────────────────────────────────────────────

LANGFUSE_ENABLED: bool = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)

# ── Initialisation conditionnelle ─────────────────────────────────────────────

if LANGFUSE_ENABLED:
    try:
        from langfuse import get_client, observe, propagate_attributes
        from langfuse.langchain import CallbackHandler as _CallbackHandler

        _client = get_client()
        logger.info(
            "✅ Langfuse tracing activé — host=%s",
            os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        def get_langfuse_handler() -> "_CallbackHandler | None":
            """
            Retourne un CallbackHandler LangChain lié au trace Langfuse courant.

            À appeler à l'intérieur d'une fonction décorée avec @observe() pour
            que le callback soit automatiquement rattaché à la bonne trace.
            Retourne None si Langfuse n'est pas activé.
            """
            return _CallbackHandler()

        def flush() -> None:
            """Envoie tous les événements en attente à Langfuse (shutdown propre)."""
            try:
                _client.flush()
                logger.debug("Langfuse — flush OK.")
            except Exception as exc:  # pragma: no cover
                logger.warning("Langfuse flush échoué : %s", exc)

    except Exception as _init_exc:
        LANGFUSE_ENABLED = False
        logger.warning(
            "⚠️  Langfuse init échoué, tracing désactivé : %s", _init_exc
        )

# ── Stubs no-op (Langfuse absent ou désactivé) ────────────────────────────────

if not LANGFUSE_ENABLED:

    def observe(*d_args, **d_kwargs):  # type: ignore[misc]
        """
        Décorateur @observe() no-op quand Langfuse n'est pas configuré.

        Supporte les deux formes :
          @observe            (sans parenthèses)
          @observe()          (avec parenthèses, avec ou sans kwargs)
          @observe(name="…")
        """

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapper

        # Appelé sans parenthèses : @observe
        if d_args and callable(d_args[0]):
            return d_args[0]
        # Appelé avec parenthèses : @observe() ou @observe(name="…")
        return decorator

    class propagate_attributes:  # type: ignore[no-redef]
        """Gestionnaire de contexte no-op quand Langfuse n'est pas configuré."""

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def get_langfuse_handler():  # type: ignore[misc]
        """Retourne None quand Langfuse n'est pas configuré."""
        return None

    def flush() -> None:  # type: ignore[misc]
        """No-op quand Langfuse n'est pas configuré."""
        pass

    logger.info("ℹ️  Langfuse tracing désactivé (LANGFUSE_PUBLIC_KEY absent).")
