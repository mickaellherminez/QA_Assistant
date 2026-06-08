"""
main.py — Orchestration principale de l'agent QA (pattern ReAct).

Cycle d'exécution :
  1. Valider et assainir l'input (input_guard).
  2. Stocker la requête en mémoire courte.
  3. Classifier l'intention.
  4. Rappeler le contexte de conversation.
  5. Si intent QA → récupérer les user stories depuis l'endpoint.
  6. Construire les messages et appeler le LLM (JSON).
  7. Valider la sortie JSON.
  8. Filtrer les données sensibles (output_filter).
  9. Stocker la réponse en mémoire.
  10. Retourner la réponse.

Langfuse :
  Chaque étape clé est encapsulée dans un span @observe().
  La fonction agent() est la trace racine ; session_id et tags (intent, status)
  sont propagés à tous les spans enfants via propagate_attributes().
"""

import logging
import os
import re

# ⚠️  Importer tracing avant les modules LLM pour que le contexte soit prêt.
from tracing import observe, propagate_attributes

from llm import call_llm_json
from memory.short_term import store, recall, clear  # noqa: F401 (clear ré-exporté)
from tools.user_stories import fetch_all, fetch_by_index, parse_us_list
from tools.validator import validate_output
from security.input_guard import validate_input
from security.output_filter import filter_response
from rag.retrieve import retrieve, build_rag_context

logger = logging.getLogger(__name__)

# ── Chargement des prompts ────────────────────────────────────────────────────
_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    path = os.path.join(_PROMPTS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


_SYSTEM_PROMPT = _load_prompt("system.md")
_CLASSIFY_PROMPT = _load_prompt("classify_intent.md")
_GENERATE_PROMPT = _load_prompt("generate_tests.md")

# ── Regex pour détecter un index US dans la requête (ex: US-006) ─────────────
_US_INDEX_PATTERN = re.compile(r"\bUS-\d{3,}\b", re.IGNORECASE)


# ── Fonctions internes ────────────────────────────────────────────────────────

@observe(name="classify-intent")
def _classify_intent(query: str) -> dict:
    """
    Identifie l'intention de la requête utilisateur.

    Returns:
        dict avec au moins {"intent": str, "confidence": float, "reason": str}
    """
    messages = [
        {"role": "system", "content": _CLASSIFY_PROMPT},
        {"role": "user",   "content": query},
    ]
    schema = '{"intent":"generate_tests|analyze_story|detect_ambiguities|general|out_of_scope","confidence":0.95,"reason":"..."}'
    try:
        return call_llm_json(messages, schema_hint=schema)
    except RuntimeError as e:
        logger.warning("Échec classification, fallback general : %s", e)
        return {"intent": "general", "confidence": 0.0, "reason": "Fallback après erreur"}


def _build_messages(
    query: str,
    context: list[dict],
    us_list: list[dict],
    intent: str,
    rag_context: str = "",
) -> list[dict]:
    """
    Construit la liste de messages à envoyer au LLM.

    Args:
        query:       requête utilisateur
        context:     historique de conversation (depuis la mémoire courte)
        us_list:     liste de user stories récupérées depuis l'endpoint
        intent:      intention classifiée
        rag_context: bonnes pratiques ISTQB récupérées via ChromaDB (optionnel)

    Returns:
        Liste de dicts {role, content} prête pour call_llm_json.
    """
    messages = []

    # 1. Prompt système principal
    messages.append({"role": "system", "content": _SYSTEM_PROMPT})

    # 2. Contexte de conversation
    if context:
        ctx_text = "\n".join(
            f"[{m['role'].upper()}] {m['content']}" for m in context
        )
        messages.append({
            "role": "system",
            "content": f"Historique de la conversation :\n{ctx_text}",
        })

    # 3. Bonnes pratiques ISTQB (RAG)
    if rag_context:
        messages.append({
            "role": "system",
            "content": (
                "Bonnes pratiques ISTQB pertinentes pour cette user story "
                "(issues de la base documentaire) :\n\n"
                + rag_context
            ),
        })

    # 4. Données User Stories
    if us_list:
        us_text = parse_us_list(us_list)
        messages.append({
            "role": "system",
            "content": f"User stories à analyser :\n\n{us_text}",
        })
    else:
        messages.append({
            "role": "system",
            "content": "Aucune user story disponible depuis l'endpoint.",
        })

    # 5. Prompt de génération si intent QA
    if intent in ("generate_tests", "analyze_story", "detect_ambiguities"):
        messages.append({"role": "system", "content": _GENERATE_PROMPT})

    # 6. Requête utilisateur
    messages.append({"role": "user", "content": query})

    return messages


@observe(name="fetch-user-stories")
def _fetch_relevant_us(query: str, intent: str) -> list[dict]:
    """
    Récupère les user stories pertinentes selon la requête et l'intent.

    - Si un index US est détecté dans la query (ex: US-006), récupère uniquement celle-là.
    - Sinon, récupère toutes les US.
    - Retourne liste vide si intent non QA.
    """
    if intent not in ("generate_tests", "analyze_story", "detect_ambiguities"):
        return []

    # Détecter un index spécifique dans la requête
    match = _US_INDEX_PATTERN.search(query)
    if match:
        index = match.group(0).upper()
        logger.info("Index US détecté dans la requête : %s", index)
        us = fetch_by_index(index)
        return [us] if us else []

    # Sinon, récupérer toutes les US
    return fetch_all()


def _build_rag_query(us_list: list[dict]) -> str:
    """
    Construit la query de recherche RAG à partir des user stories chargées.

    On utilise le contenu sémantique des US (description + critères d'acceptation)
    plutôt que la requête brute de l'utilisateur : cela donne des résultats
    ISTQB beaucoup plus pertinents (ex : BVA pour un champ password, EP pour un email).

    Args:
        us_list: liste de dicts US récupérés depuis l'endpoint.

    Returns:
        Texte concaténé à envoyer à ChromaDB pour le retrieval.
    """
    parts = []
    for us in us_list[:3]:  # limiter à 3 US pour ne pas surcharger la query
        parts.append(us.get("description", ""))
        parts.extend(us.get("acceptanceCriteria", []))
        parts.extend(us.get("constraints", []))
    return " ".join(p for p in parts if p).strip()


# ── Fonction principale ───────────────────────────────────────────────────────

@observe(name="qa-agent")
def agent(query: str, session_id: str | None = None) -> dict:
    """
    Point d'entrée principal de l'agent QA.

    Args:
        query:      requête de l'utilisateur (texte libre)
        session_id: identifiant de session optionnel (propagé à Langfuse)

    Returns:
        dict JSON structuré conforme au schéma de l'agent.
    """
    logger.info("Requête reçue : %s", query[:100])

    # 1. Valider et assainir l'input
    ok, result = validate_input(query)
    if not ok:
        logger.warning("Input rejeté par input_guard : %s", result)
        return {
            "status": "error",
            "intent": "unknown",
            "answer": result,
            "test_cases": [],
            "ambiguities": [],
            "sources": [],
            "warnings": ["Input invalide ou non autorisé."],
            "requires_human_validation": False,
        }
    query = result  # version nettoyée (tronquée si besoin)

    # 2. Stocker la requête
    store({"role": "user", "content": query})

    # 3. Classifier l'intention (span Langfuse enfant)
    intent_result = _classify_intent(query)
    intent = intent_result.get("intent", "general")
    logger.info("Intent : %s (confiance : %s)", intent, intent_result.get("confidence"))

    # Attacher session_id, intent et tags à la trace Langfuse courante
    with propagate_attributes(
        trace_name="qa-agent",
        session_id=session_id,
        tags=[f"intent:{intent}", "qa-assistant"],
        metadata={"intent": intent, "confidence": intent_result.get("confidence")},
    ):
        # 4. Rappeler le contexte (hors message courant)
        context = recall(limit=5)[:-1]  # on exclut le message qu'on vient de stocker

        # 5. Récupérer les user stories pertinentes (span Langfuse enfant)
        us_list = _fetch_relevant_us(query, intent)
        logger.info("US chargées : %d", len(us_list))

        # 6. RAG — récupérer les bonnes pratiques ISTQB pertinentes
        # (span Langfuse enfant défini dans rag/retrieve.py)
        rag_context = ""
        if us_list and intent in ("generate_tests", "analyze_story", "detect_ambiguities"):
            rag_query = _build_rag_query(us_list)
            chunks = retrieve(rag_query)
            rag_context = build_rag_context(chunks)
            logger.info("RAG — %d chunks ISTQB injectés", len(chunks))

        # 7. Construire les messages et appeler le LLM
        # (CallbackHandler Langfuse attaché automatiquement dans call_llm)
        messages = _build_messages(query, context, us_list, intent, rag_context)

        schema = (
            '{"status":"success|error|clarification_needed|out_of_scope",'
            '"intent":"generate_tests|analyze_story|detect_ambiguities|general|out_of_scope",'
            '"answer":"...","test_cases":[],"ambiguities":[],'
            '"sources":[],"warnings":[],"requires_human_validation":true}'
        )

        try:
            response = call_llm_json(messages, schema_hint=schema)
        except RuntimeError as e:
            logger.error("Erreur LLM : %s", e)
            error_response = {
                "status": "error",
                "intent": intent,
                "answer": "Une erreur est survenue lors de la génération. Veuillez réessayer.",
                "test_cases": [],
                "ambiguities": [],
                "sources": [],
                "warnings": [str(e)],
                "requires_human_validation": True,
            }
            store({"role": "assistant", "content": error_response["answer"]})
            return error_response

    # 8. Valider la sortie
    try:
        validate_output(response)
    except ValueError as e:
        logger.warning("Validation sortie échouée : %s", e)
        response["warnings"] = response.get("warnings", []) + [f"Validation: {e}"]

    # 9. Filtrer les données sensibles
    response = filter_response(response)

    # 10. Stocker la réponse
    store({"role": "assistant", "content": response.get("answer", "")})

    logger.info(
        "Réponse générée — status=%s, test_cases=%d",
        response.get("status"),
        len(response.get("test_cases", [])),
    )

    return response
