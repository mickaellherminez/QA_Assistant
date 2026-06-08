"""
api.py — Application FastAPI du QA Assistant.

Endpoints :
  POST /ask     → soumet une question à l'agent QA
  GET  /health  → vérifie l'état de l'API et du RAG
  GET  /metrics → statistiques d'utilisation depuis le démarrage
  GET  /user-stories → liste des US consommées par l'agent
  GET  /ragas   → scores RAGAS depuis les rapports JSON locaux

Documentation interactive : http://localhost:8000/docs
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api_models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetricsResponse,
    RagasReportResponse,
    RagasRunRequest,
    RagasRunResponse,
    UserStory,
)
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, ISTQB_DOCS_DIR, MODEL
from main import agent
from rag.vector_store import is_indexed
from tools.user_stories import fetch_all, fetch_by_index
from tracing import flush as langfuse_flush

logger = logging.getLogger(__name__)

# ── Compteurs in-memory (reset à chaque redémarrage) ─────────────────────────
_metrics: dict = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "response_times_ms": [],   # liste des derniers temps de réponse (max 1000)
    "test_cases_generated": 0,
}

_MAX_RESPONSE_TIMES = 1000  # évite une croissance illimitée

_REPORTS_DIR = Path(__file__).resolve().parent
_RAGAS_REPORT_FILES = {
    "reranker": "rapport_reranker.json",
    "baseline": "rapport.json",
    "v2": "rapport_v2.json",
}


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Tâches au démarrage / à l'arrêt de l'application.
    Vérifie que la base vectorielle est accessible.
    """
    logger.info("🚀 Démarrage du QA Assistant API")
    if not is_indexed():
        logger.warning(
            "⚠️  La base vectorielle ChromaDB n'est pas peuplée. "
            "Lancez : uv run python scripts/ingest_docs.py"
        )
    else:
        logger.info("✅ ChromaDB opérationnelle.")
    yield
    logger.info("🛑 Arrêt du QA Assistant API")
    langfuse_flush()  # envoyer les traces en attente avant la fermeture


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="QA Assistant API",
    description=(
        "Agent IA spécialisé en assurance qualité logicielle.\n\n"
        "Génère des cas de test fonctionnels à partir de user stories, "
        "enrichis par les bonnes pratiques ISTQB (CTFL v4.0.1 + CTAL-TA v4.0 FR)."
    ),
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — autorise toutes les origines en développement
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Gestionnaire d'erreurs global ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Erreur non gérée sur %s : %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erreur interne du serveur. Consultez les logs pour plus de détails.",
            "type": type(exc).__name__,
        },
    )


# ── POST /ask ─────────────────────────────────────────────────────────────────

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Soumettre une question à l'agent QA",
    description=(
        "Analyse la question, détecte l'intention, récupère les user stories, "
        "enrichit avec les bonnes pratiques ISTQB et génère des cas de test."
    ),
    tags=["Agent"],
)
async def ask(body: AskRequest) -> AskResponse:
    """
    Point d'entrée principal de l'agent QA.

    - **question** : question ou instruction (max 5000 chars)
    - **session_id** : optionnel, pour tracer les échanges
    """
    start_time = time.perf_counter()
    _metrics["requests_total"] += 1

    logger.info(
        "POST /ask — session_id=%s, question='%s...'",
        body.session_id or "N/A",
        body.question[:80],
    )

    try:
        result = agent(body.question, session_id=body.session_id)
    except Exception as e:
        _metrics["requests_error"] += 1
        logger.error("Erreur dans agent() : %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la requête : {type(e).__name__}",
        ) from e

    # Mise à jour des métriques
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    _metrics["response_times_ms"].append(elapsed_ms)
    if len(_metrics["response_times_ms"]) > _MAX_RESPONSE_TIMES:
        _metrics["response_times_ms"] = _metrics["response_times_ms"][-_MAX_RESPONSE_TIMES:]

    if result.get("status") == "success":
        _metrics["requests_success"] += 1
    else:
        _metrics["requests_error"] += 1

    tc_count = len(result.get("test_cases", []))
    _metrics["test_cases_generated"] += tc_count

    logger.info(
        "POST /ask terminé — status=%s, test_cases=%d, durée=%.0fms",
        result.get("status"),
        tc_count,
        elapsed_ms,
    )

    return AskResponse(**result)


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier l'état de santé de l'API",
    description="Retourne l'état de l'API, du modèle LLM et de la base vectorielle ChromaDB.",
    tags=["Monitoring"],
)
async def health() -> HealthResponse:
    """Vérification de l'état de santé."""
    try:
        indexed = is_indexed()
    except Exception as e:
        logger.warning("Erreur lors de la vérification ChromaDB : %s", e)
        indexed = False

    return HealthResponse(
        status="ok",
        rag_indexed=indexed,
        model=MODEL,
        version="0.5.0",
    )


# ── GET /metrics ──────────────────────────────────────────────────────────────

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Statistiques d'utilisation de l'API",
    description="Compteurs et temps de réponse depuis le dernier démarrage du serveur.",
    tags=["Monitoring"],
)
async def metrics() -> MetricsResponse:
    """Métriques d'utilisation in-memory."""
    times = _metrics["response_times_ms"]
    avg_ms = sum(times) / len(times) if times else 0.0

    return MetricsResponse(
        requests_total=_metrics["requests_total"],
        requests_success=_metrics["requests_success"],
        requests_error=_metrics["requests_error"],
        avg_response_time_ms=round(avg_ms, 1),
        test_cases_generated=_metrics["test_cases_generated"],
    )


# ── GET /user-stories ─────────────────────────────────────────────────────────

@app.get(
    "/user-stories",
    response_model=list[UserStory],
    summary="Lister les user stories consommées par l'agent",
    description=(
        "Retourne les US depuis la source configurée via US_API_ENDPOINT "
        "(repo GitHub fake API par défaut)."
    ),
    tags=["User Stories"],
)
async def user_stories() -> list[UserStory]:
    us_list = fetch_all()
    return [UserStory(**us) for us in us_list]


@app.get(
    "/user-stories/{index}",
    response_model=UserStory,
    summary="Récupérer une user story par index",
    tags=["User Stories"],
)
async def user_story_by_index(index: str) -> UserStory:
    us = fetch_by_index(index)
    if us is None:
        raise HTTPException(
            status_code=404,
            detail=f"User story non trouvée pour l'index '{index}'.",
        )
    return UserStory(**us)


# ── GET /ragas ────────────────────────────────────────────────────────────────

def _available_ragas_reports() -> dict[str, Path]:
    return {
        name: _REPORTS_DIR / filename
        for name, filename in _RAGAS_REPORT_FILES.items()
        if (_REPORTS_DIR / filename).exists()
    }


def _report_path_for_name(report_name: str) -> Path:
    if report_name not in _RAGAS_REPORT_FILES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Rapport '{report_name}' invalide. "
                f"Valeurs autorisées: {', '.join(_RAGAS_REPORT_FILES.keys())}."
            ),
        )
    return _REPORTS_DIR / _RAGAS_REPORT_FILES[report_name]


def _run_ragas_once(
    model: str | None,
    embedding_model: str | None,
    max_samples: int | None,
) -> dict:
    from scripts.evaluate_rag import build_dataset, print_report, run_evaluation

    # Auto-réindexe si la base est vide (cas fréquent après lancement depuis un autre cwd)
    if not is_indexed():
        logger.warning(
            "RAG index vide avant run RAGAS. Tentative d'ingestion automatique depuis %s",
            ISTQB_DOCS_DIR,
        )
        from rag.chunking import chunk_pages
        from rag.ingest import load_all_pdfs
        from rag.vector_store import index_chunks

        pages = load_all_pdfs(ISTQB_DOCS_DIR)
        if pages:
            chunks = chunk_pages(pages)
            if chunks:
                index_chunks(chunks)
                logger.info("Ingestion automatique terminée (%d chunks).", len(chunks))

    samples = build_dataset(
        max_samples=max_samples,
        candidate_model=model,
    )
    if not samples:
        raise ValueError(
            (
                "Aucun sample construit. Vérifiez l'index RAG et les chemins : "
                f"CHROMA_PERSIST_DIR='{CHROMA_PERSIST_DIR}', "
                f"ISTQB_DOCS_DIR='{ISTQB_DOCS_DIR}'. "
                "Relancez scripts/ingest_docs.py si nécessaire."
            )
        )

    result = run_evaluation(
        samples=samples,
        judge_model=model,
        embedding_model=embedding_model,
    )

    report = print_report(result, samples, model_name=model or MODEL)
    report["embedding_model"] = embedding_model or EMBEDDING_MODEL
    return report


@app.get(
    "/ragas",
    response_model=RagasReportResponse,
    summary="Lire un rapport RAGAS local",
    description="Expose les scores RAGAS JSON générés par scripts/evaluate_rag.py.",
    tags=["Monitoring"],
)
async def ragas_report(
    report: str | None = Query(
        default=None,
        description="Nom logique du rapport (reranker | baseline | v2).",
    ),
) -> RagasReportResponse:
    available = _available_ragas_reports()
    if not available:
        raise HTTPException(
            status_code=404,
            detail="Aucun rapport RAGAS trouvé dans le dossier agent/.",
        )

    selected = report or ("reranker" if "reranker" in available else next(iter(available)))
    if selected not in available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Rapport '{selected}' introuvable. "
                f"Disponibles: {', '.join(available.keys())}"
            ),
        )

    report_file = available[selected]
    try:
        payload = json.loads(report_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Rapport RAGAS invalide ({report_file.name}): JSON malformé.",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Rapport RAGAS invalide ({report_file.name}): objet JSON attendu.",
        )

    scores = payload.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}

    return RagasReportResponse(
        selected_report=selected,
        report_file=report_file.name,
        available_reports=list(available.keys()),
        scores=scores,
        global_score=payload.get("global_score"),
        n_samples=payload.get("n_samples"),
        model=payload.get("model"),
    )


@app.post(
    "/ragas/run",
    response_model=RagasRunResponse,
    summary="Lancer une évaluation RAGAS en direct",
    description=(
        "Exécute le benchmark RAGAS immédiatement avec options de modèle/samples, "
        "et sauvegarde optionnellement le rapport JSON."
    ),
    tags=["Monitoring"],
)
async def ragas_run(body: RagasRunRequest) -> RagasRunResponse:
    start = time.perf_counter()

    try:
        run_payload = await run_in_threadpool(
            _run_ragas_once,
            body.model,
            body.embedding_model,
            body.max_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Erreur pendant l'exécution RAGAS live: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Échec du run RAGAS: {type(exc).__name__}: {exc}",
        ) from exc

    selected_report = body.report or "reranker"
    persisted = False
    report_file_name = None

    if body.persist_report:
        report_path = _report_path_for_name(selected_report)
        report_file_name = report_path.name
        report_path.write_text(
            json.dumps(
                {
                    "scores": run_payload.get("scores", {}),
                    "global_score": run_payload.get("global_score"),
                    "n_samples": run_payload.get("n_samples"),
                    "model": run_payload.get("model"),
                    "embedding_model": run_payload.get("embedding_model"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        persisted = True

    available_reports = list(_available_ragas_reports().keys())
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    return RagasRunResponse(
        status="success",
        report=selected_report,
        report_file=report_file_name,
        persisted=persisted,
        duration_ms=duration_ms,
        available_reports=available_reports,
        scores=run_payload.get("scores", {}),
        global_score=run_payload.get("global_score"),
        n_samples=run_payload.get("n_samples", 0),
        model=run_payload.get("model", body.model or MODEL),
        embedding_model=run_payload.get(
            "embedding_model",
            body.embedding_model or EMBEDDING_MODEL,
        ),
    )
