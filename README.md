# QA Assistant — Génération automatique de cas de test fonctionnels

> Agent IA spécialisé en assurance qualité logicielle.
> Génère des cas de test fonctionnels à partir de user stories,
> enrichis par les bonnes pratiques **ISTQB** (CTFL v4.0.1 + CTAL-TA v4.0 FR).

---

## Table des matières

1. [Fonctionnement technique](#1-fonctionnement-technique)
2. [Pertinence métier & ROI](#2-pertinence-métier--roi)
3. [Architecture & Pipeline RAG](#3-architecture--pipeline-rag)
4. [KPIs & Métriques RAGAS](#4-kpis--métriques-ragas)
5. [Sécurité & RGPD](#5-sécurité--rgpd)
6. [Innovation — Cross-encoder Reranker](#6-innovation--cross-encoder-reranker)
7. [Prérequis](#7-prérequis)
8. [Installation](#8-installation)
9. [Démarrage de chaque composant](#9-démarrage-de-chaque-composant)
10. [URLs disponibles](#10-urls-disponibles)
11. [Tests](#11-tests)
12. [Évaluation RAGAS](#12-évaluation-ragas)
13. [Structure du projet](#13-structure-du-projet)

---

## 1. Fonctionnement technique

```text
Requête utilisateur
       ↓
┌─────────────────────────────────────────────────────┐
│  Security Guard (12 patterns anti-injection EN/FR)  │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  Classify Intent (GPT-4o-mini)                      │
│  generate_tests | analyze_story | detect_ambiguities│
│  general | out_of_scope                             │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  Fetch User Stories (endpoint REST)                 │
│  fetch_by_index(US-006) ou fetch_all()              │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  RAG ISTQB Pipeline                                 │
│  ChromaDB bi-encoder → filter → cross-encoder rerank│
│  341 chunks · CTFL v4.0.1 EN + CTAL-TA v4.0 FR     │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  LLM GPT-4o-mini (T=0) + prompts ISTQB             │
│  → cas de test positifs / négatifs / limites        │
└─────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────┐
│  Validate JSON + Output Filter (masquage PII)       │
└─────────────────────────────────────────────────────┘
       ↓
  Réponse structurée JSON (status, test_cases, sources…)
```

**Tests : 107 passent · 0 échec**

---

## 2. Pertinence métier & ROI

### Problème réel

La rédaction manuelle de cas de test fonctionnels est une tâche chronophage, répétitive et sujette aux oublis. Un analyste QA expérimenté consacre en moyenne **2 à 4 heures** pour produire un jeu de tests complet (positifs + négatifs + limites) pour une seule user story.

### Solution apportée

L'agent génère automatiquement, en **~15 secondes** :

- Des cas de test positifs, négatifs et limites
- Structurés selon les techniques ISTQB (EP, BVA, Decision Table, Error Guessing)
- Avec données fictives, préconditions, étapes détaillées et résultat attendu
- Systématiquement marqués `status: "à valider"` (validation humaine obligatoire)

### ROI estimé

| Indicateur | Manuel | Avec QA Assistant | Gain |
|---|---|---|---|
| Temps / US (6 TC) | 2–4 h | ~15 s | ×480–×960 |
| Couverture techniques ISTQB | Variable | Systématique | ✅ |
| Traçabilité US → TC | Manuelle | Automatique | ✅ |

> Les cas de test restent soumis à validation humaine. L'agent accélère la rédaction, pas la responsabilité QA.

---

## 3. Architecture & Pipeline RAG

### 7 couches du système

| Couche | Technologie | Rôle |
|---|---|---|
| 1. Entrée / Sécurité | Python regex (12 patterns) | Anti-injection, validation longueur |
| 2. Mémoire | Module Python (max 10 messages) | Contexte de conversation |
| 3. Classification | GPT-4o-mini | Détection d'intention |
| 4. User Stories | REST GET + requests | Récupération des US depuis l'endpoint |
| 5. RAG ISTQB | ChromaDB + LangChain + reranker | Bonnes pratiques ISTQB |
| 6. Génération | GPT-4o-mini (T=0) | Cas de test JSON |
| 7. Sortie / Sécurité | Pydantic + regex | Validation JSON + masquage PII |

### Pipeline RAG détaillé

```text
PDF ISTQB
  ├── CTFL v4.0.1 (EN, 78 pages)
  └── CTAL-TA v4.0 FR (81 pages)
         ↓ PyMuPDF
    Extraction texte (141 pages utiles)
         ↓ RecursiveCharacterTextSplitter
    341 chunks (size=1500, overlap=150)
         ↓ text-embedding-3-small (OpenAI)
    Vecteurs 1536 dimensions
         ↓ ChromaDB (cosine, persistant)
    Base vectorielle indexée
         ↓ similarity_search (top-5, seuil ≥ 0.10)
    Chunks candidats filtrés
         ↓ cross-encoder/ms-marco-MiniLM-L-6-v2
    Reranking précis (local, ~50ms)
         ↓ build_rag_context()
    Contexte ISTQB injecté dans le prompt
```

### Choix techniques justifiés

| Choix | Justification |
|---|---|
| `text-embedding-3-small` | Meilleur rapport qualité/coût pour documents techniques |
| Cosine distance (ChromaDB) | Scores bornés [0,1] — interprétables comme similarité |
| Seuil 0.10 (calibré) | Scores ISTQB max ~0.25 — 0.10 filtre le bruit sans perdre de signal |
| GPT-4o-mini T=0 | Déterminisme maximal pour la génération de TC |
| Cross-encoder reranker | +49% context_precision, +20% context_recall (mesuré RAGAS) |

---

## 4. KPIs & Métriques RAGAS

Évaluation automatique sur **5 questions ISTQB** avec GPT-4o-mini comme juge.

### Résultats avec cross-encoder reranker

| Métrique RAGAS | Score | Interprétation |
|---|---|---|
| context_precision | 0.583 | Les chunks les plus pertinents sont bien en tête |
| faithfulness | 0.756 | Les réponses sont ancrées dans les sources ISTQB |
| answer_relevancy | 0.931 | Les réponses répondent précisément aux questions |
| context_recall | 0.600 | 60% de la connaissance ISTQB attendue est couverte |
| **Score global** | **0.718** | Pipeline RAG de bonne qualité |

### Impact du reranker (comparaison)

| Métrique | Sans reranker | Avec reranker | Gain |
|---|---|---|---|
| context_precision | 0.390 | 0.583 | +49% ✅ |
| context_recall | 0.500 | 0.600 | +20% ✅ |
| Score global | 0.658 | 0.718 | +9% ✅ |

### Métriques opérationnelles

| Indicateur | Valeur |
|---|---|
| Tests unitaires & intégration | 107 / 107 passent |
| Chunks ISTQB indexés | 341 (2 documents, 141 pages) |
| Temps de génération TC | ~15 s (5–10 TC) |
| Temps de reranking | ~50 ms (local, CPU) |
| Coût par requête | ~0.002 $ (GPT-4o-mini) |

---

## 5. Sécurité & RGPD

### Protection des entrées

12 patterns anti-injection compilés (EN + FR) :

- `ignore all previous instructions` / `oublie tes instructions précédentes`
- Role reassignment, exfiltration, jailbreak DAN, balises `<system>`, etc.
- Troncature silencieuse à 5000 caractères
- Rejet immédiat avant tout appel LLM → `status: "error"`

### Protection des sorties

Masquage automatique des données sensibles dans `answer`, `warnings` et `données_fictives` :

- `EMAIL` → `[EMAIL]`
- `TELEPHONE` → `[TELEPHONE]`
- `CARTE` (bancaire) → `[CARTE]`
- `NIR` (sécurité sociale) → `[NIR]`

### Conformité RGPD

- `status: "à valider"` sur chaque cas de test → validation humaine obligatoire
- `requires_human_validation: true` dans toutes les réponses
- `données_fictives` uniquement (jamais de données réelles)
- Aucune persistance des requêtes utilisateur (mémoire RAM, reset au redémarrage)

### Gestion d'erreurs

| Scénario | Comportement |
|---|---|
| LLM indisponible | `status: "error"` + message explicite |
| Endpoint US en timeout | Réponse structurée sans US (graceful degradation) |
| ChromaDB vide | RAG désactivé, génération sans contexte ISTQB |
| JSON LLM malformé | Nettoyage automatique (strip fences, parse défensif) |

---

## 6. Innovation — Cross-encoder Reranker

Le pipeline RAG standard utilise un **bi-encoder** (ChromaDB) qui compare requête et chunks dans un espace vectoriel commun — rapide mais imprécis pour l'ordonnancement.

Le **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) évalue chaque paire `(requête, chunk)` individuellement, produisant un score de pertinence bien plus précis :

```text
ChromaDB retrieval  →  top-5 chunks (bi-encoder, cosine)
        ↓
Cross-encoder rerank → scores précis par paire (query, chunk)
        ↓
Chunks réordonnés   →  le plus pertinent en tête du contexte LLM
```

- **Modèle** : `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 Mo, local)
- **Latence** : ~50 ms sur CPU (non bloquant)
- **Gain mesuré** : context_precision +49%, context_recall +20%, score global +9%
- **Désactivable** : `RAG_RERANKER_ENABLED=false` dans `.env`

---

## 7. Prérequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- Clé API OpenAI avec accès à `gpt-4o-mini` et `text-embedding-3-small`
- ~500 Mo d'espace disque (modèle cross-encoder + ChromaDB)

---

## 8. Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/mickaellherminez/QA_Assistant.git
cd QA_Assistant

# 2. Installer les dépendances
cd agent
uv sync

# 3. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env et renseigner OPENAI_API_KEY

# 4. Indexer les PDFs ISTQB dans ChromaDB (une seule fois, ~2s)
uv run python scripts/ingest_docs.py
```

---

## 9. Démarrage de chaque composant

### A — Source User Stories (GitHub Raw API)

Par défaut, l'agent consomme directement le repo public :

```txt
https://raw.githubusercontent.com/mickaellherminez/github-user-stories-fake-api/main/data/user-stories.json
```

Aucun serveur local de mock n'est requis.

Optionnel : pour surcharger la source, modifiez `US_API_ENDPOINT` dans `agent/.env`.

### B — API FastAPI · Agent QA (Terminal 1)

```bash
cd agent
uv run python run.py

# Production (sans reload, accessible réseau)
uv run python run.py --no-reload --host 0.0.0.0
```

### C — Docker (alternative à B)

```bash
# Build + démarrage
OPENAI_API_KEY=sk-... docker compose up --build

# Première fois : indexer les PDFs dans le container
docker compose --profile ingest up ingest
```

### D — Évaluation RAGAS

```bash
cd agent

# Dry-run — vérifie le RAG sans appel LLM juge (gratuit)
uv run python scripts/evaluate_rag.py --dry-run

# Évaluation complète (~0.03 $)
uv run python scripts/evaluate_rag.py --out rapport.json
```

### E — Tests unitaires & intégration

```bash
cd agent
uv run pytest -v   # verbose
uv run pytest -q   # résumé
```

### F — Frontend Nuxt (UI User Stories + métriques)

```bash
cd frontend
npm install
npm run dev
```

Variables optionnelles (frontend) :

- `QA_API_BASE` (défaut `http://localhost:8000`)
- `USER_STORIES_API_BASE` (défaut `https://raw.githubusercontent.com/mickaellherminez/github-user-stories-fake-api/main/data`)

---

## 10. URLs disponibles

| URL | Méthode | Description |
|---|---|---|
| `http://localhost:8000/docs` | GET | Swagger UI — documentation interactive |
| `http://localhost:8000/redoc` | GET | ReDoc — documentation alternative |
| `http://localhost:8000/ask` | POST | Soumettre une question à l'agent QA |
| `http://localhost:8000/health` | GET | État de l'API + ChromaDB |
| `http://localhost:8000/metrics` | GET | Statistiques depuis le démarrage |
| `http://localhost:8000/user-stories` | GET | Liste des user stories consommées par l'agent |
| `http://localhost:8000/user-stories/US-006` | GET | Détail d'une user story par index |
| `http://localhost:8000/ragas` | GET | Rapport RAGAS local (par défaut `reranker`) |
| `http://localhost:8000/openapi.json` | GET | Schéma OpenAPI (machine-readable) |
| `http://localhost:3000` | GET | Frontend Nuxt (liste US + génération de tests) |
| `https://raw.githubusercontent.com/mickaellherminez/github-user-stories-fake-api/main/data/user-stories.json` | GET | Dataset User Stories (source par défaut) |

### Exemple d'appel `/ask`

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Generate test cases for US-006"}' \
  | python -m json.tool
```

Réponse attendue :

```json
{
  "status": "success",
  "intent": "generate_tests",
  "answer": "6 cas de test générés pour US-006 — Account creation.",
  "test_cases": [
    {
      "id": "TC-001",
      "titre": "Création d'un compte avec des données valides",
      "catégorie": "positive",
      "préconditions": "L'utilisateur accède au formulaire d'inscription.",
      "étapes": ["Saisir un email valide.", "Saisir un mot de passe conforme.", "Cliquer sur Créer."],
      "données_fictives": {"email": "alice@example.com", "password": "Secure123!"},
      "résultat_attendu": "Compte créé. Email de confirmation envoyé.",
      "priorité": "high",
      "user_story": "US-006",
      "status": "à valider"
    }
  ],
  "sources": ["US-006"],
  "requires_human_validation": true
}
```

---

## 11. Tests

```bash
cd agent
uv run pytest -v
```

| Fichier | Tests | Couverture |
|---|---|---|
| `test_agent_integration.py` | 19 | Pipeline complet `main.agent()` |
| `test_api.py` | 14 | Endpoints FastAPI |
| `test_rag.py` | 25 | Ingest, chunking, retrieve, reranker |
| `test_security.py` | 17 | Anti-injection + output filter |
| `test_tools.py` | 13 | US fetch/parse + validator JSON |
| `test_memory.py` | 9 | Mémoire courte |
| **Total** | **107** | **107 passent ✅** |

---

## 12. Évaluation RAGAS

| Métrique | Score |
|---|---|
| context_precision | 0.583 |
| faithfulness | 0.756 |
| answer_relevancy | 0.931 |
| context_recall | 0.600 |
| **Score global** | **0.718** |

---

## 13. Structure du projet

```text
QA_Assistant/
├── pdf/                               # PDFs ISTQB (non commités)
│   ├── ISTQB_CTFL_Syllabus_v4.0.1.pdf
│   └── ISTQB-CTAL-TA-Syllabus-v4.0-FR_final.pdf
├── user_stories_45_generated.json     # 45 user stories (US-006 → US-050)
├── Dockerfile                         # Image multi-stage python:3.11-slim
├── docker-compose.yml                 # Services api + ingest
├── .dockerignore
└── agent/
    ├── config.py                      # Configuration centralisée (.env)
    ├── llm.py                         # Abstraction LangChain / GPT-4o-mini
    ├── main.py                        # Orchestration ReAct (agent principal)
    ├── api.py                         # API FastAPI (/ask, /health, /metrics)
    ├── api_models.py                  # Modèles Pydantic v2
    ├── run.py                         # Point d'entrée uvicorn
    ├── memory/
    │   └── short_term.py              # Mémoire de session (max 10 messages)
    ├── tools/
    │   ├── user_stories.py            # Fetch + parse des US
    │   └── validator.py               # Validation JSON de sortie
    ├── security/
    │   ├── input_guard.py             # Anti-injection (12 patterns EN/FR)
    │   └── output_filter.py           # Masquage PII (EMAIL, TEL, CARTE, NIR)
    ├── rag/
    │   ├── ingest.py                  # Extraction PDF (PyMuPDF)
    │   ├── chunking.py                # Découpage texte (LangChain)
    │   ├── vector_store.py            # ChromaDB (cosine, persistant)
    │   ├── retrieve.py                # Retrieval + filtrage + reranking
    │   └── reranker.py                # Cross-encoder ms-marco-MiniLM-L-6-v2
    ├── prompts/
    │   ├── system.md                  # Prompt système QA
    │   ├── classify_intent.md         # Classification d'intention
    │   └── generate_tests.md          # Génération de cas de test
    ├── scripts/
    │   ├── ingest_docs.py             # CLI d'ingestion des PDFs
    │   └── evaluate_rag.py            # Évaluation RAGAS (4 métriques)
    ├── tests/
    │   ├── conftest.py                # Fixture clean_memory
    │   ├── test_agent_integration.py  # Tests intégration pipeline
    │   ├── test_api.py                # Tests API FastAPI
    │   ├── test_rag.py                # Tests RAG + reranker
    │   ├── test_security.py           # Tests sécurité
    │   ├── test_tools.py              # Tests outils
    │   └── test_memory.py             # Tests mémoire
    ├── pyproject.toml                 # Dépendances (uv)
    └── .env.example                   # Template de configuration
```

## Tags git

| Tag | Contenu |
|---|---|
| `v0.1.0` | Socle : config, LLM LangChain, prompts |
| `v0.2.0` | Mémoire courte + orchestration ReAct |
| `v0.3.0` | Sécurité input/output |
| `v0.4.0` | Pipeline RAG ISTQB (ChromaDB) |
| `v0.5.0` | API FastAPI |
| `v0.6.0` | Tests unitaires + intégration |
| `v0.7.0` | Évaluation RAGAS + Dockerfile |
| `v0.8.0` | Cross-encoder reranker (+9% score global) |
