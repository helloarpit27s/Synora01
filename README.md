# Synora

**A holistic health and wellness agent: symptom triage, condition-aware nutrition, a coaching chat, and live wearable anomaly detection, in one calm place.**

Built for a hackathon with React 19 + Vite on the front, FastAPI on the back, and NVIDIA NIM for language, vision and retrieval models.

> **The model never decides anything that matters.** Urgency comes from a pure function. Nutrition verdicts come from condition rules. Anomalies come from arithmetic. The LLM writes prose, reads photos, and breaks routing ties, and nothing else.

---

## Why Synora

Most health apps either bury people in generic advice or let a chatbot guess at what's wrong. Both are risky. Synora takes the opposite approach: every safety-critical decision is made by deterministic, testable code, and the AI is limited to explaining and assisting on top of decisions that are already made.

## Features

| Feature | What it does |
| --- | --- |
| **Symptom triage** | Pick a body region and symptoms, get an urgency band from a rule engine. Red-flag symptoms and escalation combinations (chest pressure with breathlessness, stroke signs, GI bleeding) jump straight to the top band. |
| **Condition-aware nutrition** | Search foods or snap a meal photo and get a per-condition verdict (glycaemic load for diabetes, sodium for hypertension, trigger foods and fat for reflux, and so on) with a running day total. |
| **Coaching chat** | A streaming coach that screens for red flags, routes intent, retrieves from a clinical corpus, answers with citations, and checks its own output before you see it. |
| **Wearable anomaly detection** | Four live streams (heart rate, HRV, blood oxygen, glucose) with an online rolling z-score detector. Alerts state the deviation and the baseline and never guess a cause. |
| **Care team** | Urgency gauge, video callback, clinician availability and medicines. |
| **Library** | Preventive care that's due, plus reading filtered by your conditions. |
| **Accessibility first** | Skip link, focus management, 48px touch targets, high-contrast mode, three text sizes, reduced-motion support. |

## Pages

| Route | What lives there |
| --- | --- |
| `/` | Dashboard: today's log, the morning anomaly, urgency at a glance |
| `/chat` | Coaching chat with routing, citations and a reasoning panel |
| `/triage` | Body map, symptom form, deterministic urgency result |
| `/nutrition` | Food search, meal photo, per-condition verdicts |
| `/vitals` | Wearable streams, flagged windows, detector settings |
| `/care` | Urgency gauge, video callback, clinician availability, medicines |
| `/library` | Preventive care and condition-filtered reading |
| `/settings` | Conditions, text size, theme, contrast, motion |

---

## How it works

```
browser (React)                         backend (FastAPI)                NVIDIA NIM
─────────────────                       ─────────────────                ──────────
lib/triage.js      decides urgency
lib/nutrition.js   scores food          POST /api/triage        ─────►  reasoning model
lib/anomaly.js     watches streams      POST /api/meal/analyse  ─────►  vision model
lib/api.js         ──────────────────►  POST /api/route         ─────►  guard model
  · explainTriage()                     POST /api/chat/stream   ─────►  reasoning + RAG
  · analyseMealPhoto()                  POST /api/rag/search
  · streamChat()      (SSE)             GET  /api/vitals/*
                                        (API key lives here only)
```

### Three layers of safety

1. **Deterministic rules first.** Red-flag symptoms force the `immediate` band on both the client's local engine and the backend's. When the band is `immediate`, no model is called at all; a fixed emergency message is returned.
2. **Guardrail screening.** User text is screened before it reaches the reasoning model, and the model's output is screened before it reaches the browser for diagnosis claims, dose advice and false reassurance. If a streamed answer fails, the UI receives a `retraction` event instead of the bad text.
3. **Prompts as reviewable code.** `backend/app/nim/prompts.py` forbids naming a diagnosis, recommending a dose, contradicting the band, or telling anyone they are fine.

### The chat pipeline

Before a single token is generated, every message goes through:

```
red-flag screen → intent routing → RAG retrieval → model → output guardrail check
```

Each stage is surfaced to the UI as its own server-sent event:

```
event: route        {"route":"wellness","confidence":0.85,"decided_by":"rule",...}
event: citations    {"retriever":"nim","hits":[{"id":"sodium-blood-pressure#0",...}]}
event: reasoning    {"text":"..."}        # only when thinking=true
event: content      {"text":"..."}        # the answer, token by token
event: retraction   {"text":"...","violations":[...]}
event: done         {"reason":"complete","model_called":true,"safe":true}
```

### Highlights under the hood

- **Intent routing** uses a strict precedence: `red flag > clinical lexicon > admin > wellness > model tiebreak > clinical`. Ties break toward clinical because under-triage is the expensive error. The model is only consulted when no lexicon matches, at temperature 0.
- **Meal recognition** sends the photo to a vision model with a strict JSON schema, then validates it. Vision models often return macros that contradict their own calorie figure, so when the gap exceeds 20% the macro-derived value wins and confidence is downgraded.
- **Anomaly detection** is an online rolling z-score with a minimum run length to suppress single-sample blips. It's O(window) per sample with no model involved, and `stats()` exposes p50/p95/max latency so the speed is checkable rather than claimed.
- **RAG** ingests markdown with frontmatter, chunks on paragraphs with overlap, embeds with NeMo Retriever (correct `input_type` for queries vs passages), and reranks the shortlist. BM25 covers the no-key path. Retrieved passages carry ids so answers return verifiable citations.
- **Resilient NIM client** with a shared async rate limiter, separate `complete()` and `stream()` paths, and exponential backoff on 429 and 5xx errors.

---

## Tech stack

| Layer | Technology |
| --- | --- |
| Frontend | React 19, React Router 7, Vite 6, Vitest, ESLint |
| Backend | FastAPI, Uvicorn, httpx, Pydantic, NumPy |
| AI | NVIDIA NIM (OpenAI-compatible endpoint) |
| Models (configurable) | `nvidia/llama-3.3-nemotron-super-49b-v1` (reasoning), `meta/llama-3.1-8b-instruct` (light), `meta/llama-3.2-90b-vision-instruct` (vision), `nvidia/nv-embedqa-e5-v5` (embeddings), `nvidia/llama-3.2-nv-rerankqa-1b-v2` (rerank), `nvidia/llama-3.1-nemoguard-8b-content-safety` (guard) |
| Retrieval | Numpy vector store + BM25 fallback over a markdown corpus |

---

## Getting started

### Prerequisites

- Node.js 20.11 or newer
- Python 3.10 or newer
- *(Optional)* an NVIDIA API key (`nvapi-...`) from [build.nvidia.com](https://build.nvidia.com)

### 1. Backend

```bash
cd backend
cp .env.example .env          # add your nvapi- key and confirm the model ids
pip install -r requirements.txt
python -m app.rag.ingest      # build the index (optional, startup does it too)
uvicorn app.main:app --reload --port 8000
```

Interactive API docs are at `http://localhost:8000/docs`.

### 2. Frontend

```bash
cd frontend
cp .env.example .env          # only needed if the backend isn't on localhost:8000
npm install
npm run dev                   # http://localhost:5173, proxies /api to :8000
```

### Works without a key

Synora runs fully **without** an API key. Triage scoring, routing, nutrition rules, anomaly detection and lexical search are all local. You only lose the plain-language summaries, meal photo analysis, dense retrieval and the chat's model replies.

### Confirm model ids

The NVIDIA catalogue changes often. Verify the defaults before trusting them:

```bash
curl -s https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer $NVIDIA_API_KEY" | jq '.data[].id'
```

---

## Configuration

Backend environment variables (`backend/.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | none | NIM key, server-side only |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM endpoint |
| `NIM_MODEL_*` | see above | Reasoning, light, vision, embed, rerank, guard model ids |
| `NIM_MIN_INTERVAL_S` | `1.5` | Pacing between upstream calls |
| `NIM_TIMEOUT_S` | `45` | Request timeout |
| `NIM_MAX_RETRIES` | `3` | Retries on 429 / 5xx |
| `RAG_TOP_K` / `RAG_RERANK_TO` / `RAG_MIN_SCORE` | `8` / `4` / `0.15` | Retrieval tuning |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://localhost:3000` | CORS allowlist |

Frontend (`frontend/.env`): `API_ORIGIN` sets where the Vite dev server proxies `/api` (default `http://localhost:8000`).

---

## API reference

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/api/health` | Key status, resolved model ids, index mode |
| `POST` | `/api/route` | Deterministic intent classification (clinical / wellness / admin) |
| `POST` | `/api/triage` | Rule-scored urgency, grounded non-diagnostic summary, citations |
| `POST` | `/api/meal/analyse` | Meal photo to macros to per-condition verdicts |
| `POST` | `/api/rag/search` | Retrieval with scores and sources |
| `POST` | `/api/rag/reindex` | Rebuild the index after editing the corpus |
| `GET` | `/api/vitals/snapshot` | Batch stream plus detected anomalies |
| `GET` | `/api/vitals/stream` | SSE: live samples and anomaly alerts |
| `POST` | `/api/chat/stream` | SSE coach: screen, route, retrieve, stream, check |

A quick manual check against a running server: `bash backend/scripts/smoke.sh`

---

## Project structure

```
synora/
├── backend/
│   ├── app/
│   │   ├── api/         # FastAPI routes (chat, triage, meal, rag, route, vitals, health)
│   │   ├── services/    # triage, nutrition, vision, router, vitals logic
│   │   ├── safety/      # red-flag lexicon and output guardrails
│   │   ├── rag/         # ingest, store, retriever
│   │   ├── nim/         # NIM client and reviewable prompts
│   │   └── data/corpus/ # guideline-style markdown documents
│   ├── tests/
│   └── scripts/smoke.sh
└── frontend/
    └── src/
        ├── pages/       # Dashboard, Chat, Triage, Nutrition, Vitals, CareTeam, Library, Settings
        ├── components/  # BodyDiagram, Gauge, Sparkline, MoodPicker, ...
        ├── lib/         # triage, nutrition, anomaly, api (SSE client)
        ├── state/       # Health and Preferences contexts
        ├── data/        # sample profile, foods, vitals, articles
        └── styles/      # design tokens and layout
```

## Testing

```bash
cd backend && pytest          # guardrails, nutrition, RAG, router, triage, vitals (no key needed)
cd frontend && npm test       # triage rules, nutrition scoring, anomaly detector
cd frontend && npm run lint
```

---

## What's real and what's sample data

**Real:** the triage rule engine, nutrition scoring rules, anomaly detector, intent routing, the entire interface, and the full backend path including RAG over the corpus in `backend/app/data/corpus/`.

**Sample:** the user profile, food table, wearable streams (seeded pseudo-random so they're reproducible), clinician availability and the article list. Each can be swapped for a real source without touching a component, since the data shapes are the contract.

## Accessibility

Skip link, focus moved to `<main>` on every route change, visible focus rings, 48px minimum interactive height, native `<dialog>` for focus trapping, `prefers-reduced-motion` respected with a manual override, a high-contrast mode, and three text sizes that scale the whole type system. Body text defaults to 17px at 1.7 line height.

## Before this goes near real patients

- Keep `NVIDIA_API_KEY` server-side. Never prefix it with `VITE_`.
- Keep `ALLOWED_ORIGINS` an explicit allowlist. Never reflect the Origin header or wildcard with credentials.
- Put auth in front of `/api/rag/reindex`, since it costs embedding calls.
- Never log symptom text, chat messages or meal photos.
- Session data currently lives in `localStorage`. Replace it with a server-side record with proper consent, retention and audit.

## Team

*Add team members, roles and hackathon name here.*

---

*Synora does not diagnose, prescribe, or replace a clinician. In an emergency, call your local emergency number.*