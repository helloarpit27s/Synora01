# ZenHealth agent API

Backend for the Holistic Health & Wellness Agent: deterministic symptom triage, condition-aware
nutrition, multimodal meal recognition, RAG over a clinical corpus, and real-time wearable anomaly
detection. FastAPI, NVIDIA NIM, no other infrastructure required to run it.

**The model never decides anything that matters.** Urgency comes from a pure function. Nutrition
verdicts come from condition rules. Anomalies come from arithmetic. The LLM writes prose, reads
photos, and breaks routing ties — nothing else.

---

## Run it

```bash
cp .env.example .env          # add your nvapi- key and confirm the model ids
pip install -r requirements.txt
python -m app.rag.ingest      # build the index (optional — startup does it too)
uvicorn app.main:app --reload --port 8000
pytest                        # 60+ assertions, no key needed
```

It runs **without a key**. Triage, routing, nutrition rules, anomaly detection and lexical RAG are
all local. You lose the plain-language summaries, meal photos, dense retrieval and reranking.

Interactive docs at `http://localhost:8000/docs`.

---

## Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/api/health` | Key status, resolved model ids, index mode |
| `POST` | `/api/route` | Deterministic intent classification (clinical / wellness / admin) |
| `POST` | `/api/triage` | Rule-scored urgency + grounded non-diagnostic summary + citations |
| `POST` | `/api/meal/analyse` | Meal photo → macros → per-condition verdicts |
| `POST` | `/api/rag/search` | Retrieval with scores and sources |
| `POST` | `/api/rag/reindex` | Rebuild the index after editing the corpus |
| `GET` | `/api/vitals/snapshot` | Batch stream + detected anomalies (first paint) |
| `GET` | `/api/vitals/stream` | SSE: live samples and anomaly alerts |
| `POST` | `/api/chat/stream` | SSE coach: red-flag screen → route → RAG → stream → output check |

### SSE contract

`/api/chat/stream` emits named events so the frontend can render each one differently:

```
event: route        {"route":"wellness","confidence":0.85,"decided_by":"rule",...}
event: citations    {"retriever":"nim","hits":[{"id":"sodium-blood-pressure#0",...}]}
event: reasoning    {"text":"..."}        # only when thinking=true
event: content      {"text":"..."}        # the answer, token by token
event: retraction   {"text":"...","violations":[...]}   # output failed the diagnosis check
event: done         {"reason":"complete","model_called":true,"safe":true}
```

`/api/vitals/stream` emits `sample`, `anomaly`, `stats`, `done`.

---

## How the requirements map to code

**Clinical safety and escalation.** `app/safety/redflags.py` is a curated regex lexicon that runs
before any model sees the text — cardiac, stroke, breathing, bleeding, neuro, abdominal, obstetric.
`app/services/triage.py` holds five escalation rules that force the band to `immediate` regardless of
severity or duration. When the band is `immediate` the model is never called; a fixed sentence is
returned. `app/safety/guardrails.py` screens every model output for diagnosis assertions, dose
advice, and false reassurance, and substitutes a safe fallback when one gets through. Streaming
answers that fail the check emit a `retraction` event rather than being quietly dropped.

**Intent routing.** `app/services/router.py` decides by precedence:
`red flag > clinical lexicon > admin > wellness > model tiebreak > clinical`. Ties break toward
clinical because under-triage is the expensive error. The model is only consulted for text that
matches no lexicon at all, and at temperature 0. `test_router.py` asserts repeatability.

**Multimodal meal recognition.** `app/services/vision.py` sends the photo to the VLM with a strict
JSON schema, then validates: plausibility bounds per field, and a macro-versus-calorie consistency
check. Vision models routinely return macros that contradict their own calorie figure; when the gap
exceeds 20% the macro-derived value wins and confidence is downgraded, because the macros are what
the condition rules consume.

**Constraint enforcement.** `app/services/nutrition.py` decides what a food means for this person —
glycaemic load for diabetes, sodium against a configurable target for hypertension, trigger tags and
fat for reflux, sodium and protein for CKD. The model is never asked for a verdict, so it cannot be
talked out of one. A 240 kcal bowl of white rice still warns for diabetes and passes for blood
pressure; that test is in `test_nutrition.py`.

**Sensor anomaly detection.** `app/services/vitals.py` runs an online rolling z-score with a minimum
run length to suppress single-sample blips. O(window) per sample, no allocation, no model. Measured
p95 detection latency is roughly 2 microseconds per sample over 500 samples, and `stats()` exposes
p50/p95/max so the number is checkable rather than claimed. Alert text states the deviation and the
baseline and names no cause. Swap `synthetic_stream()` for a Kafka consumer; `RollingDetector.push()`
is the entire contract.

**RAG.** `app/rag/` ingests markdown with frontmatter, chunks on paragraphs with overlap, embeds via
NeMo Retriever (`input_type` set correctly for queries versus passages — getting this wrong quietly
wrecks recall), stores vectors in a numpy matrix, and reranks the shortlist through `/v1/ranking`.
BM25 covers the no-key path. Retrieved passages are injected with their ids so the model can cite
them and the API can return verifiable citations. Corpus lives in `app/data/corpus/` — six
guideline-style documents; replace with your PubMed or guideline set and run `/api/rag/reindex`.

---

## About your `NimReq`

`app/nim/client.py` is that function rebuilt for a server process. Three things changed:

1. **`time.sleep()` blocks the event loop.** Under FastAPI that freezes every concurrent request,
   not just the one calling NIM. The 1.5 s pacing is now a shared async rate limiter.
2. **The streaming and non-streaming paths were tangled.** In the original, `if not stream: return
   completion.choices[0]...` sits inside `if elapsed < 1.5`, so when the call takes longer than 1.5 s
   with `stream=False`, execution falls through and returns a generator that iterates a
   non-streaming object. They are separate functions here: `complete()` and `stream()`.
3. **No retries.** NIM rate-limits, and 429 is a normal Tuesday. There is now exponential backoff on
   429 and 5xx, capped at `NIM_MAX_RETRIES`.

Also worth keeping from your version: `stream()` still separates `reasoning_content` from `content`,
but yields them as tagged tuples so the frontend can render thinking in its own panel instead of
splicing it into the answer.

---

## Deployment notes

- `NVIDIA_API_KEY` is server-side only. Nothing in this repo exposes it to a browser.
- `ALLOWED_ORIGINS` is an explicit allowlist. Never reflect the Origin header, never wildcard with
  credentials.
- Add auth in front of `/api/rag/reindex` — it costs embedding calls.
- Never log symptom text, notes, or meal photos. The error handler deliberately returns a generic
  message rather than leaking internals.
- `localStorage`-style session state does not exist here; add a real record store with consent,
  retention and audit before this touches a real patient.
- Model ids are settings, not constants. Confirm them against the live catalogue:
  `curl -s $NIM_BASE_URL/models -H "Authorization: Bearer $NVIDIA_API_KEY" | jq '.data[].id'`

---

*ZenHealth does not diagnose, prescribe, or replace a clinician. In an emergency, call your local
emergency number.*
