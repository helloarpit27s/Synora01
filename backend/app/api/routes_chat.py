import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.nim import NimError, NimNotConfigured, nim
from app.nim.prompts import WELLNESS_COACH
from app.rag.retriever import format_context, retrieve
from app.safety.guardrails import SAFE_FALLBACK, check_output_text
from app.safety.redflags import EMERGENCY_TEXT, screen_text
from app.schemas import ChatRequest
from app.services.router import classify

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


@router.post("/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    """Streaming coach with routing and guardrails in front of the model.

    Order matters: red flags are checked before a single token is generated, so
    an emergency never waits on the model, and the model never gets the chance
    to talk someone out of calling for help.
    """
    last = body.messages[-1].content

    async def gen():
        flags = screen_text(last)
        if flags.triggered:
            yield _sse("route", {"route": "clinical", "red_flag": True, "categories": flags.categories})
            yield _sse("content", {"text": EMERGENCY_TEXT})
            yield _sse("done", {"reason": "red_flag", "model_called": False})
            return

        decision = await classify(last)
        yield _sse("route", decision)

        if decision["route"] == "clinical":
            yield _sse(
                "content",
                {"text": "That sounds like something for the symptom check rather than coaching — "
                         "it scores urgency against fixed rules and tells you whether to see someone. "
                         "Open it and I will carry this message across."},
            )
            yield _sse("done", {"reason": "handoff", "model_called": False})
            return

        context = ""
        if body.use_rag:
            hits, retriever, latency = await retrieve(last)
            context = format_context(hits)
            yield _sse(
                "citations",
                {
                    "retriever": retriever,
                    "latency_ms": round(latency, 1),
                    "hits": [{"id": h.id, "title": h.title, "source": h.source, "score": h.score} for h in hits],
                },
            )

        system = WELLNESS_COACH
        if body.condition_ids:
            system += f"\n\nThe person's conditions on file: {', '.join(body.condition_ids)}. " \
                      f"Their constraints override any general advice you would otherwise give."

        history = [{"role": m.role, "content": m.content} for m in body.messages[-6:]]
        messages = [{"role": "system", "content": system}]
        if context:
            messages.append({"role": "system", "content": f"CONTEXT:\n{context}"})
        messages.extend(history)

        buffer: list[str] = []
        try:
            async for kind, text in nim.stream(messages=messages, temperature=0.4, thinking=body.thinking):
                if kind == "reasoning":
                    yield _sse("reasoning", {"text": text})
                    continue
                buffer.append(text)
                yield _sse("content", {"text": text})
        except NimNotConfigured:
            yield _sse("content", {"text": "The coaching model is not configured on this server, so I can only "
                                           "give you what the rules and the reference library already hold."})
            yield _sse("done", {"reason": "not_configured", "model_called": False})
            return
        except NimError as exc:
            yield _sse("error", {"message": f"Model call failed: {exc}"})
            yield _sse("done", {"reason": "error", "model_called": True})
            return

        # Post-hoc check on the assembled answer. Streaming means the client has
        # already seen the text, so a violation is reported as a correction
        # rather than silently swallowed.
        verdict = check_output_text("".join(buffer))
        if not verdict.safe:
            yield _sse("retraction", {"text": SAFE_FALLBACK, "violations": verdict.violations})
        yield _sse("done", {"reason": "complete", "model_called": True, "safe": verdict.safe})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
