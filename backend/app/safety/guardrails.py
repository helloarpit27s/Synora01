"""Input and output guardrails.

Layered on purpose:
  in  → red flags (regex) → NemoGuard content safety → model
  out → diagnosis detector (regex) → NemoGuard → client

The regex layers are not a substitute for the guard model; they are the layer
that still works when the guard model is unreachable, and they are the layer
you can unit-test.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.nim import NimError, NimNotConfigured, nim
from app.nim.prompts import GUARD_CLASSIFIER

log = logging.getLogger("zenhealth.safety")

# Phrases that assert a diagnosis. The model is told not to produce these; this
# catches it when the instruction does not hold.
DIAGNOSIS_PATTERNS = [
    r"\byou (have|likely have|probably have|are suffering from|are experiencing)\s+(a|an|the)?\s*\b(?!symptom|symptoms|a symptom|trouble|difficulty)\w*(itis|osis|emia|pathy|oma|syndrome|disease|infection|failure|attack)\b",
    r"\b(this|that|it) (is|sounds like|looks like|is likely|is probably) (a|an)\s+\w*(itis|osis|emia|pathy|oma|syndrome|disease|infection|attack)\b",
    r"\byou (are|'re) (diagnosed with|suffering from)\b",
    r"\bI (diagnose|can diagnose)\b",
    r"\byou (should|need to) (take|start|stop|increase|decrease|double|halve)\s+\w*(mg|mcg|ml|tablets?|pills?|doses?)\b",
    r"\b(take|start|stop|increase|double)\s+\d+\s*(mg|mcg|ml|tablets?|pills?)\b",
    r"\byou (are|'re) (fine|healthy|okay|ok)\b",
    r"\bnothing to worry about\b",
    r"\byou don'?t need (to see|a) (a )?(doctor|clinician|gp)\b",
]
COMPILED_DIAGNOSIS = [re.compile(p, re.I) for p in DIAGNOSIS_PATTERNS]

SAFE_FALLBACK = (
    "I can only give general information here, not an assessment of what is causing this. "
    "If it is worrying you or getting worse, that is reason enough to have a clinician look at it."
)


@dataclass
class GuardResult:
    safe: bool
    screened_by_model: bool = False
    categories: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


def check_output_text(text: str) -> GuardResult:
    """Fast, offline, testable. Runs on every model output before it ships."""
    violations = [p.pattern for p in COMPILED_DIAGNOSIS if p.search(text or "")]
    return GuardResult(safe=not violations, violations=violations)


async def screen_with_model(text: str, role: str = "user") -> GuardResult:
    """NemoGuard pass. Fails open with a loud log rather than blocking a
    person describing real symptoms — but an unguarded deployment should be a
    decision someone made, not one that happened quietly."""
    if not text.strip():
        return GuardResult(safe=True)
    try:
        raw, _ = await nim.complete(
            messages=[
                {"role": "system", "content": GUARD_CLASSIFIER},
                {"role": role if role in ("user", "assistant") else "user", "content": text[:4000]},
            ],
            model=nim.s.nim_model_guard,
            temperature=0.0,
            max_tokens=128,
            json_only=True,
        )
        verdict = parse_json(raw, {"unsafe": False, "categories": []})
        return GuardResult(
            safe=not bool(verdict.get("unsafe")),
            screened_by_model=True,
            categories=list(verdict.get("categories") or []),
        )
    except NimNotConfigured:
        return GuardResult(safe=True)
    except NimError as exc:
        log.error("guard model unreachable: %s — proceeding with regex layer only", exc)
        return GuardResult(safe=True)


def parse_json(text: str, fallback: dict | None = None) -> dict:
    """Models fence JSON no matter how firmly you ask."""
    if not text:
        return fallback or {}
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return fallback or {}
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return fallback or {}
