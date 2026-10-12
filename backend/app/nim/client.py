"""Async NIM client.

This is your `NimReq` rebuilt for a server process. Three things changed and
each one matters once more than one request is in flight:

1. `time.sleep()` blocks the whole event loop. The 1.5s pacing you had is now a
   shared async rate limiter, so one slow call does not freeze every other
   request on the box.
2. Your original falls through to the streaming branch when `elapsed >= 1.5`
   and `stream=False`, and then iterates a non-streaming completion object.
   Streaming and non-streaming are separate functions here.
3. Retries with backoff on 429/5xx. NIM rate limits, and a health agent that
   dies on the first 429 is not a health agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from app.config import get_settings

log = logging.getLogger("zenhealth.nim")


class NimError(RuntimeError):
    """Upstream failed in a way the caller has to handle."""


class NimNotConfigured(NimError):
    """No API key. Callers fall back to local logic instead of 500-ing."""


@dataclass
class Usage:
    """Per-call telemetry. The mentor brief asks about latency, so measure it."""

    model: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    meta: dict[str, Any] = field(default_factory=dict)


class _Pacer:
    """Minimum interval between upstream calls, shared across all requests."""

    def __init__(self, min_interval: float) -> None:
        self._min = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self._min <= 0:
            return
        async with self._lock:
            gap = time.monotonic() - self._last
            if gap < self._min:
                await asyncio.sleep(self._min - gap)
            self._last = time.monotonic()


class NimClient:
    def __init__(self) -> None:
        self.s = get_settings()
        self._pacer = _Pacer(self.s.nim_min_interval_s)
        self._http: httpx.AsyncClient | None = None

    # ── lifecycle ────────────────────────────────────────────────────────
    async def open(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.s.nim_base_url,
                timeout=httpx.Timeout(self.s.nim_timeout_s, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.s.nvidia_api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _require(self) -> httpx.AsyncClient:
        if not self.s.configured:
            raise NimNotConfigured("NVIDIA_API_KEY is not set")
        if self._http is None:
            raise NimError("client not opened")
        return self._http

    # ── core ─────────────────────────────────────────────────────────────
    async def _post(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        http = self._require()
        last: Exception | None = None
        for attempt in range(1, self.s.nim_max_retries + 1):
            await self._pacer.wait()
            try:
                res = await http.post(path, json=payload)
                if res.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"{res.status_code}: {res.text[:200]}", request=res.request, response=res
                    )
                res.raise_for_status()
                return res.json(), attempt
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last = exc
                if attempt == self.s.nim_max_retries:
                    break
                backoff = min(8.0, 0.75 * 2 ** (attempt - 1))
                log.warning("NIM %s failed (attempt %s/%s): %s", path, attempt, self.s.nim_max_retries, exc)
                await asyncio.sleep(backoff)
        raise NimError(f"{path} failed after {self.s.nim_max_retries} attempts: {last}")

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        thinking: bool = False,
        json_only: bool = False,
    ) -> tuple[str, Usage]:
        """Single-shot completion. Deterministic by default: temperature 0."""
        model = model or self.s.nim_model_reasoning
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if thinking:
            payload["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 8192,
            }
        if json_only:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        data, attempts = await self._post("/chat/completions", payload)
        latency = (time.perf_counter() - started) * 1000

        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return text, Usage(
            model=model,
            latency_ms=round(latency, 1),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            attempts=attempts,
        )

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        thinking: bool = False,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yields ``(kind, text)`` where kind is 'reasoning' or 'content'.

        Keeping the two apart lets the frontend render thinking in a muted
        panel instead of splicing it into the answer, which is what your
        original generator ended up doing.
        """
        model = model or self.s.nim_model_reasoning
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if thinking:
            payload["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 8192,
            }

        http = self._require()
        await self._pacer.wait()
        async with http.stream("POST", "/chat/completions", json=payload) as res:
            if res.status_code >= 400:
                body = (await res.aread()).decode(errors="replace")
                raise NimError(f"stream failed {res.status_code}: {body[:200]}")
            async for line in res.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    return
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if reasoning := delta.get("reasoning_content"):
                    yield "reasoning", reasoning
                if content := delta.get("content"):
                    yield "content", content

    async def vision(
        self,
        *,
        system: str,
        text: str,
        image_data_url: str,
        model: str | None = None,
        max_tokens: int = 900,
    ) -> tuple[str, Usage]:
        model = model or self.s.nim_model_vision
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]
        return await self.complete(messages=messages, model=model, temperature=0.0, max_tokens=max_tokens)

    # ── retrieval primitives ─────────────────────────────────────────────
    async def embed(self, texts: list[str], *, input_type: str = "passage") -> list[list[float]]:
        """NeMo Retriever embeddings.

        ``input_type`` is not optional for these models — queries and passages
        are embedded differently, and mixing them up quietly wrecks recall.
        """
        data, _ = await self._post(
            "/embeddings",
            {
                "model": self.s.nim_model_embed,
                "input": texts,
                "input_type": input_type,
                "encoding_format": "float",
                "truncate": "END",
            },
        )
        rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
        return [r["embedding"] for r in rows]

    async def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        """Returns ``(original_index, logit)`` sorted best first."""
        data, _ = await self._post(
            "/ranking",
            {
                "model": self.s.nim_model_rerank,
                "query": {"text": query},
                "passages": [{"text": p} for p in passages],
            },
        )
        ranked = data.get("rankings", [])
        return [(r["index"], float(r.get("logit", 0.0))) for r in ranked]


nim = NimClient()
