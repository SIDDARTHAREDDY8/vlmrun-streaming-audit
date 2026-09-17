# vlmrun-streaming-audit

A small measurement tool that answers one question about any OpenAI-compatible
SSE chat-completions endpoint: **is the streaming real, or simulated?**

## The problem this audits

VLM Run's Gateway "streams" OCR/adapter model responses over SSE — but the
streaming is **simulated**: the finished reply is re-chunked at the gateway, so
clients get zero time-to-first-token benefit. Genuine token-by-token streaming
exists only for native vLLM chat models.

Citations (from VLM Run's own public repo):

- **vlm-run/vlmrun-python-sdk PR #214** (merged 2026-08-30, closes tracker
  **VLM-916**) — the PR body states the gateway does *"simulated streaming for
  OCR/adapter models"* while *"real token streaming"* exists only for
  *"native vLLM chat models"*.
- **Code comment** in `vlmrun/cli/_cli/gateway.py`
  (commit `806f8c9224859166803a425621187651c1025507`).

Why it matters: Document OCR is the centerpiece of VLM Run's pricing page
("Dirt-cheap Document OCR"), and their "Orchestration Built-In" fans
multi-page PDFs out per page. With simulated streaming there is no incremental
result delivery — and the streaming UX is misleading, because the client waits
the full generation time before the first "token" arrives.

The actual serving code is private (`vlm-lab`), so this repo does **not** fix
the gateway. It gives VLM Run (or anyone) a way to *measure* the behavior.

## What the tool does

`stream_audit.py` POSTs a streaming chat-completions request, timestamps every
SSE chunk arrival, and reports:

| Metric | Meaning |
|---|---|
| TTFB | time from request send to first data chunk |
| inter-chunk gaps (p50 / p95 / max) | spacing between consecutive chunks |
| total duration | time from send to last chunk |
| chunk count | number of `data:` events (`[DONE]` excluded) |

It then **classifies** the stream:

- **real streaming** — chunks spread across the generation window
  (steady token cadence, like native vLLM output).
- **simulated streaming** — a long wait, then the whole reply arrives in one
  burst of chunks with near-zero gaps: the signature of a finished reply
  being re-chunked at the gateway.

The heuristic is simple and principled:

> **simulated** ⟺ **TTFB > 80% of total duration** AND **p95 inter-chunk gap < 5 ms**

Rationale: a real token generator emits its first token well before generation
finishes (TTFB ≪ total) with decode-latency-sized gaps (tens of ms). A
gateway that re-chunks a finished reply can't send the first byte until the
reply is done (TTFB ≈ total), and the "chunks" then arrive back-to-back at
network speed (sub-ms gaps).

## How to run

Python 3, stdlib only — no dependencies to install.

```bash
# 1. Run the demo: audits both mock servers, prints the comparison table
python3 run_demo.py
```

```bash
# 2. Audit any OpenAI-compatible SSE endpoint yourself
python3 stream_audit.py --url http://localhost:18080/v1/chat/completions \
    --model mock-real --prompt "Say hello"

# with a bearer token (or set VLM_AUDIT_API_KEY):
python3 stream_audit.py --url https://your-gateway/v1/chat/completions \
    --model your-model --prompt "Transcribe this page." \
    --api-key "$API_KEY" --json
```

`mock_servers.py` provides the two reference servers used by the demo (stdlib
`http.server` only):

- `:18080` — **real** streaming: 24 token chunks at a steady ~45 ms cadence
  after a short 150 ms prefill pause.
- `:18081` — **simulated** streaming: waits 1.2 s (the "full reply" being
  generated), then flushes all 24 re-chunked chunks in a single burst.

## How VLM Run could run this against their own gateway

VLM Run holds the API keys, so this is a one-liner for them:

```bash
python3 stream_audit.py \
  --url https://gateway.vlm.run/v1/openai/chat/completions \
  --model <an-OCR-or-adapter-model> \
  --prompt "<a real document task>" \
  --api-key "$VLM_RUN_API_KEY"
```

(Adjust the path if their OpenAI-compatible mapping differs.) Run it once
against an OCR/adapter model and once against a native vLLM chat model: if the
OCR run classifies as *simulated* while the chat run classifies as *real*,
the measurement reproduces the behavior described in PR #214 / VLM-916.

## Caveat — read this before citing numbers

**Everything in this repo was validated against local mock servers, not
against VLM Run's live gateway.** The demo numbers prove the *classifier*
works (it cleanly separates spaced token cadence from burst-after-delay);
they are not measurements of VLM Run's infrastructure. No request carrying
credentials was ever made to the live gateway, so no live measurement is
taken or claimed here — the one-liner above is for VLM Run to run themselves,
since they hold the keys.
