#!/usr/bin/env python3
"""Audit whether an OpenAI-compatible SSE chat-completions endpoint really streams.

Connects to an SSE endpoint, timestamps every chunk arrival, and reports
time-to-first-byte (TTFB), inter-chunk gaps (p50/p95/max), total duration and
chunk count. It then CLASSIFIES the stream:

  * "real streaming"      - chunks arrive spread across the generation window
                            (steady token cadence, like native vLLM output).
  * "simulated streaming" - a long wait, then the whole reply arrives in one
                            burst of chunks with near-zero gaps. That is the
                            signature of a finished reply being re-chunked at
                            a gateway, so the client gets no time-to-first-
                            token benefit.

The classification heuristic is deliberately simple and principled:

    simulated  <=>  TTFB > 80% of total duration
                    AND p95 inter-chunk gap (after first byte) < 5 ms

Rationale: if the server generates tokens one at a time, the first token
arrives well before generation finishes (TTFB << total) and the gaps between
chunks reflect real decode latency (tens of ms). If instead the gateway waits
for the full reply and then re-chunks it, the first byte arrives only when the
reply is done (TTFB ~= total) and the "chunks" arrive back-to-back at network
speed (sub-ms gaps).

Stdlib only. Usage:

    python3 stream_audit.py --url http://localhost:18080/v1/chat/completions \\
        --model mock-real --prompt "Say hello"

    python3 stream_audit.py --url https://gateway.example.com/v1/chat/completions \\
        --model my-ocr-model --prompt "..." --api-key "$API_KEY"
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

TTFB_RATIO_THRESHOLD = 0.80   # TTFB must exceed 80% of total duration ...
GAP_P95_THRESHOLD_S = 0.005  # ... and p95 inter-chunk gap must be < 5 ms ...


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def audit(url, model, messages, api_key=None, timeout=120):
    """POST a streaming chat-completions request and time every SSE chunk.

    Returns a dict with timing stats and a 'verdict' key:
    'real streaming' | 'simulated streaming' | 'inconclusive'.
    """
    body = {"model": model, "messages": messages, "stream": True}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        },
    )
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)

    t_send = time.monotonic()
    arrivals = []          # arrival times (seconds since send) of data events
    done_seen = False

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError("endpoint returned HTTP %s" % resp.status)
        for raw in resp:                       # line-delimited SSE events
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith(":"):
                continue                       # heartbeat / comment
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done_seen = True
                break
            arrivals.append(time.monotonic() - t_send)

    t_end = time.monotonic() - t_send
    n = len(arrivals)
    ttfb = arrivals[0] if n else None
    total = (arrivals[-1] if n else t_end)
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]

    report = {
        "url": url,
        "model": model,
        "chunk_count": n,
        "done_terminator_seen": done_seen,
        "ttfb_s": ttfb,
        "total_duration_s": total,
        "ttfb_to_total_ratio": (ttfb / total) if (ttfb is not None and total > 0) else None,
        "gap_p50_s": percentile(gaps, 50),
        "gap_p95_s": percentile(gaps, 95),
        "gap_max_s": max(gaps) if gaps else None,
    }
    report["verdict"] = classify(report)
    return report


def classify(report):
    """Apply the streaming heuristic. Returns a verdict string."""
    n = report["chunk_count"]
    ratio = report["ttfb_to_total_ratio"]
    p95 = report["gap_p95_s"]
    if n < 2 or ratio is None or p95 is None:
        return "inconclusive (too few chunks to classify)"
    simulated = (ratio > TTFB_RATIO_THRESHOLD) and (p95 < GAP_P95_THRESHOLD_S)
    return "simulated streaming" if simulated else "real streaming"


def fmt_seconds(v):
    return ("%.1f ms" % (v * 1000.0)) if v is not None else "n/a"


def print_report(report):
    print("endpoint : %s" % report["url"])
    print("model    : %s" % report["model"])
    print("chunks   : %d   (SSE [DONE] terminator: %s)"
          % (report["chunk_count"], "yes" if report["done_terminator_seen"] else "no"))
    print("TTFB     : %s" % fmt_seconds(report["ttfb_s"]))
    print("total    : %s" % fmt_seconds(report["total_duration_s"]))
    print("TTFB/total ratio : %s"
          % ("%.3f" % report["ttfb_to_total_ratio"]
             if report["ttfb_to_total_ratio"] is not None else "n/a"))
    print("gap p50  : %s" % fmt_seconds(report["gap_p50_s"]))
    print("gap p95  : %s" % fmt_seconds(report["gap_p95_s"]))
    print("gap max  : %s" % fmt_seconds(report["gap_max_s"]))
    print("verdict  : %s" % report["verdict"].upper())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit SSE streaming behaviour.")
    ap.add_argument("--url", required=True,
                    help="chat-completions endpoint URL (SSE)")
    ap.add_argument("--model", default="default",
                    help="model name to request")
    ap.add_argument("--prompt", default="Say hello in one short sentence.",
                    help="user prompt to send")
    ap.add_argument("--api-key", default=None,
                    help="bearer token (or set VLM_AUDIT_API_KEY)")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--json", action="store_true",
                    help="emit the raw report as JSON")
    args = ap.parse_args(argv)

    api_key = args.api_key
    if api_key is None:
        import os
        api_key = os.environ.get("VLM_AUDIT_API_KEY")

    try:
        report = audit(args.url, args.model,
                       [{"role": "user", "content": args.prompt}],
                       api_key=api_key, timeout=args.timeout)
    except Exception as exc:  # surface cleanly for CLI use
        print("audit failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
