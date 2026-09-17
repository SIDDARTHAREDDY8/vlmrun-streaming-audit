#!/usr/bin/env python3
"""Demo: run the streaming audit against both mock servers and compare.

Spins up the real-streaming and simulated-streaming mocks on localhost,
audits each with stream_audit.audit(), and prints a side-by-side table
showing that the classifier tells them apart.

    python3 run_demo.py
"""

import time

from mock_servers import (RealStreamingHandler, SimulatedStreamingHandler,
                          serve)
from stream_audit import audit

REAL_PORT = 18080
SIM_PORT = 18081
PROMPT = "Say hello in one short sentence."


def row(label, real, sim, fmt):
    return "%-22s %s %s" % (label, fmt(real), fmt(sim))


def main():
    real_srv = serve(RealStreamingHandler, REAL_PORT)
    sim_srv = serve(SimulatedStreamingHandler, SIM_PORT)
    time.sleep(0.3)  # let the sockets bind

    messages = [{"role": "user", "content": PROMPT}]
    print("auditing real-streaming mock ...")
    r_real = audit("http://127.0.0.1:%d/v1/chat/completions" % REAL_PORT,
                   "mock-real", messages)
    print("auditing simulated-streaming mock ...")
    r_sim = audit("http://127.0.0.1:%d/v1/chat/completions" % SIM_PORT,
                  "mock-sim", messages)

    real_srv.shutdown()
    sim_srv.shutdown()

    ms = lambda v: ("%.1f ms" % (v * 1000.0)) if v is not None else "n/a"

    print()
    print("=" * 78)
    print("STREAMING AUDIT: REAL vs SIMULATED (mock servers, 24 token chunks each)")
    print("=" * 78)
    print("%-22s %-24s %-24s" % ("metric", "REAL (spaced tokens)",
                                 "SIMULATED (burst after delay)"))
    print("-" * 78)
    print(row("chunk count", r_real["chunk_count"], r_sim["chunk_count"], str))
    print(row("TTFB", r_real["ttfb_s"], r_sim["ttfb_s"], ms))
    print(row("total duration", r_real["total_duration_s"],
              r_sim["total_duration_s"], ms))
    print(row("TTFB/total ratio", r_real["ttfb_to_total_ratio"],
              r_sim["ttfb_to_total_ratio"], lambda v: "%.3f" % v))
    print(row("inter-chunk gap p50", r_real["gap_p50_s"],
              r_sim["gap_p50_s"], ms))
    print(row("inter-chunk gap p95", r_real["gap_p95_s"],
              r_sim["gap_p95_s"], ms))
    print(row("inter-chunk gap max", r_real["gap_max_s"],
              r_sim["gap_max_s"], ms))
    print("-" * 78)
    print(row("classifier verdict", r_real["verdict"], r_sim["verdict"],
              lambda v: v.upper()))
    print("=" * 78)
    print()
    print("Heuristic: simulated <=> TTFB > 80% of total AND p95 gap < 5 ms.")
    print("Validated against local mocks only - see README.md for the caveat.")


if __name__ == "__main__":
    main()
