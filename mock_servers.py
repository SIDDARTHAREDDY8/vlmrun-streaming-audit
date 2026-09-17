#!/usr/bin/env python3
"""Two tiny stdlib-only SSE mock servers for validating stream_audit.py.

  * RealStreamingHandler  - emits one token chunk at a time with a steady
                            ~45 ms cadence, like a native vLLM chat model.
  * SimulatedStreamingHandler - waits ~1.2 s ("generating" the full reply),
                            then re-chunks the finished reply and flushes it
                            in a single burst, mimicking a gateway that
                            simulated-streams OCR/adapter model responses.

Both speak the OpenAI chat-completions SSE wire format
(`data: {...}` per chunk, `data: [DONE]` terminator) and accept any POST body.

Run standalone to expose both servers for manual probing:

    python3 mock_servers.py                 # :18080 real, :18081 simulated

Or import and use serve() from run_demo.py:

    from mock_servers import serve, RealStreamingHandler, SimulatedStreamingHandler
    srv = serve(RealStreamingHandler, 18080)
    ...
    srv.shutdown()
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKENS = [" The", " quick", " brown", " fox", " jumps", " over",
          " the", " lazy", " dog", " in", " the", " moon", "light", "."]
N_TOKENS = 24
REAL_FIRST_TOKEN_DELAY_S = 0.15   # prefill-ish pause before token 1
REAL_TOKEN_GAP_S = 0.045          # steady decode cadence
SIMULATED_GENERATION_DELAY_S = 1.2  # "generate whole reply", then burst


def _chunk(i):
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mock",
        "choices": [{"index": 0,
                     "delta": {"content": TOKENS[i % len(TOKENS)]},
                     "finish_reason": None}],
    }
    return ("data: %s\n\n" % json.dumps(payload)).encode("utf-8")


DONE = b"data: [DONE]\n\n"


class _BaseHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited body: simplest reliable SSE

    def _drain_and_headers(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def log_message(self, *args):  # keep demo output clean
        pass


class RealStreamingHandler(_BaseHandler):
    """Genuine token-by-token streaming: spaced chunks across the window."""

    def do_POST(self):
        self._drain_and_headers()
        time.sleep(REAL_FIRST_TOKEN_DELAY_S)
        for i in range(N_TOKENS):
            self.wfile.write(_chunk(i))
            self.wfile.flush()
            time.sleep(REAL_TOKEN_GAP_S)
        self.wfile.write(DONE)
        self.wfile.flush()


class SimulatedStreamingHandler(_BaseHandler):
    """Simulated streaming: long wait, then the whole re-chunked reply at once."""

    def do_POST(self):
        self._drain_and_headers()
        # The "model" finishes generating before the first byte is sent...
        time.sleep(SIMULATED_GENERATION_DELAY_S)
        # ...then the gateway re-chunks the finished reply into one burst.
        burst = b"".join(_chunk(i) for i in range(N_TOKENS))
        self.wfile.write(burst)
        self.wfile.flush()
        self.wfile.write(DONE)
        self.wfile.flush()


def serve(handler_cls, port):
    """Start a mock server in a daemon thread; returns the server object."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the SSE mock servers.")
    ap.add_argument("--real-port", type=int, default=18080)
    ap.add_argument("--sim-port", type=int, default=18081)
    args = ap.parse_args(argv)

    real = serve(RealStreamingHandler, args.real_port)
    sim = serve(SimulatedStreamingHandler, args.sim_port)
    print("real-streaming mock      : http://127.0.0.1:%d/v1/chat/completions"
          % args.real_port)
    print("simulated-streaming mock : http://127.0.0.1:%d/v1/chat/completions"
          % args.sim_port)
    print("Ctrl-C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        real.shutdown()
        sim.shutdown()


if __name__ == "__main__":
    main()
