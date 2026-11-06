"""A tiny stand-in for llama-server used by the tests.

Behaviour depends on the model file's name:
    *always_a*  -> replies "Answer: A"
    *always_b*  -> reasons, mentions other letters, ends with "Answer: B"
    *no_letter* -> replies without any option letter
    *crash*     -> exits immediately (simulates a failed model load)
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
model = args[args.index("-m") + 1]
port = int(args[args.index("--port") + 1])

if "crash" in model:
    print("error: failed to load model", flush=True)
    sys.exit(1)

print("load_tensors: Metal_Mapped model buffer size =  4403.49 MiB", flush=True)
print("llama_kv_cache: Metal KV buffer size =   512.00 MiB", flush=True)
print("sched_reserve: Metal compute buffer size =   258.50 MiB", flush=True)

if "always_a" in model:
    reply = "Answer: A"
elif "always_b" in model:
    reply = "Option A seems plausible and C is possible, but on balance...\nAnswer: B"
else:
    reply = "I would refer this patient to a specialist."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        json.loads(self.rfile.read(length))
        timings = {"predicted_per_second": 21.5, "prompt_per_second": 310.0}
        if self.path == "/v1/chat/completions":
            self._send({
                "choices": [{"message": {"role": "assistant", "content": reply}}],
                "usage": {"completion_tokens": 5},
                "timings": timings,
            })
        elif self.path == "/completion":
            self._send({"content": reply, "tokens_predicted": 5, "timings": timings})
        else:
            self._send({"error": "not found"}, 404)


time.sleep(0.3)  # pretend to load
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
