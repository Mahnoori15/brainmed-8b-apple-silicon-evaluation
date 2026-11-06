"""Running GGUF models through llama-server and measuring speed and memory."""

import json
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

from .scoring import SYSTEM_PROMPT, build_prompt, extract_answer

_BUFFER_RE = re.compile(r"(\S+)\s+(model|KV|compute) buffer size\s*=\s*([\d.]+)\s*MiB", re.I)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class MemorySampler(threading.Thread):
    """Samples a process's resident memory in the background and keeps the peak."""

    def __init__(self, pid: int, interval: float = 0.25):
        super().__init__(daemon=True)
        self.process = psutil.Process(pid)
        self.interval = interval
        self.peak_rss = 0
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            except psutil.Error:
                break
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)


def parse_buffer_sizes(log_text: str) -> dict:
    """Sum llama.cpp's reported model / KV / compute buffer sizes (MiB)."""
    totals = {"model": 0.0, "kv": 0.0, "compute": 0.0}
    for _device, kind, size in _BUFFER_RE.findall(log_text):
        totals[kind.lower()] += float(size)
    totals["total"] = sum(totals.values())
    return totals


class LlamaServer:
    """Context manager that starts llama-server for one model and stops it after."""

    def __init__(self, binary, model_path, log_path, context=4096, gpu_layers=99,
                 extra_args=(), startup_timeout=300):
        self.binary = binary
        self.model_path = str(Path(model_path).expanduser())
        self.log_path = Path(log_path)
        self.context = context
        self.gpu_layers = gpu_layers
        self.extra_args = list(extra_args)
        self.startup_timeout = startup_timeout
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.process = None
        self.sampler = None
        self.load_seconds = None

    def __enter__(self):
        command = [
            self.binary, "-m", self.model_path,
            "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.context), "-ngl", str(self.gpu_layers),
            *self.extra_args,
        ]
        self._log = open(self.log_path, "w")
        start = time.time()
        self.process = subprocess.Popen(
            command, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
        )
        self.sampler = MemorySampler(self.process.pid)
        self.sampler.start()

        deadline = start + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                self._shutdown()
                raise RuntimeError(
                    f"llama-server exited during startup (code {self.process.returncode}). "
                    f"See {self.log_path}"
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    if response.status == 200:
                        self.load_seconds = time.time() - start
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.5)
        self._shutdown()
        raise RuntimeError(f"llama-server did not become ready in {self.startup_timeout}s")

    def __exit__(self, *exc):
        self._shutdown()

    def _shutdown(self):
        if self.sampler:
            self.sampler.stop()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if getattr(self, "_log", None) and not self._log.closed:
            self._log.close()

    def memory_report(self) -> dict:
        log_text = self.log_path.read_text(errors="replace") if self.log_path.exists() else ""
        buffers = parse_buffer_sizes(log_text)
        return {
            "peak_rss_mib": round(self.sampler.peak_rss / 2**20, 1) if self.sampler else None,
            "llama_buffers_mib": {k: round(v, 1) for k, v in buffers.items()},
        }

    def ask(self, question, chain_of_thought=False, max_tokens=64, use_chat=True,
            timeout=300) -> dict:
        """Ask one question; return the reply text, token counts and timings."""
        prompt = build_prompt(question, chain_of_thought)
        if use_chat:
            endpoint = "/v1/chat/completions"
            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "seed": 0,
            }
        else:
            endpoint = "/completion"
            payload = {
                "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}\n\n",
                "temperature": 0,
                "n_predict": max_tokens,
                "seed": 0,
            }

        request = urllib.request.Request(
            self.url + endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.time()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        latency = time.time() - start

        if use_chat:
            message = body["choices"][0]["message"]
            text = message.get("content") or message.get("reasoning_content") or ""
            output_tokens = (body.get("usage") or {}).get("completion_tokens")
        else:
            text = body.get("content", "")
            output_tokens = body.get("tokens_predicted")

        timings = body.get("timings") or {}
        return {
            "text": text,
            "latency_s": round(latency, 3),
            "output_tokens": output_tokens,
            "gen_tok_s": timings.get("predicted_per_second"),
            "prompt_tok_s": timings.get("prompt_per_second"),
        }


def run_model(model, questions, out_dir, llama_server="llama-server",
              chain_of_thought=False, max_tokens=64, done_ids=(), on_record=None):
    """Run every question for one model. Returns (records, run_info).

    model: dict with "name", "path" and optional "context", "gpu_layers",
           "chat" (default true) and "extra_args".
    done_ids: question ids already answered (for resuming an interrupted run).
    on_record: callback receiving each new record as soon as it is ready.
    """
    out_dir = Path(out_dir)
    log_path = out_dir / f"llama-server_{model['name']}.log"
    use_chat = model.get("chat", True)
    records = []
    todo = [q for q in questions if q["id"] not in set(done_ids)]

    with LlamaServer(
        llama_server, model["path"], log_path,
        context=model.get("context", 4096),
        gpu_layers=model.get("gpu_layers", 99),
        extra_args=model.get("extra_args", []),
    ) as server:
        print(f"  loaded in {server.load_seconds:.1f}s; {len(todo)} questions to run")
        for number, question in enumerate(todo, start=1):
            record = {
                "model": model["name"],
                "id": question["id"],
                "subject": question["subject"],
                "answer": question["answer"],
            }
            try:
                reply = server.ask(question, chain_of_thought, max_tokens, use_chat)
                record.update(reply)
                record["predicted"] = extract_answer(reply["text"], question["options"])
                record["error"] = None
            except Exception as error:  # keep going; the failure is recorded
                record.update({"text": "", "predicted": None, "error": str(error)})
            record["correct"] = record["predicted"] == question["answer"]
            records.append(record)
            if on_record:
                on_record(record)
            if number % 10 == 0 or number == len(todo):
                right = sum(r["correct"] for r in records)
                print(f"  {number}/{len(todo)}  running accuracy {right / number:.1%}")

        run_info = {"load_seconds": round(server.load_seconds, 1), **server.memory_report()}
    return records, run_info
