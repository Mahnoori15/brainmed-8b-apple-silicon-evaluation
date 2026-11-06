"""Runs the whole benchmark against fake llama-server processes."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "run_benchmark.py"


def make_fixture(tmp_path):
    """Ten made-up questions: six with answer A, four with answer B."""
    path = tmp_path / "questions.jsonl"
    with open(path, "w") as handle:
        for i in range(10):
            handle.write(json.dumps({
                "question": f"Synthetic test question {i}?",
                "options": {"A": f"alpha {i}", "B": f"bravo {i}", "C": f"charlie {i}", "D": f"delta {i}"},
                "answer_idx": "A" if i < 6 else "B",
                "meta_info": "step1" if i % 2 else "step2&3",
            }) + "\n")
    return path


def make_server_wrapper(tmp_path):
    wrapper = tmp_path / "llama-server"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_llama_server.py'} \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return wrapper


def make_models(tmp_path, names):
    lines = []
    for name in names:
        model_file = tmp_path / f"{name}.gguf"
        model_file.write_bytes(b"")
        lines += ["[[models]]", f'name = "{name}"', f'path = "{model_file}"', ""]
    config = tmp_path / "models.toml"
    config.write_text("\n".join(lines))
    return config


def run(args, tmp_path):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=120, cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_full_run(tmp_path):
    dataset = make_fixture(tmp_path)
    server = make_server_wrapper(tmp_path)
    config = make_models(tmp_path, ["always_a", "always_b", "no_letter", "crash"])
    out = tmp_path / "run"

    result = run(["--models", str(config), "--dataset-file", str(dataset), "--n", "10",
                  "--llama-server", str(server), "--out", str(out)], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAILED" in result.stdout  # the crashing model is reported, not fatal

    summary = json.loads((out / "summary.json").read_text())
    models = summary["models"]
    assert set(models) == {"always_a", "always_b", "no_letter"}
    assert models["always_a"]["accuracy"] == 0.6
    assert models["always_b"]["accuracy"] == 0.4
    assert models["no_letter"]["accuracy"] == 0.0
    assert models["no_letter"]["invalid_answers"] == 10
    assert models["always_a"]["median_generation_tok_s"] == 21.5
    assert models["always_a"]["llama_buffers_mib"]["total"] == 5174.0
    assert models["always_a"]["peak_rss_mib"] > 0

    pair = next(c for c in summary["comparisons"]
                if {c["model_a"], c["model_b"]} == {"always_a", "always_b"})
    assert (pair["only_a_correct"], pair["only_b_correct"]) == (6, 4)

    report = (out / "report.md").read_text()
    assert "| always_a | **60.0%** (6/10)" in report
    for chart in ("accuracy.png", "speed.png", "memory.png"):
        assert (out / chart).stat().st_size > 1000

    # Re-running the same command resumes: nothing is asked twice.
    lines_before = len((out / "predictions.jsonl").read_text().splitlines())
    rerun = run(["--models", str(config), "--dataset-file", str(dataset), "--n", "10",
                 "--llama-server", str(server), "--out", str(out), "--only", "always_a"], tmp_path)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert "already complete" in rerun.stdout
    assert len((out / "predictions.jsonl").read_text().splitlines()) == lines_before

    # The report can be rebuilt without running models.
    (out / "report.md").unlink()
    rebuilt = run(["--report-only", str(out)], tmp_path)
    assert rebuilt.returncode == 0, rebuilt.stdout + rebuilt.stderr
    assert (out / "report.md").exists()
