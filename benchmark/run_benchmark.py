"""
Benchmark local GGUF medical LLMs on MedQA-style questions.

Examples:
    # 150 questions, every model in models.toml
    python benchmark/run_benchmark.py --models benchmark/models.toml --n 150

    # quick check with 10 questions and one model
    python benchmark/run_benchmark.py --models benchmark/models.toml --n 10 --only BrainMed-8B

    # rebuild the report from an earlier run without re-running models
    python benchmark/run_benchmark.py --report-only results/benchmark/2026-11-06_1200

Runs are resumable: if a run is interrupted, re-run the same command with
--out pointing at the same folder and finished questions are skipped.
"""

import argparse
import json
import shutil
import sys
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from medbench.data import download_medqa, load_questions, sample_questions  # noqa: E402
from medbench.report import write_report  # noqa: E402
from medbench.runner import run_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, help="TOML file listing the models to compare")
    parser.add_argument("--n", type=int, default=150, help="Number of questions (default 150)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42)")
    parser.add_argument("--cot", action="store_true", help="Ask for step-by-step reasoning first")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens per answer (default 64, or 1024 with --cot)")
    parser.add_argument("--only", nargs="+", help="Run only these model names")
    parser.add_argument("--dataset-file", type=Path,
                        help="Local JSONL in MedQA format instead of downloading MedQA")
    parser.add_argument("--llama-server", default="llama-server", help="Path to llama-server")
    parser.add_argument("--out", type=Path, help="Output folder (default results/benchmark/<timestamp>)")
    parser.add_argument("--report-only", type=Path, metavar="RUN_DIR",
                        help="Rebuild the report from an existing run folder")
    return parser.parse_args()


def load_predictions(path: Path) -> dict:
    by_model = defaultdict(list)
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                by_model[record["model"]].append(record)
    return dict(by_model)


def main():
    args = parse_args()

    if args.report_only:
        run_dir = args.report_only
        state = json.loads((run_dir / "run.json").read_text())
        records = load_predictions(run_dir / "predictions.jsonl")
        records = {name: records[name] for name in state["model_order"] if name in records}
        write_report(records, state["run_info"], state["settings"], run_dir)
        print(f"Report written to {run_dir / 'report.md'}")
        return

    if not args.models:
        sys.exit("--models is required (see benchmark/models.example.toml)")
    config = tomllib.loads(args.models.read_text())
    models = config.get("models", [])
    if args.only:
        models = [m for m in models if m["name"] in args.only]
    if not models:
        sys.exit("No models to run. Check the [[models]] entries and --only names.")
    for model in models:
        if not Path(model["path"]).expanduser().is_file():
            sys.exit(f"Model file not found for {model['name']}: {model['path']}")
    if shutil.which(args.llama_server) is None:
        sys.exit(f"llama-server not found ({args.llama_server}). Install with `brew install llama.cpp`.")

    max_tokens = args.max_tokens or (1024 if args.cot else 64)
    dataset_path = args.dataset_file or download_medqa()
    questions = sample_questions(load_questions(dataset_path), args.n, args.seed)

    out_dir = args.out or REPO_ROOT / "results" / "benchmark" / datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    state_path = out_dir / "run.json"

    settings = {
        "dataset": "custom file" if args.dataset_file else "MedQA-USMLE 4-option (test)",
        "questions": len(questions),
        "seed": args.seed,
        "chain_of_thought": args.cot,
        "max_tokens": max_tokens,
    }
    state = json.loads(state_path.read_text()) if state_path.exists() else {"run_info": {}}
    if state.get("settings") and state["settings"] != settings:
        sys.exit(f"{out_dir} holds a run with different settings; choose another --out folder.")
    state["settings"] = settings
    state["model_order"] = [m["name"] for m in models]

    existing = load_predictions(predictions_path)
    print(f"Benchmark: {len(questions)} questions, {len(models)} model(s) -> {out_dir}")

    with open(predictions_path, "a") as predictions:
        def save(record):
            predictions.write(json.dumps(record) + "\n")
            predictions.flush()

        for model in models:
            name = model["name"]
            done = {r["id"] for r in existing.get(name, [])}
            if len(done) >= len(questions):
                print(f"\n{name}: already complete, skipping")
                continue
            print(f"\n{name}")
            try:
                _, run_info = run_model(
                    model, questions, out_dir, args.llama_server,
                    chain_of_thought=args.cot, max_tokens=max_tokens,
                    done_ids=done, on_record=save,
                )
                state["run_info"][name] = run_info
            except RuntimeError as error:
                print(f"  FAILED: {error}")
            state_path.write_text(json.dumps(state, indent=2))

    records = load_predictions(predictions_path)
    question_ids = {q["id"] for q in questions}
    records = {
        m["name"]: [r for r in records[m["name"]] if r["id"] in question_ids]
        for m in models if m["name"] in records
    }
    if not records:
        sys.exit("No results to report.")
    summary = write_report(records, state["run_info"], settings, out_dir)

    print("\nResults")
    for name, m in summary["models"].items():
        print(f"  {name:<24} {m['accuracy']:6.1%}  "
              f"(95% CI {m['ci95'][0]:.1%}-{m['ci95'][1]:.1%}, {m['invalid_answers']} invalid)")
    print(f"\nReport: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
