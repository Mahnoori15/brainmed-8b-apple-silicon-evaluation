# BrainMed-8B: Apple Silicon Local Evaluation

## Overview

This repository documents my individual work deploying and evaluating
BrainMed-8B locally on an Apple Silicon Mac.

BrainMed-8B was developed as part of a team project. This repository
does not claim ownership of the original BrainMed-8B model or the
original team repository.

My focus in this repository is local deployment, GGUF inference,
medical question testing, performance observation, and error analysis.

---

## My Contributions

My individual work includes:

- Setting up the BrainMed environment on Apple Silicon
- Investigating Python and PyTorch compatibility
- Testing Apple Silicon MPS support
- Obtaining and testing the Q4_K_M GGUF version
- Running BrainMed-8B locally with llama.cpp
- Designing and running a small medical smoke test
- Evaluating model responses
- Analysing incorrect and partially correct answers
- Recording local inference performance
- Documenting the deployment process

---

## Model

**Model:** BrainMed-8B

**Quantization:** Q4_K_M GGUF

**Model size:** approximately 4.6 GB

The GGUF model is not included in this repository because of its
large file size.

The model was stored locally at:

```text
~/Models/BrainMed-8B/BrainMed-8B.Q4_K_M.gguf
```

---

## Hardware

- Apple Silicon MacBook Air (ARM64)
- 16 GB RAM
- Apple Metal / MPS acceleration

---

## How to Run

1. Install llama.cpp:

   ```bash
   brew install llama.cpp
   ```

2. Run the smoke test:

   ```bash
   python scripts/evaluate_brainmed.py --model ~/Models/BrainMed-8B/BrainMed-8B.Q4_K_M.gguf
   ```

   You can also set the model path with the `BRAINMED_MODEL` environment
   variable, and a different llama.cpp binary with `--llama-bin` or `LLAMA_BIN`.

The script asks four clinical questions (acute chest pain, eclampsia,
Kawasaki disease and a patient-safety question) with greedy decoding
(`--temp 0`), and saves:

- `results/evaluation_results.txt`: a readable report
- `results/evaluation_results.json`: responses, timings, exit codes and llama.cpp logs

If llama.cpp fails or returns nothing, the error and the last lines of its
log are recorded for that question, so problems can be diagnosed.

---

## Repository Structure

| Path | Purpose |
| --- | --- |
| `scripts/evaluate_brainmed.py` | Smoke-test evaluation script |
| `results/` | Evaluation output |
| `benchmark/` | MedQA multi-model benchmark |
| `MODEL.md` | Model and hardware details |

---

## MedQA Benchmark

`benchmark/` grows the four-question smoke test into a proper, repeatable
benchmark that compares several local models on 100–200 USMLE-style
questions from MedQA.

**What it measures for each model**

- **Accuracy**, with a 95% confidence interval (Wilson), overall and by exam step
- **Invalid answers**: replies where no option letter could be extracted
- **Speed**: generation and prompt-processing tokens/second, latency per question, load time
- **Memory**: model, KV-cache and compute buffers reported by llama.cpp, plus peak process RSS
- **Significance**: exact McNemar test for each pair of models, to show whether a
  gap in accuracy is likely to be real or just noise from the sample size

**How it works**

Each model is loaded once in `llama-server` (llama.cpp) with Metal offload.
Questions are sent through its chat API at temperature 0, and the answer letter is
extracted automatically. The same seeded random sample of questions is used for
every model, so the comparison is paired and reproducible.

The MedQA test split ([Jin et al., 2020](https://arxiv.org/abs/2009.13081);
`GBaker/MedQA-USMLE-4-options` on Hugging Face, CC-BY-4.0) is downloaded on the first
run and cached in `~/.cache/medbench`. It is not stored in this repository.

### Run it

Requires Python 3.11+ and llama.cpp (`brew install llama.cpp`).

```bash
pip install -r benchmark/requirements.txt
cp benchmark/models.example.toml benchmark/models.toml   # edit the GGUF paths

# quick check: 10 questions, one model
python benchmark/run_benchmark.py --models benchmark/models.toml --n 10 --only BrainMed-8B

# full comparison: 150 questions, all models
python benchmark/run_benchmark.py --models benchmark/models.toml --n 150
```

Options: `--cot` asks for step-by-step reasoning before the answer (slower, often
more accurate); `--seed` changes the question sample; `--max-tokens` caps reply length.

Results go to `results/benchmark/<date_time>/`:
`report.md` (tables and charts), `summary.json`, `predictions.jsonl` (every
reply) and the llama-server logs. If a run is interrupted, re-run the same
command with `--out` pointing at that folder and it carries on where it stopped.

For a fair comparison, use the same quantisation (e.g. Q4_K_M) for every model.

### Tests

```bash
python -m pytest benchmark/tests
```

The tests run the full pipeline against a fake `llama-server`, so they need no
model files. They also run automatically on GitHub for every pull request.

> This benchmark measures performance on exam-style questions. It is not an
> assessment of clinical safety, and no model here should be used for medical
> decisions.
