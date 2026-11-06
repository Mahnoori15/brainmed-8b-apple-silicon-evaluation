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
| `MODEL.md` | Model and hardware details |
