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