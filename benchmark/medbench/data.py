"""Loading and sampling MedQA-style multiple-choice questions.

The default dataset is MedQA-USMLE (4-option test split) from Hugging Face
(GBaker/MedQA-USMLE-4-options, CC-BY-4.0; Jin et al., 2020). It is downloaded
at run time and cached locally; it is never committed to this repository.

Each question is a dict:
    {"id": str, "question": str, "options": {"A": ..., "B": ..., ...},
     "answer": "A".."D", "subject": str}
"""

import json
import random
import urllib.request
from pathlib import Path

MEDQA_URL = (
    "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/"
    "resolve/main/phrases_no_exclude_test.jsonl"
)
CACHE_DIR = Path.home() / ".cache" / "medbench"


def download_medqa(cache_dir: Path = CACHE_DIR) -> Path:
    """Download the MedQA test split once and return the local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "medqa_usmle_4opt_test.jsonl"
    if not path.exists():
        print(f"Downloading MedQA test split to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MEDQA_URL, tmp)
        tmp.rename(path)
    return path


def load_questions(path: Path) -> list:
    """Read a JSONL file of MedQA-format records into normalised questions."""
    questions = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            options = record["options"]
            answer = record.get("answer_idx") or record.get("answer")
            if answer not in options:
                raise ValueError(f"{path}:{line_number + 1}: answer {answer!r} not in options")
            questions.append(
                {
                    "id": str(record.get("id", line_number)),
                    "question": record["question"].strip(),
                    "options": {key: str(value).strip() for key, value in sorted(options.items())},
                    "answer": answer,
                    "subject": record.get("meta_info") or "unknown",
                }
            )
    return questions


def sample_questions(questions: list, n: int, seed: int = 42) -> list:
    """Pick a reproducible random sample of n questions (all if n >= len)."""
    if n >= len(questions):
        return list(questions)
    return random.Random(seed).sample(questions, n)
