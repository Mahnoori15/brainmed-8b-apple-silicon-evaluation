"""Prompting, answer extraction and statistics."""

import math
import re

SYSTEM_PROMPT = (
    "You are a medical expert answering USMLE-style multiple-choice questions."
)

DIRECT_INSTRUCTION = (
    "Choose the single best answer. Reply with the letter only, in the form "
    "'Answer: X'."
)

COT_INSTRUCTION = (
    "Think through the question step by step, then finish with a final line "
    "in the form 'Answer: X', where X is the letter of the single best answer."
)


def build_prompt(question: dict, chain_of_thought: bool = False) -> str:
    """Format a question and its options as the user message."""
    options = "\n".join(f"{key}. {text}" for key, text in question["options"].items())
    instruction = COT_INSTRUCTION if chain_of_thought else DIRECT_INSTRUCTION
    return f"{question['question']}\n\n{options}\n\n{instruction}"


_ANSWER_PATTERNS = [
    # "Answer: B", "final answer is (B)", "answer - B."
    re.compile(r"answer\s*(?:is|:|-)?\s*(?:option\s*)?\(?\**([A-E])\**\)?(?![A-Za-z])", re.I),
    # "B) ..." / "(B)" / "B." at the very start of the reply
    re.compile(r"^\s*\(?\**([A-E])\**[\).:\s]", re.I),
    # the reply is just a letter
    re.compile(r"^\s*\(?([A-E])\)?\s*$", re.I),
]


def extract_answer(text: str, options: dict) -> str | None:
    """Return the chosen option letter, or None if no answer can be found.

    The last "Answer: X" in the reply wins (so reasoning that mentions other
    letters earlier doesn't count). If no letter is found, an exact match of
    an option's text is accepted as a fallback.
    """
    if not text:
        return None
    text = text.strip()
    valid = set(options)

    matches = [m.group(1).upper() for m in _ANSWER_PATTERNS[0].finditer(text)]
    matches = [letter for letter in matches if letter in valid]
    if matches:
        return matches[-1]

    for pattern in _ANSWER_PATTERNS[1:]:
        match = pattern.search(text)
        if match and match.group(1).upper() in valid:
            return match.group(1).upper()

    lowered = text.lower()
    hits = [key for key, value in options.items() if value.lower() in lowered]
    if len(hits) == 1:
        return hits[0]
    return None


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    only_a: questions model A got right and model B got wrong (and vice versa).
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def median(values: list) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def summarise(records: list) -> dict:
    """Aggregate one model's prediction records into summary metrics."""
    total = len(records)
    correct = sum(r["correct"] for r in records)
    invalid = sum(r["predicted"] is None for r in records)
    errors = sum(bool(r.get("error")) for r in records)
    low, high = wilson_interval(correct, total)

    by_subject = {}
    for record in records:
        bucket = by_subject.setdefault(record["subject"], [0, 0])
        bucket[0] += record["correct"]
        bucket[1] += 1

    return {
        "questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "ci95": [low, high],
        "invalid_answers": invalid,
        "request_errors": errors,
        "by_subject": {
            subject: {"correct": c, "total": t, "accuracy": c / t}
            for subject, (c, t) in sorted(by_subject.items())
        },
        "median_latency_s": median([r.get("latency_s") for r in records]),
        "median_generation_tok_s": median([r.get("gen_tok_s") for r in records]),
        "median_prompt_tok_s": median([r.get("prompt_tok_s") for r in records]),
        "mean_output_tokens": (
            sum(r.get("output_tokens") or 0 for r in records) / total if total else 0
        ),
    }


def pairwise_comparisons(records_by_model: dict) -> list:
    """McNemar comparisons for every pair of models on their shared questions."""
    names = list(records_by_model)
    results = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            right_a = {r["id"]: r["correct"] for r in records_by_model[a]}
            right_b = {r["id"]: r["correct"] for r in records_by_model[b]}
            shared = right_a.keys() & right_b.keys()
            only_a = sum(right_a[q] and not right_b[q] for q in shared)
            only_b = sum(right_b[q] and not right_a[q] for q in shared)
            results.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "shared_questions": len(shared),
                    "only_a_correct": only_a,
                    "only_b_correct": only_b,
                    "p_value": mcnemar_exact(only_a, only_b),
                }
            )
    return results
