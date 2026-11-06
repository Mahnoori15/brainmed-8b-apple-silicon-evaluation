import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medbench.runner import parse_buffer_sizes  # noqa: E402
from medbench.scoring import (  # noqa: E402
    build_prompt,
    extract_answer,
    mcnemar_exact,
    summarise,
    wilson_interval,
)

OPTIONS = {"A": "Aspirin", "B": "Magnesium sulfate", "C": "Diazepam", "D": "Phenytoin"}


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("Answer: B", "B"),
        ("answer: b", "B"),
        ("The answer is (C).", "C"),
        ("Final answer: **D**", "D"),
        ("B", "B"),
        ("(A)", "A"),
        ("B. Magnesium sulfate is first line.", "B"),
        ("A is wrong and C is wrong.\nAnswer: B", "B"),
        ("Answer: A\nOn reflection, answer: D", "D"),
        ("Magnesium sulfate", "B"),
        ("I would refer to a specialist.", None),
        ("Answer: E", None),  # not one of the options
        ("", None),
    ],
)
def test_extract_answer(reply, expected):
    assert extract_answer(reply, OPTIONS) == expected


def test_prompt_lists_options_and_instruction():
    prompt = build_prompt({"question": "Q?", "options": OPTIONS})
    assert "B. Magnesium sulfate" in prompt
    assert "Answer: X" in prompt


def test_wilson_interval():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # 10 vs 0 discordant pairs: p = 2 * 0.5**10
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


def test_summarise():
    records = [
        {"id": "1", "subject": "step1", "predicted": "A", "correct": True, "latency_s": 1.0, "gen_tok_s": 20},
        {"id": "2", "subject": "step1", "predicted": None, "correct": False, "latency_s": 3.0, "gen_tok_s": 30},
        {"id": "3", "subject": "step2&3", "predicted": "C", "correct": True, "latency_s": 2.0, "gen_tok_s": 10},
    ]
    s = summarise(records)
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["invalid_answers"] == 1
    assert s["by_subject"]["step1"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert s["median_latency_s"] == 2.0
    assert s["median_generation_tok_s"] == 20


def test_parse_buffer_sizes():
    log = (
        "load_tensors: Metal_Mapped model buffer size =  4403.49 MiB\n"
        "load_tensors:   CPU_Mapped model buffer size =   281.81 MiB\n"
        "llama_kv_cache: Metal KV buffer size =   512.00 MiB\n"
        "sched_reserve: Metal compute buffer size =   258.50 MiB\n"
    )
    sizes = parse_buffer_sizes(log)
    assert sizes["model"] == pytest.approx(4685.30)
    assert sizes["kv"] == 512.0
    assert sizes["total"] == pytest.approx(5455.80)
