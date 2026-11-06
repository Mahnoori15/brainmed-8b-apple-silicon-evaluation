"""Turn prediction records into summary.json, report.md and charts."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .scoring import pairwise_comparisons, summarise

# Categorical palette (validated; fixed order, colour follows the model's
# position in models.toml). More than three models fold back to slot order,
# so keep comparisons to three models per chart for best colour separation.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _fmt(value, pattern="{:.1f}", missing="–"):
    return missing if value is None else pattern.format(value)


def build_summary(records_by_model: dict, run_info: dict, settings: dict) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings,
        "models": {
            name: {**summarise(records), **run_info.get(name, {})}
            for name, records in records_by_model.items()
        },
        "comparisons": pairwise_comparisons(records_by_model),
    }


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar_chart(path, names, values, title, xlabel, labels, errors=None, reference=None,
               reference_label=None, xmax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 0.7 * len(names) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    positions = list(range(len(names)))[::-1]
    colors = [SERIES[i % len(SERIES)] for i in range(len(names))]
    ax.barh(positions, values, height=0.5, color=colors, edgecolor=SURFACE, linewidth=2)
    if errors:
        ax.errorbar(values, positions, xerr=errors, fmt="none", ecolor=TEXT_PRIMARY,
                    elinewidth=1.2, capsize=4)
    if reference is not None:
        ax.axvline(reference, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(reference, max(positions) + 0.6, f" {reference_label}", color=TEXT_SECONDARY,
                fontsize=8, va="center")
    top = xmax or max([v for v in values if v] + [1]) * 1.25
    for i, (pos, value, label) in enumerate(zip(positions, values, labels)):
        offset = (errors[1][i] if errors else 0) + top * 0.015
        ax.text(value + offset, pos, label, va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_yticks(positions)
    ax.set_yticklabels(names, color=TEXT_PRIMARY, fontsize=10)
    ax.set_xlim(0, top)
    ax.set_ylim(-0.6, max(positions) + (0.9 if reference is not None else 0.6))
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def make_charts(summary: dict, out_dir: Path) -> list:
    """Draw accuracy, speed and memory charts. Returns the file names written."""
    names = list(summary["models"])
    models = summary["models"]
    written = []

    acc = [models[n]["accuracy"] * 100 for n in names]
    lows = [acc[i] - models[n]["ci95"][0] * 100 for i, n in enumerate(names)]
    highs = [models[n]["ci95"][1] * 100 - acc[i] for i, n in enumerate(names)]
    chance = 25.0  # 4-option questions
    _bar_chart(out_dir / "accuracy.png", names, acc, "MedQA accuracy (95% CI)", "Accuracy (%)",
               [f"{a:.1f}%" for a in acc], errors=[lows, highs], reference=chance,
               reference_label="random guess", xmax=100)
    written.append("accuracy.png")

    speed = [models[n].get("median_generation_tok_s") for n in names]
    if any(speed):
        _bar_chart(out_dir / "speed.png", names, [s or 0 for s in speed],
                   "Generation speed (median)", "Tokens per second",
                   [_fmt(s, "{:.1f} tok/s") for s in speed])
        written.append("speed.png")

    memory = [(models[n].get("llama_buffers_mib") or {}).get("total") or
              models[n].get("peak_rss_mib") for n in names]
    if any(memory):
        gib = [(m or 0) / 1024 for m in memory]
        _bar_chart(out_dir / "memory.png", names, gib, "Memory used by llama.cpp",
                   "GiB (model + KV cache + compute buffers)", [f"{g:.2f} GiB" for g in gib])
        written.append("memory.png")
    return written


def write_markdown(summary: dict, charts: list, path: Path):
    settings = summary["settings"]
    models = summary["models"]
    lines = [
        "# Medical LLM benchmark report",
        "",
        f"Generated {summary['generated_at']} · dataset: {settings['dataset']} · "
        f"{settings['questions']} questions (seed {settings['seed']}) · "
        f"mode: {'chain-of-thought' if settings['chain_of_thought'] else 'direct answer'} · "
        f"max tokens: {settings['max_tokens']}",
        "",
        "## Results",
        "",
        "| Model | Accuracy | 95% CI | Invalid answers | Median gen speed | Median latency | Load time | Memory (llama.cpp) | Peak RSS |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in models.items():
        buffers = (m.get("llama_buffers_mib") or {}).get("total")
        lines.append(
            f"| {name} | **{m['accuracy']:.1%}** ({m['correct']}/{m['questions']}) "
            f"| {m['ci95'][0]:.1%}–{m['ci95'][1]:.1%} "
            f"| {m['invalid_answers']} "
            f"| {_fmt(m.get('median_generation_tok_s'), '{:.1f} tok/s')} "
            f"| {_fmt(m.get('median_latency_s'), '{:.2f} s')} "
            f"| {_fmt(m.get('load_seconds'), '{:.1f} s')} "
            f"| {_fmt(buffers / 1024 if buffers else None, '{:.2f} GiB')} "
            f"| {_fmt(m.get('peak_rss_mib') / 1024 if m.get('peak_rss_mib') else None, '{:.2f} GiB')} |"
        )
    lines += ["", "Random guessing scores 25% on 4-option questions.", ""]

    for chart in charts:
        lines += [f"![{chart[:-4]}]({chart})", ""]

    subjects = sorted({s for m in models.values() for s in m["by_subject"]})
    if len(subjects) > 1:
        lines += ["## Accuracy by exam step", "", "| Model | " + " | ".join(subjects) + " |",
                  "| --- |" + " --- |" * len(subjects)]
        for name, m in models.items():
            cells = []
            for subject in subjects:
                s = m["by_subject"].get(subject)
                cells.append(f"{s['accuracy']:.1%} ({s['correct']}/{s['total']})" if s else "–")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")

    if summary["comparisons"]:
        lines += [
            "## Is the difference real?",
            "",
            "Exact McNemar test on the questions both models answered. "
            "p < 0.05 suggests the accuracy gap is unlikely to be chance.",
            "",
            "| Model A | Model B | Only A right | Only B right | p-value |",
            "| --- | --- | --- | --- | --- |",
        ]
        for c in summary["comparisons"]:
            lines.append(
                f"| {c['model_a']} | {c['model_b']} | {c['only_a_correct']} "
                f"| {c['only_b_correct']} | {c['p_value']:.3f} |"
            )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Answers are extracted automatically; replies with no recognisable "
        "option letter count as wrong and are listed as invalid.",
        "- Memory (llama.cpp) is the model, KV-cache and compute buffer sizes "
        "llama.cpp reports at load time. On Apple Silicon, GPU (Metal) memory is "
        "shared with the CPU and memory-mapped weights may not all show up in "
        "peak RSS, so the llama.cpp figure is the more reliable one.",
        "- This is a research benchmark on exam-style questions, not a measure of "
        "clinical safety.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_report(records_by_model: dict, run_info: dict, settings: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    summary = build_summary(records_by_model, run_info, settings)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    charts = make_charts(summary, out_dir)
    write_markdown(summary, charts, out_dir / "report.md")
    return summary
