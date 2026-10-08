"""
enerdatasynth.resources.wind.evaluation.results_report
==========================================================
Wind's counterpart to `resources.pjm_load.evaluation.results_report` — regenerates a
results-summary markdown, added for parity. Built entirely from
`resources/wind/manifest.json`'s real, already-measured numbers (from the upstream
project's own validation) — no new comparison is run, since there's no real wind data
in this repository to compare against (see `docs/models/wind.md` §2).
"""

from __future__ import annotations

from pathlib import Path

from enerdatasynth.utils.config import DATA_ROOT
from enerdatasynth.resources.wind.api import list_models
from enerdatasynth.resources.wind.evaluation.failure_analysis import analyze


def generate_results_summary() -> Path:
    models = list_models()
    out_dir = DATA_ROOT / "reports" / "wind"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Wind Results Summary",
        "",
        "Generated from `resources/wind/manifest.json` — real numbers measured by the "
        "upstream project, not a new comparison run in this repository (there is no "
        "real wind data here to compare against; see "
        "[`docs/models/wind.md`](../../../docs/models/wind.md) §2).",
        "",
    ]

    for preset in sorted(models):
        entry = models[preset]
        metrics = entry.get("metrics", {})
        lines += [
            f"## Preset `{preset}`",
            "",
            f"- Channels: {entry.get('channel_description', 'n/a')}",
            f"- Sequence length: {entry.get('seq_len')} steps @ {entry.get('freq')}",
        ]
        if metrics:
            lines += [
                f"- Measured: {metrics.get('measured_at', 'n/a')}",
                f"- Training recipe: {metrics.get('training_recipe', 'n/a')}",
                f"- Generation settings: {metrics.get('generation_settings', 'n/a')}",
                "",
            ]
            gates = metrics.get("gates_by_regime", {})
            if gates:
                lines.append("| Regime | marginal_wasserstein | acf_rmse | regime_accuracy |")
                lines.append("|---|---|---|---|")
                for regime, gate in gates.items():
                    lines.append(
                        f"| {regime} | {gate.get('marginal_wasserstein', float('nan')):.3f} "
                        f"| {gate.get('acf_rmse', float('nan')):.3f} "
                        f"| {gate.get('regime_accuracy', float('nan')):.2f} |"
                    )
                lines.append("")
        else:
            # This preset's manifest entry has no structured `metrics` dict
            # (a real inconsistency in the vendored manifest, not fixed here —
            # see failure_analysis.py's comment). Its real numbers live only in
            # the entry's own known_limitation narrative, quoted below instead
            # of silently showing "n/a" for everything.
            lines.append(
                "_No structured metrics for this preset in the manifest — its real, "
                "measured history is only in the known-limitation narrative:_"
            )
            lines.append("")
            if entry.get("known_limitation"):
                lines.append(f"> {entry['known_limitation']}")
                lines.append("")

        failure_result = analyze(preset)
        lines.append(f"**Flagged issues:** {len(failure_result['issues'])} "
                     f"(see `failure_analysis_{preset}.md`)")
        lines.append("")

    out_path = out_dir / "results_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


if __name__ == "__main__":
    path = generate_results_summary()
    print(f"Saved: {path}")
