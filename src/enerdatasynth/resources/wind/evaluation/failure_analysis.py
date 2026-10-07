"""
enerdatasynth.resources.wind.evaluation.failure_analysis
============================================================
Wind's counterpart to `resources.pjm_load.evaluation.failure_analysis` — automated
failure-mode flagging, added for parity (PJM load had this; wind didn't).

**This does not run a new comparison against real data — there is none to compare
against (see `docs/models/wind.md` §2).** Instead it flags issues already present in
`resources/wind/manifest.json`'s real, measured fidelity-gate numbers (from the
original upstream project's own validation), against the same kind of documented
thresholds PJM load's `failure_analysis.py` uses. Nothing here is a new measurement;
it's automation of a check a human would otherwise do by reading the manifest by eye.
"""

from __future__ import annotations

import json
from pathlib import Path

from enerdatasynth.utils.config import DATA_ROOT
from enerdatasynth.resources.wind.api import list_models

# Thresholds — documented, adjustable. `regime_accuracy` for a 3-way bucket
# (low/mid/high) has a 1/3 no-conditioning-effect chance floor; anything at or
# below that means the conditioning isn't steering generation at all.
REGIME_ACCURACY_CHANCE_FLOOR = 1.0 / 3.0
REGIME_ACCURACY_WARN_MARGIN = 0.10  # within 10pp of chance is still "not really working"


def analyze(preset: str) -> dict:
    """Flag known-limitation-style issues in `preset`'s manifest entry."""
    entry = list_models()[preset]
    metrics = entry.get("metrics", {})
    gates = metrics.get("gates_by_regime", {})

    issues = []
    for regime, gate in gates.items():
        acc = gate.get("regime_accuracy")
        if acc is not None and acc <= REGIME_ACCURACY_CHANCE_FLOOR + REGIME_ACCURACY_WARN_MARGIN:
            issues.append({
                "issue": f"Regime steering weak or absent ({regime})",
                "metric": (
                    f"regime_accuracy={acc:.2f} for '{regime}' "
                    f"(chance floor for 3-way conditioning is {REGIME_ACCURACY_CHANCE_FLOOR:.2f})"
                ),
                "possible_cause": (
                    "Conditioning signal for wind_regime is too weak relative to the "
                    "generation settings, or (for 'mid' specifically) the bucket sits "
                    "at the boundary between 'low' and 'high' in a way the model "
                    "consistently confuses — a documented, unresolved issue across "
                    "every generation-setting configuration tried (see manifest's own "
                    "known_limitation field for this preset)."
                ),
                "potential_future_fix": (
                    "Try guidance_scope='wind_regime' with a higher guidance_scale "
                    "specifically for this regime, or retrain with a stronger "
                    "cond_dropout_prob so classifier-free guidance has more signal "
                    "to amplify. Not attempted in this repository."
                ),
            })

    # Manifest structure is inconsistent across presets, not a bug here: 1ch/3ch
    # nest `known_limitation` inside `metrics`; 2ch has no `metrics` dict at all and
    # keeps it at the entry's top level instead. Check both so 2ch's real,
    # well-documented limitation isn't silently dropped.
    known_limitation = metrics.get("known_limitation") or entry.get("known_limitation")
    if known_limitation:
        issues.append({
            "issue": "Documented known limitation (from manifest.json)",
            "metric": known_limitation,
            "possible_cause": "See the manifest entry's own account of what was tried.",
            "potential_future_fix": "See the manifest entry's own account of what was tried.",
        })

    return {"preset": preset, "issues": issues}


def write_failure_analysis_md(result: dict) -> Path:
    out_dir = DATA_ROOT / "reports" / "wind"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Wind Failure Analysis",
        "",
        f"Preset: `{result['preset']}`",
        "",
        "**Not a new measurement** — flags issues already present in this preset's "
        "`manifest.json` entry (real numbers from the upstream project's own "
        "validation), against documented thresholds. See "
        "[`docs/models/wind.md`](../../../docs/models/wind.md) §8 for the full",
        "per-regime table this is derived from.",
        "",
    ]
    if not result["issues"]:
        lines.append("No flagged issues for this preset above threshold.")
    else:
        for issue in result["issues"]:
            lines += [
                f"## {issue['issue']}",
                "",
                f"**Observed issue:** {issue['metric']}",
                "",
                f"**Possible cause:** {issue['possible_cause']}",
                "",
                f"**Potential future fix:** {issue['potential_future_fix']}",
                "",
            ]

    out_path = out_dir / f"failure_analysis_{result['preset']}.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


if __name__ == "__main__":
    for preset in ("1ch", "2ch", "3ch"):
        result = analyze(preset)
        path = write_failure_analysis_md(result)
        print(f"{preset}: {len(result['issues'])} issue(s), saved to {path}")
