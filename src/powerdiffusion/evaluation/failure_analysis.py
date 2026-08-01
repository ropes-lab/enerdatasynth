"""
src/powerdiffusion/evaluation/failure_analysis.py
====================================================
Automatically detects and summarizes common failure modes by
comparing real vs generated profiles:
  - Peak underestimation / overestimation
  - Ramp mismatches
  - Flattened profiles (generated variance collapse)
  - Seasonal mismatches (if month info is available)

Writes reports/failure_analysis.md with: observed issue, possible
cause, potential future fix — for each detected problem.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from powerdiffusion.utils.config import PROCESSED_DIR, GENERATED_DIR, REPORTS_DIR, N_MONTHS, N_DOW


# Thresholds for flagging an issue — simple, documented, adjustable.
PEAK_REL_ERROR_THRESHOLD   = 0.10   # 10% relative error in peak load
RAMP_REL_ERROR_THRESHOLD   = 0.15   # 15% relative error in ramp std
VARIANCE_COLLAPSE_RATIO    = 0.5    # generated variance < 50% of real -> flattening
SEASONAL_REL_ERROR_THRESHOLD = 0.10


def analyze(run_id: str) -> dict:
    X_real = np.load(PROCESSED_DIR / "X_daily_profiles_val.npy")
    X_gen  = np.load(GENERATED_DIR / f"X_generated_{run_id}.npy")
    Z      = np.load(PROCESSED_DIR / "Z_conditions_val.npy")

    issues = []

    # ── Peak under/over-estimation ───────────────────────────────────
    peak_real = X_real.max(axis=1)
    peak_gen  = X_gen.max(axis=1)
    peak_rel_error = (peak_gen.mean() - peak_real.mean()) / (peak_real.mean() + 1e-8)

    if abs(peak_rel_error) > PEAK_REL_ERROR_THRESHOLD:
        direction = "underestimation" if peak_rel_error < 0 else "overestimation"
        issues.append({
            "issue": f"Peak {direction}",
            "metric": f"Mean real peak = {peak_real.mean():.4f}, "
                      f"mean generated peak = {peak_gen.mean():.4f} "
                      f"({peak_rel_error*100:+.1f}% relative error)",
            "possible_cause": (
                "Diffusion model trained for too few epochs, or the "
                "reverse sampling process is not fully converging to "
                "sharp peaks (regression to the mean is common in "
                "under-trained diffusion models)."
                if direction == "underestimation" else
                "Model may be overshooting due to noise schedule "
                "instability or insufficient denoising steps at "
                "inference time."
            ),
            "potential_future_fix": (
                "Train to convergence (more epochs), increase diffusion "
                "steps T, or check that the reverse sampler is using "
                "the full T steps rather than a truncated schedule."
            ),
        })

    # ── Ramp mismatch ─────────────────────────────────────────────────
    ramp_real = np.diff(X_real, axis=1)
    ramp_gen  = np.diff(X_gen, axis=1)
    ramp_std_real = ramp_real.std()
    ramp_std_gen  = ramp_gen.std()
    ramp_rel_error = (ramp_std_gen - ramp_std_real) / (ramp_std_real + 1e-8)

    if abs(ramp_rel_error) > RAMP_REL_ERROR_THRESHOLD:
        direction = "too small" if ramp_rel_error < 0 else "too large"
        issues.append({
            "issue": f"Ramp rate mismatch ({direction})",
            "metric": f"Real ramp std = {ramp_std_real:.4f}, "
                      f"generated ramp std = {ramp_std_gen:.4f} "
                      f"({ramp_rel_error*100:+.1f}% relative error)",
            "possible_cause": (
                "Generated profiles are smoother/flatter than reality "
                "(common in early-training diffusion models that "
                "haven't learned high-frequency structure)."
                if ramp_rel_error < 0 else
                "Generated profiles are noisier than real load — "
                "incomplete denoising or too few reverse steps."
            ),
            "potential_future_fix": (
                "Train longer; consider whether hidden_dim is large "
                "enough to capture hour-to-hour transition structure; "
                "verify noise schedule (beta_start/beta_end) is "
                "appropriate for this data's scale."
            ),
        })

    # ── Flattened profiles (variance collapse) ───────────────────────
    var_real = X_real.var(axis=0).mean()
    var_gen  = X_gen.var(axis=0).mean()
    if var_gen < VARIANCE_COLLAPSE_RATIO * var_real:
        issues.append({
            "issue": "Flattened / mode-collapsed profiles",
            "metric": f"Mean per-hour variance — real: {var_real:.5f}, "
                      f"generated: {var_gen:.5f} "
                      f"(generated is {var_gen/(var_real+1e-8)*100:.1f}% of real)",
            "possible_cause": (
                "Model is regressing toward the conditional mean rather "
                "than sampling diverse realistic profiles — typically a "
                "sign of under-training or a reverse sampler that isn't "
                "injecting enough stochastic noise at each step."
            ),
            "potential_future_fix": (
                "Train to convergence; verify the reverse sampler adds "
                "noise at every step (not just step 0); consider "
                "evaluating multiple samples per condition vector to "
                "check sample diversity, not just the mean."
            ),
        })

    # ── Seasonal mismatch (using one-hot month columns 0-11 of Z) ────
    month_idx = np.argmax(Z[:, :N_MONTHS], axis=1)  # 0-11
    seasonal_issues = []
    for season_name, months in [
        ("Winter", [11, 0, 1]), ("Spring", [2, 3, 4]),
        ("Summer", [5, 6, 7]),  ("Fall", [8, 9, 10]),
    ]:
        mask = np.isin(month_idx, months)
        if mask.sum() < 5:
            continue
        real_mean = X_real[mask].mean()
        gen_mean  = X_gen[mask].mean()
        rel_err   = (gen_mean - real_mean) / (real_mean + 1e-8)
        if abs(rel_err) > SEASONAL_REL_ERROR_THRESHOLD:
            seasonal_issues.append(
                f"{season_name}: real mean={real_mean:.4f}, "
                f"generated mean={gen_mean:.4f} ({rel_err*100:+.1f}%)"
            )
    if seasonal_issues:
        issues.append({
            "issue": "Seasonal mismatch",
            "metric": "; ".join(seasonal_issues),
            "possible_cause": (
                "Model is not fully utilizing the one-hot month "
                "conditioning — possibly due to limited training "
                "epochs, or the conditioning signal being too weak "
                "relative to the noise schedule at the early-training "
                "stage."
            ),
            "potential_future_fix": (
                "Train to convergence; verify the conditioning vector "
                "is reaching the network correctly; consider per-season "
                "validation metrics during training to catch this earlier."
            ),
        })

    return {"run_id": run_id, "issues": issues}


def write_failure_analysis_md(result: dict) -> Path:
    lines = [
        "# Failure Analysis",
        "",
        f"Run: `{result['run_id']}`",
        "",
    ]
    if not result["issues"]:
        lines.append("No failure modes detected above threshold for this run.")
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

    out_path = REPORTS_DIR / "failure_analysis.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


if __name__ == "__main__":
    from powerdiffusion.utils.config import RunConfig
    cfg = RunConfig()
    result = analyze(cfg.run_id)
    path = write_failure_analysis_md(result)
    print(f"Saved: {path}")
    print(f"Issues detected: {len(result['issues'])}")
