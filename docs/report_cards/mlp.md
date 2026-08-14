# Report Card — MLP Denoiser

**Status: Original baseline / legacy architecture. Currently the package's de facto
default** (see LIMITATIONS — this is not by design, it's the absence of a bundled
Transformer checkpoint).

---

**MODEL**
Conditional DDPM denoiser using a feedforward MLP (residual blocks) over the
concatenated noisy profile, timestep embedding, and conditioning vector.
`src/enerdatasynth/resources/pjm_load/models/denoiser.py`'s `SimpleMLPDenoiser`.

**DATA**
PJM zone PS. Two data lineages exist for this architecture: the Week 3/6 research runs
(6-dim "Z5" conditioning) and the package's own hyperparameter-sweep checkpoints
(23-dim one-hot conditioning, `data/reports/pjm_load/results_summary.md`, 2,785 train / 600
validation days). See `docs/data_card.md`.

**CONDITIONING**
Both representations exist for this architecture (unlike the Transformer, which is
research-validated on 6-dim only): Week 6/7 comparisons used 6-dim "Z5"; the bundled
`data/checkpoints/pjm_load/*` sweep checkpoints were trained on the current 23-dim one-hot
representation, which is why they load without a dimension mismatch today.

**ARCHITECTURE**
Flat-vector treatment of the 24-hour profile (no explicit temporal axis) — the one
architectural choice the Week 6 comparison was specifically designed to test against
sequence-aware alternatives. 4 residual blocks, width 256, ~688,408 parameters (Week 6
comparison config) — the specific bundled `T100_H256_L4` checkpoint has 224,024
parameters (different hyperparameters: `hidden_dim=256, n_layers=4`, `T=100`, not
directly comparable to the Week 6 number).

**TRAINING**
Same DDPM training loss and procedure as the Transformer. The bundled sweep checkpoints
were produced by `enerdatasynth`'s own hyperparameter sweep (`enerdatasynth sweep`), not
by the Week 3/6 research scripts directly — best bundled run: `T100_H256_L4`, best
validation loss 0.08164 at epoch 40, 6.7s training time (`data/reports/pjm_load/results_summary.md`
— note this is a much smaller/faster sweep configuration than the Week 6 comparison run;
the two are not the same experiment).

**SAMPLING**
Same corrected ancestral DDPM sampler as the Transformer (`src/enerdatasynth/resources/pjm_load/diffusion/ddpm.py`)
— architecture-agnostic.

**EVALUATION**
Same 7-metric suite as the Transformer.

**RESULTS**

*Week 6/7 research comparison (6-dim conditioning, trained to convergence):*

| Metric | MLP | Transformer | Best classical |
|---|---|---|---|
| Hourly | 0.0399 | **0.0243** | 1.3535 |
| FirstDiff | 0.0185 | **0.0031** | 0.1633 |
| ACF | 0.0429 | **0.0325** | 0.1084 |
| RMSE | 0.0259 | **0.0195** | 0.1808 |
| Peak | 0.0115 | **0.0089** | 0.3124 |
| Energy | **0.0008** | 0.1405 | 4.2818 |
| Ramp | 0.016 | **0.0025** | 0.0057 |

MLP wins only on Energy among the two diffusion models — and loses to the Transformer on
6 of 7 metrics at ~6× more parameters.

*Package sweep result (23-dim conditioning, `T100_H256_L4`, from `data/reports/pjm_load/results_summary.md`,
its own known failure modes):* ramp-rate std overshoot (+289.8% vs. real), seasonal mean
mismatch (winter −10.8%, summer −14.9%). This is a genuinely different, earlier-stage
result from the Week 6/7 comparison above — not directly comparable, and notably worse
on shape metrics than either research-lineage result.

**FAILURE MODES**
Per the package's own `data/reports/pjm_load/failure_analysis.md`: ramp-rate mismatch (too noisy)
and seasonal under-utilization of the month conditioning — both attributed to
under-training relative to the Transformer/Week-7 convergence work.

**LIMITATIONS**
1. Loses to the Transformer on 6 of 7 metrics in the controlled comparison.
2. The specific bundled checkpoint (`T100_H256_L4`) is from an earlier, smaller,
   less-converged sweep than the Week 6/7 research runs — its failure modes (ramp/
   seasonal) are worse than either research-lineage MLP result.
3. **No calibration.json is bundled** for any MLP checkpoint — Energy's apparent
   advantage over the Transformer in the Week 6/7 table has not been tested against a
   calibrated Transformer under the same conditions.
4. **This is currently what `enerdatasynth.model("pjm_load", ...)` actually returns**,
   despite the Transformer being the documented recommendation — see
   `docs/research_status.md` §D for exactly why.
