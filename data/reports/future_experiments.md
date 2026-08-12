# Future Experiments

**None of the items below are implemented in this baseline.** They are
documented here only as recommendations for future work, per the PI's
explicit instruction to establish a simple, reproducible baseline first.

---

## One-Hot vs Cyclic Encoding

**Current baseline:** month and day-of-week are one-hot encoded.

**Future experiment:** replace one-hot with cyclic sin/cos pairs
(`sin(2*pi*month/12)`, `cos(2*pi*month/12)`, and the equivalent for
day-of-week). This would make December and January geometrically
adjacent, and Sunday and Monday geometrically adjacent, rather than
treating every category as equally distant.

**Why not now:** the PI confirmed one-hot is acceptable for the baseline.

---

## Attention Layers

**Current baseline:** simple feedforward MLP, no attention mechanism.

**Future experiment:** add self-attention over the 24 hourly positions,
or cross-attention between the condition vector and the profile, to let
the model learn which hours are most influenced by which conditioning
variables.

**Why not now:** explicitly excluded from the baseline scope.

---

## MW Anchor (Multi-Day Continuity)

**Current baseline:** each day is generated independently; no
information about the previous day's ending load is used.

**Future experiment:** add the previous day's final hour (h23) load
value as an additional conditioning input, to reduce discontinuities
when stitching multiple generated days into a continuous sequence.

**Why not now:** explicitly excluded from the baseline scope. Becomes
relevant once multi-day (e.g., 7-day) synthetic sequences are needed.

---

## Unified PJM Model

**Current baseline:** one model trained on a single zone (PS).

**Future experiment:** a single model conditioned on zone identity,
trained across all PJM zones simultaneously, to test whether pooling
data helps zones with limited or noisy data.

**Why not now:** explicitly excluded from the baseline scope. This is
later-phase work after the per-zone baseline is validated.

---

## Advanced Conditioning

**Current baseline:** month, day-of-week, holiday flag, Tmax, Tmin,
and load spread — six conditioning signals only.

**Future experiment:** additional weather variables (humidity), hourly
(rather than daily) temperature resolution, or other derived features.

**Why not now:** explicitly excluded from the baseline scope to keep
the conditioning vector minimal and interpretable.
