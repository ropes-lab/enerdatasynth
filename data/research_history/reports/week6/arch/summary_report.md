# Week 6 — Architecture Comparison

Controlled experiment: only the denoiser backbone changes. Identical data, Z5 conditioning, T=1000 linear schedule, epochs, lr, batch, seed, and sampler.

| Arch | Params | Best val | RMSE | Hourly | FirstDiff | ACF | Peak | Energy | Ramp | Min |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp | 688,408 | 0.01407 | 0.0241 | 0.0333 | 0.017 | 0.0448 | 0.0116 | 0.0833 | 0.0149 | 2.8 |
| cnn | 158,401 | 0.01022 | 0.0231 | 0.0287 | 0.0006 | 0.0467 | 0.0009 | 0.1072 | 0.0006 | 118.0 |
| attention | 60,417 | 0.01589 | 0.022 | 0.0423 | 0.0179 | 0.0408 | 0.0092 | 0.0582 | 0.0153 | 2.9 |
| transformer | 110,401 | 0.01087 | 0.0215 | 0.0364 | 0.0013 | 0.0388 | 0.0003 | 0.075 | 0.0011 | 4.4 |

*Lowest validation loss:* **cnn** (0.01022).
*Best generated profile RMSE:* **transformer** (0.0215).

Note: validation loss is the noise-prediction MSE and is not directly comparable to generation quality. Judge architectures on the generated metrics, not val loss alone.