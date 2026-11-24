# Mooring-tension extreme quantiles by neural operator — minimal reproduction

Code and data accompanying

> **Extreme quantile prediction of mooring tension for floating offshore wind turbines
> using neural operator learning: a comparative study between semi-submersible and TLP**

This repository contains the smallest set of files needed to load one trained network,
run it on one simulated case and reproduce the single-case forecast of Fig. 26.

## What is here

```
data/E3D1_s1.npz      one simulated case: 3600 s of platform motion, thrust and line tension
data/norm.json        normalisation constants used at training time
models/semi_don.pt    trained branch–trunk neural operator (semi-submersible, seed 0)
models/semi_lstm.pt   trained quantile-regression LSTM (same encoder, different output head)
metrics/*.json        training history and evaluation scores for the two models
models.py             network definitions (QuantileDeepONet, QRLSTM)
reproduce_fig26.py    loads the case and the operator, queries 13 horizons x 6 levels
```

Two platforms are compared in the paper. Only the semi-submersible is shipped here to keep
the download small; the TLP case and weights are available from the authors on request.

## Environment

```
python 3.11
torch 2.2.1        # weights were saved with torch.save; torch >= 2.6 defaults to
                   # weights_only=True and will refuse to load them
numpy, matplotlib
```

```
pip install -r requirements.txt
```

## Run

```
python reproduce_fig26.py
```

The script prints a table of predicted quantiles at one instant and writes
`figures/reproduced_fig26.png`.

At `t = 2000 s` on line 1, the predicted 0.99-quantile rises from about **4.8 MN** at `H = 5 s`
to about **5.7 MN** at `H = 600 s`, while the measured 60 s maximum at that instant is **4.74 MN**.
The ordering of the six levels and the trend with the horizon are what the figure shows.

Absolute values may differ in the second decimal between machines: the LSTM reduction order in
float32 is not bit-reproducible, and repeated runs of this script differ by about 1 %.
That is well inside the model error reported in the paper (RMSE 0.08 MN at a 5 s lead).

## What the case file contains

| key | shape | meaning |
|---|---|---|
| `X` | (18001, 10) | measured signals at 5 Hz: surge, sway, heave, roll, pitch, yaw, rotor thrust, three nacelle accelerations |
| `Xs` | (721, 10) | the same channels low-pass filtered at 0.05 Hz and decimated to 0.2 Hz |
| `FT` | (18001, 3) | tension of the three fairleads, MN |
| `M` | (18001, 13, 3) | forward maximum of `FT` over the 13 horizons, MN |
| `eta` | (18001,) | wave elevation at the platform, m |
| `th_wave`, `th_curr` | (3,) | wave and current direction **relative to each line**, degrees |
| `meta_*` | scalar | sea state: `U`, `Hs`, `Tp`, `gamma`, seeds, and the case identifier |

The simulation was run in OpenFAST v3.5.1 coupling ElastoDyn, AeroDyn, HydroDyn and MoorDyn v2.0.0,
with all four second-order hydrodynamic switches off. The case is E3D1 (1-year return period,
heading D1), first wave seed, of the matrix described in Section 2 of the paper.

## Licence

Code: MIT. Data and weights: CC BY 4.0.

## Citation

If you use these files, please cite the paper above.
