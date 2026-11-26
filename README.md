# Mooring-tension extreme quantiles by neural operator — minimal reproduction

Code, weights and data accompanying

> **Extreme quantile prediction of mooring tension with a query-coordinate neural operator:
> semi-submersible versus tension-leg platform**

This repository contains the smallest set of files needed to load the trained networks,
run them on one simulated case per platform and query the predicted extreme quantiles
at a single instant.

## What is here

```
data/E3D1_s1.npz        semi-submersible case: 3600 s of platform motion, thrust and line tension
data/E1D1_s1.npz        TLP case, same channels
data/norm_semi.json     normalisation constants of the semi-submersible runs
data/norm_tlp.json      normalisation constants of the TLP runs
models/semi_don.pt      branch–trunk neural operator   (semi-submersible, seed 0)
models/semi_lstm.pt     quantile-regression LSTM       (same encoder, different output head)
models/tlp_don.pt       branch–trunk neural operator   (TLP, seed 0)
models/tlp_lstm.pt      quantile-regression LSTM       (TLP, seed 0)
metrics/*.json          training history, args and scores of the four runs
models.py               network definitions (QuantileDeepONet, QRLSTM)
query_case.py           queries one network for 13 horizons × 6 levels at one instant
figures/                example output of the four combinations
```

The four runs are the seed-0 members of the ensembles reported in Table 4
(`d_lstmbr`, `l1`, `tlp_don`, `tlp_lstm`); the `tag` field inside each metrics file keeps
that internal run name. The operator has 100 513 parameters and the LSTM 132 207,
sharing the same recurrent encoder.

Only one case per platform is shipped, to keep the download small. The other 55
(semi-submersible) and 59 (TLP) cases are available from the corresponding author on
reasonable request.

## Environment

```
python 3.11
torch 2.2.1        # the weights were saved with torch.save; torch >= 2.6 defaults to
                   # weights_only=True and will refuse to load them
numpy, matplotlib
```

```
pip install -r requirements.txt
```

## Run

```
python query_case.py                          # semi-submersible, operator
python query_case.py --platform tlp           # TLP, operator
python query_case.py --platform tlp --model lstm
python query_case.py --platform semi --line 2 --time 3000
```

The script prints the 13 × 6 table of predicted quantiles next to the realised
forward maximum $M(t;H)$ carried by the case file, and writes
`figures/query_<platform>_<model>.png`. `--no-projection` disables the monotone
projection described below.

At `t = 2000 s` on line 1 with the operator:

| | semi-submersible | TLP |
|---|---|---|
| 0.99-quantile at `H = 5 s` | about 4.8 MN | about 8.5 MN |
| 0.99-quantile at `H = 600 s` | about 5.6 MN | about 9.5 MN |
| realised `M(t;600)` | 4.9 MN | 8.2 MN |

The ordering of the six levels and the trend with the horizon are what the figure shows.
This is a single-instant query, not a redraw of a figure in the paper: it checks that the
shipped weights and the shipped case reproduce the magnitudes reported in Section 4.

Absolute values may differ in the second decimal between machines: the LSTM reduction
order in float32 is not bit-reproducible, and repeated runs of the same command differ by
up to 2 % at the longest horizon (9.4 to 9.6 MN for the 0.99 level above). That is
well inside the model error reported in the paper (RMSE 0.08 MN at a 5 s lead).

### Two things that are easy to get wrong

**The tension scale is platform dependent.** The semi-submersible is normalised by its
mooring-line MBL, 22.286 MN; the TLP has no MBL and is normalised by the static
pretension of one tendon, 6.97 MN, stored as `scale_mn` in `data/norm_tlp.json`. Using
the semi-submersible value on the TLP inflates the prediction by a factor of 3.2. The
script reads `scale_mn` and aborts if it disagrees with the platform table.

**The normalisation constants are not interchangeable.** `norm_semi.json` and
`norm_tlp.json` have different channel means, standard deviations and tension scales
(`ft_mean` 0.128 vs 1.015). Pairing a case with the wrong file loads without error and
produces plausible-looking numbers that are wrong.

### Monotone projection

Quantiles must not cross as the horizon or the level increases, so both directions are
projected with a running maximum applied once, `np.maximum.accumulate` over `H` then over
`α`, as described in Section 3. There is no penalty term in the loss. `--no-projection`
shows the raw network output, in which adjacent levels can occasionally invert at long
horizons.

## What the case files contain

| key | shape | meaning |
|---|---|---|
| `X` | (18001, 10) | measured signals at 5 Hz: surge, sway, heave, roll, pitch, yaw, rotor thrust, three nacelle accelerations |
| `Xs` | (721, 10) | the same channels low-pass filtered at 0.05 Hz and decimated to 0.2 Hz |
| `FT` | (18001, *n*) | tension of the *n* mooring lines, MN; *n* = 3 (semi-submersible), 9 (TLP) |
| `M` | (18001, 13, *n*) | forward maximum of `FT` over the 13 horizons, MN |
| `eta` | (18001,) | wave elevation at the platform, m |
| `th_wave`, `th_curr` | (*n*,) | wave and current direction **relative to each line**, degrees |
| `dt` | scalar | sampling interval of `X`, s (0.2, i.e. 5 Hz) |
| `H_grid` | (13,) | the horizons of the third axis of `M`, s |
| `meta_*` | scalar | sea state: `U`, `Hs`, `Tp`, `gamma`, seeds, and the case identifier |

The simulations were run in OpenFAST v3.5.1 coupling ElastoDyn, AeroDyn, HydroDyn and
MoorDyn v2.0.0, with all four second-order hydrodynamic switches off. The cases are
E3D1 (1-year return period, heading D1, first wave seed) and E1D1, of the matrix
described in Section 2 of the paper.

## What is not here

The evaluation pipeline that produces the skill scores of Tables 4, 7 and 9 needs the
full set of 56 (semi-submersible) and 60 (TLP) cases, the climatological baselines built
from the training split, and the saved outputs of the 3-seed ensembles. Those, together
with the scripts for the coordinate-head control, the channel occlusion and the reduced
training budget, are available on request.

## Licence

Code: MIT. Data and weights: CC BY 4.0.

## Citation

If you use these files, please cite the paper above.
