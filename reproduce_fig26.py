#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal reproduction of the single-case quantile forecast (Fig. 26 of the paper).

Loads one simulated case, one trained network and the normalisation constants,
then queries the operator at a single instant for all 13 horizons and 6 levels.
The figure compares the predicted quantiles with the measured 60 s maximum.

    python reproduce_fig26.py

Requires torch 2.2 (weights were saved with torch.save; torch >= 2.6 loads with
weights_only=True by default and will refuse them).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)      # deterministic reduction order in the LSTM

torch.manual_seed(0)

from models import QuantileDeepONet

HERE = Path(__file__).resolve().parent
H_GRID = np.array([5, 8, 12, 18, 27, 40, 60, 90, 130, 200, 300, 450, 600], float)
ALPHA = np.array([0.50, 0.80, 0.90, 0.95, 0.98, 0.99])
LINE = 0                      # which mooring line to show
T_QUERY = 2000.0              # instant of the query, in seconds


def trunk_coords(alpha, H, th_w, th_c):
    """Query coordinates of the trunk, shape (n_theta, n_H, n_alpha, 6).

    Last axis: (cos th_w, sin th_w, cos th_c, sin th_c, (ln H - ln 60)/2, Lambda(alpha)/5),
    with Lambda = -ln(-ln alpha) the Gumbel reduced variate.
    """
    lam = -np.log(-np.log(alpha))
    rw, rc = np.radians(th_w), np.radians(th_c)
    cw, sw = np.cos(rw)[:, None, None], np.sin(rw)[:, None, None]
    cc, sc = np.cos(rc)[:, None, None], np.sin(rc)[:, None, None]
    lnh = ((np.log(H) - np.log(60.0)) / 2.0)[None, :, None]
    lamn = (lam / 5.0)[None, None, :]
    return np.stack(np.broadcast_arrays(cw, sw, cc, sc, lnh, lamn), axis=-1).astype(np.float32)


def main():
    z = np.load(HERE / "data" / "E3D1_s1.npz")
    norm = json.loads((HERE / "data" / "norm.json").read_text())
    args = json.loads((HERE / "metrics" / "semi_don.json").read_text())["args"]

    x_mean = np.asarray(norm["x_mean"], np.float32)
    x_std = np.asarray(norm["x_std"], np.float32)
    X = (z["X"] - x_mean) / x_std                 # fast channel, 5 Hz
    Xs = (z["Xs"] - x_mean) / x_std               # slow channel, 0.2 Hz

    env = np.array([z["meta_U"] / 50.0, z["meta_Hs"] / 12.0, z["meta_Tp"] / 15.0,
                    np.sin(np.radians(z["meta_curr_off"])),
                    np.cos(np.radians(z["meta_curr_off"]))], np.float32)

    dt = float(z["dt"])
    i = int(round(T_QUERY / dt))
    if i < 1475:
        raise SystemExit("query time too early: the slow window needs 300 s of history")
    fast = X[i - 99:i + 1][None]                  # 20 s at 5 Hz
    k = i // 25                                   # slow sample index (Xs[k] <-> t = 5k s)
    slow = Xs[k - 59:k + 1][None]                 # 300 s at 0.2 Hz
    env = env[None]

    # th_wave / th_curr in the npz are already relative to each line (CHAIN_AZ - wave_line)
    th_w = float(z["th_wave"][LINE])
    th_c = float(z["th_curr"][LINE])
    q = trunk_coords(ALPHA, H_GRID, np.array([th_w]), np.array([th_c]))[0]
    q = torch.from_numpy(q.reshape(1, len(H_GRID) * len(ALPHA), 6))

    model = QuantileDeepONet(
        p=int(args["width"]),
        use_theta=not args.get("no_theta", False),
        use_theta_c=not args.get("no_theta_c", False),
        use_H=not args.get("no_H", False),
        use_lambda=not args.get("no_lambda", False),
        single_scale=args.get("single_scale", False),
        use_fast=not args.get("no_fast", False),
        linear_lambda=args.get("linear_lambda", False),
        branch_kind=args.get("branch", "lstm"),
        raw_alpha=args.get("raw_alpha", False),
        dropout=float(args.get("dropout", 0.0)),
    )
    ck = torch.load(HERE / "models" / "semi_don.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        out = model(torch.from_numpy(fast), torch.from_numpy(slow),
                    torch.from_numpy(env[None] if env.ndim == 1 else env), q)
    MBL = 22.286                      # MN, Table 6
    pred = (out.numpy()[0] * float(norm["ft_std"]) + float(norm["ft_mean"])) * MBL
    pred = pred.reshape(len(H_GRID), len(ALPHA))

    meas = z["M"][i, :, LINE]        # already in MN
    print(f"case {z['meta_sid']}{z['meta_did']}, t = {T_QUERY:g} s, line {LINE + 1}")
    print(f"{'H (s)':>7} " + " ".join(f"a={a:<5g}" for a in ALPHA) + "   measured")
    for j, h in enumerate(H_GRID):
        print(f"{h:7.0f} " + " ".join(f"{v:7.3f}" for v in pred[j]) + f"   {meas[j]:7.3f}")

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(ALPHA)))
    for m, (a, c) in enumerate(zip(ALPHA, cols)):
        ax.plot(H_GRID, pred[:, m], "-o", ms=2.6, lw=1.0, color=c, label=rf"$\alpha$ = {a:g}")
    ax.plot(H_GRID, meas, "k^", ms=3.4, label="measured 60 s max")
    ax.set_xscale("log")
    ax.set_xlabel("Forecast horizon $H$ (s)")
    ax.set_ylabel("Mooring tension (MN)")
    ax.legend(fontsize=6.6, ncol=2, frameon=False)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    out_png = HERE / "figures" / "reproduced_fig26.png"
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
