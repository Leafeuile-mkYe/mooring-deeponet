#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Query one trained network at a single instant, for both platforms.

Loads one simulated case, one trained network and the normalisation constants,
then queries the network at one instant for all 13 horizons and 6 levels. The
printed table and the figure compare the predicted quantiles with the realised
forward maximum M(t; H) carried by the case file.

    python query_case.py                    # semi-submersible, operator
    python query_case.py --platform tlp     # TLP, operator
    python query_case.py --platform tlp --model lstm --line 3

This is a single-instant query, not a redraw of any figure in the paper: it
checks that the shipped weights and the shipped case reproduce the magnitudes
and the ordering reported in Section 4.

Requires torch 2.2 (the weights were saved with torch.save; torch >= 2.6 loads
with weights_only=True by default and will refuse them).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(1)      # deterministic reduction order in the LSTM
torch.manual_seed(0)

from models import QuantileDeepONet, QRLSTM

HERE = Path(__file__).resolve().parent
H_GRID = np.array([5, 8, 12, 18, 27, 40, 60, 90, 130, 200, 300, 450, 600], float)
ALPHA = np.array([0.50, 0.80, 0.90, 0.95, 0.98, 0.99])
T_QUERY = 2000.0              # instant of the query, in seconds

# scale_mn: the tension unit the network was trained on. The semi-submersible
# has an MBL of 22.286 MN; the TLP has no MBL and is scaled by the static
# pretension of one tendon, 6.97 MN. Using the wrong one is a factor of 3.2.
PLAT = {
    "semi": dict(case="E3D1_s1", norm="norm_semi.json", scale=22.286,
                 n_lines=3, name="semi-submersible"),
    "tlp": dict(case="E1D1_s1", norm="norm_tlp.json", scale=6.97,
                n_lines=9, name="TLP"),
}


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


def build_network(kind, args, norm):
    """Rebuild the network from the training arguments stored in metrics/*.json."""
    width = int(args["width"])
    dropout = float(args.get("dropout") or 0.0)      # older runs store null
    if kind == "don":
        net = QuantileDeepONet(
            p=width,
            use_theta=not args.get("no_theta", False),
            use_theta_c=not args.get("no_theta_c", False),
            use_H=not args.get("no_H", False),
            use_lambda=not args.get("no_lambda", False),
            single_scale=args.get("single_scale", False),
            use_fast=not args.get("no_fast", False),
            linear_lambda=args.get("linear_lambda", False),
            branch_kind=args.get("branch", "lstm"),
            raw_alpha=args.get("raw_alpha", False),
            dropout=dropout,
        )
    else:
        net = QRLSTM(hidden=width, n_H=len(H_GRID), n_A=len(ALPHA),
                     per_H=bool(args.get("per_H", False)),
                     single_scale=bool(args.get("single_scale", False)),
                     use_fast=not args.get("no_fast", False),
                     dropout=dropout,
                     ft_mean=float(norm["ft_mean"]), ft_std=float(norm["ft_std"]))
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="semi", choices=sorted(PLAT))
    ap.add_argument("--model", default="don", choices=["don", "lstm"])
    ap.add_argument("--line", type=int, default=0, help="0-based line index")
    ap.add_argument("--time", type=float, default=T_QUERY, help="query instant, s")
    ap.add_argument("--no-projection", action="store_true",
                    help="skip the monotone projection over H and the level")
    args_cli = ap.parse_args()

    cfg = PLAT[args_cli.platform]
    if not 0 <= args_cli.line < cfg["n_lines"]:
        raise SystemExit(f"{cfg['name']} has {cfg['n_lines']} lines; --line must be "
                         f"0..{cfg['n_lines'] - 1}")

    z = np.load(HERE / "data" / f"{cfg['case']}.npz")
    norm = json.loads((HERE / "data" / cfg["norm"]).read_text())
    stem = f"{args_cli.platform}_{args_cli.model}"
    args = json.loads((HERE / "metrics" / f"{stem}.json").read_text())["args"]

    # the tension scale must come from the same place the training used
    stored = float(norm.get("scale_mn", cfg["scale"]))
    if abs(stored - cfg["scale"]) > 1e-6:
        raise SystemExit(f"scale mismatch for {cfg['name']}: norm says {stored}, "
                         f"script expects {cfg['scale']}")

    x_mean = np.asarray(norm["x_mean"], np.float32)
    x_std = np.asarray(norm["x_std"], np.float32)
    X = (z["X"] - x_mean) / x_std                 # fast channel, 5 Hz
    Xs = (z["Xs"] - x_mean) / x_std               # slow channel, 0.2 Hz

    env = np.array([z["meta_U"] / 50.0, z["meta_Hs"] / 12.0, z["meta_Tp"] / 15.0,
                    np.sin(np.radians(z["meta_curr_off"])),
                    np.cos(np.radians(z["meta_curr_off"]))], np.float32)

    dt = float(z["dt"])
    i = int(round(args_cli.time / dt))
    if i < 1475:
        raise SystemExit("query time too early: the slow window needs 300 s of history")
    fast = X[i - 99:i + 1][None]                  # 20 s at 5 Hz
    k = i // 25                                   # slow sample index (Xs[k] <-> t = 5k s)
    slow = Xs[k - 59:k + 1][None]                 # 300 s at 0.2 Hz
    env = env[None]

    # th_wave / th_curr in the npz are already relative to each line
    th_w = float(z["th_wave"][args_cli.line])
    th_c = float(z["th_curr"][args_cli.line])
    q = trunk_coords(ALPHA, H_GRID, np.array([th_w]), np.array([th_c]))[0]
    q = torch.from_numpy(q.reshape(1, len(H_GRID) * len(ALPHA), 6))

    net = build_network(args_cli.model, args, norm)
    ck = torch.load(HERE / "models" / f"{stem}.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    net.eval()

    with torch.no_grad():
        out = net(torch.from_numpy(fast), torch.from_numpy(slow),
                  torch.from_numpy(env), q)
    pred = (out.numpy()[0] * float(norm["ft_std"]) + float(norm["ft_mean"])) * cfg["scale"]
    pred = pred.reshape(len(H_GRID), len(ALPHA))
    if not args_cli.no_projection:
        # monotone projection over H then over the level, as in Section 3:
        # a running maximum, applied once, no penalty term in the loss
        pred = np.maximum.accumulate(np.maximum.accumulate(pred, 0), 1)

    meas = z["M"][i, :, args_cli.line]        # already in MN
    print(f"case {z['meta_sid']}{z['meta_did']} ({cfg['name']}), "
          f"t = {args_cli.time:g} s, line {args_cli.line + 1} of {cfg['n_lines']}, "
          f"model = {args_cli.model}")
    print(f"{'H (s)':>7} " + " ".join(f"a={a:<5g}" for a in ALPHA) + "   measured")
    for j, h in enumerate(H_GRID):
        print(f"{h:7.0f} " + " ".join(f"{v:7.3f}" for v in pred[j]) + f"   {meas[j]:7.3f}")

    # a scale error would show up as a factor of 3.2, not a few percent
    p_flat = pred.reshape(-1)
    m_flat = np.repeat(meas, len(ALPHA))
    ok = p_flat > 0
    ratio = np.median(p_flat[ok] / np.maximum(m_flat[ok], 1e-6))
    print(f"\nmedian predicted/measured = {ratio:.2f}  (0.8-1.3 expected)")
    if not 0.5 < ratio < 2.0:
        raise SystemExit("predictions are off by a factor: check scale_mn / norm file")

    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(ALPHA)))
    for m, (a, c) in enumerate(zip(ALPHA, cols)):
        ax.plot(H_GRID, pred[:, m], "-o", ms=2.6, lw=1.0, color=c, label=rf"$\alpha$ = {a:g}")
    ax.plot(H_GRID, meas, "k^", ms=3.4, label="realised $M(t;H)$")
    ax.set_xscale("log")
    ax.set_xticks([5, 20, 60, 200, 600])
    ax.set_xticklabels(["5", "20", "60", "200", "600"])
    ax.set_xlabel("Forecast horizon $H$ (s)")
    ax.set_ylabel("Mooring tension (MN)")
    ax.set_title(f"{cfg['name']}, line {args_cli.line + 1}", fontsize=8)
    ax.legend(fontsize=6.8, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5), handlelength=1.6)
    ax.grid(alpha=0.3, lw=0.5)
    fig.subplots_adjust(left=0.16, right=0.70, top=0.90, bottom=0.17)
    out_png = HERE / "figures" / f"query_{args_cli.platform}_{args_cli.model}.png"
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
