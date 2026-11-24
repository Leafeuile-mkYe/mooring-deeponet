#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mooring_DeepONet —— 模型与损失

两个模型吃同一份输入，输出同一张分位面 (n_H, n_α)，因此可以直接比较：

    QuantileDeepONet   branch(u) · trunk(θ, Λ(α), lnH)      算子结构
    QRLSTM             LSTM(u) → MLP → 全部分位数展平        常规做法

分位数损失用 pinball。单调性（对 α、对 H）**不在训练里加惩罚项**，
训练后用保序重排（rearrangement）投影，既严格单调又不限制表达能力。

用法:
    from models import QuantileDeepONet, QRLSTM, pinball_loss, rearrange
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CH_IN = 10          # dataset 的 X 通道数
N_ENV = 5


# ----------------------------------------------------------------------------- 组件
class ConvEncoder(nn.Module):
    """1D 卷积编码器，输入 (B, C, L)，输出 (B, d)。"""

    def __init__(self, c_in: int, d: int, widths=(32, 48, 64), k=5):
        super().__init__()
        layers, c = [], c_in
        for w in widths:
            layers += [nn.Conv1d(c, w, k, padding=k // 2), nn.GELU(),
                       nn.MaxPool1d(2)]
            c = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * (100 // 2 ** 3), d))
        # 注意：slow 通道长度 60，池化 3 次后是 7，所以下面用自适应池化兜底
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(c, d)

    def forward(self, x):
        h = self.body(x)
        return self.proj(self.pool(h).squeeze(-1))


class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, n=2, act=nn.GELU):
        super().__init__()
        layers, d = [], d_in
        for _ in range(n):
            layers += [nn.Linear(d, d_hid), act()]
            d = d_hid
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------- DeepONet
class QuantileDeepONet(nn.Module):
    """分位数 DeepONet。

    branch: 快通道 20 s@5 Hz + 慢通道 300 s@0.2 Hz + 工况标量 → b ∈ R^p
    trunk : (cosθ, sinθ, lnH, Λ(α)) → t ∈ R^p

    输出先经 softplus 保证张力非负（物理量，单位 T/MBL），再线性变换到
    标准化空间给损失用。**顺序不能反**：若先标准化再加 softplus，
    目标会落到负值区，网络只能靠把输出推向无穷大来逼近，训练必崩。
    """

    def __init__(self, p: int = 64, use_theta: bool = True,
                 use_theta_c: bool = True,
                 use_H: bool = True, use_lambda: bool = True,
                 single_scale: bool = False, use_fast: bool = True,
                 linear_lambda: bool = False, branch_kind: str = "conv",
                 raw_alpha: bool = False, dropout: float = 0.0,
                 ft_mean: float = 0.1316, ft_std: float = 0.054):
        super().__init__()
        self.p = p
        self.use_theta = use_theta
        self.use_theta_c = use_theta_c
        self.use_H = use_H
        self.use_lambda = use_lambda
        self.single_scale = single_scale
        self.use_fast = use_fast
        self.linear_lambda = linear_lambda
        self.branch_kind = branch_kind
        self.raw_alpha = raw_alpha
        self.drop = nn.Dropout(dropout)

        # branch_kind="lstm"：与 QRLSTM 用完全相同的时序编码器，两个模型的差别
        # 只剩「输出头」——乘法算子 vs 拼接。否则卷积 vs LSTM 的差异会混进来。
        if branch_kind == "lstm":
            self.enc_fast = nn.LSTM(N_CH_IN, p, batch_first=True) if use_fast else None
            self.enc_slow = None if single_scale else nn.LSTM(N_CH_IN, p, batch_first=True)
        else:
            self.enc_fast = ConvEncoder(N_CH_IN, p) if use_fast else None
            self.enc_slow = None if single_scale else ConvEncoder(N_CH_IN, p)
        self.enc_env = MLP(N_ENV, 32, p)
        nb = 1 + int(use_fast) + int(not single_scale)
        # 20 个训练算例下 branch 极易记住「算例身份」，dropout 作用在融合后的
        # 隐表示与 trunk 输出上，等效于对乘积的两个因子同时做子采样。
        self.branch_fuse = MLP(p * nb, 128, p)

        d_in = (2 * int(use_theta) + 2 * int(use_theta_c)
                + int(use_H) + int(use_lambda and not linear_lambda))
        # linear_lambda: trunk 输出 2p（截距与斜率各 p 维），α 依赖显式线性
        self.trunk = MLP(d_in, 64, (2 * p if linear_lambda else p), n=2)

        # b0 初始化到 softplus(b0) ≈ ft_mean，避免起步时整体偏移
        b0 = float(np.log(np.expm1(max(ft_mean, 1e-3))))
        self.b0 = nn.Parameter(torch.tensor([b0]))
        self.register_buffer("ft_mean", torch.tensor(float(ft_mean)))
        self.register_buffer("ft_std", torch.tensor(float(ft_std)))

    def _enc(self, net, x):
        if self.branch_kind == "lstm":
            h, _ = net(x.transpose(1, 2))
            return h[:, -1]
        return net(x)

    def branch(self, fast, slow, env):
        parts = [self.enc_env(env)]
        if not self.single_scale:
            parts.append(self._enc(self.enc_slow, slow))
        if self.use_fast:
            parts.append(self._enc(self.enc_fast, fast))
        return self.drop(self.branch_fuse(torch.cat(parts, -1)))

    def trunk_in(self, q):
        """从 dataset 给的 6 维坐标里按开关取子集。

        q = (cos θ_w, sin θ_w, cos θ_c, sin θ_c, lnH, Λ/5)

        raw_alpha=True 时把 Λ/5 换成裸 α（线性拉伸到 [0,1]）——消融 A3 要比较的是
        「用裸 α 代替 Λ(α)」，而不是「去掉 α」（后者会退化成中位数模型）。
        α 可以从 Λ 反算：α = exp(−exp(−Λ))。
        """
        parts = []
        if self.use_theta:
            parts.append(q[..., 0:2])          # 浪向相对锚链方位
        if self.use_theta_c:
            parts.append(q[..., 2:4])          # 流向相对锚链方位
        if self.use_H:
            parts.append(q[..., 4:5])
        if self.use_lambda and not self.linear_lambda:
            v = q[..., 5:6]
            if self.raw_alpha:
                alpha = torch.exp(-torch.exp(-5.0 * v))
                v = (alpha - 0.5) / 0.5
            parts.append(v)
        return torch.cat(parts, -1)

    def forward(self, fast, slow, env, q):
        b = self.branch(fast, slow, env)               # (B, p)
        t = self.drop(self.trunk(self.trunk_in(q)))    # (B, Q, p) 或 (B, Q, 2p)
        if self.linear_lambda:
            # Q = base(u,θ,H) + slope(u,θ,H)·Λ(α)
            #   极值渐近 Gumbel 时分位数对 Λ 严格线性；写成显式线性有两个好处：
            #   1) 外推到 α=0.999 不会跑偏（MLP 在训练区间外不可控）
            #   2) 斜率被全部 6 个 α 共同约束——α=0.99 的 pinball 权重只有 0.01，
            #      单靠它定不出斜率，中部权重大，能把尾部一起带准
            p = self.p
            t0, t1 = t[..., :p], t[..., p:]
            base = torch.einsum("bp,bqp->bq", b, t0) / (p ** 0.5) + self.b0
            slope = torch.einsum("bp,bqp->bq", b, t1) / (p ** 0.5)
            lam = q[..., 5]
            q_phys = F.softplus(base) + F.softplus(slope) * lam
        else:
            z = torch.einsum("bp,bqp->bq", b, t) / (self.p ** 0.5)
            q_phys = F.softplus(z + self.b0)           # T/MBL，≥ 0
        return (q_phys - self.ft_mean) / self.ft_std   # 标准化，供损失使用


class QRLSTM(nn.Module):
    """分位数回归 LSTM 基线。

    与 DeepONet 保持同样的输入与输出形状，差别只在「α 是输出维度」。
    per_H=True 时只输出单个 H 的分位面（对应「每时程训一个模型」的常规做法）。
    """

    def __init__(self, hidden: int = 96, n_H: int = 13, n_A: int = 6,
                 per_H: bool = False, single_scale: bool = False,
                 use_fast: bool = True, dropout: float = 0.0,
                 ft_mean: float = 0.1316, ft_std: float = 0.054):
        super().__init__()
        self.per_H = per_H
        self.single_scale = single_scale
        self.use_fast = use_fast
        self.n_H, self.n_A = n_H, n_A

        self.lstm_f = nn.LSTM(N_CH_IN, hidden, batch_first=True) if use_fast else None
        self.lstm_s = None if single_scale else nn.LSTM(N_CH_IN, hidden, batch_first=True)
        self.enc_env = MLP(N_ENV, 32, hidden)
        nh = hidden * (1 + int(use_fast) + int(not single_scale))
        n_out = n_A if per_H else n_H * n_A
        # +4：链与浪、流的相对方位（cos θw, sin θw, cos θc, sin θc）。
        # 不给的话三根链输入完全相同，模型无法区分 4.3 / 2.0 / 2.0 MN
        # 三个水平，只能预测链间平均——实测相关系数只有 0.098。
        self.head = MLP(nh + 4, 192, n_out, n=2)
        self.drop = nn.Dropout(dropout)

        b0 = float(np.log(np.expm1(max(ft_mean, 1e-3))))
        self.b0 = nn.Parameter(torch.tensor([b0]))
        self.register_buffer("ft_mean", torch.tensor(float(ft_mean)))
        self.register_buffer("ft_std", torch.tensor(float(ft_std)))

    def forward(self, fast, slow, env, q=None):
        parts = []
        if self.use_fast:
            hf, _ = self.lstm_f(fast.transpose(1, 2))
            parts.append(hf[:, -1])
        if not self.single_scale:
            hs, _ = self.lstm_s(slow.transpose(1, 2))
            parts.append(hs[:, -1])
        parts.append(self.enc_env(env))
        if q is not None:
            parts.append(q[:, 0, :4])      # 角度对与 (H, α) 无关，取任一查询即可
        else:
            parts.append(torch.zeros(env.shape[0], 4, device=env.device))
        z = self.head(self.drop(torch.cat(parts, -1)))
        q_phys = F.softplus(z + self.b0)
        return (q_phys - self.ft_mean) / self.ft_std


# ----------------------------------------------------------------------------- 损失
def pinball_loss(pred: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor,
                 n_H: int, h_keep=None) -> torch.Tensor:
    """pred (B, n_H*n_α) 已按 (H, α) 展平；target (B, n_H)；alpha (n_α,)。

    h_keep: 可选布尔张量 (n_H,)，只对这些时程计损失——用于「留 H」实验，
    检验 trunk 的 lnH 坐标能否插值到没给过标签的时程。
    """
    B, _ = pred.shape
    p = pred.view(B, n_H, -1)
    a = alpha.view(1, 1, -1).to(device=pred.device, dtype=pred.dtype)
    y = target.view(B, n_H, 1)
    r = y - p
    loss = torch.maximum(a * r, (a - 1.0) * r)
    if h_keep is not None:
        loss = loss[:, h_keep, :]
    return loss.mean()


def rearrange(pred: torch.Tensor, n_H: int) -> torch.Tensor:
    """保序投影：沿 α 与 H 两个方向做累计最大值，保证严格单调。

    分位数对 α 单调（定义），极值对 H 单调（更长的窗口最大值不可能更小）。
    两者都是数学必然，用投影实现比惩罚项干净得多。
    """
    B, _ = pred.shape
    p = pred.view(B, n_H, -1)
    p = torch.cummax(p, dim=2).values      # 对 α
    p = torch.cummax(p, dim=1).values      # 对 H
    return p.reshape(B, -1)


def crps_from_quantiles(pred: torch.Tensor, target: torch.Tensor,
                        alpha: torch.Tensor, n_H: int) -> torch.Tensor:
    """用 pinball 的积分近似 CRPS：CRPS = 2 ∫ ρ_α dα ≈ 2·mean_α ρ_α。"""
    return 2.0 * pinball_loss(pred, target, alpha, n_H)


# ----------------------------------------------------------------------------- 自检
if __name__ == "__main__":
    import sys
    from pathlib import Path

    B, nH, nA = 8, 13, 6
    fast = torch.randn(B, N_CH_IN, 100)
    slow = torch.randn(B, N_CH_IN, 60)
    env = torch.randn(B, N_ENV)
    q = torch.randn(B, nH * nA, 6)
    y = torch.rand(B, nH)
    al = torch.linspace(0.5, 0.99, nA)

    for name, m in [
        ("DeepONet", QuantileDeepONet(p=64)),
        ("DeepONet-w/o θ", QuantileDeepONet(p=64, use_theta=False)),
        ("DeepONet-single-scale", QuantileDeepONet(p=64, single_scale=True)),
        ("QR-LSTM", QRLSTM(hidden=96, n_H=nH, n_A=nA)),
    ]:
        out = m(fast, slow, env, q)
        loss = pinball_loss(out, y, al, nH)
        loss.backward()
        n_par = sum(p.numel() for p in m.parameters())
        print(f"{name:<24} out={tuple(out.shape)}  loss={loss.item():.4f}  参数={n_par/1e3:.0f}k")

    # 单调性检查
    z = torch.rand(4, nH * nA) * 3
    zr = rearrange(z, nH)
    a = zr.view(4, nH, nA)
    print("trunk 输入维:", QuantileDeepONet(p=64).trunk.net[0].in_features, "（含两个角度对）")
    print("rearrange 后对 α 单调:", bool((a.diff(dim=2) >= 0).all()))
    print("rearrange 后对 H 单调:", bool((a.diff(dim=1) >= 0).all()))

    # per_H 变体
    m = QRLSTM(hidden=64, n_H=nH, n_A=nA, per_H=True)
    print("per_H out:", tuple(m(fast, slow, env).shape), "（单 H，n_α=6）")
