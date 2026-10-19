# -*- coding: utf-8 -*-
"""
因子组合实验：15 个正交因子 → 组合，看 IC 能到哪里。

组合方式（权重一律用训练期 mine=2016-2021 算，测试期 val=2022-2026 应用）：
  等权 / IC 加权 / ICIR 加权 / 最优权重(w ∝ C⁻¹ρ)

数学：ρ_F = wᵀρ / sqrt(wᵀCw)，独立因子等权可 √K 提高；最优 w ∝ C⁻¹ρ。

用法：python combine.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csi500_alpha.data_loader import Csi500DataLoader
from csi500_alpha.vm import StackVM

MINE_END = "2021-12-31"


def rank_spearman(f: np.ndarray, r: np.ndarray, mask: np.ndarray) -> float:
    def rank_cols(x):
        return np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float64)

    fr = rank_cols(f)
    rr = rank_cols(r)
    fr = np.where(mask, fr, np.nan)
    rr = np.where(mask, rr, np.nan)
    with np.errstate(all="ignore"):
        fc = fr - np.nanmean(fr, axis=0, keepdims=True)
        rc = rr - np.nanmean(rr, axis=0, keepdims=True)
        num = np.nansum(fc * rc, axis=0)
        den = np.sqrt(np.nansum(fc ** 2, axis=0) * np.nansum(rc ** 2, axis=0))
    return float(np.nanmean(num / (den + 1e-12)))


def daily_ics(f: np.ndarray, r: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """逐日截面 Spearman，返回每天一个 IC。"""
    def rank_cols(x):
        return np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float64)

    fr = np.where(mask, rank_cols(f), np.nan)
    rr = np.where(mask, rank_cols(r), np.nan)
    with np.errstate(all="ignore"):
        fc = fr - np.nanmean(fr, axis=0, keepdims=True)
        rc = rr - np.nanmean(rr, axis=0, keepdims=True)
        num = np.nansum(fc * rc, axis=0)
        den = np.sqrt(np.nansum(fc ** 2, axis=0) * np.nansum(rc ** 2, axis=0))
    return num / (den + 1e-12)


def main():
    loader = Csi500DataLoader().load_data()
    vm = StackVM(n_features=20)
    feat = loader.feat_tensor
    target = loader.target_ret.numpy()
    mask = loader.mask.numpy()
    dates = loader.dates

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_results", "csi500_batched_factors.json"), encoding="utf-8") as f:
        factors = json.load(f)

    # 执行 15 因子 → [K, S, T]
    sigs = []
    for fac in factors:
        f = vm.execute(fac["tokens"], feat)
        if f is None:
            raise RuntimeError("因子执行失败")
        sigs.append(f.numpy())
    S = np.stack(sigs)                       # [K, S, T]
    K = S.shape[0]

    # 截面 z-score（每因子每天）
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-12
    Z = (S - mu) / sd                        # [K, S, T]

    mine_idx = dates <= pd.Timestamp(MINE_END)
    val_idx = dates > pd.Timestamp(MINE_END)
    Zm = Z[:, :, mine_idx]                   # [K, S, Tm]
    Zv = Z[:, :, val_idx]                    # [K, S, Tv]
    tm = target[:, mine_idx]
    tv = target[:, val_idx]
    mm = mask[:, mine_idx]
    mv = mask[:, val_idx]

    # ---- 训练期：ρ（IC 向量）+ C（因子相关矩阵）+ ICIR ----
    rho = np.array([rank_spearman(Zm[k], tm, mm) for k in range(K)])
    # 逐日 IC 序列 → ICIR
    icir = []
    for k in range(K):
        ic = daily_ics(Zm[k], tm, mm)
        icir.append(float(np.nanmean(ic) / (np.nanstd(ic) + 1e-12)))
    icir = np.array(icir)

    # 因子相关矩阵 C（训练期截面相关的时间平均）
    C = np.zeros((K, K))
    Tm = Zm.shape[2]
    for t in range(Tm):
        Zt = Zm[:, :, t]                     # [K, S]
        Zt = Zt - Zt.mean(axis=1, keepdims=True)
        C += Zt @ Zt.T / Zt.shape[1]
    C /= Tm

    # ---- 组合权重 ----
    w_equal = np.ones(K) / K
    w_ic = np.abs(rho) / (np.abs(rho).sum() + 1e-12)
    w_icir = np.abs(icir) / (np.abs(icir).sum() + 1e-12)
    ridge = 0.1 * np.trace(C) / K
    w_opt = np.linalg.solve(C + ridge * np.eye(K), rho)
    w_opt = w_opt / (np.abs(w_opt).sum() + 1e-12)   # 归一（允许正负）

    def combo_ic(w):
        c = (w[:, None, None] * Zv).sum(axis=0)     # [S, Tv]
        return rank_spearman(c, tv, mv)

    print(f"\n因子数 K={K}，单因子训练期 IC 均值={rho.mean():.4f}，max={rho.max():.4f}")
    print(f"\n{'组合方式':<14}{'验证期 IC':>12}{'vs 单因子均值':>14}")
    print("-" * 44)
    base = rho.mean()
    for label, w in [("等权", w_equal), ("IC加权", w_ic), ("ICIR加权", w_icir), ("最优 C⁻¹ρ", w_opt)]:
        ic = combo_ic(w)
        print(f"{label:<14}{ic:>12.4f}{ic/base:>14.2f}x")


if __name__ == "__main__":
    main()
