# -*- coding: utf-8 -*-
"""
85 因子线性组合（统一口径：5 日收益 label）。

重点对比两种处理负 IC 因子的方式：
  A. 带符号反转：权重用 ρ 的符号（w ∝ ρ / Σ|ρ| 或 C⁻¹ρ），负 IC 因子自动反转；
  B. 剔除：只用训练期 IC>0 的因子子集。

用法：python combine85.py
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

    fr = np.where(mask, rank_cols(f), np.nan)
    rr = np.where(mask, rank_cols(r), np.nan)
    with np.errstate(all="ignore"):
        fc = fr - np.nanmean(fr, axis=0, keepdims=True)
        rc = rr - np.nanmean(rr, axis=0, keepdims=True)
        num = np.nansum(fc * rc, axis=0)
        den = np.sqrt(np.nansum(fc ** 2, axis=0) * np.nansum(rc ** 2, axis=0))
    return float(np.nanmean(num / (den + 1e-12)))


def main():
    loader = Csi500DataLoader().load_data()
    vm = StackVM(n_features=20)
    feat = loader.feat_tensor
    target = loader.forward_return(5).numpy()   # 5 日收益
    mask = loader.mask.numpy()
    dates = loader.dates

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_results", "csi500_batched_factors.json"), encoding="utf-8") as f:
        factors = json.load(f)

    sigs = []
    for fac in factors:
        f = vm.execute(fac["tokens"], feat)
        if f is None:
            raise RuntimeError("执行失败")
        sigs.append(f.numpy())
    S = np.stack(sigs)                          # [K, S, T]
    K = S.shape[0]
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-12
    Z = (S - mu) / sd

    mine_idx = dates <= pd.Timestamp(MINE_END)
    val_idx = dates > pd.Timestamp(MINE_END)
    Zm = Z[:, :, mine_idx]
    Zv = Z[:, :, val_idx]
    tm = target[:, mine_idx]
    tv = target[:, val_idx]
    mm = mask[:, mine_idx]
    mv = mask[:, val_idx]

    # 训练期 ρ（带符号 IC 向量）+ ICIR + C
    rho = np.array([rank_spearman(Zm[k], tm, mm) for k in range(K)])
    icir = []
    for k in range(K):
        # 逐日 IC
        fr = np.where(mm, np.argsort(np.argsort(Zm[k], axis=0), axis=0).astype(np.float64), np.nan)
        rr = np.where(mm, np.argsort(np.argsort(tm, axis=0), axis=0).astype(np.float64), np.nan)
        with np.errstate(all="ignore"):
            fc = fr - np.nanmean(fr, axis=0, keepdims=True)
            rc = rr - np.nanmean(rr, axis=0, keepdims=True)
            num = np.nansum(fc * rc, axis=0)
            den = np.sqrt(np.nansum(fc ** 2, axis=0) * np.nansum(rc ** 2, axis=0))
        ic = num / (den + 1e-12)
        icir.append(float(np.nanmean(ic) / (np.nanstd(ic) + 1e-12)))
    icir = np.array(icir)

    C = np.zeros((K, K))
    Tm = Zm.shape[2]
    for t in range(Tm):
        Zt = Zm[:, :, t] - Zm[:, :, t].mean(axis=1, keepdims=True)
        C += Zt @ Zt.T / Zt.shape[1]
    C /= Tm

    pos = rho > 0

    def combo_ic(w, Zsub=None):
        Zz = Zv if Zsub is None else Zv[Zsub]
        c = (w[:, None, None] * Zz).sum(axis=0)
        return rank_spearman(c, tv, mv)

    # 全部 85 因子（带符号）
    w_equal = np.ones(K) / K
    w_ic = rho / (np.abs(rho).sum() + 1e-12)
    w_icir = icir / (np.abs(icir).sum() + 1e-12)
    ridge = 0.1 * np.trace(C) / K
    w_opt = np.linalg.solve(C + ridge * np.eye(K), rho)
    w_opt = w_opt / (np.abs(w_opt).sum() + 1e-12)

    # 只用正 IC 子集（剔除负 IC）
    Kpos = int(pos.sum())
    w_equal_pos = np.ones(Kpos) / Kpos
    rho_pos = rho[pos]
    Cpos = C[np.ix_(pos, pos)]
    ridge_p = 0.1 * np.trace(Cpos) / Kpos
    w_opt_pos = np.linalg.solve(Cpos + ridge_p * np.eye(Kpos), rho_pos)
    w_opt_pos = w_opt_pos / (np.abs(w_opt_pos).sum() + 1e-12)

    print(f"\n因子数 K={K}，训练期 IC>0 的因子 {Kpos} 个，IC<0 的 {K - Kpos} 个")
    print(f"单因子训练期 IC 均值={rho.mean():.4f}，|IC|均值={np.abs(rho).mean():.4f}")
    print(f"\n{'组合方式':<20}{'验证期 IC':>12}")
    print("-" * 36)
    print(f"{'等权(全部,含负IC)':<20}{combo_ic(w_equal):>12.4f}")
    print(f"{'IC加权(带符号反转)':<20}{combo_ic(w_ic):>12.4f}")
    print(f"{'ICIR加权(带符号)':<20}{combo_ic(w_icir):>12.4f}")
    print(f"{'最优 C⁻¹ρ(带符号)':<20}{combo_ic(w_opt):>12.4f}")
    print(f"{'等权(仅正IC子集)':<20}{combo_ic(w_equal_pos, pos):>12.4f}")
    print(f"{'最优(仅正IC子集)':<20}{combo_ic(w_opt_pos, pos):>12.4f}")
    print(f"\n参照：LightGBM(横截面z标签) 0.016~0.022")


if __name__ == "__main__":
    main()
