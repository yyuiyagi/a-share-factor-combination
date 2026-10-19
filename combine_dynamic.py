# -*- coding: utf-8 -*-
"""
动态组合：滚动窗口算因子权重，定期更新，看能否超过静态最优（0.0353）。

权重方式：
  - 动态 IC 加权：滚动 252 天 IC（用符号，捕捉因子方向翻转），每 63 天更新一次；
  - 动态 ICIR 加权：滚动 IC 均值/标准差（考虑稳定性）。

用法：python combine_dynamic.py
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

LOOKBACK = 252
REBALANCE = 63
VAL_START = "2022-01-01"


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


def daily_ics(f: np.ndarray, r: np.ndarray, mask: np.ndarray) -> np.ndarray:
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

    sigs = []
    for fac in factors:
        f = vm.execute(fac["tokens"], feat)
        if f is None:
            raise RuntimeError("因子执行失败")
        sigs.append(f.numpy())
    S = np.stack(sigs)                       # [K, S, T]
    K = S.shape[0]
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-12
    Z = (S - mu) / sd                        # [K, S, T]
    S_n = Z.shape[1]
    T = Z.shape[2]

    # 滚动：每 REBALANCE 天更新权重，权重用过去 LOOKBACK 天算
    combo_ic = np.full((S_n, T), np.nan)     # 组合信号 [股票, 时间]
    combo_icir = np.full((S_n, T), np.nan)
    t = LOOKBACK
    while t < T:
        end = min(t + REBALANCE, T)
        win = slice(t - LOOKBACK, t)
        w_ic = np.zeros(K)
        w_icir = np.zeros(K)
        for k in range(K):
            ic = rank_spearman(Z[k, :, win], target[:, win], mask[:, win])
            ic_series = daily_ics(Z[k, :, win], target[:, win], mask[:, win])
            icir = float(np.nanmean(ic_series) / (np.nanstd(ic_series) + 1e-12))
            w_ic[k] = ic
            w_icir[k] = icir
        w_ic = w_ic / (np.abs(w_ic).sum() + 1e-12)
        w_icir = w_icir / (np.abs(w_icir).sum() + 1e-12)
        combo_ic[:, t:end] = (w_ic[:, None, None] * Z[:, :, t:end]).sum(axis=0)
        combo_icir[:, t:end] = (w_icir[:, None, None] * Z[:, :, t:end]).sum(axis=0)
        t = end

    val = dates >= pd.Timestamp(VAL_START)
    tv = target[:, val]
    mv = mask[:, val]
    ic_dyn = rank_spearman(combo_ic[:, val], tv, mv)
    icir_dyn = rank_spearman(combo_icir[:, val], tv, mv)

    print(f"\n===== 动态组合（滚动 {LOOKBACK} 天，每 {REBALANCE} 天更新）=====")
    print(f"动态 IC 加权  验证期 IC = {ic_dyn:.4f}")
    print(f"动态 ICIR 加权 验证期 IC = {icir_dyn:.4f}")
    print(f"\n对比：")
    print(f"  静态 IC 加权   : 0.0337")
    print(f"  静态最优 C⁻¹ρ : 0.0353")


if __name__ == "__main__":
    main()
