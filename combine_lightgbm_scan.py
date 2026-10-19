# -*- coding: utf-8 -*-
"""
LightGBM 规模扫描：验证"树太大 → 过拟合"的解释，看小树能否回到线性最优(0.0353)水平。

对比四个规模（叶子数 × 树数 × 最小叶子样本数）：
  极小树 / 小树 / 中树 / 大树(原)

用法：python combine_lightgbm_scan.py
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
    target = loader.target_ret.numpy()
    mask = loader.mask.numpy()
    dates = loader.dates

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_results", "csi500_batched_factors.json"), encoding="utf-8") as f:
        factors = json.load(f)

    sigs = []
    for fac in factors:
        f = vm.execute(fac["tokens"], feat)
        sigs.append(f.numpy())
    S = np.stack(sigs)                       # [K, S, T]
    K = S.shape[0]
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-12
    Z = (S - mu) / sd

    mine_idx = dates <= pd.Timestamp(MINE_END)
    val_idx = dates > pd.Timestamp(MINE_END)

    def to_xy(sl, target_sl, mask_sl):
        X = Z[:, :, sl].transpose(2, 1, 0).reshape(-1, K)
        y = target_sl.T.reshape(-1)
        m = mask_sl.T.reshape(-1).astype(bool)
        return X, y, m

    X_tr, y_tr, m_tr = to_xy(mine_idx, target[:, mine_idx], mask[:, mine_idx])
    X_val, y_val, m_val = to_xy(val_idx, target[:, val_idx], mask[:, val_idx])
    X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
    X_val_f = X_val[m_val]
    Tv = int(val_idx.sum())
    S_n = target.shape[0]

    import lightgbm as lgb
    configs = [
        ("极小树", dict(n_estimators=20, num_leaves=4, min_child_samples=200)),
        ("小树", dict(n_estimators=50, num_leaves=8, min_child_samples=100)),
        ("中树", dict(n_estimators=100, num_leaves=31, min_child_samples=20)),
        ("大树(原)", dict(n_estimators=200, num_leaves=63, min_child_samples=20)),
    ]

    print(f"\n{'规模':<12}{'验证期 IC':>12}")
    print("-" * 28)
    for label, kw in configs:
        model = lgb.LGBMRegressor(
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=7, n_jobs=-1, verbose=-1, **kw,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val_f)
        pred_2d_flat = np.full(S_n * Tv, np.nan)
        pred_2d_flat[m_val] = pred
        pred_2d = pred_2d_flat.reshape(Tv, S_n).T
        ic = rank_spearman(pred_2d, target[:, val_idx], mask[:, val_idx])
        print(f"{label:<12}{ic:>12.4f}")

    print(f"\n参照：线性最优 C⁻¹ρ = 0.0353")


if __name__ == "__main__":
    main()
