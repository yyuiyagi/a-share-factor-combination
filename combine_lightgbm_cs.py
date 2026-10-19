# -*- coding: utf-8 -*-
"""
验证假设：之前 LightGBM 崩是因为"标签用了绝对收益(未横截面标准化)"，
而非"信噪比太低"。这里把标签(5日收益)做横截面 z-score，重跑 LightGBM。

对比：绝对收益标签 vs 横截面 z-score 标签。
用法：python combine_lightgbm_cs.py
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


def cs_zscore(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """横截面 z-score（每天，跨股票），mask 之外视为 NaN。"""
    xm = np.where(mask, x, np.nan)
    mu = np.nanmean(xm, axis=0, keepdims=True)
    sd = np.nanstd(xm, axis=0, keepdims=True)
    return (x - mu) / (sd + 1e-12)


def main():
    loader = Csi500DataLoader().load_data()
    vm = StackVM(n_features=20)
    feat = loader.feat_tensor
    target = loader.forward_return(5).numpy()     # 5 日收益
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
    S = np.stack(sigs)                            # [K, S, T]
    K = S.shape[0]
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-12
    Z = (S - mu) / sd

    mine_idx = dates <= pd.Timestamp(MINE_END)
    val_idx = dates > pd.Timestamp(MINE_END)

    # 标签横截面标准化
    target_cs = cs_zscore(target, mask)

    def to_xy(sl, y):
        X = Z[:, :, sl].transpose(2, 1, 0).reshape(-1, K)
        yy = y[:, sl].T.reshape(-1)
        m = mask[:, sl].T.reshape(-1).astype(bool)
        return X[m], yy[m], m

    import lightgbm as lgb
    configs = [
        ("小树", dict(n_estimators=50, num_leaves=8, min_child_samples=100)),
        ("中树", dict(n_estimators=100, num_leaves=31, min_child_samples=20)),
        ("大树", dict(n_estimators=200, num_leaves=63, min_child_samples=20)),
    ]

    print(f"\n因子数 K={K}")
    print(f"{'规模':<10}{'绝对收益标签':>14}{'横截面z标签':>14}")
    print("-" * 42)
    for label, kw in configs:
        row = []
        for y_tag in [target, target_cs]:
            X_tr, y_tr, m_tr = to_xy(mine_idx, y_tag)
            X_val, y_val, m_val = to_xy(val_idx, y_tag)
            model = lgb.LGBMRegressor(
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=7, n_jobs=-1, verbose=-1, **kw,
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            Tv = int(val_idx.sum())
            S_n = target.shape[0]
            flat = np.full(S_n * Tv, np.nan)
            flat[m_val] = pred
            pred_2d = flat.reshape(Tv, S_n).T
            # 注意：横截面 z 标签预测出来的是"z 排名"，与横截面 z 的真实收益比较
            y_true = y_tag[:, val_idx]
            ic = rank_spearman(pred_2d, y_true, mask[:, val_idx])
            row.append(ic)
        print(f"{label:<10}{row[0]:>14.4f}{row[1]:>14.4f}")

    print(f"\n参照：线性最优 C⁻¹ρ ≈ 0.0353（15 因子时）")


if __name__ == "__main__":
    main()
