# -*- coding: utf-8 -*-
"""
确认 RL 挖出的 15 个因子是否"过时/过拟合"：逐年 IC 时间序列分析。

挖矿结构（batched.py）：mine=2016-2021（PPO 优化），val=2022-2026（独立验证）。
本脚本把 15 个因子在全样本的 IC 按年展开，区分：
  - 过拟合：挖矿期(2016-2021)高、验证期(2022-2026)骤降
  - 过时(alpha decay)：验证期 IC 逐年缓降

用法：python validate_temporal.py
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


def rank_spearman(f: np.ndarray, r: np.ndarray, mask: np.ndarray | None = None) -> float:
    """截面 Spearman 均值（可选 mask 剔除停牌日）。"""
    def rank_cols(x):
        return np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float64)

    fr = rank_cols(f)
    rr = rank_cols(r)
    if mask is not None:
        fr = np.where(mask, fr, np.nan)
        rr = np.where(mask, rr, np.nan)
    with np.errstate(all="ignore"):
        fc = fr - np.nanmean(fr, axis=0, keepdims=True)
        rc = rr - np.nanmean(rr, axis=0, keepdims=True)
        num = np.nansum(fc * rc, axis=0)
        den = np.sqrt(np.nansum(fc ** 2, axis=0) * np.nansum(rc ** 2, axis=0))
    ic = num / (den + 1e-12)
    return float(np.nanmean(ic))


def main():
    loader = Csi500DataLoader().load_data()
    vm = StackVM(n_features=20)
    feat = loader.feat_tensor
    target = loader.forward_return(5)   # 85 因子版用 5 日收益 label
    mask = loader.mask.numpy()

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_results", "csi500_batched_factors.json"), encoding="utf-8") as f:
        factors = json.load(f)

    dates = loader.dates
    years = dates.year.values
    year_list = sorted(set(years.tolist()))
    mine_years = [y for y in year_list if y <= int(MINE_END[:4])]
    val_years = [y for y in year_list if y > int(MINE_END[:4])]

    print(f"\n挖矿期 {mine_years}  vs  验证期 {val_years}\n")

    rows = []
    for i, fac in enumerate(factors):
        f = vm.execute(fac["tokens"], feat)
        if f is None:
            continue
        f = f.numpy()
        t = target.numpy()
        yearly = {}
        for y in year_list:
            idx = np.where(years == y)[0]
            yearly[y] = rank_spearman(f[:, idx], t[:, idx], mask[:, idx])
        mine_ic = float(np.mean([yearly[y] for y in mine_years]))
        val_ic = float(np.mean([yearly[y] for y in val_years]))
        rows.append({"idx": i, "mine": mine_ic, "val": val_ic, "drop": mine_ic - val_ic, **yearly})

    df = pd.DataFrame(rows)
    print(f"{'#':<3}{'mine(16-21)':>12}{'val(22-26)':>12}{'衰减':>8}" +
          "".join(f"{y:>8}" for y in val_years))
    print("-" * 90)
    for _, r in df.iterrows():
        print(f"{int(r['idx']):<3}{r['mine']:>12.4f}{r['val']:>12.4f}{r['drop']:>8.4f}" +
              "".join(f"{r[y]:>8.4f}" for y in val_years))

    print("\n===== 汇总 =====")
    print(f"挖矿期平均 IC: {df['mine'].mean():.4f}")
    print(f"验证期平均 IC: {df['val'].mean():.4f}")
    print(f"平均衰减(mine-val): {df['drop'].mean():.4f}")
    print(f"验证期仍为正的因子: {(df['val'] > 0).sum()}/{len(df)}")

    # 验证期逐年趋势（过时检测）：看 val 各年 IC 是走平还是下滑
    print("\n验证期逐年 IC 均值（过时检测）：")
    for y in val_years:
        print(f"  {y}: {df[y].mean():+.4f}")


if __name__ == "__main__":
    main()
