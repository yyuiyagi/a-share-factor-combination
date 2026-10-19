# -*- coding: utf-8 -*-
"""
验证 csi500_batched_factors.json 的 15 个因子可复现：
  1. 加载 Csi500DataLoader（data_tx 数据 + 20 叶子特征）
  2. 用 StackVM 执行每条公式的 token 序列
  3. 计算每个因子的截面 RankIC（未来 1 日收益）
  4. 与 json 里记录的 oos_median 对比（量级应接近）

用法：python verify_factors.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csi500_alpha.data_loader import Csi500DataLoader
from csi500_alpha.vm import StackVM


def rank_spearman_mean(f: torch.Tensor, r: torch.Tensor) -> float:
    """因子与收益的截面 Spearman 均值（向量化）。"""
    f = f.numpy().astype(np.float64)
    r = r.numpy().astype(np.float64)

    def rank_cols(x):
        return np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float64)

    fr = rank_cols(f)
    rr = rank_cols(r)
    fc = fr - fr.mean(axis=0, keepdims=True)
    rc = rr - rr.mean(axis=0, keepdims=True)
    num = (fc * rc).sum(axis=0)
    den = np.sqrt((fc ** 2).sum(axis=0) * (rc ** 2).sum(axis=0))
    return float((num / (den + 1e-12)).mean())


def main():
    print("加载数据 ...")
    loader = Csi500DataLoader().load_data()
    print(loader.summary())

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_results", "csi500_batched_factors.json"), encoding="utf-8") as f:
        factors = json.load(f)

    vm = StackVM(n_features=20)
    feat = loader.feat_tensor      # [股票, 特征, 时间]
    target = loader.target_ret     # [股票, 时间] 未来 1 日收益

    print(f"\n{'公式(截断)':<52}{'json_oos':>10}{'复现IC':>10}")
    print("-" * 74)
    for fac in factors:
        tokens = fac["tokens"]
        f = vm.execute(tokens, feat)
        if f is None:
            print(f"{fac['decoded'][:50]:<52}{fac['oos_median']:>10.4f}{'执行失败':>10}")
            continue
        ic = rank_spearman_mean(f, target)
        print(f"{fac['decoded'][:50]:<52}{fac['oos_median']:>10.4f}{ic:>10.4f}")


if __name__ == "__main__":
    main()
