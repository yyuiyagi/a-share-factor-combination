# -*- coding: utf-8 -*-
"""
鲁棒性检验（过拟合检测的第二道防线）：

  测试 1 特征微扰：给 20 个叶子特征加高斯噪声 N(0, σ²)，σ∈{0.1, 0.3}，看因子 IC 是否崩。
  测试 2 参数微扰：把公式里的 20 日窗口算子（MEAN20/STD20/RANK20）换成 15/25 日，
                    看因子 IC 是否稳定（过拟合的因子对窗口极其敏感）。

用法：python validate_robustness.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csi500_alpha.data_loader import Csi500DataLoader
from csi500_alpha.ops import OPS, _ts_mean, _ts_std, _ts_rank

# 扩展算子表：追加 15/25 窗口的时序算子（token 40~45）
EXTRA_OPS = [
    ("MEAN15", lambda a: _ts_mean(a, 15), 1, "ts"),
    ("MEAN25", lambda a: _ts_mean(a, 25), 1, "ts"),
    ("STD15", lambda a: _ts_std(a, 15), 1, "ts"),
    ("STD25", lambda a: _ts_std(a, 25), 1, "ts"),
    ("RANK15", lambda a: _ts_rank(a, 15), 1, "ts"),
    ("RANK25", lambda a: _ts_rank(a, 25), 1, "ts"),
]
FULL_OPS = list(OPS) + EXTRA_OPS
N_FEAT = 20

# 原 token：MEAN20=35, STD20=36, RANK20=37
REPLACE_15 = {35: N_FEAT + len(OPS) + 0, 36: N_FEAT + len(OPS) + 2, 37: N_FEAT + len(OPS) + 4}
REPLACE_25 = {35: N_FEAT + len(OPS) + 1, 36: N_FEAT + len(OPS) + 3, 37: N_FEAT + len(OPS) + 5}


class ExtStackVM:
    def __init__(self):
        self.op_map = {}
        self.arity = {}
        for i, (_, func, a, _) in enumerate(FULL_OPS):
            self.op_map[N_FEAT + i] = func
            self.arity[N_FEAT + i] = a

    def execute(self, tokens, feat):
        stack = []
        try:
            for tok in tokens:
                tok = int(tok)
                if tok < N_FEAT:
                    stack.append(feat[:, tok, :])
                elif tok in self.op_map:
                    a = self.arity[tok]
                    if len(stack) < a:
                        return None
                    args = [stack.pop() for _ in range(a)]
                    args.reverse()
                    res = self.op_map[tok](*args)
                    res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            return stack[0] if len(stack) == 1 else None
        except Exception:
            return None


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


def main():
    torch.manual_seed(0)
    loader = Csi500DataLoader().load_data()
    vm = ExtStackVM()
    feat = loader.feat_tensor          # [S, 20, T]
    target = loader.target_ret
    mask = loader.mask.numpy()

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_results", "csi500_batched_factors.json"), encoding="utf-8") as f:
        factors = json.load(f)

    def ic_of(f):
        return rank_spearman(f.numpy(), target.numpy(), mask)

    print("=" * 100)
    print("测试 1：特征微扰（高斯噪声）—— 因子 IC 是否随噪声崩掉")
    print("=" * 100)
    print(f"{'#':<4}{'原IC':>9}{'σ=0.1':>9}{'σ=0.3':>9}{'σ=0.3降幅':>10}")
    print("-" * 60)
    for i, fac in enumerate(factors):
        f0 = vm.execute(fac["tokens"], feat)
        ic0 = ic_of(f0)
        ics = []
        for sigma in [0.1, 0.3]:
            noise = torch.randn_like(feat) * sigma
            f = vm.execute(fac["tokens"], feat + noise)
            ics.append(ic_of(f))
        drop = (ic0 - ics[1]) / (abs(ic0) + 1e-12)
        print(f"{i:<4}{ic0:>9.4f}{ics[0]:>9.4f}{ics[1]:>9.4f}{drop:>10.0%}")

    print("\n" + "=" * 100)
    print("测试 2：参数微扰（20 日窗口 → 15 / 25 日）—— 因子 IC 是否稳定")
    print("=" * 100)
    print(f"{'#':<4}{'原20窗口':>10}{'15窗口':>9}{'25窗口':>9}{'相对波动':>10}")
    print("-" * 60)
    for i, fac in enumerate(factors):
        f0 = vm.execute(fac["tokens"], feat)
        ic0 = ic_of(f0)
        toks15 = [REPLACE_15.get(t, t) for t in fac["tokens"]]
        toks25 = [REPLACE_25.get(t, t) for t in fac["tokens"]]
        ic15 = ic_of(vm.execute(toks15, feat))
        ic25 = ic_of(vm.execute(toks25, feat))
        spread = (max(ic0, ic15, ic25) - min(ic0, ic15, ic25)) / (abs(ic0) + 1e-12)
        print(f"{i:<4}{ic0:>10.4f}{ic15:>9.4f}{ic25:>9.4f}{spread:>10.0%}")


if __name__ == "__main__":
    main()
