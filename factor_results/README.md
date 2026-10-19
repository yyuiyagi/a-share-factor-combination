# 因子结果文件夹

本文件夹集中存放 CSI500 因子挖掘的全部成果，方便后续组合实验使用。

## 文件清单

### 说明文档
| 文件 | 内容 |
|------|------|
| `因子挖掘记录.md` | 完整实验记录（有效秩对比、Alpha158 质量检测、复现要点） |
| `特征与算子说明.md` | 20 个叶子特征 + 20 个算子的完整含义（词表 token 映射） |

### 因子文件
| 文件 | 内容 |
|------|------|
| `csi500_batched_factors.json` | **15 个正交代表因子**（分批贪心，后续组合实验的输入） |
| `factor_library.json` | 汇总（15 代表 + top10 + top500 + 元数据） |
| `csi500_top500_factors.json` / `.csv` | 500 因子（按 OOS 中位数排序） |
| `csi500_factors.json` | 最新 select() 的 top10 |
| `csi500_pool.json` | 12800 条原始候选池（最新一次挖矿） |

### Alpha158 参照
| 文件 | 内容 |
|------|------|
| `alpha158_ic.json` / `.csv` | 158 个 Alpha158 因子的逐因子 IC 质量检测 |
| `alpha158_eigenvalues.npy` / `alpha158_corr_matrix.npy` | Alpha158 相关矩阵与特征值 |

### 相关矩阵 / 特征值
| 文件 | 内容 |
|------|------|
| `csi500_corr_matrix.npy` / `csi500_eigenvalues.npy` | top500 相关矩阵与特征值 |

## 核心结论速览

- **有效秩**：单次 3.70 / 弱新颖性 2.43 / **分批贪心 6.45** / Alpha158 9.06。
- **质量**：分批挖的 15 个代表 OOS 中位数 **+0.025、100% 正 IC**；Alpha158 中位数 −0.006、仅 42% 正。
- **组合实验输入**：`csi500_batched_factors.json`（15 个正交代表，带 tokens 可复现 + decoded 公式 + oos_median/mine_ic）。

## 复现

- 数据加载：`csi500_alpha/data_loader.py`（`Csi500DataLoader`）
- 叶子/算子：`csi500_alpha/features.py` / `csi500_alpha/ops.py` / `csi500_alpha/vm.py`
- 挖矿：`csi500_alpha/engine.py`（PPO）或 `csi500_alpha/batched.py`（分批正交贪心）
- 验证：`segment_eval`（样本外按年分段，取 IC 中位数）
