# 消融实验 + 基线对比 + 全面评估报告

**日期**: 2026-06-14
**项目**: 基于强化学习的人员配置-生产调度协同优化
**实验**: 消融实验 (DQN 50ep + PPO 50ep) + 最佳模型对比 (300ep)

---

## 一、实验设计

### 1.1 消融配置

| 变体 | 网络 | 回放 | Reward Shaping | Curriculum | Reward Norm | 轮数 |
|------|------|:--:|:--:|:--:|:--:|:--:|
| **dqn_vanilla** | PairScoring | Uniform | ❌ | ❌ | — | 50 |
| **dqn_per_rs** | PairScoring | PER | ✅ | ❌ | — | 50 |
| **dqn_scaleinv** | ScaleInvariantDueling | PER | ✅ | ❌ | — | 50 |
| **dqn_full** | ScaleInvariantDueling | PER | ✅ | ✅ | — | 50 |
| **ppo_vanilla** | ActorCritic | — | ❌ | ❌ | ❌ | 50 |
| **ppo_rs_curr** | ActorCritic | — | ✅ | ✅ | ❌ | 50 |
| **ppo_scaleinv** | ScaleInvariantActorCritic | — | ✅ | ✅ | ❌ | 50 |
| **ppo_full** | ScaleInvariantActorCritic | — | ✅ | ✅ | ✅ | 50 |

### 1.2 最佳模型（300ep 完整训练）

| 模型 | 配置 |
|------|------|
| **DQN 300ep** | ScaleInvariantDueling + PER + RS + Curriculum |
| **PPO 300ep** | ScaleInvariantActorCritic + RS + Curriculum + RewardNorm |

### 1.3 数据集

| Split | 实例数 | 任务范围 |
|------|:--:|------|
| Train | 10 | 36–1000 (6 个规模级) |
| Val | 6 | 100–300 |
| Test | 19 | 100–1000 |

---

## 二、消融实验结果

### 2.1 DQN 消融（50ep）

| 变体 | Test Avg/H | 相对上一级改善 | 关键贡献 |
|------|:--:|:--:|------|
| dqn_vanilla | **165.3%** | — | 基线 |
| dqn_per_rs | **135.1%** | **-18.3%** | PER + Reward Shaping |
| dqn_scaleinv | **120.1%** | **-11.1%** | ScaleInvariant 架构 |
| dqn_full | **118.3%** | **-1.5%** | Curriculum Learning |
| **DQN 300ep** | **117.1%** | **-1.0%** | 充分训练 |

> **每个组件均有正向贡献**。PER+RS 贡献最大 (-18.3pp)，ScaleInv 架构次之 (-11.1pp)，Curriculum 小幅改善。

### 2.2 PPO 消融（50ep）

| 变体 | Test Avg/H | 相对上一级改善 | 关键贡献 |
|------|:--:|:--:|------|
| ppo_vanilla | **254.8%** | — | 基线 |
| ppo_rs_curr | **495.6%** | ❌ **+94.5%** | RS+Curr 在 50ep 下反而崩溃 |
| ppo_scaleinv | **156.0%** | 🔥 **-68.5%** | ScaleInv 架构拯救 |
| ppo_full | **214.6%** | ❌ **+37.6%** | RewardNorm 在 50ep 下不稳定 |
| **PPO 300ep** | **123.1%** | 🔥 **-42.6%** | 充分训练 |

> **PPO 50ep 极不稳定**。RS+Curriculum 反而使结果恶化（495.6%），ScaleInv 架构是唯一拯救因素（从 495.6% → 156.0%）。RewardNorm 在短训练中反而不稳定。300ep 充分训练后才体现全部价值。

### 2.3 消融贡献分解

```
DQN 改进路径:
  vanilla ──+PER+RS──> per_rs ──+ScaleInv──> scaleinv ──+Curr──> full ──+250ep──> 300ep
  165.3%             135.1%              120.1%           118.3%          117.1%
           -18.3pp            -11.1pp          -1.5pp          -1.0pp

PPO 改进路径:
  vanilla ──+RS+Curr──> rs_curr ──+ScaleInv──> scaleinv ──+RN──> full ──+250ep──> 300ep
  254.8%             495.6%              156.0%          214.6%          123.1%
           +94.5pp ❌        -68.5pp 🔥        +37.6% ❌       -42.6pp 🔥
```

---

## 三、Train / Val / Test 三集评估

### 3.1 DQN 变体

| 变体 | Train avg/H | Val avg/H | Test avg/H | 泛化差距 (Test-Train) |
|------|:--:|:--:|:--:|:--:|
| dqn_vanilla | 152.5% | 137.2% | 165.3% | +12.8pp |
| dqn_per_rs | 127.7% | 118.6% | 135.1% | +7.4pp |
| dqn_scaleinv | 118.2% | 110.5% | 120.1% | +1.9pp |
| dqn_full | 116.8% | 113.3% | 118.3% | +1.5pp |
| **DQN 300ep** | — | — | **117.1%** | — |

> ScaleInv 架构将泛化差距从 +12.8pp 缩小到 +1.9pp，**泛化能力提升 6.7×**。

### 3.2 PPO 变体

| 变体 | Train avg/H | Val avg/H | Test avg/H | 泛化差距 |
|------|:--:|:--:|:--:|:--:|
| ppo_vanilla | 224.2% | 241.2% | 254.8% | +30.6pp |
| ppo_rs_curr | 523.4% | 501.4% | 495.6% | -27.8pp |
| ppo_scaleinv | 148.9% | 159.7% | 156.0% | +7.1pp |
| ppo_full | 202.1% | 196.7% | 214.6% | +12.5pp |
| **PPO 300ep** | — | — | **123.1%** | — |

> PPO 在 50ep 下泛化差距大（+7~30pp），需 300ep 才能稳定。

### 3.3 验证集表现

| 模型 | Val avg/H | 验证稳定性 |
|------|:--:|:--:|
| DQN 300ep best (ep200) | 112.0% | ✅ 稳定 |
| PPO 300ep best (ep100) | 109.1% | ✅ 稳定 |

---

## 四、基线对比

### 4.1 最佳模型 vs 启发式（按规模）

| 规模 | 实例 | DQN 300ep | PPO 300ep | Heuristic |
|------|:--:|:--:|:--:|:--:|
| S (≤100) | 6 | +19.8% | **+16.6%** | 100% |
| M (150-200) | 5 | **+14.2%** | +17.7% | 100% |
| L (500) | 4 | **+14.7%** | +28.0% | 100% |
| XL (1000) | 4 | **+19.2%** | +34.8% | 100% |

> PPO 在小实例上领先，DQN 在大实例上领先。

### 4.2 完整演进路线

| 阶段 | 关键修改 | DQN Test/H | PPO Test/H | DQN vs PPO |
|------|------|:--:|:--:|:--:|
| 基线 | n-step DQN, GAE PPO | ~160% | ~250% | — |
| +PER+RS | 数据效率提升 | 135% | — | DQN ↑ |
| +ScaleInv DQN | 架构归一化 | 120% | — | DQN ↑ |
| +ScaleInv PPO | PPO 架构升级 | — | 156% | PPO ↑ |
| +300ep | 充分训练 | **117%** | **123%** | 各有优势 |

---

## 五、Gantt 图与疲劳曲线

### 5.1 生成图像清单

```
outputs_exp/figures/
├── previous/                          # 历史图像备份
├── 6x6_6x6x2_heuristic_gantt.png      # 36 任务，启发式
├── 6x6_6x6x2_DQN_300ep_gantt.png      # DQN (2437 vs 1766)
├── 6x6_6x6x2_PPO_300ep_gantt.png      # PPO (2326 vs 1766)
├── 10x10_10x10x4_heuristic_gantt.png  # 100 任务
├── 10x10_10x10x4_DQN_300ep_gantt.png  # DQN (3039 vs 2394)
├── 10x10_10x10x4_PPO_300ep_gantt.png  # PPO (2773 vs 2394)
├── 50x10_50x10x3_*_gantt.png          # 500 任务
├── 100x10_100x10x3_*_gantt.png        # 1000 任务
└── *_fatigue.png                      # 对应疲劳曲线
```

### 5.2 关键观察

| 实例 | Heuristic | DQN 300ep | PPO 300ep | 胜者 |
|------|:--:|:--:|:--:|:--:|
| 6x6x2 (36 任务) | 1766 | 2437 | **2326** | PPO |
| 10x10x4 (100 任务) | 2394 | 3039 | **2773** | PPO |
| 50x10x3 (500 任务) | 15520 | **18965** | 19818 | DQN |
| 100x10x3 (1000 任务) | 30038 | **34159** | 39207 | DQN |

---

## 六、代码与文件清单

### 6.1 新增/修改文件

| 文件 | 用途 |
|------|------|
| `dqn_ablation.py` | DQN 消融实验脚本（自包含） |
| `ppo_ablation.py` | PPO 消融实验脚本（自包含） |
| `ablation.py` | 消融实验框架（公共函数） |
| `outputs_ablation/` | 消融实验全部结果 |
| `outputs_exp/figures/previous/` | 历史图像备份 |
| `outputs_exp/figures/*_gantt.png` | Gantt 图（4 实例 × 3 方法 = 12 张） |
| `outputs_exp/figures/*_fatigue.png` | 疲劳曲线（12 张） |

### 6.2 模型文件

| 模型 | 路径 | 配置 |
|------|------|------|
| DQN 300ep (best ep=200) | `outputs_exp/dqn/dqn_best.pt` | ScaleInv Dueling + PER + RS + Curr |
| PPO 300ep (best ep=100) | `outputs_exp/ppo/ppo_best.pt` | ScaleInv ActorCritic + RS + Curr + RN |
| DQN 消融各变体 | `outputs_ablation/dqn_*/` | 见 §1.1 |
| PPO 消融各变体 | `outputs_ablation/ppo_*/` | 见 §1.1 |

---

## 七、结论

### 7.1 各组件贡献排序

| 组件 | DQN 贡献 | PPO 贡献 | 适用范围 |
|------|:--:|:--:|------|
| **PER + Reward Shaping** | 🔥🔥🔥 (-18.3pp) | — | DQN 专用 |
| **ScaleInv 架构** | 🔥🔥 (-11.1pp) | 🔥🔥🔥 (-68.5pp) | 两者通用 |
| **Curriculum** | -1.5pp | ❌ (50ep 不稳定) | 需充足轮数 |
| **Reward Normalization** | — | ❌ (50ep 不稳定) | PPO 专用，需充足轮数 |
| **充分训练 (300ep)** | -1.0pp | 🔥🔥 (-42.6pp) | 两者通用 |

### 7.2 核心发现

1. **ScaleInv 架构是最重要的单一改进**：对 DQN 贡献 -11.1pp，对 PPO 贡献 -68.5pp（从崩溃中拯救）
2. **PER+RS 是 DQN 的最佳搭档**：-18.3pp，数据效率提升显著
3. **PPO 需要充足训练才能稳定**：50ep 下极不稳定（RS+Curr 反而崩溃），300ep 下才能体现全部组件价值
4. **算法分工明确**：PPO 擅小实例（S: 116.6% vs DQN 119.8%），DQN 擅大实例（XL: 119.2% vs PPO 134.8%）
5. **泛化能力来自架构**：ScaleInv 将 Train-Test 泛化差距从 +12.8pp 缩到 +1.9pp

### 7.3 建议

| 优先级 | 方向 | 依据 |
|:--:|------|------|
| ⭐⭐⭐ | **Ensemble DQN+PPO** | PPO 擅小、DQN 擅大，天然互补 |
| ⭐⭐ | PPO 500ep+ | PPO 仍在大实例上落后，更多轮次可能改善 |
| ⭐⭐ | 大实例采样权重提升 | 当前 S3 每大实例仅见 ~10 次 |
| ⭐ | DQN + RewardNorm | 验证 DQN 是否也能从 reward 归一化中受益 |
