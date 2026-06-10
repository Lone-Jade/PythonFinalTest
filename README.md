# 基于强化学习的人员配置-生产调度协同优化

> Reinforcement Learning-Based Personnel Allocation & Production Scheduling Collaborative Optimization

---

> **同济大学 Python 人工智能程序设计 课程论文**
>
> *Tongji University — Python Artificial Intelligence Programming — Course Paper*

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 目录 | Table of Contents

- [中文](#中文)
  - [1. 项目概述](#1-项目概述)
  - [2. 问题建模](#2-问题建模)
  - [3. 实现的模型](#3-实现的模型)
  - [4. 对比基线](#4-对比基线)
  - [5. 快速开始](#5-快速开始)
  - [6. 使用指南](#6-使用指南)
  - [7. 实验结果总结](#7-实验结果总结)
  - [8. 文件结构](#8-文件结构)
  - [9. 核心结论](#9-核心结论)
- [English](#english)
  - [1. Project Overview](#1-project-overview)
  - [2. Problem Formulation](#2-problem-formulation)
  - [3. Implemented Models](#3-implemented-models)
  - [4. Baselines](#4-baselines)
  - [5. Quick Start](#5-quick-start)
  - [6. Usage Guide](#6-usage-guide)
  - [7. Results Summary](#7-results-summary)
  - [8. File Structure](#8-file-structure)
  - [9. Key Findings](#9-key-findings)

---

# 中文

## 1. 项目概述

本项目解决**柔性作业车间调度问题（FJSP）**：在考虑操作员疲劳度动态变化的情况下，同时优化**最大完工时间（Makespan）**和**操作员疲劳度**。将人员配置与生产调度建模为联合优化 MDP，应用深度强化学习求解。

### 核心特性

- **两种 DRL 算法**：Dueling Double DQN（+ PER + N-step）和 PPO（+ GAE）
- **Dueling DQN V2** 🆕：优化变体，含 Noisy Nets、硬目标更新、余弦学习率退火
- **疲劳模型**：α-β 疲劳动态 + 加工效率影响
- **Action Masking**：保证调度动作合法性
- **事件驱动仿真**：高效离散事件模拟环境
- **全面基准测试**：10 个问题规模 × 5 种方法 × 2 轮训练

### 问题规模

数据集命名规则 `{作业数}x{机器数}x{操作员数}.csv`：

| 规模 | 示例 | 状态维度 | 动作维度 | 说明 |
|------|------|---------|---------|------|
| 小规模 | 6×6×2 | 43 | 12 | 微型车间 |
| 中规模 | 10×10×3 | 70 | 30 | 典型配置 |
| 大规模 | 30×10×3 | 130 | 90 | 工业规模 |
| 超大规模 | 100×10×6 | 310 | 600 | 压力测试 |

---

## 2. 问题建模

### MDP 形式化

| 组件 | 描述 |
|------|------|
| **状态空间** | `3M + 3W + 3N + 1` 维向量：机器状态(空闲+剩余时间)、操作员状态(空闲+疲劳度)、工件状态(进度+下一机器+剩余工时)、机器需求信号、操作员负载信号、当前时间 |
| **动作空间** | `N × W` 维 (job_id, worker_id) 对，将工件下一工序分配给指定操作员 |
| **奖励函数** | 即时：`-(实际加工时间) / 归一化尺度`；终端：`-λ × Σ max(0, F - F_threshold)` + makespan 惩罚 |
| **状态转移** | 事件驱动：选择动作 → 分配工序 → 推进仿真时钟 → 释放资源 → 到达下一决策点 |

### 疲劳模型

```
工作时:   dF/dt = +α   (α=0.02，疲劳积累)
空闲时:   dF/dt = -β   (β=0.025，疲劳恢复)
加工影响:  实际时间 = 基础时间 × (1 + γ × F)   (γ=0.05)
终端惩罚:  -λ × Σ max(0, F - F_threshold)   (λ=2.0)
```

- 疲劳在工作中积累、空闲时恢复
- 高疲劳 → 实际加工时间延长 → 完工时间推迟
- 终端惩罚 λ=2.0 防止过度使用单一操作员

---

## 3. 实现的模型

### 3.1 Dueling Double DQN + PER + N-step Returns

**文件**: [`agent.py`](XIWEI_code/agent.py) · [`train_dqn.py`](XIWEI_code/train_dqn.py)

**网络架构**:
```
输入层(state_dim) → [256, 128] 共享特征 →
  Value Stream: 64 → 1          (估计 V(s))
  Advantage Stream: 64 → N×W    (估计 A(s,a))
输出: Q(s,a) = V(s) + A(s,a) − mean(A(s,a))
```

**关键技术特性**:

| 特性 | 实现 |
|------|------|
| Double DQN | 解耦动作选择与价值评估，减少 Q 值高估 |
| Dueling Architecture | 分离 V(s) 和 A(s,a)，大动作空间下更稳定 |
| Prioritized Experience Replay | SumTree 实现，优先采样 TD 误差大的样本 |
| N-step Returns | N=3，加速信用传播 |
| Action Masking | 无效动作 Q 值掩码为 −∞ |
| 目标网络 | Polyak 软更新 τ=0.005 |
| ε-greedy 探索 | 线性衰减：1.0 → 0.02，衰减 100k 步 |

**超参数**: lr=3e-4, γ=0.95, batch=64, memory=50k, hidden=[256,128,64]

### 3.2 DQN V2（优化版）🆕

**文件**: [`config.py`](XIWEI_code/config.py) (DQNConfigV2) · [`train_dqn_optimized.py`](XIWEI_code/train_dqn_optimized.py)

相比原始 DQN 的 **8 项关键改进**：

| 改进项 | 原始 DQN | DQN V2 | 改进理由 |
|--------|----------|--------|---------|
| **探索机制** | ε-greedy | Noisy Nets + ε-greedy | 状态依赖探索（Rainbow DQN 核心组件） |
| **目标网络** | 软更新 τ=0.005 | 硬拷贝每 1000 步 | 更清晰的学习信号 |
| **N-step** | 3 | 7 | 更快信用传播通过长序列 |
| **折扣因子 γ** | 0.95 | 0.99 | 更好的长期价值估计 |
| **学习率** | 3e-4 常数 | 1e-3→1e-5 余弦退火 | 快速初始学习 + 精细收敛 |
| **网络结构** | [256, 128, 64] | [512, 256, 128, 64] | 更大容量捕捉复杂状态 |
| **批次/记忆池** | 64 / 50k | 128 / 100k | 更稳定梯度 + 多样经验 |
| **PER α** | 0.6 | 0.7 | 更强优先级采样 |

**核心结果**: 300 集时在 **4/10** 数据集上优于原始 DQN，最佳改进 **-9.1%**（10×10×3）。

### 3.3 PPO + GAE

**文件**: [`agent_ppo.py`](XIWEI_code/agent_ppo.py) · [`train_ppo.py`](XIWEI_code/train_ppo.py)

**网络架构**:
```
输入层(state_dim) → [256, 128] 共享特征 →
  Actor Head:  128 → N×W  (策略分布 logits)
  Critic Head: 128 → 1    (状态价值 V(s))
```

**关键技术特性**:

| 特性 | 实现 |
|------|------|
| Actor-Critic | 共享特征提取器 |
| GAE | λ=0.95，低方差优势估计 |
| PPO Clipping | ε=0.2，限制策略更新幅度 |
| Action Masking | 无效动作 logits 设为 −1e10 |
| Orthogonal 初始化 | 稳定训练 |
| Entropy Bonus | 系数 0.01，鼓励探索 |
| Rollout Buffer | 每 10 个 episode 进行一次 PPO 更新 |

**超参数**: lr=3e-4, γ=0.99, clip=0.2, hidden=[256,128]

### 3.4 DQN vs PPO 方法对比

| 维度 | DQN | PPO |
|------|-----|-----|
| **学习范式** | 值函数逼近 (Q-learning) | 直接策略优化 (Policy Gradient) |
| **输出** | Q(s,a) 值函数 | π(a\|s) 策略分布 + V(s) |
| **信用分配** | N-step bootstrapping | 完整 episode 的 GAE 回报 |
| **样本效率** | 高（经验回放重用样本） | 较低（on-policy，更新后丢弃） |
| **探索机制** | ε-greedy（值无关探索） | Entropy bonus + 随机采样（策略驱动） |
| **稳定性** | 需要目标网络 + PER 稳定 | PPO clip 天然稳定 |
| **训练速度** | 快（每步更新） | 较慢（批量更新），但 episode 级更快 |

---

## 4. 对比基线

| 方法 | 描述 | 实现位置 |
|------|------|---------|
| **Greedy SPT** | 每次选择实际加工时间最短的 (工件, 操作员) 组合（考虑疲劳影响） | [`environment.py`](XIWEI_code/environment.py) `GreedyScheduler` |
| **Round-Robin** | 轮询分配操作员，公平负载分配 | [`train_baselines.py`](XIWEI_code/train_baselines.py) |
| **Random** | 随机选择有效动作（30 次平均） | [`train_baselines.py`](XIWEI_code/train_baselines.py) |

---

## 5. 快速开始

```bash
# 安装依赖
pip install torch numpy matplotlib

# 训练 DQN（默认 10×10×3，2000 集）
cd XIWEI_code
python train_dqn.py

# 训练 PPO
python train_ppo.py --data 10x10x3 --episodes 2000

# 多规模批量训练（5 个规模 × 2 种方法）
python train_all.py --episodes 300

# 训练 DQN V2（优化版）
python train_dqn_optimized.py --episodes 300

# 评估已训练模型
python test.py --model checkpoints/dqn_10x10x3_final.pt --data 10x10x3
```

---

## 6. 使用指南

### 训练命令

```bash
# DQN — 标准训练
python train_dqn.py --data 6x6x2 --episodes 500 --lr 5e-4

# DQN — 消融实验（禁用 PER）
python train_dqn.py --data 10x10x3 --no_per

# DQN V2 — 优化训练（支持多数据集同时训练）
python train_dqn_optimized.py --episodes 300 --datasets 10x10x3 30x10x3

# PPO
python train_ppo.py --data 6x6x2 --episodes 500 --lr 3e-4 --gamma 0.99

# 多规模批量训练
python train_all.py --episodes 1000 --datasets 10x5x3 10x10x6 15x10x2
```

### 评估命令

```bash
# 评估所有基线方法
python train_baselines.py

# 加载 checkpoint 测试
python test.py --model checkpoints/ppo_15x10x2_ep1000.pt --data 15x10x2

# 疲劳惩罚权重 λ 敏感性分析
python test.py --data 6x6x2 --compare_weights --episodes 500
```

### 可视化

```bash
# 自动扫描 logs/ 生成所有图表
python visualize.py

# 指定日志文件
python visualize.py --log logs/train_10x10x3_nstep3.json

# 比较多次训练运行
python visualize.py --compare logs/

# DQN V2 专项分析图表
python analyze_dqn_v2.py
```

### Python API 示例

```python
from config import Config
from utils import load_csv_data, get_data_path, get_state_dim, get_action_dim
from environment import JSPEnvironment
from agent import DQNAgent
from agent_ppo import PPOAgent

# 加载数据，创建环境
config = Config()
data = load_csv_data('data/csv_output/10x10x3.csv')
env = JSPEnvironment(data, config.env)

# ── DQN 推理 ──
agent = DQNAgent(get_state_dim(data), get_action_dim(data), config.dqn)
agent.load('XIWEI_code/checkpoints/dqn_10x10x3_final.pt')

state = env.reset()
done = False
while not done:
    mask = env._get_action_mask()
    action = agent.select_action(state, mask, epsilon=0.0)
    state, reward, done, _ = env.step(action)

print(f"DQN Makespan: {env.get_makespan():.1f}")

# ── PPO 推理 ──
env.reset()
ppo = PPOAgent(get_state_dim(data), get_action_dim(data), config.ppo)
ppo.load('XIWEI_code/checkpoints/ppo_10x10x3_final.pt')

state = env.reset()
done = False
while not done:
    mask = env._get_action_mask()
    action, _ = ppo.evaluate(state, mask)
    state, reward, done, _ = env.step(action)

print(f"PPO Makespan: {env.get_makespan():.1f}")
```

---

## 7. 实验结果总结

### 7.1 第一轮：300 集 × 5 规模（Phase 3）

| 数据集 | Greedy | DQN | PPO | 最佳 RL vs Greedy |
|--------|--------|-----|-----|-------------------|
| 6×6×2 | **2,686** | 2,858 | 3,350 | DQN +6.4% |
| 10×10×3 | **6,834** | 10,159 | 9,822 | PPO +43.7% |
| 15×10×3 | **9,341** | 13,976 | 13,582 | PPO +45.4% |
| 20×10×3 | **12,968** | 20,395 | 19,348 | PPO +49.2% |
| 30×10×3 | **19,276** | 32,339 | 27,259 | PPO +41.4% |

> **Greedy SPT 5/5 全胜**。PPO 在 3/5 数据集上优于 DQN，且随规模增大优势扩大。

**图表**: [`charts/dataset_comparison_makespan.png`](XIWEI_code/charts/dataset_comparison_makespan.png)

### 7.2 第二轮：1000 集 × 5 规模（Phase 4）

| 数据集 | Greedy | DQN | PPO | 最佳 RL vs Greedy |
|--------|--------|-----|-----|-------------------|
| 10×5×3 | **3,419** | 4,628 | 4,254 | PPO +24.4% |
| 10×10×6 | **5,325** | 8,692 | 7,388 | PPO +38.7% |
| 15×10×2 | **10,194** | 14,886 | **10,659** | **PPO +4.6%** ⭐ |
| 15×5×3 | **4,648** | 5,394 | 5,144 | PPO +10.7% |
| 20×5×3 | **6,457** | 9,609 | 7,228 | PPO +11.9% |

> **PPO 5/5 完胜 DQN**。🏆 **15×10×2** 实现最佳 RL 结果：PPO best MS 仅差 Greedy 3.2%，且疲劳不到 Greedy 的一半。

**图表**: [`charts/method_comparison_15x10x2.png`](XIWEI_code/charts/method_comparison_15x10x2.png)

### 7.3 DQN V2 优化实验（300 集 × 10 规模）🆕

| 数据集 | DQN 原始 | DQN V2 | V2 vs 原始 | 判定 |
|--------|----------|--------|------------|------|
| 10×10×3 | 10,159 | 9,238 | **−9.1%** | ✅ 更优 |
| 10×10×6 | 8,692 | 8,172 | **−6.0%** | ✅ 更优 |
| 30×10×3 | 32,339 | 30,979 | **−4.2%** | ✅ 更优 |
| 20×5×3 | 9,609 | 9,281 | **−3.4%** | ✅ 更优 |
| 15×10×2 | 14,886 | 15,147 | +1.8% | ❌ 更差 |
| 20×10×3 | 20,395 | 21,008 | +3.0% | ❌ 更差 |
| 15×10×3 | 13,976 | 14,824 | +6.1% | ❌ 更差 |
| 10×5×3 | 4,628 | 5,058 | +9.3% | ❌ 更差 |
| 15×5×3 | 5,394 | 6,152 | +14.1% | ❌ 更差 |
| 6×6×2 | 2,858 | 3,474 | +21.6% | ❌ 更差 |

> **DQN V2 在 4/10 数据集上胜出**。大动作空间（≥30 维）上最有效；小规模上网络过参数化。

**详细分析**: [`analysis_dqn_v2.md`](XIWEI_code/analysis_dqn_v2.md)  
**图表**: [`charts/dqn_v2_vs_original_makespan.png`](XIWEI_code/charts/dqn_v2_vs_original_makespan.png)

### 7.4 综合排名

| 排名 | 方法 | 相对 Greedy 平均差距 | 训练速度 | 稳定性 |
|------|------|---------------------|---------|--------|
| 🥇 | **Greedy SPT** | 0% | 即时 | 完美 |
| 🥈 | **PPO** | +5~45%（随集数改善） | 快（DQN 的 4-8×） | 逐步提升 |
| 🥉 | **DQN V2** | +35~62% | 慢（大网络） | 混合（4/10 胜） |
| 4 | **DQN** | +6~68% | 中等 | 随规模退化 |

---

## 8. 文件结构

```
PythonFinalTest/
├── README.md                          # 本文件（中英双语）
├── CLAUDE.md                          # Claude Code 指令
├── 基于强化学习的人员配置-生产调度协同优化(2).pdf  # 参考论文
├── data/
│   ├── basic_data.xlsx                # 原始数据
│   └── csv_output/                    # CSV 数据集（40 个规模）
│       ├── 6x6x2.csv
│       ├── 10x10x3.csv
│       ├── 30x10x3.csv
│       └── ... (37 more)
│
└── XIWEI_code/                        # 核心代码
    ├── __init__.py
    ├── config.py                      # 超参数配置（Env, DQN, DQNConfigV2, PPO, Train）
    ├── environment.py                 # JSP 环境模拟器 + GreedyScheduler 基线
    ├── agent.py                       # DQN 智能体（DuelingDQN + NoisyLinear + SumTree PER）
    ├── agent_ppo.py                   # PPO 智能体（ActorCritic + GAE）
    ├── train_dqn.py                   # DQN 训练（N-step + PER + 早停）
    ├── train_dqn_optimized.py         # DQN V2 训练（NoisyNets + 硬更新）🆕
    ├── train_ppo.py                   # PPO 训练（Rollout + PPO Update）
    ├── train_all.py                   # 多规模批量训练编排器
    ├── train_baselines.py             # Greedy / Random / Round-Robin 评估
    ├── test.py                        # 模型测试 + λ 敏感性分析
    ├── visualize.py                   # 图表生成（训练曲线、方法对比）
    ├── analyze_dqn_v2.py              # DQN V2 分析脚本 🆕
    ├── utils.py                       # 数据加载 + 工具函数
    │
    ├── checkpoints/                   # 训练模型权重（.pt，共 32 个）
    │   ├── dqn_6x6x2_*.pt            # 原始 DQN（Phase 1）
    │   ├── dqn_{ds}_ep300.pt         # DQN 300 集 × 5（Phase 3）
    │   ├── dqn_{ds}_ep1000.pt        # DQN 1000 集 × 5（Phase 4）
    │   ├── dqn_v2_{ds}_ep300.pt      # DQN V2 300 集 × 10 🆕
    │   ├── ppo_6x6x2_*.pt            # 原始 PPO（Phase 1）
    │   ├── ppo_{ds}_ep300.pt         # PPO 300 集 × 5（Phase 3）
    │   └── ppo_{ds}_ep1000.pt        # PPO 1000 集 × 5（Phase 4）
    │
    ├── logs/                          # 训练日志（JSON，共 25+ 个）
    │   ├── train_{ds}_dqn.json        # DQN 训练日志
    │   ├── train_{ds}_ppo.json        # PPO 训练日志
    │   ├── train_{ds}_dqn_v2_ep300.json  # DQN V2 日志 🆕
    │   ├── dqn_v2_combined_results.json  # V2 汇总 🆕
    │   ├── dqn_v2_analysis.json          # V2 分析数据 🆕
    │   └── combined_results.json         # 跨方法结果汇总
    │
    ├── charts/                        # 可视化图表（PNG，共 20 张）
    │   ├── training_curves_*.png       # DQN 训练曲线（×2）
    │   ├── method_comparison_*.png     # 各规模方法对比（×10）
    │   ├── dataset_comparison_*.png    # 跨规模对比（×2）
    │   └── dqn_v2_*.png               # DQN V2 分析（×6）🆕
    │
    ├── result_XIWEI.md                # 完整实验报告（10 章节）
    └── analysis_dqn_v2.md             # DQN V2 优化分析 🆕
```

### 模型命名规范

| 格式 | 示例 | 含义 |
|------|------|------|
| `dqn_{ds}_ep{N}.pt` | `dqn_10x10x3_ep300.pt` | DQN 训练 N 集 |
| `dqn_v2_{ds}_ep{N}.pt` | `dqn_v2_10x10x3_ep300.pt` | DQN V2 训练 N 集 |
| `ppo_{ds}_ep{N}.pt` | `ppo_10x10x3_ep300.pt` | PPO 训练 N 集 |
| `train_{ds}_dqn.json` | `train_10x10x3_dqn.json` | DQN 训练日志 |
| `train_{ds}_dqn_v2_ep{N}.json` | `train_10x10x3_dqn_v2_ep300.json` | DQN V2 训练日志 |

> **所有模型/日志/图表独立命名 -- 跨实验无任何文件覆盖。**

---

## 9. 核心结论

### 已验证的发现

1. **Greedy SPT 是最强基线**：10 个规模全部最优（100% 胜率）。问题结构天然适合"每次选最短加工时间"的贪心策略。
2. **PPO 扩展性优于 DQN**：策略梯度方法随规模退化更平缓（1.25×→1.49× Greedy），而 DQN 的 Q 函数在大动作空间中退化严重（1.06×→1.68×）。
3. **更多训练帮助 PPO、损害 DQN**：1000 集时 PPO 策略稳定（eval ≈ best），DQN 反而退化（回放池陈旧样本主导）。
4. **DQN V2 在大问题上有效**：Noisy Nets + 硬更新 + 大网络在 ≥30 维动作空间上改善 DQN。小规模上网络过参数化导致退步。
5. **15×10×2 是最佳配置**：PPO best MS 仅差 Greedy 3.2%，疲劳不到 Greedy 的一半（1.015 vs 2.169）。少操作员场景是 RL 最容易追近贪心启发式的情况。

### 待改进方向

1. **RL 与 Greedy 仍有差距**：最佳 PPO 仍差 3.2%，多数规模上差距显著（+10~45%）
2. **小规模 DQN V2 过参数化**：6×6×2 上退步 21.6%，需自适应网络规模
3. **疲劳管理需改进**：RL 方法在多目标权衡上表现不一致
4. **训练时间**：大规模 DQN 在 CPU 上极慢（30×10×3 需 ~35 分钟 / 300 集）

### 改进路线

| 方向 | 具体措施 |
|------|---------|
| **奖励塑形** | 更精细的步间信用分配，解决稀疏奖励问题 |
| **网络架构** | 图神经网络编码调度图结构、Transformer 编码工件关系 |
| **算法扩展** | SAC 处理连续疲劳控制、多智能体 RL 分布式调度 |
| **泛化能力** | size-agnostic 模型架构、跨规模迁移学习 |
| **训练加速** | GPU 加速、并行环境采样（PPO 天然支持向量化） |
| **实用部署** | 实时数据流集成、在线微调、人机协同调度接口 |

---

# English

## 1. Project Overview

This project solves the **Flexible Job Shop Scheduling Problem (FJSP)** with **worker fatigue dynamics**. We model personnel allocation and production scheduling as a joint MDP and apply deep reinforcement learning to simultaneously minimize **makespan** and **worker fatigue**.

### Core Features

- **Two DRL algorithms**: Dueling Double DQN (+ PER + N-step) and PPO (+ GAE)
- **Dueling DQN V2** 🆕: Optimized variant with Noisy Nets, hard target updates, cosine LR annealing
- **Fatigue model**: α-β dynamics with processing time impact
- **Action masking**: Guarantees valid scheduling actions
- **Event-driven simulation**: Efficient discrete-event environment
- **Comprehensive benchmarking**: 10 problem scales × 5 methods × 2 training regimes

### Problem Scales

Datasets follow `{Jobs}x{Machines}x{Workers}.csv` naming:

| Scale | Example | State Dim | Action Dim | Description |
|-------|---------|-----------|------------|-------------|
| Small | 6×6×2 | 43 | 12 | Tiny workshop |
| Medium | 10×10×3 | 70 | 30 | Typical configuration |
| Large | 30×10×3 | 130 | 90 | Industrial scale |
| Extra Large | 100×10×6 | 310 | 600 | Stress test |

---

## 2. Problem Formulation

### MDP Definition

| Component | Description |
|-----------|-------------|
| **State** | `3M + 3W + 3N + 1` dims: machine status (free + remaining time), worker status (free + fatigue), job progress (completion% + next machine + remaining work), machine demand signal, worker load signal, current time |
| **Action** | `N × W` pairs of (job_id, worker_id) — assign next operation of a job to a specified worker |
| **Reward** | Immediate: `-(actual processing time) / reward_scale`; Terminal: `-λ × Σ max(0, F - F_threshold)` + makespan penalty |
| **Transition** | Event-driven: choose action → assign operation → advance simulation clock → release resources → arrive at next decision point |

### Fatigue Model

```
Working:  dF/dt = +α   (α=0.02, fatigue accumulates)
Idle:     dF/dt = -β   (β=0.025, fatigue recovers)
Impact:   actual_time = base_time × (1 + γ × F)   (γ=0.05)
Penalty:  -λ × Σ max(0, F - F_threshold)   (λ=2.0)
```

- Fatigue accumulates during work, recovers during idle
- High fatigue → longer actual processing time → later completions
- Terminal penalty λ=2.0 discourages excessive fatigue on any worker

---

## 3. Implemented Models

### 3.1 Dueling Double DQN + PER + N-step Returns

**Files**: [`agent.py`](XIWEI_code/agent.py) · [`train_dqn.py`](XIWEI_code/train_dqn.py)

**Network**:
```
Input(state_dim) → [256, 128] shared features →
  Value Stream:     64 → 1       (estimates V(s))
  Advantage Stream: 64 → N×W     (estimates A(s,a))
Output: Q(s,a) = V(s) + A(s,a) − mean(A(s,a))
```

**Key features**: Double DQN | SumTree PER | N=3 step returns | ε-greedy (1.0→0.02) | Polyak soft update (τ=0.005) | Action masking

**Hyperparameters**: lr=3e-4, γ=0.95, batch=64, memory=50k, hidden=[256,128,64]

### 3.2 DQN V2 (Optimized) 🆕

**Files**: [`config.py`](XIWEI_code/config.py) (DQNConfigV2) · [`train_dqn_optimized.py`](XIWEI_code/train_dqn_optimized.py)

**8 key improvements** over baseline DQN:

| Improvement | Baseline | V2 | Rationale |
|-------------|----------|-----|-----------|
| **Exploration** | ε-greedy | Noisy Nets + ε-greedy | State-dependent exploration (Rainbow) |
| **Target update** | Soft τ=0.005 | Hard copy / 1000 steps | Cleaner learning signal |
| **N-step** | 3 | 7 | Faster credit propagation |
| **Gamma** | 0.95 | 0.99 | Long-horizon value estimation |
| **LR schedule** | Constant 3e-4 | Cosine 1e-3→1e-5 | Fast start + fine convergence |
| **Network** | [256,128,64] | [512,256,128,64] | More capacity |
| **Batch/Memory** | 64/50k | 128/100k | Stable gradients + diverse experience |
| **PER α** | 0.6 | 0.7 | Stronger prioritization |

**Result**: Wins on **4/10** datasets at 300ep (best: **-9.1%** on 10×10×3).

### 3.3 PPO + GAE

**Files**: [`agent_ppo.py`](XIWEI_code/agent_ppo.py) · [`train_ppo.py`](XIWEI_code/train_ppo.py)

**Network**:
```
Input(state_dim) → [256, 128] shared features →
  Actor Head:  128 → N×W  (policy logits)
  Critic Head: 128 → 1    (state value V(s))
```

**Key features**: GAE λ=0.95 | PPO Clip ε=0.2 | Orthogonal init | Entropy bonus 0.01 | 10-episode rollout → 8 PPO epochs

**Hyperparameters**: lr=3e-4, γ=0.99, clip=0.2, hidden=[256,128]

### 3.4 DQN vs PPO Comparison

| Dimension | DQN | PPO |
|-----------|-----|-----|
| **Paradigm** | Value-based (Q-learning) | Policy gradient |
| **Output** | Q(s,a) values | π(a\|s) + V(s) |
| **Credit assignment** | N-step bootstrapping | Full-episode GAE returns |
| **Sample efficiency** | High (experience replay) | Lower (on-policy, discard after update) |
| **Exploration** | ε-greedy (value-agnostic) | Entropy bonus + stochastic sampling |
| **Stability** | Needs target net + PER | Naturally stable via clipping |
| **Training speed** | Fast per-step, slow per-episode | Batch update, faster per-episode |

---

## 4. Baselines

| Method | Description | Source |
|--------|-------------|--------|
| **Greedy SPT** | Greedily picks (job, worker) with shortest actual processing time | [`environment.py`](XIWEI_code/environment.py) |
| **Round-Robin** | Cycles through workers in round-robin order | [`train_baselines.py`](XIWEI_code/train_baselines.py) |
| **Random** | Random valid action (30-run average) | [`train_baselines.py`](XIWEI_code/train_baselines.py) |

---

## 5. Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib

# Train DQN (default 10×10×3, 2000 episodes)
cd XIWEI_code
python train_dqn.py

# Train PPO
python train_ppo.py --data 10x10x3 --episodes 2000

# Multi-scale batch training (5 datasets × 2 methods)
python train_all.py --episodes 300

# Train DQN V2 (optimized)
python train_dqn_optimized.py --episodes 300

# Evaluate a trained model
python test.py --model checkpoints/dqn_10x10x3_final.pt --data 10x10x3
```

---

## 6. Usage Guide

### Training

```bash
# DQN — standard
python train_dqn.py --data 6x6x2 --episodes 500 --lr 5e-4

# DQN — without PER (ablation)
python train_dqn.py --data 10x10x3 --no_per

# DQN V2 — optimized (supports multi-dataset)
python train_dqn_optimized.py --episodes 300 --datasets 10x10x3 30x10x3

# PPO
python train_ppo.py --data 6x6x2 --episodes 500 --lr 3e-4 --gamma 0.99

# Multi-scale batch training
python train_all.py --episodes 1000 --datasets 10x5x3 10x10x6 15x10x2
```

### Evaluation

```bash
# Evaluate all baselines
python train_baselines.py

# Load and test a checkpoint
python test.py --model checkpoints/ppo_15x10x2_ep1000.pt --data 15x10x2

# Fatigue penalty weight λ sensitivity analysis
python test.py --data 6x6x2 --compare_weights --episodes 500
```

### Visualization

```bash
# Auto-scan logs/ and generate all charts
python visualize.py

# Specify a log file
python visualize.py --log logs/train_10x10x3_nstep3.json

# Compare multiple runs
python visualize.py --compare logs/

# DQN V2 specific analysis
python analyze_dqn_v2.py
```

### Python API

```python
from config import Config
from utils import load_csv_data, get_data_path, get_state_dim, get_action_dim
from environment import JSPEnvironment
from agent import DQNAgent
from agent_ppo import PPOAgent

# Load data and create environment
config = Config()
data = load_csv_data('data/csv_output/10x10x3.csv')
env = JSPEnvironment(data, config.env)

# ── DQN inference ──
agent = DQNAgent(get_state_dim(data), get_action_dim(data), config.dqn)
agent.load('XIWEI_code/checkpoints/dqn_10x10x3_final.pt')

state = env.reset()
done = False
while not done:
    mask = env._get_action_mask()
    action = agent.select_action(state, mask, epsilon=0.0)
    state, reward, done, _ = env.step(action)

print(f"DQN Makespan: {env.get_makespan():.1f}")

# ── PPO inference ──
env.reset()
ppo = PPOAgent(get_state_dim(data), get_action_dim(data), config.ppo)
ppo.load('XIWEI_code/checkpoints/ppo_10x10x3_final.pt')

state = env.reset()
done = False
while not done:
    mask = env._get_action_mask()
    action, _ = ppo.evaluate(state, mask)
    state, reward, done, _ = env.step(action)

print(f"PPO Makespan: {env.get_makespan():.1f}")
```

---

## 7. Results Summary

### 7.1 Round 1: 300 Episodes × 5 Scales (Phase 3)

| Dataset | Greedy | DQN | PPO | Best RL vs Greedy |
|---------|--------|-----|-----|-------------------|
| 6×6×2 | **2,686** | 2,858 | 3,350 | DQN +6.4% |
| 10×10×3 | **6,834** | 10,159 | 9,822 | PPO +43.7% |
| 15×10×3 | **9,341** | 13,976 | 13,582 | PPO +45.4% |
| 20×10×3 | **12,968** | 20,395 | 19,348 | PPO +49.2% |
| 30×10×3 | **19,276** | 32,339 | 27,259 | PPO +41.4% |

> **Greedy SPT wins 5/5**. PPO beats DQN on 3/5, with margin growing with scale.

**Chart**: [`charts/dataset_comparison_makespan.png`](XIWEI_code/charts/dataset_comparison_makespan.png)

### 7.2 Round 2: 1000 Episodes × 5 Scales (Phase 4)

| Dataset | Greedy | DQN | PPO | Best RL vs Greedy |
|---------|--------|-----|-----|-------------------|
| 10×5×3 | **3,419** | 4,628 | 4,254 | PPO +24.4% |
| 10×10×6 | **5,325** | 8,692 | 7,388 | PPO +38.7% |
| 15×10×2 | **10,194** | 14,886 | **10,659** | **PPO +4.6%** ⭐ |
| 15×5×3 | **4,648** | 5,394 | 5,144 | PPO +10.7% |
| 20×5×3 | **6,457** | 9,609 | 7,228 | PPO +11.9% |

> **PPO wins 5/5 over DQN**. 🏆 **15×10×2** achieves best RL result: PPO best MS only +3.2% vs Greedy, with fatigue <50% of Greedy's level.

**Chart**: [`charts/method_comparison_15x10x2.png`](XIWEI_code/charts/method_comparison_15x10x2.png)

### 7.3 DQN V2 Optimization (300 Episodes × 10 Scales) 🆕

| Dataset | DQN Orig | DQN V2 | V2 vs Orig | Verdict |
|---------|----------|--------|------------|---------|
| 10×10×3 | 10,159 | 9,238 | **−9.1%** | ✅ Win |
| 10×10×6 | 8,692 | 8,172 | **−6.0%** | ✅ Win |
| 30×10×3 | 32,339 | 30,979 | **−4.2%** | ✅ Win |
| 20×5×3 | 9,609 | 9,281 | **−3.4%** | ✅ Win |
| 15×10×2 | 14,886 | 15,147 | +1.8% | ❌ Loss |
| 20×10×3 | 20,395 | 21,008 | +3.0% | ❌ Loss |
| 15×10×3 | 13,976 | 14,824 | +6.1% | ❌ Loss |
| 10×5×3 | 4,628 | 5,058 | +9.3% | ❌ Loss |
| 15×5×3 | 5,394 | 6,152 | +14.1% | ❌ Loss |
| 6×6×2 | 2,858 | 3,474 | +21.6% | ❌ Loss |

> **DQN V2 wins 4/10**. Most effective on large action spaces (≥30 dims). Over-parameterized on small problems.

**Analysis**: [`analysis_dqn_v2.md`](XIWEI_code/analysis_dqn_v2.md)  
**Chart**: [`charts/dqn_v2_vs_original_makespan.png`](XIWEI_code/charts/dqn_v2_vs_original_makespan.png)

### 7.4 Overall Rankings

| Rank | Method | Avg Gap to Greedy | Training Speed | Stability |
|------|--------|-------------------|----------------|-----------|
| 🥇 | **Greedy SPT** | 0% | Instant | Perfect |
| 🥈 | **PPO** | +5~45% (improves w/ episodes) | Fast (4-8× vs DQN) | Stabilizes |
| 🥉 | **DQN V2** | +35~62% | Slow (large network) | Mixed (4/10 wins) |
| 4 | **DQN** | +6~68% | Medium | Degrades w/ scale |

---

## 8. File Structure

```
PythonFinalTest/
├── README.md                          # This file (bilingual EN/ZH)
├── CLAUDE.md                          # Claude Code instructions
├── 基于强化学习的人员配置-生产调度协同优化(2).pdf  # Reference paper
├── data/
│   ├── basic_data.xlsx                # Source data
│   └── csv_output/                    # CSV datasets (40 scales)
│
└── XIWEI_code/                        # Core code
    ├── config.py                      # Hyperparams (Env, DQN, DQNConfigV2, PPO, Train)
    ├── environment.py                 # JSP simulator + GreedyScheduler
    ├── agent.py                       # DQN agent (DuelingDQN + NoisyLinear + SumTree PER)
    ├── agent_ppo.py                   # PPO agent (ActorCritic + GAE)
    ├── train_dqn.py                   # DQN training (N-step + PER + early stop)
    ├── train_dqn_optimized.py         # DQN V2 training (NoisyNets + hard updates) 🆕
    ├── train_ppo.py                   # PPO training (Rollout + PPO Update)
    ├── train_all.py                   # Multi-scale batch training orchestrator
    ├── train_baselines.py             # Baseline evaluation
    ├── test.py                        # Model testing + λ sensitivity
    ├── visualize.py                   # Chart generation
    ├── analyze_dqn_v2.py              # DQN V2 analysis 🆕
    ├── utils.py                       # Data loading + utilities
    ├── checkpoints/                   # Model weights (32 .pt files)
    ├── logs/                          # Training logs (25+ JSON files)
    └── charts/                        # Visualization charts (20 PNGs)
```

### Model Naming Convention

| Pattern | Example | Meaning |
|---------|---------|---------|
| `dqn_{ds}_ep{N}.pt` | `dqn_10x10x3_ep300.pt` | DQN trained N episodes |
| `dqn_v2_{ds}_ep{N}.pt` | `dqn_v2_10x10x3_ep300.pt` | DQN V2 trained N episodes |
| `ppo_{ds}_ep{N}.pt` | `ppo_10x10x3_ep300.pt` | PPO trained N episodes |
| `train_{ds}_dqn.json` | `train_10x10x3_dqn.json` | DQN training log |
| `train_{ds}_dqn_v2_ep{N}.json` | `train_10x10x3_dqn_v2_ep300.json` | DQN V2 training log |

> **All models/logs/charts are independently named — no files are overwritten across experiments.**

---

## 9. Key Findings

### What Works

1. **Greedy SPT is the strongest baseline**: Wins 100% on all 10 scales. Shortest-processing-time-first is naturally optimal for this problem structure.
2. **PPO scales better than DQN**: PPO's policy gradient degrades gracefully with scale (1.25×→1.49× Greedy), while DQN's Q-function struggles in large action spaces (1.06×→1.68×).
3. **More training helps PPO, hurts DQN**: At 1000ep, PPO eval ≈ best training MS (strategy stabilizes), while DQN degrades (stale replay buffer samples dominate).
4. **DQN V2 excels on large problems**: Noisy Nets + hard updates + larger network improve DQN on action spaces ≥30 dims. Over-parameterized on small scales.
5. **15×10×2 is the sweet spot**: PPO best MS only +3.2% vs Greedy, with fatigue <50% of Greedy's level. Fewer workers = easier RL learning.

### What Needs Work

1. **RL still trails Greedy**: Best PPO is +3.2% behind Greedy; most scales show significant gaps (+10~45%).
2. **DQN V2 over-parameterization**: [512,256,128,64] network regresses +21.6% on 6×6×2. Adaptive network sizing needed.
3. **Fatigue management**: RL methods are inconsistent on the makespan-fatigue trade-off across scales.
4. **Training time**: Large-scale DQN training is very slow on CPU (30×10×3 takes ~35 min for 300ep).

### Improvement Roadmap

| Direction | Action |
|-----------|--------|
| **Reward shaping** | Finer per-step credit assignment for sparse rewards |
| **Network architecture** | Graph Neural Networks for scheduling topology, Transformers for job relations |
| **Algorithm extension** | SAC for continuous fatigue control, Multi-agent RL for distributed scheduling |
| **Generalization** | Size-agnostic model architecture, cross-scale transfer learning |
| **Training acceleration** | GPU training, parallel environment sampling (PPO naturally supports vectorized envs) |
| **Deployment** | Real-time data integration, online fine-tuning, human-in-the-loop interface |

---

## 📚 参考 | References

- [完整实验报告 | Full Experiment Report](XIWEI_code/result_XIWEI.md) — 10 章节详细分析
- [DQN V2 优化分析 | DQN V2 Optimization Analysis](XIWEI_code/analysis_dqn_v2.md) — 参数优化专项报告
- [基于强化学习的人员配置-生产调度协同优化(2).pdf](基于强化学习的人员配置-生产调度协同优化(2).pdf) — 参考论文

---

*最后更新 | Last updated: 2026-06-10 · 分支 | Branch: XIWEI_code · 20 模型已训练 | 20 models trained · 10 规模已测试 | 10 scales tested · 20 图表已生成 | 20 charts generated*
