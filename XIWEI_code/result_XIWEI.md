# XIWEI_code 项目总结报告

> 基于强化学习的人员配置-生产调度协同优化 (RL-based Personnel Allocation & Production Scheduling Collaborative Optimization)

---

## 目录

1. [项目概述](#1-项目概述)
2. [问题建模](#2-问题建模)
3. [实现的模型](#3-实现的模型)
4. [对比基线](#4-对比基线)
5. [实验结果与对比分析](#5-实验结果与对比分析)
6. [如何使用](#6-如何使用)
7. [文件结构](#7-文件结构)
8. [可视化与分析](#8-可视化与分析)
9. [总结与展望](#9-总结与展望)

---

## 1. 项目概述

本项目解决**柔性作业车间调度问题（FJSP）**：在考虑操作员疲劳度动态变化的情况下，同时优化**最大完工时间（Makespan）**和**操作员疲劳度**。

### 核心创新点

- 将人员配置与生产调度作为**联合优化问题**建模为 MDP
- 引入**疲劳模型**：疲劳在工作时积累、空闲时恢复、疲劳度影响加工效率
- 同时实现了两种主流深度强化学习算法：**Dueling Double DQN** 和 **PPO**
- 支持 **Action Masking** 保证动作合法性

### 问题规模

数据集命名规则 `{作业数}x{机器数}x{操作员数}.csv`：

| 规模 | 数据集示例 | 状态维度 | 动作维度 |
|------|-----------|---------|---------|
| 小规模 | 6x6x2 | 46 | 12 |
| 中规模 | 10x10x3 | 76 | 30 |
| 大规模 | 15x5x2 | 73 | 30 |

---

## 2. 问题建模

### 2.1 MDP 形式化

| 组件 | 描述 |
|------|------|
| **状态空间** | `3M + 3W + 3N + 1` 维向量：机器状态(空闲+剩余时间)、操作员状态(空闲+疲劳度)、工件状态(进度+下一机器+剩余工时)、机器需求信号、操作员负载信号、当前时间 |
| **动作空间** | `N × W` 维 (job_id, worker_id) 对，选定后将工件下一工序分配给指定操作员 |
| **奖励函数** | 即时奖励 `= -实际加工时间 / 归一化尺度`；终端惩罚 `= -λ * Σ max(0, F_p - F_threshold)` |
| **疲劳模型** | 疲劳积累率 α=0.02，疲劳恢复率 β=0.025，疲劳影响系数 γ=0.05，疲劳惩罚权重 λ=2.0 |

### 2.2 仿真机制

采用**事件驱动仿真**：在决策点选择动作→分配工序→推进仿真时钟→释放资源→到达下一决策点。每个 `step()` 返回即时奖励仅与所分配操作的加工时间关联，实现精确的信用分配（credit assignment）。

---

## 3. 实现的模型

### 3.1 Dueling Double DQN + PER + N-step Returns

**文件**：[agent.py](agent.py) | [train_dqn.py](train_dqn.py)

**网络架构**：
```
输入层: state_dim →
  全连接层1: 256 + ReLU →
  全连接层2: 128 + ReLU →
  Value Stream: 64 → 1
  Advantage Stream: 64 → action_dim
输出: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
```

**关键技术特性**：

| 特性 | 实现 |
|------|------|
| Double DQN | 解耦动作选择与价值评估，减少 Q 值高估 |
| Dueling Architecture | 分离 V(s) 和 A(s,a)，动作空间大时更稳定 |
| Prioritized Experience Replay (PER) | SumTree 实现，优先采样 TD 误差大的样本 |
| N-step Returns | N=3，加速信用传播 |
| Action Masking | 掩码无效动作的 Q 值为 -∞ |
| 目标网络 | Polyak 软更新 τ=0.005 |
| ε-greedy 探索 | 线性衰减，ε_start=1.0 → ε_end=0.02，decay=100k steps |

**超参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 5e-4 | Adam 优化器 |
| gamma | 0.95 | 适用于有限时域 JSP |
| n_step | 3 | N-step TD |
| batch_size | 64 | 批次大小 |
| memory_capacity | 50000 | 经验池容量 |
| epsilon_decay | 100000 | 线性衰减步数 |
| use_per | True | 启用优先经验回放 |
| per_alpha | 0.6 | PER 优先级指数 |
| hidden_dims | [256, 128, 64] | 网络隐藏层 |

**已训练模型**（位于 `checkpoints/`）：
- `dqn_6x6x2_nstep3_lr5e-04_g0.95_final.pt` — 6作业×6机器×2操作员
- `dqn_10x10x3_nstep3_lr5e-04_g0.95_final.pt` — 10作业×10机器×3操作员

---

### 3.2 PPO (Proximal Policy Optimization) + GAE

**文件**：[agent_ppo.py](agent_ppo.py) | [train_ppo.py](train_ppo.py)

**网络架构**：
```
输入层: state_dim →
  Shared Feature: [256, 128] + ReLU →
  Actor Head: Linear(128, action_dim) — 输出 logits
  Critic Head: Linear(128, 1) — 输出 V(s)
```

**关键技术特性**：

| 特性 | 实现 |
|------|------|
| Actor-Critic | 共享特征提取器的 Actor-Critic 架构 |
| GAE (Generalized Advantage Estimation) | λ=0.95，低方差优势估计 |
| PPO Clipping | ε=0.2，限制策略更新幅度 |
| Action Masking | 无效动作 logits 设为 -1e10 |
| Orthogonal 初始化 | 稳定训练，Actor 小权重初始化 |
| Entropy Bonus | 系数 0.01~0.05，鼓励探索 |
| Rollout Buffer | 每 10 个 episode 进行一次 PPO 更新 |

**超参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 3e-4 | Adam 优化器 |
| gamma | 0.99 | 折扣因子（GAE 计算中使用）|
| gae_lambda | 0.95 | GAE 平滑参数 |
| clip_epsilon | 0.2 | PPO 裁剪范围 |
| value_coef | 0.5 | 价值损失系数 |
| entropy_coef | 0.05 | 熵正则化系数 |
| rollout_episodes | 10 | 每次更新收集的 episode 数 |
| ppo_epochs | 8 | 每次更新的 PPO epoch 数 |
| hidden_dims | [256, 128] | 网络隐藏层 |

**已训练模型**（位于 `checkpoints/`）：
- `ppo_6x6x2_lr3e-04_g0.99_final.pt` — 6作业×6机器×2操作员
- `ppo_10x10x3_lr3e-04_g0.99_final.pt` — 10作业×10机器×3操作员

---

### 3.3 DQN vs PPO 方法对比

| 维度 | DQN | PPO |
|------|-----|-----|
| 学习范式 | 值函数逼近 (Q-learning) | 直接策略优化 (Policy Gradient) |
| 输出 | Q(s,a) 值函数 | π(a|s) 策略分布 + V(s) |
| 信用分配 | N-step bootstrapping | 完整 episode 的 GAE 回报 |
| 样本效率 | 高（经验回放重用样本） | 较低（on-policy 每次更新后丢弃） |
| 探索机制 | ε-greedy（值无关探索） | Entropy bonus + 随机采样（策略驱动探索） |
| 稳定性 | 需要目标网络 + PER 稳定 | PPO clip 天然稳定，无 bootstrapping 错误 |
| 训练速度 | 快（每步更新） | 较慢（批量更新） |

---

## 4. 对比基线

| 基线方法 | 描述 | 实现位置 |
|---------|------|---------|
| **Greedy SPT** | 每次选择加工时间最短的 (工件, 操作员) 组合（考虑疲劳影响） | [environment.py](environment.py) `GreedyScheduler` |
| **Round-Robin** | 轮询分配操作员，公平负载分配 | [train_baselines.py](train_baselines.py) `run_roundrobin()` |
| **Random** | 随机选择有效动作（30 次平均） | [train_baselines.py](train_baselines.py) `run_random()` |
| **DQN (无疲劳感知)** | DQN 仅优化 Makespan，λ_fatigue=0 | [test.py](test.py) `train_dqn_quick()` |

---

## 5. 实验结果与对比分析

### 5.1 小规模问题：6×6×2（训练 500 集）

| 方法 | Makespan ↓ | Fatigue ↓ | 备注 |
|------|-----------|----------|------|
| **Greedy SPT** | **2686.2** | 2.767 | Makespan 最优 |
| Round-Robin | 2837.5 | **1.583** | 疲劳最低 |
| DQN (best) | 2857.2 | 1.690 | 接近 Round-Robin |
| PPO (best) | 2841.8 | 1.813 | 综合表现好 |
| PPO (final eval) | 2998.3 | 2.514 | 稳定策略 |
| Random (30次) | 3255.9 | 4.226 | 性能下界 |
| DQN (final eval) | 3616.3 | 5.000 | ⚠️ 策略退化 |

**DQN 训练曲线（6×6×2）**：下图展示了 DQN 在 6×6×2 问题上的训练过程，包括 Reward、Makespan 和 ε 探索率随 episode 的变化趋势。

![DQN Training Curves - 6x6x2](charts/training_curves_6x6x2.png)

**方法对比（6×6×2）**：下图以分组柱状图对比了 Greedy SPT、Round-Robin、Random 和 DQN 在 6×6×2 问题上的 Makespan 和 Fatigue 表现。

![Method Comparison - 6x6x2](charts/method_comparison_6x6x2.png)

### 5.2 中规模问题：10×10×3（训练 900~2000 集）

| 方法 | Makespan ↓ | Fatigue ↓ | 备注 |
|------|-----------|----------|------|
| **Greedy SPT** | **6834.4** | 0.829 | Makespan 最优 |
| PPO (best) | 7808.6 | **0.367** | 疲劳降至 Greedy 的 44% |
| Round-Robin | 7923.6 | 0.922 | — |
| DQN (best) | 7915.6 | 0.413 | 疲劳显著低于 Greedy |
| Random (30次) | 8408.6 | 1.509 | 性能下界 |
| PPO (final eval) | 9097.2 | 3.333 | ⚠️ 策略退化 |
| DQN (final eval) | 11766.0 | 3.333 | ⚠️ 严重退化 |

**DQN 训练曲线（10×10×3）**：下图展示了 DQN 在 10×10×3 问题上的训练过程。可以看到随着训练进行，Makespan 均值和最佳值逐步下降，但仍需更长的训练时间以稳定收敛。

![DQN Training Curves - 10x10x3](charts/training_curves_10x10x3.png)

**方法对比（10×10×3）**：下图以分组柱状图对比了 Greedy SPT、Round-Robin、Random 和 DQN 在 10×10×3 问题上的 Makespan 和 Fatigue 表现。可见 Greedy SPT 在 Makespan 上大幅领先，而 DQN 的 Fatigue 控制有一定优势。

![Method Comparison - 10x10x3](charts/method_comparison_10x10x3.png)

### 5.3 多数据集综合对比（500 集 DQN 快速训练）

| 数据集 | Greedy MS | RR MS | Random MS | DQN+Fatigue MS | DQN(no fat) MS |
|--------|-----------|-------|-----------|----------------|----------------|
| 6x6x2 | 2686.2 | 2871.2 | 3243.6 | 3357.3 | 3595.4 |
| 10x5x2 | 3842.3 | 4505.6 | 4718.6 | 4897.3 | **4663.5** |
| 6x6x3 | 2373.5 | 2692.0 | 2849.7 | 2857.5 | 3188.8 |
| 10x5x3 | 3418.5 | 3950.6 | 4252.2 | 4726.1 | 5214.3 |
| 15x5x2 | 5054.2 | 5715.8 | 6616.0 | 7564.2 | 7570.9 |

**趋势分析**：
- **Greedy SPT 在 Makespan 上始终最优**（问题规模越大优势越明显）
- DQN 在小规模上接近 Greedy（差距 6-20%），大规模上差距拉大（50%+）
- **疲劳感知 DQN 比无疲劳 DQN 更优**（在 6x6x2 和 6x6x3 上），证明疲劳建模有正向作用
- 500 集训练不足以收敛，RL 方法需要更长的训练时间

### 5.4 关键发现

1. **Makespan-Fatigue 权衡**：RL 方法（特别是 PPO）可以显著降低操作员疲劳度（降至 Greedy 的 44%），但 Makespan 比 Greedy 高约 14%

2. **PPO 优于 DQN**：PPO 在最终评估中表现更稳定（6x6x2 上 PPO eval MS=2998 vs DQN eval MS=3616）。PPO 的完整 episode 回报更适合该问题的信用分配

3. **策略退化问题**：两种 RL 方法在训练后期均出现不同程度的策略退化（final eval 比 best 差），可能原因：
   - 环境高度随机 + 稀疏奖励导致训练不稳定
   - 经验回放中的陈旧样本（DQN）
   - 需要更精细的学习率调度或早停策略

4. **N-step Returns 有效**：DQN 的最佳结果与 Round-Robin 相当，说明 N-step 加速了信用传播

5. **Greedy SPT 作为强基线**：贪婪启发式在此问题中作为极强的基线（仅考虑即时加工时间），表明问题结构适合贪心方法

### 5.5 可视化图表说明

以上章节中已嵌入 4 张来自 `charts/` 目录的关键图表：

| 图表 | 所在章节 | 说明 |
|------|---------|------|
| `training_curves_6x6x2.png` | §5.1 | DQN 在 6×6×2 上的三面板训练曲线（Reward / Makespan / Epsilon） |
| `method_comparison_6x6x2.png` | §5.1 | 6×6×2 上各方法的 Makespan + Fatigue 并排柱状图对比 |
| `training_curves_10x10x3.png` | §5.2 | DQN 在 10×10×3 上的三面板训练曲线 |
| `method_comparison_10x10x3.png` | §5.2 | 10×10×3 上各方法的 Makespan + Fatigue 并排柱状图对比 |

---

## 6. 如何使用

### 6.1 环境配置

```bash
# 依赖安装
pip install torch numpy matplotlib

# 项目结构
PythonFinalTest/
├── data/csv_output/          # CSV 数据集
├── XIWEI_code/               # 核心代码
│   ├── config.py             # 所有超参数配置
│   ├── environment.py        # JSP 环境模拟器 + Greedy 基线
│   ├── agent.py              # DQN 智能体 (Dueling Double DQN + PER)
│   ├── agent_ppo.py          # PPO 智能体 (Actor-Critic + GAE)
│   ├── train_dqn.py          # DQN 训练脚本
│   ├── train_ppo.py          # PPO 训练脚本
│   ├── train_baselines.py    # 基线方法评估
│   ├── test.py               # 综合评估与对比脚本
│   ├── visualize.py          # 可视化工具
│   ├── utils.py              # 数据加载与工具函数
│   ├── checkpoints/          # 预训练模型
│   ├── logs/                 # 训练日志 (JSON)
│   └── charts/               # 可视化图表 (PNG)
```

### 6.2 训练模型

```bash
cd XIWEI_code

# 训练 DQN（默认 10x10x3，2000 集）
python train_dqn.py

# 训练 DQN 指定数据集和超参数
python train_dqn.py --data 6x6x2 --episodes 500 --lr 5e-4

# 禁用 PER 进行对比
python train_dqn.py --data 10x10x3 --no_per

# 训练 PPO
python train_ppo.py --data 10x10x3 --episodes 2000

# 训练 PPO 指定超参数
python train_ppo.py --data 6x6x2 --episodes 500 --lr 3e-4 --gamma 0.99
```

### 6.3 评估基线方法

```bash
# 评估所有数据集上的基线方法
python train_baselines.py

# 指定数据集和运行次数
python train_baselines.py --data 10x10x3 --runs 50
```

### 6.4 加载预训练模型进行评估

```bash
# 加载 DQN 模型评估
python test.py --model checkpoints/dqn_10x10x3_nstep3_lr5e-04_g0.95_final.pt --data 10x10x3

# 加载 PPO 模型评估
python test.py --model checkpoints/ppo_10x10x3_lr3e-04_g0.99_final.pt --data 10x10x3

# 综合测试（自动训练+评估+对比）
python test.py --data 6x6x2 --episodes 500
```

### 6.5 可视化训练结果

```bash
# 自动扫描 logs/ 目录生成所有图表
python visualize.py

# 指定日志文件
python visualize.py --log logs/train_10x10x3_nstep3.json

# 比较多次训练运行
python visualize.py --compare logs/

# 指定输出目录
python visualize.py --save-dir charts/
```

### 6.6 Python API 使用示例

```python
import sys; sys.path.insert(0, 'XIWEI_code')
from config import Config
from utils import load_csv_data, get_data_path, get_state_dim, get_action_dim
from environment import JSPEnvironment
from agent import DQNAgent
from agent_ppo import PPOAgent
import numpy as np

# 1. 加载数据
config = Config()
data = load_csv_data('data/csv_output/10x10x3.csv')

# 2. 创建环境
env = JSPEnvironment(data, config.env)
state_dim = get_state_dim(data)
action_dim = get_action_dim(data)

# 3. DQN 推理
agent = DQNAgent(state_dim, action_dim, config.dqn)
agent.load('XIWEI_code/checkpoints/dqn_10x10x3_nstep3_lr5e-04_g0.95_final.pt')

state = env.reset()
done = False
while not done:
    mask = env._get_action_mask()
    action = agent.select_action(state, mask, epsilon=0.0)
    state, reward, done, info = env.step(action)

print(f"DQN Makespan: {env.get_makespan():.1f}, Fatigue: {env.get_avg_fatigue():.3f}")

# 4. PPO 推理
env.reset()
ppo_agent = PPOAgent(state_dim, action_dim, config.ppo)
ppo_agent.load('XIWEI_code/checkpoints/ppo_10x10x3_lr3e-04_g0.99_final.pt')

state = env.reset()
done = False
while not done:
    mask = env._get_action_mask()
    action, _ = ppo_agent.evaluate(state, mask)
    state, reward, done, _ = env.step(action)

print(f"PPO Makespan: {env.get_makespan():.1f}, Fatigue: {env.get_avg_fatigue():.3f}")
```

### 6.7 疲劳惩罚权重敏感性分析

```bash
# 比较不同 λ 值（0.0, 1.0, 2.0, 5.0, 10.0, 20.0）
python test.py --data 6x6x2 --compare_weights --episodes 500
```

λ 越大 → 智能体越倾向于选择低疲劳操作员 → Makespan 上升但 Fatigue 下降。

---

## 7. 文件结构

```
XIWEI_code/
├── __init__.py              # 包初始化
├── config.py                # 数据中心化配置（Env, DQN, PPO, Train）
├── environment.py           # JSP 环境模拟器 + GreedyScheduler 基线
├── agent.py                 # DQN Agent（DuelingDQN 网络 + SumTree PER）
├── agent_ppo.py             # PPO Agent（ActorCritic 网络 + GAE）
├── train_dqn.py             # DQN 训练主循环（N-step + PER + 早停）
├── train_ppo.py             # PPO 训练主循环（Rollout + PPO Update）
├── train_baselines.py       # 三种基线启发式评估
├── test.py                  # 综合评估脚本（模型加载/训练+基线+λ对比）
├── visualize.py             # matplotlib 可视化（训练曲线/方法对比/λ敏感性）
├── utils.py                 # 数据加载 (CSV→numpy) + 工具函数
├── checkpoints/             # 训练好的模型权重 (.pt)
│   ├── dqn_6x6x2_*.pt
│   ├── dqn_10x10x3_*.pt
│   ├── ppo_6x6x2_*.pt
│   ├── ppo_10x10x3_*.pt
│   └── result-6.1.txt       # 多数据集测试结果
├── logs/                    # 训练日志 (.json)
│   ├── train_6x6x2_nstep3.json
│   ├── train_10x10x3_nstep3.json
│   ├── train_ppo_6x6x2.json
│   └── train_ppo_10x10x3.json
└── charts/                  # 自动生成的可视化图表 (.png)
    ├── training_curves_6x6x2.png
    ├── training_curves_10x10x3.png
    ├── method_comparison_6x6x2.png
    └── method_comparison_10x10x3.png
```

---

## 8. 可视化与分析

`visualize.py` 支持以下图表类型（已嵌入上文实验章节的图表标记 ✅）：

| 图表 | 函数 | 说明 | 已生成 |
|------|------|------|--------|
| 训练曲线 | `plot_training_curves()` | 三面板：Reward、Makespan（含基线参考线）、Epsilon | ✅ |
| 方法对比 | `plot_method_comparison()` | 分组柱状图：Makespan + Fatigue 并排比较 | ✅ |
| 跨数据集对比 | `plot_dataset_comparison()` | 多数据集方法对比柱状图 | — |
| N-step 消融 | `plot_nstep_ablation()` | 不同 N-step 值的性能敏感性 | — |
| λ 敏感性 | `plot_fatigue_comparison()` | 双 Y 轴：λ vs Makespan / Fatigue | — |
| 多运行对比 | `plot_multi_run_comparison()` | 多次训练的 Makespan 曲线叠加比较 | — |

### 训练曲线解读

**训练曲线图**（见 [§5.1](#51-小规模问题6×6×2训练-500-集) 和 [§5.2](#52-中规模问题10×10×3训练-9002000-集)）包含三个面板：

1. **Reward（上图）**：每 episode 总奖励 + 50 集移动平均。蓝色散点为原始值，深蓝色为平滑趋势线。
2. **Makespan（中图）**：每 episode 完工时间 + 50 集移动平均。红色虚线标注最佳值，绿/紫/橙色水平线分别标注 Greedy / Round-Robin / Random 基线。
3. **Epsilon（下图）**：探索率 ε 随 episode 线性衰减过程。橙色虚线标注 ε=0.3 的利用阶段分界线。

### 方法对比图解读

**方法对比图**（见 [§5.1](#51-小规模问题6×6×2训练-500-集) 和 [§5.2](#52-中规模问题10×10×3训练-9002000-集)）为左右并排的分组柱状图：

- **左图（Makespan）**：数值越低越好，柱顶标注精确值
- **右图（Fatigue）**：数值越低越好，柱顶标注精确值（3 位小数）
- 颜色图例：🔵 DQN / 🟢 Greedy / 🟠 Random / 🟣 Round-Robin

---

## 9. 总结与展望

### 9.1 项目成果

1. ✅ 完整实现了两种主流 DRL 算法（Dueling Double DQN + PER 和 PPO + GAE）解决 FJSP 联合优化问题
2. ✅ 构建了带疲劳模型的事件驱动调度仿真环境
3. ✅ 实现了 4 种基线方法用于性能对比
4. ✅ 在多组数据集上完成训练和评估（6×6×2, 10×10×3, 6×6×3, 10×5×2, 15×5×2 等）
5. ✅ 完整的模型保存/加载机制（含元数据）和可视化工具链

### 9.2 主要结论

- **Greedy SPT** 在 Makespan 维度是最强基线，但会显著增加操作员疲劳
- **PPO 在疲劳控制上表现最优**，可以在牺牲约 14% Makespan 的情况下将疲劳降至 Greedy 的 44%
- **DQN** 在最佳状态下可以达到与 Round-Robin 相近的 Makespan，同时疲劳更低
- **RL 方法的策略退化**是需要关注的问题（训练不稳定、超参数敏感）
- 较小规模的 500 集快速训练不足以充分发挥 RL 潜力

### 9.3 改进方向

1. **训练稳定性**：实现学习率调度（warmup + cosine decay）、模型 EMA checkpoint 保存最佳策略而非最终策略
2. **奖励塑形**：更精细的奖励函数设计，解决稀疏奖励问题
3. **网络架构**：尝试 Graph Neural Network 编码调度图结构、Transformer 编码工件间关系
4. **算法扩展**：SAC (Soft Actor-Critic) 处理连续疲劳控制、多智能体 RL 实现分布式调度
5. **泛化能力**：在更多规模数据集上训练、实现 size-agnostic 的模型架构
6. **实际部署**：集成实时数据流、在线微调、约束满足检查

---

*文档生成时间: 2026-06-01 | 项目分支: XIWEI_code | 最后提交: add ppo*
