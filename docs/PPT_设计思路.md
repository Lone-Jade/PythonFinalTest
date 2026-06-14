# PPT 设计思路 — 用于 AI 生成 PPT

> **汇报**: 基于强化学习的人员配置-生产调度协同优化  
> **时长**: 10 min | **页数**: 14 | **分工人数**: 5 | **语言**: 中英双语，每页文字精简

---

## 全局样式

- 配色：DQN = `#2196F3` 蓝, PPO = `#00BCD4` 青, Heuristic = `#4CAF50` 绿, 消融改善 = `#E91E63` 红
- 每页顶部标题中英双语，正文要点每条 ≤ 15 字（中）+ ≤ 10 词（英）
- 数据来源：`./FinalCode/outputs_exp/`、`./FinalCode/outputs_ablation/`

---

## Slide 1 — 封面（30s, Person A）

```
标题:  基于强化学习的人员配置-生产调度协同优化
      RL-based Personnel Allocation & Production Scheduling

副标题: DQN · PPO · Scale-Invariant Network · Ablation Study

成员:  [5 人姓名]

课程:  [课程名]  |  日期: 2026.06
```

**数据/图片**: 无

---

## Slide 2 — 问题背景（1 min, Person A）

```
标题: 问题背景 / Problem Background

左半部分 (疲劳机制):
  ┌─────────────────────────┐
  │ 工人疲劳 / Worker Fatigue │
  │ · 连续工作 → 疲劳累积      │
  │   Work → Fatigue ↑       │
  │ · 主动休息 → 疲劳恢复      │
  │   Rest → Fatigue ↓       │
  │ · 高疲劳 → 强制休息       │
  │   F > 0.8 → Force Rest   │
  │ · 高疲劳 → 加工时间 ×1.6  │
  │   Fatigue slows work     │
  └─────────────────────────┘

右半部分 (调度约束):
  ┌─────────────────────────┐
  │ 生产调度 / Scheduling     │
  │ · Job-Shop: 机器+工序约束 │
  │ · 工人分配 ≠ 机器分配     │
  │   Worker ≠ Machine       │
  │ · 规模: 36 → 1000 任务   │
  │   Scale: 36→1000 tasks   │
  └─────────────────────────┘

底部居中:
  ↙ 双向耦合 / Bidirectional Coupling ↘
  工人状态影响调度效率 ↔ 调度决策影响工人疲劳
  Worker state ↔ Schedule quality

核心挑战 / Challenges:
  1. NP-hard 组合优化 / combinatorial optimization
  2. 5× 规模泛化 / scale generalization (36→1000)
  3. 传统启发式难适应动态变化 / heuristics fail on dynamics
```

**数据**: 无

**图片**: 无（用图标/箭头表示耦合关系）

---

## Slide 3 — 方法总览 + SMDP 建模（1 min, Person A）

```
标题: 方法总览 / Method Overview

上半部分 (流程图):
  ┌──────────┐    ┌──────────────┐    ┌──────────┐
  │ 数据输入   │ → │  RL 智能体     │ → │ 调度决策   │
  │ 40 实例   │    │ DQN + PPO     │    │ 工人+机器  │
  │ 36→1000  │    │ 协同优化       │    │ 协同分配   │
  │ Data In   │    │ Co-optimize   │    │ Decision   │
  └──────────┘    └──────────────┘    └──────────┘

下半部分 (SMDP 建模):
  ┌─────────────────────────────────────────┐
  │ 状态 / State (19 dims)                   │
  │  Global(6) + Worker(4) + Action(9)      │
  │                                          │
  │ 动作 / Action (可变数量 / variable)       │
  │  WAIT | REST | ASSIGN_Job_j             │
  │                                          │
  │ 奖励 / Reward                            │
  │  基础 / Basic: 时间惩罚 + Makespan       │
  │  + Reward Shaping: 完成/效率/进度/停滞   │
  │                    Completion/Efficiency  │
  │                                          │
  │ 疲劳 / Fatigue:                          │
  │  F_new = F + α - γ·rest                 │
  │  ProcTime × (1 + β·σ(F - θ))           │
  └─────────────────────────────────────────┘

参考文献 (小字底部):
  [1] Mnih et al. Nature 2015 (DQN)
  [2] Hessel et al. AAAI 2018 (Rainbow: n-step+PER+Dueling)
  [3] Schulman et al. 2017 (PPO)
  [4] Liu,Fan,Zhao,Shen,Zhang. RCIM 2023 (DRL+Worker Fatigue)
  [5] Liu et al. C&OR 2025 (DDQN+PER for JSSP)
```

**数据**: 无

**图片**: 无（用流程图）

---

## Slide 4 — DQN vs PPO（1 min, Person B）

```
标题: 算法对比 / Algorithm Comparison

左右两栏对照:

┌─────────────── DQN ───────────────┐ ┌─────────────── PPO ────────────────┐
│ · ε-greedy 探索 / exploration      │ │ · 随机策略采样 / stochastic policy │
│ · Q(s,a) 价值函数 / value function │ │ · π(a|s) 策略分布 / policy dist.   │
│ · n-step return (n=10)            │ │ · GAE 优势估计 / advantage est.    │
│ · PER 优先回放 / prioritized replay│ │ · EMA RewardNorm 归一化            │
│ · 目标网络 / target net (500步)    │ │ · Clip Ratio = 0.20                │
│ · 每步更新 / update per step       │ │ · 每 rollout 更新 / per rollout     │
│                                    │ │                                    │
│ 擅长大状态空间 / large state space │ │ 擅长精细策略 / fine-grained policy │
│ 信用分配强 / credit assignment     │ │ 探索能力强 / exploration            │
└────────────────────────────────────┘ └────────────────────────────────────┘

共同设计 / Shared Design:
  · Curriculum Learning: 3 阶段渐进 / 3-stage progressive
  · Reward Shaping:  密集中间奖励 / dense intermediate rewards
  · 跨规模训练集 / Cross-scale training: 10 实例 (36→1000 tasks)
```

**数据**: 无

**代码提示**:
```python
# DQN update (agents.py)
loss = (weights_t * F.smooth_l1_loss(pred, target, reduction='none')).mean()
# PPO update (agents.py)
ratio = torch.exp(logp - old_logp)
loss = -torch.min(ratio*adv, torch.clamp(ratio, 1-ε, 1+ε)*adv)
```

---

## Slide 5 — 核心创新：ScaleInv 架构（1 min, Person B）

```
标题: 核心创新 / Core Innovation — Scale-Invariant Network

问题 / Problem:
  普通 MLP → 特征分布随规模偏移 / feature distribution shifts with scale
  小实例 (6×6): time/t_ref ~ 0.01-2.0  vs  大实例 (100×10): time/t_ref ~ 0.1-50.0
  → 泛化到未见规模时性能暴跌 / fails on unseen scales

架构 / Architecture:
  ┌─────────────────────────────────────┐
  │  Input: [global(6), worker(4),      │
  │          action(9)] = 19 dims       │
  │         │                           │
  │    ┌────▼────────┐                  │
  │    │ LayerNorm   │ ← 消除规模差异    │
  │    │ (input)     │   normalize scale │
  │    └────┬────────┘                  │
  │    ┌────▼────┐ ┌──▼───────────┐     │
  │    │ State    │ │ Action       │     │
  │    │ Encoder  │ │ Encoder      │     │
  │    │ (10 dim) │ │ (19 dim)     │     │
  │    │ LN+ReLU  │ │ LN+ReLU      │     │
  │    └────┬────┘ └──┬───────────┘     │
  │         ▼         ▼                 │
  │      V(s)    concat(s,a)→A(s,a)     │
  │                                        │
  │   Q = V(s) + A(s,a) - mean(A)        │
  └─────────────────────────────────────┘

效果 / Results (Test set avg/H):
  DQN: 135.1% → 120.1% (-11.1pp)
  PPO: 495.6% → 156.0% (-68.5pp) 🔥 从崩溃中拯救 / saved from collapse
```

**数据表**:
| 指标 / Metric | 无 ScaleInv / Without | 有 ScaleInv / With | 改善 / Δ |
|:--|:--:|:--:|:--:|
| DQN Test/H | 135.1% | 120.1% | -11.1pp |
| PPO Test/H | 495.6% | 156.0% | -68.5pp |
| 泛化差距 / Generalization Gap | +12.8pp | +1.9pp | -85% |

**代码提示**:
```python
# models.py — ScaleInvariantDuelingNetwork
self.input_norm = nn.LayerNorm(feature_dim)
self.state_encoder = nn.Sequential(
    nn.Linear(10, 128), nn.LayerNorm(128), nn.ReLU(), ...)
self.action_encoder = nn.Sequential(
    nn.Linear(19, 128), nn.LayerNorm(128), nn.ReLU(), ...)
```

---

## Slide 6 — 实验设置（1 min, Person C）

```
标题: 实验设置 / Experimental Setup

数据集划分 / Data Split (零重叠 / Zero Overlap, 35 实例):
  ┌──────────────────────────────────────────┐
  │ Train (10)    │ Val (6)    │ Test (19)   │
  │ 36─1000 任务   │ 100─300     │ 100─1000    │
  │ 6 个规模级     │            │             │
  │ 6 scale levels│            │             │
  └──────────────────────────────────────────┘

课程学习 / Curriculum Learning:
  Stage 1 (ep 1-100)   → ≤50 任务, 2 实例
  Stage 2 (ep 101-200) → ≤100 任务, 4 实例
  Stage 3 (ep 201-300) → 全部 / All, 10 实例

消融设计 / Ablation Design (8 variants × 50ep):
  DQN: vanilla → +PER+RS → +ScaleInv → +Curriculum
  PPO: vanilla → +RS+Curr → +ScaleInv → +RewardNorm
  每次只改一个组件 / one component per step

模型参数 / Model Params:
  lr=3e-4  γ=0.99  hidden=128  batch=64  n_step=10
  per_α=0.6  β=0.4  ε:1.0→0.02  ppo_epochs=4  clip=0.20
```

**数据**: 无（结构图）

---

## Slide 7 — 消融实验（1.5 min, Person C）⭐ 核心页

```
标题: 消融实验 / Ablation Study — 各组件贡献 / Component Contributions

左侧 — DQN 消融瀑布图 / Waterfall (Test set avg/H):
  ┌────────────────────────────────────────┐
  │ vanilla (PairScoring)         165.3%   │ ← 基线 / baseline
  │   └─ +PER+RewardShaping  →   135.1%   │   -30.2pp ████████████
  │   └─ +ScaleInv (no PER)  →   124.9%   │   -40.4pp ████████████████
  │   └─ +PER +ScaleInv      →   120.1%   │   -45.2pp ██████████████████
  │   └─ +Curriculum         →   118.3%   │   -47.0pp ███████████████████
  │   └─ +250ep Training     →   117.1%   │   -48.2pp ████████████████████
  └────────────────────────────────────────┘

右侧 — PPO 消融瀑布图 / Waterfall:
  ┌────────────────────────────────────────┐
  │ vanilla (MLP, 50ep)          254.8%    │ ← 基线
  │   └─ +ScaleInv only (150ep) → 292.3% ❌│   无RS时 ScaleInv 反而有害!
  │   └─ +RS+Curr+ScaleInv(150ep)→198.7%  │   RS 拯救 PPO
  │   └─ +RewardNorm + 250ep    → 123.1% 🔥│   -75.6pp (充分训练)
  └────────────────────────────────────────┘

底部总结 / Summary:
  ┌──────────────────────────────────────────────────┐
  │ 组件 / Component      DQN 独立贡献     PPO 贡献    │
  │ ScaleInv Architecture   -40.4pp 🔥🔥🔥🔥   必须RS! │
  │ PER + RS                -30.2pp 🔥🔥🔥     —       │
  │ ScaleInv+PER 交互       -45.2pp (重叠)   —        │
  │ Curriculum (无ScaleInv)   -3.5pp          —       │
  │ RewardNorm                 —          需充足训练   │
  └──────────────────────────────────────────────────┘
  ScaleInv = DQN 最大单一贡献者 (-40.4pp) / PPO 必须 RS 才能学习
```

**数据**: `./FinalCode/outputs_ablation/{variant}/test/` 各 JSON 文件

**图片**: 建议用 AI 生成瀑布图（waterfall chart），红=恶化，绿=改善

---

## Slide 8 — 训练曲线（0.5 min, Person D）

```
标题: 训练过程 / Training Process — DQN 300ep

  ┌─────────────────────────────────────────────┐
  │  [插入训练曲线图: reward + makespan dual Y]   │
  │  X轴: Episode (1-300)                       │
  │  左Y (蓝): Total Reward                     │
  │  右Y (红): Makespan                         │
  │                                              │
  │  关键节点 / Key Milestones:                   │
  │  ep 1:   ε=1.00  reward=-8.5  mk=2111       │
  │  ep 50:  ε=0.48  reward=-1.0  mk=1816       │
  │  ep 100: ε=0.22  reward=-1.1  mk=1893       │
  │  ep 200: ε=0.05  reward=-6.3  mk=2994       │
  │  ep 300: ε=0.02  reward=+80  mk=34076       │
  │          (100x10x3, 1000 tasks)              │
  └─────────────────────────────────────────────┘

  Curriculum 阶段标注:
  S1 (浅色): ep 1-100, ≤50 tasks, 2 instances
  S2 (中色): ep 101-200, ≤100 tasks, 4 instances
  S3 (深色): ep 201-300, ALL, 10 instances

DQN vs PPO 最终 (ep 300, on 100x10x3):
  DQN: reward=+80.0  mk=34076
  PPO: reward=+78.7  mk=34086
  → 训练集表现接近 / similar training performance
```

**数据**: `./FinalCode/outputs_exp/dqn/train_log_live.jsonl` (300 行), `./FinalCode/outputs_exp/ppo/train_log_live.jsonl`

**图片**: 用 training log 数据绘制的双 Y 轴曲线图（或用 `visualize.py` 生成）

---

## Slide 9 — Gantt 对比：小 + 中规模（1.5 min, Person D）⭐ 核心页

```
标题: 调度可视化 / Scheduling Visualization — Small & Medium Scale

页面布局: 2 行 × 3 列

第 1 行 — 6×6×2 (36 tasks, 最小规模 / smallest):
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ 启发式       │  │ DQN 300ep   │  │ PPO 300ep   │
  │ Heuristic    │  │             │  │             │
  │ mk=1766      │  │ mk=2437     │  │ mk=2326     │
  │ [gantt图]    │  │ [gantt图]    │  │ [gantt图]    │
  │ 基线 / base  │  │ +38% vs H   │  │ +32% vs H   │
  └─────────────┘  └─────────────┘  └─────────────┘

第 2 行 — 10×10×4 (100 tasks, 中规模 / medium):
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ 启发式       │  │ DQN 300ep   │  │ PPO 300ep   │
  │ Heuristic    │  │             │  │             │
  │ mk=2394      │  │ mk=3039     │  │ mk=2773     │
  │ [gantt图]    │  │ [gantt图]    │  │ [gantt图]    │
  │ 基线 / base  │  │ +27% vs H   │  │ +16% vs H ← 更优/better │
  └─────────────┘  └─────────────┘  └─────────────┘

结论 / Takeaway: PPO 在小/中规模上更接近启发式 / PPO closer to heuristic on small/medium
```

**图片路径**:
```
第1行左: ./FinalCode/outputs_exp/figures/6x6_6x6x2_heuristic_gantt.png
第1行中: ./FinalCode/outputs_exp/figures/6x6_6x6x2_DQN_300ep_gantt.png
第1行右: ./FinalCode/outputs_exp/figures/6x6_6x6x2_PPO_300ep_gantt.png
第2行左: ./FinalCode/outputs_exp/figures/10x10_10x10x4_heuristic_gantt.png
第2行中: ./FinalCode/outputs_exp/figures/10x10_10x10x4_DQN_300ep_gantt.png
第2行右: ./FinalCode/outputs_exp/figures/10x10_10x10x4_PPO_300ep_gantt.png
```

---

## Slide 10 — Gantt 对比：大 + 超大规模（1.5 min, Person D）⭐ 核心页

```
标题: 调度可视化 / Scheduling Visualization — Large & Extreme Scale

页面布局: 2 行 × 3 列

第 1 行 — 50×10×3 (500 tasks, 大规模 / large):
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ 启发式       │  │ DQN 300ep   │  │ PPO 300ep   │
  │ Heuristic    │  │             │  │             │
  │ mk=15520     │  │ mk=18965    │  │ mk=19818    │
  │ [gantt图]    │  │ [gantt图]    │  │ [gantt图]    │
  │ 基线 / base  │  │ +22% vs H ← 更优 │ +28% vs H │
  └─────────────┘  └─────────────┘  └─────────────┘

第 2 行 — 100×10×3 (1000 tasks, 超大规模 / extreme):
  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ 启发式       │  │ DQN 300ep   │  │ PPO 300ep   │
  │ Heuristic    │  │             │  │             │
  │ mk=30038     │  │ mk=34159    │  │ mk=39207    │
  │ [gantt图]    │  │ [gantt图]    │  │ [gantt图]    │
  │ 基线 / base  │  │ +14% vs H ← 明显更优 │ +31% vs H │
  └─────────────┘  └─────────────┘  └─────────────┘

结论 / Takeaway: DQN 在大/超大规模上明显优于 PPO / DQN dominates on large scale
                 → 算法分工明确 / clear algorithm specialization
```

**图片路径**:
```
第1行左: ./FinalCode/outputs_exp/figures/50x10_50x10x3_heuristic_gantt.png
第1行中: ./FinalCode/outputs_exp/figures/50x10_50x10x3_DQN_300ep_gantt.png
第1行右: ./FinalCode/outputs_exp/figures/50x10_50x10x3_PPO_300ep_gantt.png
第2行左: ./FinalCode/outputs_exp/figures/100x10_100x10x3_heuristic_gantt.png
第2行中: ./FinalCode/outputs_exp/figures/100x10_100x10x3_DQN_300ep_gantt.png
第2行右: ./FinalCode/outputs_exp/figures/100x10_100x10x3_PPO_300ep_gantt.png
```

---

## Slide 11 — DQN vs PPO 按规模对比（0.5 min, Person D）

```
标题: DQN vs PPO — 按规模 / by Scale

  ┌────────────────────────────────────────────┐
  │  [分组柱状图 / Grouped Bar Chart]            │
  │                                            │
  │  X轴: S(≤100)  M(150-200)  L(500)  XL(1000)│
  │  Y轴: Makespan / Heuristic (%)             │
  │  100% 线 = Heuristic baseline              │
  │                                            │
  │  蓝色柱 = DQN   青色柱 = PPO                │
  │                                            │
  │  数据表:                                    │
  │  ┌──────────┬──────┬──────┬──────┬──────┐  │
  │  │ 规模     │  S   │  M   │  L   │  XL  │  │
  │  │ Scale    │ ≤100 │150-200│ 500  │ 1000 │  │
  │  ├──────────┼──────┼──────┼──────┼──────┤  │
  │  │ Instances│  6   │  5   │  4   │  4   │  │
  │  │ DQN/H    │119.8%│114.2%│114.7%│119.2%│  │
  │  │ PPO/H    │116.6%│117.7%│128.0%│134.8%│  │
  │  │ Winner   │ PPO  │ DQN  │ DQN  │ DQN  │  │
  │  └──────────┴──────┴──────┴──────┴──────┘  │
  └────────────────────────────────────────────┘

结论 / Key Insight:
  PPO 小实例领先 / wins on small → 精细策略梯度 / fine-grained policy gradient
  DQN 大实例领先 / wins on large → PER + n-step 信用分配 / credit assignment
  → 天然适合 Ensemble / natural for ensemble
```

**数据来源**: `./FinalCode/outputs_exp/test_results/dqn_dqn_best.json` + `ppo_ppo_best.json`

---

## Slide 12 — 最终结果 + 演进路线（1 min, Person E）

```
标题: 最终结果 / Final Results & 演进路线 / Evolution

上半部分 — 演进路线 / Evolution (8 stages):
  ┌─────────────────────────────────────────────────────┐
  │ ① n-step  ② +RS   ③ +PER   ④ +Dueling+Curr        │
  │   DQN基线    PPO↑    DQN↑     500ep                 │
  │     ↓         ↓       ↓         ↓                   │
  │ ⑤ +ScaleInv  ⑥ +跨规模  ⑦ +ScaleInv  ⑧ +300ep     │
  │   DQN 122%    训练集     PPO+RN      最终           │
  │               DQN 130%   PPO 123%    DQN 117%       │
  └─────────────────────────────────────────────────────┘

下半部分 — 关键指标 / Key Metrics:
  ┌────────────────────────────────────────────────┐
  │ 指标 / Metric          基线 → 最终    改善 / Δ  │
  │────────────────────────────────────────────────│
  │ DQN 100x10x6 / H       ~60% → 26%    -34pp    │
  │ PPO 崩溃 / Crashes     多次 → 0      ✅ 根治   │
  │ 泛化差距 / Gen. Gap    12.8pp→1.9pp  -85%     │
  │ 训练覆盖 / Train Cover 36-200→36-1000  5×     │
  └────────────────────────────────────────────────┘

最佳模型 / Best Models:
  DQN: ScaleInv Dueling + PER + RS + Curr  (best ep=200)
  PPO: ScaleInv ActorCritic + RS + Curr + RN (best ep=100)
```

---

## Slide 13 — 结论（1 min, Person E）

```
标题: 结论 / Conclusions

  1️⃣  ScaleInv 架构 = 最大突破 / largest breakthrough
      LayerNorm + 分离编码 → 跨规模泛化 -85% / generalization gap -85%

  2️⃣  PER + Reward Shaping = DQN 最佳搭档 / best DQN combo
      数据效率 ↑，Test/H -30.2pp

  3️⃣  算法分工明确 / clear algorithm specialization
      PPO 擅小 / small (116.6%)  ·  DQN 擅大 / large (114.7%)
      → 天然 Ensemble / natural for ensemble

  4️⃣  PPO 从崩溃到可用 / from collapse to usable
      ScaleInv + RewardNorm 根治崩溃 / eliminated crashes
      PPO/H: 495.6% → 156.0% → 123.1%

  5️⃣  消融实验量化了各组件贡献 / ablation quantified each component
      为后续研究提供技术路线图 / roadmap for future work

局限 / Limitations:
  · 1000 任务 gap 仍存 / gap remains (DQN+26%, PPO+43%)
  · PPO 需更长训练 / needs more episodes
```

---

## Slide 14 — 致谢（30s, Person E）

```
标题: 感谢聆听 / Thank You

居中:
  欢迎提问 & 讨论
  Questions & Discussion

底部小字:
  代码 / Code: github.com/Lone-Jade/PythonFinalTest
  报告 / Reports: result6.12.md ~ result6.13.6.md
  模型 / Models: ./FinalCode/outputs_exp/dqn/ + ./FinalCode/outputs_exp/ppo/
  可视化 / Figures: ./FinalCode/outputs_exp/figures/ (36 images)

  分工 / Team:
  Person A — 问题引入+方法总览 / Problem & Overview
  Person B — 环境建模+算法设计+ScaleInv / Methodology
  Person C — 实验设计+消融实验 / Experiments & Ablation
  Person D — 最终结果+可视化分析 / Results & Visualization
  Person E — 结论总结+致谢 / Conclusions & Acknowledgements
```

---

## 附录 A — 图片路径汇总

```
Gantt 图 (./FinalCode/outputs_exp/figures/):
  6x6_6x6x2_heuristic_gantt.png    6x6_6x6x2_DQN_300ep_gantt.png    6x6_6x6x2_PPO_300ep_gantt.png
  10x10_10x10x4_heuristic_gantt.png 10x10_10x10x4_DQN_300ep_gantt.png 10x10_10x10x4_PPO_300ep_gantt.png
  50x10_50x10x3_heuristic_gantt.png 50x10_50x10x3_DQN_300ep_gantt.png 50x10_50x10x3_PPO_300ep_gantt.png
  100x10_100x10x3_heuristic_gantt.png 100x10_100x10x3_DQN_300ep_gantt.png 100x10_100x10x3_PPO_300ep_gantt.png

疲劳曲线 (./FinalCode/outputs_exp/figures/):
  同上 *_fatigue.png (12 张)

历史图像 (./FinalCode/outputs_exp/figures/previous/):
  12 张早期模型图像
```

## 附录 B — 数据文件汇总

```
消融结果:
  ./FinalCode/outputs_ablation/{variant}/test/dqn_{variant}_best.json    (DQN × 6)
  ./FinalCode/outputs_ablation/{variant}/test/ppo_{variant}_best.json    (PPO × 4)
  ./FinalCode/outputs_ablation/test/heuristic_rest_aware.json             (baseline)

最佳模型结果:
  ./FinalCode/outputs_exp/test_results/dqn_dqn_best.json
  ./FinalCode/outputs_exp/test_results/ppo_ppo_best.json
  ./FinalCode/outputs_exp/test_results/heuristic_rest_aware.json

训练日志:
  ./FinalCode/outputs_exp/dqn/train_log_live.jsonl  (300 episodes)
  ./FinalCode/outputs_exp/ppo/train_log_live.jsonl  (300 episodes)

模型文件:
  ./FinalCode/outputs_exp/dqn/dqn_best.pt   (ScaleInv Dueling, 71,464 params)
  ./FinalCode/outputs_exp/ppo/ppo_best.pt   (ScaleInv ActorCritic, 88,232 params)
```

## 附录 C — 在 FinalCode 目录下生成数据汇总的 Python 命令

```bash
cd FinalCode

# 消融汇总
python -c "
import json, numpy as np
with open('./FinalCode/outputs_ablation/test/heuristic_rest_aware.json') as f:
    h = {r['instance']: r for r in json.load(f)}

for algo, variants in [('dqn', ['dqn_vanilla','dqn_per_rs','dqn_scaleinv_no_per',
    'dqn_scaleinv','dqn_pair_curr','dqn_full']),
    ('ppo', ['ppo_vanilla','ppo_rs_curr','ppo_scaleinv','ppo_full'])]:
    for v in variants:
        try:
            with open(f'./FinalCode/outputs_ablation/{v}/test/{algo}_{v}_best.json') as f:
                rows = json.load(f)
            ratios = [r['makespan']/h[r['instance']]['makespan'] for r in rows]
            print(f'{v:<25s} avg/H={np.mean(ratios):.1%}')
        except: pass
"

# 300ep 结果
python -c "
import json, numpy as np
with open('./FinalCode/outputs_ablation/test/heuristic_rest_aware.json') as f:
    h = {r['instance']: r for r in json.load(f)}
for algo in ['dqn','ppo']:
    with open(f'./FinalCode/outputs_exp/test_results/{algo}_{algo}_best.json') as f:
        rows = json.load(f)
    ratios = [r['makespan']/h[r['instance']]['makespan'] for r in rows if r['instance'] in h]
    print(f'{algo}_300ep: avg/H={np.mean(ratios):.1%}, n={len(ratios)}')
"
```
