# 基于强化学习的人员配置-生产调度协同优化

> Reinforcement Learning-based Personnel Allocation & Production Scheduling Co-Optimization

本项目实现了一个面向作业车间调度的强化学习实验框架。在传统 JSP 的机器约束和工序顺序约束基础上，引入工人、疲劳累积、疲劳导致的加工时间增长、主动休息和强制休息机制。

---

## 目录

1. [项目概述](#一项目概述)
2. [文件结构](#二文件结构)
3. [神经网络架构](#三神经网络架构)
4. [模型参数](#四模型参数)
5. [数据集分配](#五数据集分配)
6. [模型修改历程](#六模型修改历程)
7. [最终结果](#七最终结果)
8. [图像清单](#八图像清单)
9. [结论与分析](#九结论与分析)
10. [快速开始](#十快速开始)

---

## 一、项目概述

- **建模方式**: SMDP（半马尔可夫决策过程），离散事件驱动
- **动作空间**: 以工人为中心 → `WAIT` / `REST` / `ASSIGN_JOB_j`
- **状态特征**: 19 维 = global(6) + worker(4) + action(9)
- **核心约束**: 每道工序一旦开始不可中断；工人疲劳累积→加工时间增长→强制休息
- **算法**: DQN（PER + n-step）+ PPO（GAE + RewardNorm）
- **数据**: `data/basic_data.xlsx` — 40 个实例，规模 6×6×2 到 100×10×6

---

## 二、文件结构

### 2.1 核心程序

| 文件 | 说明 |
|------|------|
| `config.py` | `EnvConfig`（疲劳参数、奖励权重）+ `TrainConfig`（RL 超参数） |
| `data_loader.py` | 从 Excel 解析 Job-Shop 实例（工件数×机器数×工人数） |
| `env.py` | SMDP 调度环境：状态转移、奖励计算、工人疲劳/休息机制 |
| `models.py` | 6 种神经网络架构（DQN 3 种 + PPO 3 种） |
| `agents.py` | DQNAgent（PER + n-step）+ PPOAgent（GAE + EMA RewardNorm） |
| `heuristics.py` | 启发式策略：SPT、Rest-Aware、Random、Fatigue |
| **`experiment.py`** | **主实验脚本**：训练→验证→测试→可视化，一键运行 |
| `test.py` | 模型加载、策略评估、JSON/CSV 输出 |
| `train.py` | 早期训练脚本（已被 experiment.py 取代） |
| `evaluate.py` | 独立评估脚本 |
| `visualize.py` | Gantt 图 + 疲劳曲线绘制 |
| `draw_fatigue_functions.py` | 疲劳函数可视化 |

### 2.2 消融实验

| 文件 | 说明 |
|------|------|
| `ablation.py` | 消融实验公共框架（配置工厂函数） |
| `dqn_ablation.py` | DQN 消融（4 变体 × 50ep），自包含 |
| `ppo_ablation.py` | PPO 消融（4 变体 × 50ep），自包含 |

### 2.3 输出目录

| 目录 | 内容 |
|------|------|
| `outputs_exp/dqn/` | DQN 模型（`.pt`）+ 训练日志（`.jsonl`） |
| `outputs_exp/ppo/` | PPO 模型 + 训练日志 |
| `outputs_exp/figures/` | Gantt 图 + 疲劳曲线（24 张最新） |
| `outputs_exp/figures/previous/` | 历史图像备份（12 张） |
| `outputs_exp/test_results/` | 测试集结果（JSON/CSV） |
| `outputs_exp/val_results/` | 验证集结果 |
| `outputs_ablation/` | 消融实验输出（8 变体 × 3 split = 24 子目录） |

### 2.4 实验报告（按时间顺序）

| 报告 | 内容 |
|------|------|
| `result6.12.md` | 初始实验：n-step DQN + GAE PPO |
| `result6.13.0.md` | Reward Shaping 密集奖励 |
| `result6.13.1.md` | Prioritized Experience Replay (PER) |
| `result6.13.2.md` | Dueling DQN + Curriculum Learning + 500ep |
| `result6.13.3.md` | Scale-Invariant Representation Learning |
| `result6.13.4.md` | 跨规模训练集 + 100ep 实验 |
| `result6.13.5.md` | PPO 全面升级（ScaleInv + RewardNorm）+ 300ep |
| `result6.13.6.md` | **消融实验 + 基线对比 + 全面评估（最终报告）** |

---

## 三、神经网络架构

### 3.1 DQN 系列

| 网络 | 参数量 | LayerNorm | 分离编码 | Dueling |
|------|:--:|:--:|:--:|:--:|
| `PairScoringNetwork` | 37,250 | ❌ | ❌ | ❌ |
| `DuelingPairScoringNetwork` | 74,500 | ❌ | ❌ | ✅ |
| **`ScaleInvariantDuelingNetwork`** | **71,464** | ✅ | ✅ | ✅ |

#### ScaleInvariantDuelingNetwork（最终模型）

```
Input [19]: global(6) + worker(4) + action(9)
    │
    ▼ LayerNorm(input)           ← 跨规模归一化
    │
    ├─ StateEncoder(10d) → Linear→LN→ReLU→Linear→LN→ReLU → state_emb(128d)
    ├─ ActionEncoder(19d) → Linear→LN→ReLU→Linear→LN→ReLU → action_emb(128d)
    ├─ ValueHead:  state_emb → Linear → V(s)                    [scalar]
    └─ AdvantageHead: concat(state_emb, action_emb) → Linear→LN→ReLU→Linear → A(s,a)

    Q(s,a) = V(s) + A(s,a) - mean(A)
```

### 3.2 PPO 系列

| 网络 | 参数量 | LayerNorm | 分离编码 | RewardNorm |
|------|:--:|:--:|:--:|:--:|
| `ActorCriticNetwork` | 74,500 | ❌ | ❌ | ❌ |
| **`ScaleInvariantActorCritic`** | **88,232** | ✅ | ✅ | ✅ |

#### ScaleInvariantActorCritic（最终模型）

```
Input [19]
    │
    ▼ LayerNorm(input)
    │
    ├─ StateEncoder(10d) → 同 DQN → state_emb(128d)
    ├─ ActionEncoder(19d) → 同 DQN → action_emb(128d)
    ├─ ValueHead:  state_emb → Linear→LN→ReLU→Linear → V(s)    [scalar]
    └─ ActorHead:  concat(state_emb, action_emb) → Linear→LN→ReLU→Linear → logit

    π(a|s) = Categorical(logits)
    V(s) — 仅从 state 特征预测（不受 action 影响）
```

---

## 四、模型参数

### 4.1 环境参数 (`EnvConfig`)

| 参数 | 值 | 说明 |
|------|:--:|------|
| `alpha` | 0.020 | 疲劳增长速率 |
| `gamma_rest` | 0.045 | 休息恢复速率 |
| `f_force` | 0.80 | 强制休息阈值 |
| `f_resume` | 0.50 | 恢复工作阈值 |
| `beta` | 0.60 | 疲劳→加工时间增长系数 |
| `s_time` | 1.0 | 时间惩罚 |
| `s_makespan` | 2.0 | makespan 奖励 |
| `s_job_completion` | 0.5 | 任务完成奖励 (RS) |
| `s_efficiency` | 0.05 | 低疲劳分配奖励 (RS) |
| `s_progress` | 0.02 | 有效分配奖励 (RS) |
| `s_stall` | 0.1 | 无意义等待惩罚 (RS) |

### 4.2 训练参数 (`TrainConfig`)

| 参数 | 值 | 说明 |
|------|:--:|------|
| `episodes` | 300 | 训练轮数（最终版） |
| `gamma` | 0.99 | 折扣因子 |
| `lr` | 3×10⁻⁴ | Adam 学习率 |
| `lr_decay` | 0.995 | 每轮学习率衰减 |
| `hidden_dim` | 128 | 隐藏层维度 |
| `batch_size` | 64 | DQN 批大小 |
| `replay_size` | 100,000 | 回放缓冲区 |
| `target_update` | 500 | 目标网络更新频率 |
| `epsilon_start` / `end` | 1.0 / 0.02 | ε-greedy 范围 |
| `epsilon_decay` | 0.985 | ε 每轮衰减 (0.985³⁰⁰≈0.011) |
| `n_step` | 10 | n-step return |
| `per_alpha` | 0.6 | PER 优先级指数 |
| `per_beta` | 0.4 | IS 权重起始值 |
| `rollout_steps` | 1024 | PPO rollout 步数 |
| `ppo_epochs` | 4 | PPO 更新轮数 |
| `clip_ratio` | 0.20 | PPO 裁剪比率 |
| `entropy_coef` | 0.01 | 熵正则系数 |
| `value_coef` | 0.50 | 价值损失系数 |

### 4.3 PPO Reward Normalization

| 参数 | 值 | 说明 |
|------|:--:|------|
| `ret_momentum` | 0.01 | EMA 动量（慢速适应跨规模 reward 变化） |

---

## 五、数据集分配

### 5.1 最终训练集（10 实例，6 个规模级，零重叠）

| 规模级 | 实例 | 任务数 | Workers | 选取理由 |
|:--:|------|:--:|:--:|------|
| 微小 | `6x6_6x6x2` | 36 | 2 | 最小规模，低 worker |
| 微小 | `10x5_10x5x3` | 50 | 3 | 不同形状 (10×5) |
| 小 | `15x5_15x5x2` | 75 | 2 | tall shape，低 worker |
| 小 | `10x10_10x10x4` | 100 | 4 | square shape，中 worker |
| 中 | `15x10_15x10x3` | 150 | 3 | 矩形，中 worker |
| 中 | `20x10_20x10x5` | 200 | 5 | 更大矩形，高 worker |
| 大 | `30x10_30x10x2` | 300 | 2 | 首次入训大实例 |
| 大 | `30x10_30x10x6` | 300 | 6 | 高低 worker 泛化 |
| 很大 | `50x10_50x10x3` | 500 | 3 | 大规模 |
| 极端 | **`100x10_100x10x3`** | **1000** | 3 | **闭合 train-test gap** |

### 5.2 验证集（6 实例）

`20x10_20x10x4`, `30x5_30x5x2`, `30x5_30x5x3`, `30x10_30x10x3`, `30x10_30x10x4`, `30x10_30x10x5`

### 5.3 测试集（19 实例）

`10x10_10x10x2,3,5,6` · `15x10_15x10x4,5,6` · `20x10_20x10x2,6` · `20x5_20x5x2,3` · `50x10_50x10x2,4,5,6` · `100x10_100x10x2,4,5,6`

> Train ∩ Val ∩ Test = ∅，总计 **35 个独立实例**。

---

## 六、模型修改历程

| 阶段 | 报告 | 修改 | DQN Test/H | PPO Test/H | DQN vs PPO |
|:--:|------|------|:--:|:--:|:--:|
| ① | result6.12 | n-step DQN + GAE PPO (基线) | ~160% | ~250% | PPO 全胜 |
| ② | result6.13.0 | +Reward Shaping | — | — | PPO 更受益 |
| ③ | result6.13.1 | +Prioritized Experience Replay | ~135% | — | DQN 反超 |
| ④ | result6.13.2 | +Dueling DQN + Curriculum + 500ep | ~127% | ~120% | PPO 14:5 |
| ⑤ | result6.13.3 | **+ScaleInvariant DQN** | **122%** | — | PPO 10:7 |
| ⑥ | result6.13.4 | 跨规模训练集 + 100ep | ~130% | 💀 崩溃 | DQN 19:0 |
| ⑦ | result6.13.5 | **+ScaleInv PPO + RewardNorm + 300ep** | **117%** | **123%** | DQN 13:6 |
| ⑧ | result6.13.6 | **消融实验**：各组件贡献量化 | 118% | 215% | — |

### 消融结论（DQN 50ep / PPO 50ep，Test set）

| 组件 | DQN 贡献 | PPO 贡献 | 适用范围 |
|------|:--:|:--:|------|
| PER + Reward Shaping | 🔥🔥🔥 **-18.3pp** | — | DQN 专用 |
| ScaleInv 架构 | 🔥🔥 -11.1pp | 🔥🔥🔥 **-68.5pp** | **两者通用** |
| Curriculum | -1.5pp | ❌ 短训练不稳定 | 需充足轮数 |
| Reward Normalization | — | ❌ 短训练不稳定 | PPO 专用 |
| 充分训练 (300ep) | -1.0pp | 🔥🔥 -42.6pp | 两者通用 |

---

## 七、最终结果

### 7.1 最佳模型

| 模型 | 路径 | 配置 |
|------|------|------|
| **DQN 300ep** (best ep=200) | `outputs_exp/dqn/dqn_best.pt` | ScaleInv Dueling + PER + RS + Curriculum |
| **PPO 300ep** (best ep=100) | `outputs_exp/ppo/ppo_best.pt` | ScaleInv ActorCritic + RS + Curriculum + RewardNorm |

### 7.2 测试集性能（按规模）

| 规模 | 实例数 | DQN 300ep/H | PPO 300ep/H | 胜者 |
|:--:|:--:|:--:|:--:|:--:|
| S (≤100) | 6 | +19.8% | **+16.6%** | **PPO** |
| M (150-200) | 5 | **+14.2%** | +17.7% | DQN |
| L (500) | 4 | **+14.7%** | +28.0% | **DQN** |
| XL (1000) | 4 | **+19.2%** | +34.8% | **DQN** |

> 🔬 **算法分工**: PPO（策略梯度精细控制）擅小实例，DQN（PER+n-step 信用分配）擅大实例 → 天然适合 Ensemble

### 7.3 关键指标演变

| 指标 | 初始基线 | 最终结果 | 改善 |
|------|:--:|:--:|:--:|
| DQN 100×10×6 / Heuristic | ~+60% | **+26%** | **-34pp** |
| PPO 100×10×6 / Heuristic | ~+47% | **+43%** | -4pp |
| PPO 崩溃次数 | 多次 | **0** | ✅ 根治 |
| 泛化差距 (Train→Test) | +12.8pp | **+1.9pp** | **-85%** |
| 训练集规模覆盖 | 36–200 | **36–1000** | **5×** |

---

## 八、图像清单

### 8.1 最新图像（`outputs_exp/figures/`，36 张）

**Gantt 图**（横轴=时间，纵轴=机器，颜色=工人）：

| 图像 | 实例 | 方法 | Makespan |
|------|:--:|------|:--:|
| `6x6_6x6x2_heuristic_gantt.png` | 36 任务 | 启发式 | 1766 |
| `6x6_6x6x2_DQN_300ep_gantt.png` | 36 任务 | DQN | 2437 |
| `6x6_6x6x2_PPO_300ep_gantt.png` | 36 任务 | PPO | 2326 |
| `10x10_10x10x4_heuristic_gantt.png` | 100 任务 | 启发式 | 2394 |
| `10x10_10x10x4_DQN_300ep_gantt.png` | 100 任务 | DQN | 3039 |
| `10x10_10x10x4_PPO_300ep_gantt.png` | 100 任务 | PPO | 2773 |
| `50x10_50x10x3_heuristic_gantt.png` | 500 任务 | 启发式 | 15520 |
| `50x10_50x10x3_DQN_300ep_gantt.png` | 500 任务 | DQN | 18965 |
| `50x10_50x10x3_PPO_300ep_gantt.png` | 500 任务 | PPO | 19818 |
| `100x10_100x10x3_heuristic_gantt.png` | 1000 任务 | 启发式 | 30038 |
| `100x10_100x10x3_DQN_300ep_gantt.png` | 1000 任务 | DQN | 34159 |
| `100x10_100x10x3_PPO_300ep_gantt.png` | 1000 任务 | PPO | 39207 |

**疲劳曲线**（横轴=时间，纵轴=疲劳度，红色虚线=强制休息线，绿色虚线=恢复线）：

| 图像 | 对应 Gantt |
|------|------|
| `*_heuristic_fatigue.png` | 启发式各工人疲劳曲线 |
| `*_DQN_300ep_fatigue.png` | DQN 各工人疲劳曲线 |
| `*_PPO_300ep_fatigue.png` | PPO 各工人疲劳曲线 |

### 8.2 历史图像（`outputs_exp/figures/previous/`，12 张）

早期模型（6x6x2, 6x6x3）的 `dqn_model`、`ppo_model`、`dqn_best`、`ppo_best` 图像。

---

## 九、结论与分析

### 9.1 核心发现

1. **ScaleInvariant 架构是最大的单一突破** — LayerNorm + 分离 State/Action 编码器使 PPO 从崩溃（497% gap）恢复到可用（156%），DQN 额外改善 11pp
2. **PER + Reward Shaping 是 DQN 的最佳搭档** — 贡献 -18.3pp，数据效率提升显著
3. **PPO 需要充足训练才能稳定** — 50ep 下 RewardNorm 和 Curriculum 反而恶化，300ep 后才能体现全部组件价值
4. **算法分工明确** — PPO（策略梯度精细控制）擅小实例，DQN（PER+n-step 信用分配）擅大实例 → 天然适合 Ensemble
5. **跨规模训练集是泛化的基础** — 训练集覆盖到 1000 任务后，Train→Test 泛化差距从 +12.8pp 缩到 +1.9pp（**-85%**）

### 9.2 各技术贡献量化

| 技术 | 作用 | 效果 |
|------|------|:--:|
| LayerNorm 输入归一化 | 消除不同规模实例的特征分布差异 | DQN +11pp, PPO +68pp |
| 分离 State/Action 编码 | 防止规模信息泄漏到动作选择 | 泛化能力大幅提升 |
| Pre-LN 隐藏层 | 稳定跨规模深度网络训练 | Loss 收敛更平稳 |
| EMA Reward Normalization | 统一小实例(-10)与大实例(+80)的 reward scale | PPO 不再崩溃 |
| Prioritized Experience Replay | 按 TD-error 优先采样关键 transition | DQN 数据效率 2× |
| n-step Returns (n=10) | 加速信用分配传播 | 稀疏奖励下更快收敛 |
| Curriculum Learning | 渐进增加难度 | DQN +1.5pp，需充足轮数 |

### 9.3 推荐方向

| 优先级 | 方向 | 依据 |
|:--:|------|------|
| ⭐⭐⭐ | **Ensemble DQN+PPO** | PPO 擅小、DQN 擅大，天然互补 |
| ⭐⭐ | PPO 500ep+ | PPO 仍在大实例上落后，更多轮次可能改善 |
| ⭐⭐ | 大实例采样权重提升 | S3 每大实例仅见 ~10 次 |
| ⭐ | PPO value clipping | 进一步稳定跨规模 value 学习 |
| ⭐ | DQN + RewardNorm | 验证 DQN 是否也能受益 |

---

## 十、快速开始

```bash
cd XUYANG_code

# 完整训练（300 轮 DQN + PPO）
python experiment.py

# 仅测试已有模型
python test.py --algorithm dqn --model-path outputs_exp/dqn/dqn_best.pt \
  --test-instances 10x10_10x10x5 100x10_100x10x6

# 运行消融实验
python dqn_ablation.py   # DQN 消融（4 变体 × 50ep）
python ppo_ablation.py   # PPO 消融（4 变体 × 50ep）

# 生成 Gantt 图 + 疲劳曲线
python visualize.py
```
