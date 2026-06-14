# PPO 全面升级 + 跨规模 300 轮实验报告

**日期**: 2026-06-13
**项目**: 基于强化学习的人员配置-生产调度协同优化
**修改**: ScaleInvariantActorCritic + Reward Normalization + 300 轮跨规模训练

---

## 一、修改背景

### 1.1 上次问题（result6.13.4.md）

| 问题 | 严重程度 |
|------|:--:|
| PPO 100 轮测试崩溃 | 🔴 致命 |
| PPO makespan = 200-347% × Heuristic | 🔴 |
| PPO 验证 val_makespan=500767 | 🔴 |
| DQN 100 轮 gap 偏大 (+30%) | 🟡 |
| 大实例仅见 3-4 次（100 轮太短） | 🟡 |

### 1.2 根因诊断

| 根因 | DQN 为何不受影响 | PPO 为何崩溃 |
|------|------|------|
| 架构不匹配规模 | ScaleInv Dueling 有 LayerNorm | 纯 MLP 无归一化 |
| 跨规模 reward scale | PER 按 TD-error 采样 | GAE advantage 方差爆炸 (-10 vs +80) |
| 数据效率 | 每步更新 + n-step | 每 rollout 更新 4 epoch |
| 训练轮数 | 300 轮充足 | 100 轮不够收敛 |

### 1.3 本次三项调整

1. **ScaleInvariantActorCritic** — PPO 网络架构升级
2. **Reward Normalization** — 跨规模 reward 归一化
3. **300 轮训练** — 充分收敛

---

## 二、ScaleInvariantActorCritic 架构

### 2.1 设计

```
Input: [global(6), worker(4), action(9)] = 19 dims
    │
    ▼
LayerNorm(input)
    │
    ├── StateEncoder (10 dims):  Linear→LN→ReLU→Linear→LN→ReLU → state_emb (128d)
    │
    ├── ActionEncoder (19 dims): Linear→LN→ReLU→Linear→LN→ReLU → action_emb (128d)
    │
    ├── ValueHead:  state_emb → Linear→LN→ReLU→Linear → V(s)  [scalar]
    │
    └── ActorHead:  concat(state_emb, action_emb) → Linear→LN→ReLU→Linear → logit
```

### 2.2 与旧架构对比

| 特性 | 旧 ActorCritic | **新 ScaleInvActorCritic** |
|------|:--:|:--:|
| Input 归一化 | ❌ | ✅ LayerNorm(19) |
| 分离 State/Action 编码 | ❌ 全特征混入 | ✅ 独立编码器 |
| Pre-LN 隐藏层 | ❌ | ✅ 每层前 LayerNorm |
| Value 输入 | 平均池化全特征 | ✅ 仅 State embedding |
| Actor 输入 | 原始特征 | ✅ Fused(state, action) |
| 参数量 | 74,500 | **88,232** (+18%) |

### 2.3 Reward Normalization

```python
# EMA tracking of return statistics across episodes
self.ret_mean = (1 - 0.01) * self.ret_mean + 0.01 * batch_mean
self.ret_std  = (1 - 0.01) * self.ret_std  + 0.01 * batch_std

# Normalize returns for stable value learning
norm_returns = (returns - self.ret_mean) / self.ret_std
```

效果：小实例 return ≈ -10 和大实例 return ≈ +80 都被映射到统一分布，GAE advantage 估计不再受 scale 差异影响。

---

## 三、训练配置

| 项目 | 旧 PPO 100ep | 新 PPO 300ep |
|------|:--:|:--:|
| 网络 | ActorCritic (MLP) | **ScaleInvariantActorCritic** |
| Reward Norm | ❌ | **✅ EMA running stats** |
| Episodes | 100 | **300** |
| ε 衰减 (DQN) | 0.96/ep | **0.985/ep** |
| ε 终点 (DQN) | ep 97 | ep 270+ |
| Curriculum | S1(1-33)→S2(34-66)→S3(67-100) | S1(1-100)→S2(101-200)→S3(201-300) |
| 大实例接触次数 | 3-4 次 | **10 次** |
| 训练集 | 10 跨规模实例 | 不变 |
| 测试集 | 19 实例 | 不变 |

---

## 四、训练过程

### 4.1 DQN 300ep

| 轮次 | ε | Instance | Reward | Makespan | 阶段 |
|:--:|:--:|------|:--:|:--:|:--:|
| 1 | 1.00 | 6x6x2 | -8.49 | 2111 | S1 |
| 50 | 0.48 | 10x5x3 | -1.02 | 1816 | S1 |
| 100 | 0.22 | 10x5x3 | -1.12 | 1893 | S1→S2 |
| 150 | 0.11 | 10x5x3 | -2.51 | 2130 | S2 |
| 200 | 0.05 | 10x10x4 | -6.27 | 2994 | S2→S3 |
| 250 | 0.02 | **100x10x3** | +79.53 | 34249 | S3 |
| **300** | **0.02** | **100x10x3** | **+80.00** | **34076** | S3 |

> ★ Best model: ep=200（val on 20x10x4: reward=-2.23, makespan=3039）
>
> ε 在 ep ~270 触底 0.02，最后 30 轮纯 exploitation

### 4.2 PPO 300ep（ScaleInvActorCritic + Reward Norm）

| 轮次 | Instance | Reward | Makespan | Loss | 阶段 |
|:--:|------|:--:|:--:|:--:|:--:|
| 1 | 6x6x2 | -10.40 | 2363 | 1.48 | S1 |
| 50 | 10x5x3 | -3.82 | 2086 | 0.05 | S1 |
| 100 | 10x5x3 | -1.39 | 1933 | -0.01 | S1→S2 |
| 150 | 10x5x3 | -2.14 | 2120 | 0.01 | S2 |
| 200 | 10x10x4 | -4.06 | 2852 | 0.01 | S2→S3 |
| 250 | **100x10x3** | +77.03 | 35766 | 0.33 | S3 |
| **300** | **100x10x3** | **+78.73** | **34086** | 0.05 | S3 |

> ★ Best model: ep=100（val on 20x10x4: reward=-2.19, makespan=2782）
>
> 🔥 **PPO 验证首次正常！** val_makespan=2782（vs 之前 500767 崩溃）

### 4.3 训练过程对比

| 指标 | DQN 300ep | PPO 300ep | 分析 |
|------|:--:|:--:|------|
| 最终 makespan (100x10x3) | 34076 | 34086 | 几乎相同 |
| 最终 reward (100x10x3) | +80.00 | +78.73 | 接近 |
| Best val makespan | 3039 (ep 200) | **2782 (ep 100)** | PPO 更优 |
| S3 loss 稳定性 | 0.03 | 0.05-0.33 | DQN 更稳定 |
| 收敛速度 | 渐进 | ep 100 即达最优 | PPO 早期收敛快 |

---

## 五、测试结果

### 5.1 完整对比

| Instance | Tasks | W | Heuristic | DQN 300ep | PPO 300ep | DQN/H | PPO/H | Winner |
|------|:--:|:--:|:---:|:---:|:---:|:--:|:--:|:--:|
| 10x10x2 | 100 | 2 | 4476 | 5901 | **5387** | +31.8% | +20.4% | **PPO** |
| 10x10x3 | 100 | 3 | 3461 | 4196 | **3896** | +21.2% | +12.6% | **PPO** |
| 10x10x5 | 100 | 5 | 2298 | **2356** | 2487 | +2.5% | +8.2% | DQN |
| 10x10x6 | 100 | 6 | 1796 | 2212 | **1927** | +23.2% | +7.3% | **PPO** |
| 20x5x2 | 100 | 2 | 4776 | 5801 | **5777** | +21.5% | +21.0% | **PPO** |
| 20x5x3 | 100 | 3 | 3267 | **3865** | 4249 | +18.3% | +30.1% | DQN |
| 15x10x4 | 150 | 4 | 3790 | **4000** | 4278 | +5.5% | +12.9% | DQN |
| 15x10x5 | 150 | 5 | 2956 | **3309** | 3521 | +11.9% | +19.1% | DQN |
| 15x10x6 | 150 | 6 | 2584 | **2813** | 3036 | +8.9% | +17.5% | DQN |
| 20x10x2 | 200 | 2 | 9334 | 12104 | **11877** | +29.7% | +27.2% | **PPO** |
| 20x10x6 | 200 | 6 | 3451 | 3961 | **3858** | +14.8% | +11.8% | **PPO** |
| 50x10x2 | 500 | 2 | 22442 | **25546** | 28335 | +13.8% | +26.3% | DQN |
| 50x10x4 | 500 | 4 | 11362 | **13091** | 14262 | +15.2% | +25.5% | DQN |
| 50x10x5 | 500 | 5 | 9015 | **10620** | 12001 | +17.8% | +33.1% | DQN |
| 50x10x6 | 500 | 6 | 7734 | **8661** | 9840 | +12.0% | +27.2% | DQN |
| 100x10x2 | 1000 | 2 | 46683 | **51164** | 58698 | +9.6% | +25.7% | DQN |
| 100x10x4 | 1000 | 4 | 21989 | **26465** | 29654 | +20.4% | +34.9% | DQN |
| 100x10x5 | 1000 | 5 | 17646 | **21246** | 24012 | +20.4% | +36.1% | DQN |
| 100x10x6 | 1000 | 6 | 14048 | **17753** | 20028 | +26.4% | +42.6% | DQN |

### 5.2 胜负统计

| | 胜出 | 占比 |
|------|:--:|:--:|
| DQN | **13** | 68.4% |
| PPO | **6** | 31.6% |

### 5.3 按规模分析

| 规模 | 实例数 | DQN/H | PPO/H | 胜者 |
|:--:|:--:|:--:|:--:|:--:|
| S (≤100) | 6 | +19.8% | +16.6% | **PPO 4:2** |
| M (150-200) | 5 | +14.2% | +17.7% | PPO 2:3 |
| L (500) | 4 | +14.7% | +28.0% | **DQN 4:0** |
| XL (1000) | 4 | +19.2% | +34.8% | **DQN 4:0** |

> 清晰的模式：**PPO 在小实例上占优，DQN 在大实例上统治**。这是两种算法本质差异的体现：
> - PPO 的 GAE + 策略梯度更适合精细控制（小实例上接近启发式仅 +7%）
> - DQN 的 PER + n-step 更适合大状态空间的 credit assignment

---

## 六、PPO 改进验证：100ep vs 300ep

| 指标 | PPO 100ep (旧 MLP) | PPO 300ep (ScaleInv) | 改善 |
|------|:--:|:--:|:--:|
| 验证崩溃 | ✅ 崩溃 (500767) | ✅ 正常 (2782) | 🔥 **修复** |
| S 规模 PPO/H | +204.4% | **+16.6%** | 🔥 **-91.9%** |
| M 规模 PPO/H | +212.0% | **+17.7%** | 🔥 **-91.6%** |
| L 规模 PPO/H | +216.9% | **+28.0%** | 🔥 **-87.1%** |
| XL 规模 PPO/H | +223.2% | **+34.8%** | 🔥 **-84.4%** |
| vs DQN 胜出 | 0/19 | **6/19** | 🔥 |
| 100x10x6 PPO/H | +247.2% | **+42.6%** | 🔥 |

> PPO 从全面崩溃 → 小实例反超 DQN，改善幅度惊人。

---

## 七、与 ScaleInv DQN 200ep 对比（共同实例）

| Instance | Heur | ScaleInv 200ep | **Cross-Scale 300ep** | Δ |
|------|:--:|:--:|:--:|:--:|
| 10x10x5 | 2298 | 2585 | **2356** | **-8.9%** ✅ |
| 10x10x6 | 1796 | **2094** | 2212 | +5.6% |
| 15x10x4 | 3790 | 4076 | **4000** | **-1.9%** ✅ |
| 15x10x5 | 2956 | 3479 | **3309** | **-4.9%** ✅ |
| 15x10x6 | 2584 | 3122 | **2813** | **-9.9%** ✅ |
| 20x10x2 | 9334 | 12718 | **12104** | **-4.8%** ✅ |
| 20x10x6 | 3451 | **3618** | 3961 | +9.5% |
| 50x10x2 | 22442 | 28645 | **25546** | **-10.8%** ✅ |
| 50x10x4 | 11362 | 13124 | **13091** | ≈ |
| 50x10x5 | 9015 | 10728 | **10620** | **-1.0%** ✅ |
| 50x10x6 | 7734 | 8797 | **8661** | **-1.5%** ✅ |
| 100x10x2 | 46683 | 58125 | **51164** | **-12.0%** ✅ |
| 100x10x4 | 21989 | **26125** | 26465 | +1.3% |
| 100x10x5 | 17646 | **20658** | 21246 | +2.8% |
| 100x10x6 | 14048 | **17083** | 17753 | +3.9% |

> **15 个共同实例，Cross-Scale 300ep 胜出 10 个（67%），平均改善约 -3%。**
>
> 尤其在 50x10x2（-10.8%）和 100x10x2（-12.0%）上改善显著——大实例入训练集的效果开始显现！

---

## 八、全部优化演进（最终版）

| 阶段 | 修改 | 训练轮数 | DQN/H on 100x10x6 | PPO/H on 100x10x6 | DQN vs PPO |
|------|------|:--:|:--:|:--:|:--:|
| ① | n-step (原始) | 300 | — | — | PPO 全胜 |
| ② | +Reward Shaping | 300 | — | — | PPO 20:4 |
| ③ | +PER | 100 | ~+16%¹ | — | DQN 领先 |
| ④ | +Dueling+Curr | 500 | +27% | — | PPO 14:5 |
| ⑤ | +ScaleInv DQN | 200 | **+22%** | — | PPO 10:7 |
| ⑥ | +跨规模训练集 | 100 | +26% | **+247%** 💀 | DQN 19:0 |
| **⑦** | **+ScaleInv PPO +300轮** | **300** | **+26%** | **+43%** 🔥 | **DQN 13:6** |

> ¹ 有数据重叠，结果偏乐观
>
> PPO 从崩溃中恢复，从小实例开始反超 DQN。DQN 仍在大实例上有明显优势。

---

## 九、代码修改总览

| 文件 | 修改 |
|------|------|
| `models.py:158-237` | **新增 `ScaleInvariantActorCritic`**（LayerNorm + 分离编码 + ValueHead + ActorHead） |
| `models.py:234` | 修复 value.reshape(()) 形状问题 |
| `agents.py:18` | 导入 `ScaleInvariantActorCritic` |
| `agents.py:191-280` | **PPOAgent 重构**：使用 ScaleInv 架构 + EMA reward normalization |
| `test.py:12` | 导入 `ScaleInvariantActorCritic` |
| `test.py:102-108` | PPO 模型加载自动检测 ScaleInv 架构 |
| `config.py:60` | epsilon_decay: 0.96 → **0.985**（300 轮适配） |
| `experiment.py:96` | RUN_VALIDATION → **True** |
| `experiment.py:100` | EPISODES: 100 → **300** |

---

## 十、模型文件

| 模型 | 路径 | 配置 |
|------|------|------|
| DQN Cross-Scale 300ep (best ep=200) | `outputs_exp/dqn/dqn_best.pt` | ScaleInv Dueling + PER + RS + Curr |
| PPO ScaleInv 300ep (best ep=100) | `outputs_exp/ppo/ppo_best.pt` | **ScaleInv ActorCritic + Reward Norm** + RS + Curr |

---

## 十一、结论

### 11.1 核心成果

1. 🔥 **PPO 崩溃完全修复**：ScaleInvariantActorCritic + Reward Normalization 使 PPO 从 247% gap → 43% gap（100x10x6）
2. 🎯 **PPO 小实例反超 DQN**：在 ≤200 任务实例上 PPO 赢 6 个，DQN 赢 5 个
3. 📈 **DQN 大实例仍领先**：500+ 任务实例上 DQN 8:0 全胜
4. ✅ **跨规模训练集生效**：Cross-Scale 300ep 在 67% 共同实例上超越 ScaleInv 200ep（纯小实例训练）
5. 🔬 **算法分工清晰**：PPO（策略梯度）精于小实例精细调度，DQN（价值学习+PER）擅长大实例 credit assignment

### 11.2 尚存问题

| 问题 | 证据 | 可能方向 |
|------|------|------|
| PPO 大实例仍落后 | 50x10+ 全败 | 增加 PPO rollout steps，或增大 entropy |
| DQN 小实例不如 PPO | S 规模 4:2 落后 | 降低 ε 衰减速度，增加小实例 exploration |
| 100x10x6 gap 仍 26%+ | 两者都落后启发式 | 更多大实例训练样本，或提升大实例采样权重 |

### 11.3 推荐下一步

| 优先级 | 方向 | 依据 |
|:--:|------|------|
| ⭐⭐⭐ | **Ensemble DQN+PPO** | PPO 擅小、DQN 擅大——天然互补 |
| ⭐⭐ | 提升大实例采样权重 | 当前 S3 每大实例仅见 ~10 次 |
| ⭐⭐ | PPO 500 轮 + 更大 entropy | 大实例上 PPO 仍有改善空间 |
| ⭐ | PPO value clipping | 进一步稳定跨规模 value 学习 |
