# 基于强化学习的人员配置-生产调度协同优化 —— 问题建模

## 一、问题形式化描述

### 1.1 问题定义

考虑一个柔性作业车间调度问题（Flexible Job Shop Scheduling Problem, FJSP），包含：

- $N$ 个工件（Jobs）$J = \{J_1, J_2, \ldots, J_N\}$
- $M$ 台机器（Machines）$M = \{M_1, M_2, \ldots, M_M\}$
- $W$ 名操作员（Workers）$P = \{P_1, P_2, \ldots, P_W\}$

每个工件 $J_i$ 包含 $O_i$ 道工序，工序按顺序执行。每道工序 $O_{i,j}$ 需要在指定的机器 $m_{i,j}$ 上加工，可选任意操作员 $p \in P$ 执行，加工时间为 $t_{i,j,p}$（不同操作员的技能水平不同导致加工时间不同）。

### 1.2 约束条件

1. **工序顺序约束**：同一工件的工序必须按顺序执行
2. **机器能力约束**：每台机器同一时刻只能加工一个工件
3. **操作员能力约束**：每个操作员同一时刻只能操作一台机器
4. **不可抢占约束**：一旦开始加工，不可中断

### 1.3 疲劳模型

操作员 $p$ 的疲劳度 $F_p$ 动态变化：

- **疲劳积累**：操作员连续工作时，疲劳度线性增长
  $$F_p(t + \Delta t) = F_p(t) + \alpha \cdot \Delta t$$
  其中 $\alpha$ 为疲劳积累率。

- **疲劳恢复**：操作员空闲时，疲劳度逐渐恢复
  $$F_p(t + \Delta t) = \max(0, F_p(t) - \beta \cdot \Delta t)$$
  其中 $\beta$ 为疲劳恢复率。

- **疲劳对效率的影响**：疲劳度越高，加工效率越低
  $$t_{actual} = t_{base} \cdot (1 + \gamma \cdot F_p)$$
  其中 $\gamma$ 为疲劳影响系数。

### 1.4 目标函数

$$\min \left( C_{max} + \lambda \cdot \sum_{p=1}^{W} \max(0, F_p - F_{threshold}) \right)$$

其中：
- $C_{max} = \max_i C_i$ 为最大完工时间（Makespan），$C_i$ 为工件 $J_i$ 的完工时间
- $F_p$ 为操作员 $p$ 的最终疲劳度
- $F_{threshold}$ 为疲劳度阈值，超过此阈值才受惩罚
- $\lambda$ 为疲劳惩罚权重（效率优先 vs. 健康优先的调节参数）

---

## 二、马尔可夫决策过程（MDP）建模

### 2.1 状态空间 $S$

状态向量包含以下信息：

| 组件 | 维度 | 说明 |
|------|------|------|
| 机器状态 | $M \times 2$ | 每台机器的 [是否空闲, 剩余加工时间（归一化）] |
| 操作员状态 | $W \times 2$ | 每个操作员的 [是否空闲, 当前疲劳度] |
| 工件状态 | $N \times 2$ | 每个工件的 [完成进度(0-1), 下一工序所需机器ID（归一化）] |
| 全局状态 | 1 | 当前时间（归一化） |

总状态维度：$2M + 2W + 2N + 1$

### 2.2 动作空间 $A$

动作定义为一个 $(job\_id, worker\_id)$ 对：
$$A = \{(j, w) \mid j \in \{0, \ldots, N-1\}, w \in \{0, \ldots, W-1\}\}$$

动作 $(j, w)$ 表示：将工件 $j$ 的下一道工序分配给操作员 $w$ 执行。

动作掩码：以下情况动作无效
1. 工件 $j$ 已完成所有工序
2. 工件 $j$ 下一工序所需机器当前被占用
3. 操作员 $w$ 当前正忙

### 2.3 状态转移

采用事件驱动仿真：

1. 在决策点，智能体选择有效动作 $(j, w)$
2. 将工件 $j$ 的当前工序分配至所需机器，由操作员 $w$ 执行
3. 加工时间 = $t_{j, op, w} \cdot (1 + \gamma \cdot F_w)$（受疲劳影响）
4. 操作员疲劳度在加工期间累积
5. 仿真时钟跳至下一个事件完成时间
6. 释放机器和操作员，操作员在空闲期间恢复疲劳

### 2.4 奖励函数 $R$

采用**稠密奖励 + 终端惩罚**的方案：

**步骤奖励**：
$$r_{step} = -\Delta t \cdot \left(1 + \eta \cdot \frac{1}{W}\sum_{p=1}^{W} F_p\right)$$

- $\Delta t$ 为时间推进量
- 鼓励智能体减少空闲等待和降低疲劳

**终端奖励**：
$$r_{terminal} = -\lambda_{fatigue} \cdot \frac{1}{W}\sum_{p=1}^{W} \max(0, F_p - F_{threshold})$$

- 对高疲劳状态施加最终惩罚
- $\lambda_{fatigue}$ 控制疲劳惩罚强度

**总奖励**：
$$R = \sum r_{step} + r_{terminal}$$

即：$$R = -\left(C_{max} + \sum_t \Delta t \cdot \eta \cdot \bar{F}_t + \lambda_{fatigue} \cdot \overline{F}_{penalty}\right)$$

---

## 三、强化学习算法设计

### 3.1 算法选择：Dueling Double DQN

选用 **Dueling Double DQN**，理由如下：

1. **Double DQN**：解耦动作选择和价值评估，减少Q值高估偏差
2. **Dueling Architecture**：分离状态价值 $V(s)$ 和优势函数 $A(s,a)$，在动作数量大时更稳定
3. **优先经验回放（PER）**：优先采样TD误差大的样本，提高学习效率
4. **Action Masking**：在Q值计算后屏蔽无效动作，保证策略可行性

### 3.2 网络结构

```
输入层: state_dim → 
全连接层1: 256 + ReLU →
全连接层2: 128 + ReLU →
Value Stream: 64 → 1
Advantage Stream: 64 → action_dim
输出: V(s) + (A(s,a) - mean(A(s,a)))
```

### 3.3 训练方法

- **探索策略**：$\epsilon$-greedy with exponential decay
  $$\epsilon = \epsilon_{end} + (\epsilon_{start} - \epsilon_{end}) \cdot e^{-t/\tau}$$
- **优化器**：Adam
- **损失函数**：Huber Loss（Smooth L1 Loss）
- **目标网络**：软更新 $\theta' \leftarrow \tau \theta + (1-\tau) \theta'$
- **经验回放**：容量 10000，批次大小 64

### 3.4 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 1e-3 | 学习率 |
| gamma | 0.99 | 折扣因子 |
| epsilon_start | 1.0 | 初始探索率 |
| epsilon_end | 0.01 | 最小探索率 |
| epsilon_decay | 2000 | 探索率衰减系数 |
| batch_size | 64 | 批次大小 |
| memory_capacity | 10000 | 经验池容量 |
| target_update | 0.005 | 目标网络软更新率 |
| alpha | 0.02 | 疲劳积累率 |
| beta | 0.015 | 疲劳恢复率 |
| gamma_fatigue | 0.1 | 疲劳对加工时间的影响系数 |
| F_threshold | 1.0 | 疲劳惩罚阈值 |
| lambda_fatigue | 50.0 | 疲劳惩罚权重 |
| eta | 0.1 | 步骤奖励中的疲劳系数 |

---

## 四、数据集映射

数据集命名规则 `{作业数}x{机器数}x{操作员数}.csv`，例如：
- `6x6x2.csv`：6个作业，6台机器，2个操作员（小规模）
- `10x10x6.csv`：10个作业，10台机器，6个操作员（中规模）
- `50x10x6.csv`：50个作业，10台机器，6个操作员（大规模）

训练建议：小规模数据用于快速验证和调试，中规模数据用于训练和对比实验，大规模数据用于测试泛化能力。

---

## 五、对比基线

1. **贪婪启发式**：每次选择加工时间最短的（工件，操作员）组合
2. **Round-Robin**：轮询分配操作员
3. **DQN（无疲劳感知）**：仅优化Makespan，忽略疲劳度的DQN基线
4. **随机策略**：随机选择有效动作
