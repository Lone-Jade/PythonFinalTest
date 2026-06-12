# 基于强化学习的人员配置-生产调度协同优化

本项目实现了一个面向作业车间调度的强化学习实验框架。模型在传统 JSP 的机器约束和工序顺序约束基础上，引入工人、疲劳累积、疲劳导致的加工时间增长、主动休息和强制休息机制。

## 核心建模

- 时间采用整数时间，但仿真采用离散事件驱动 SMDP，不按每个时间单位逐步推进。
- 每道工序一旦开始，不允许中断、换人或休息。
- 动作以工人为中心：当前可决策工人选择 `WAIT`、`REST` 或 `ASSIGN_JOB_j`。
- DQN/PPO 共用同一个共享打分网络，对可变数量的候选动作逐个打分。
- 奖励函数是非线性累计成本的负数，包含工期、平均疲劳、最高疲劳、机器空闲、工人空闲、强制休息和非法动作惩罚。

## 文件说明

- `data_loader.py`：解析 `basic_data.xlsx` 中的多规模调度实例。
- `env.py`：事件驱动调度环境，包含疲劳、休息、奖励和动作掩码。
- `models.py`：共享候选动作打分网络和 PPO actor-critic 网络。
- `agents.py`：DQN 与 PPO 训练逻辑。
- `heuristics.py`：SPT、随机、疲劳感知启发式基线。
- `train.py`：训练入口。
- `evaluate.py`：实例列表和启发式评估入口。
- `visualize.py`：生成甘特图和疲劳曲线。
- `config.py`：环境和训练超参数。
- `draw_fatigue_functions.py`：生成疲劳函数示意图。

## 运行命令

如果使用 VSCode，不想在命令行里改参数，可以直接打开并运行：

```text
experiment.py
```

训练集、验证集、测试集和训练轮数都在 `experiment.py` 文件顶部：

```python
TRAIN_INSTANCES = [...]
VAL_INSTANCES = [...]
TEST_INSTANCES = [...]
EPISODES = 500
```

只需要修改这些列表，然后点击 VSCode 的 Run Python File 即可。实验会自动训练 DQN/PPO，并分别输出验证集和测试集结果。

列出可用实例：

```powershell
& "G:\anaconda\envs\agri\python.exe" evaluate.py --list
```

运行启发式基线：

```powershell
& "G:\anaconda\envs\agri\python.exe" evaluate.py --instance 6x6_6x6x3 --policies spt rest_aware random
```

训练 DQN：

```powershell
& "G:\anaconda\envs\agri\python.exe" train.py --algorithm dqn --instances 6x6_6x6x3 10x5_10x5x3 --episodes 100
```

训练 PPO：

```powershell
& "G:\anaconda\envs\agri\python.exe" train.py --algorithm ppo --instances 6x6_6x6x3 10x5_10x5x3 --episodes 100
```

生成甘特图和疲劳曲线：

```powershell
& "G:\anaconda\envs\agri\python.exe" visualize.py --instance 6x6_6x6x3 --policy rest_aware
```

在测试集实例上评估启发式：

```powershell
& "G:\anaconda\envs\agri\python.exe" test.py --algorithm heuristic --heuristic rest_aware --test-instances 10x5_10x5x3 15x5_15x5x3
```

在测试集实例上评估训练好的 DQN：

```powershell
& "G:\anaconda\envs\agri\python.exe" test.py --algorithm dqn --model-path outputs\dqn\dqn_model.pt --test-instances 10x5_10x5x3 15x5_15x5x3
```

在测试集实例上评估训练好的 PPO：

```powershell
& "G:\anaconda\envs\agri\python.exe" test.py --algorithm ppo --model-path outputs\ppo\ppo_model.pt --test-instances 10x5_10x5x3 15x5_15x5x3
```

## 数据说明

加载器会解析所有合法的 `工件数x机器数x工人数` 表块。当前 Excel 中 `50x10x2` 有一个加工时间单元格填成了机器号，加载器默认跳过该异常实例，并保留其它实例。
