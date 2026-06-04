# 这是记录某次上传的详细文件
## 6.4 下午四点
### 完成的工作
5 个规模 × 5 种方法系统对比
|规模|	状态维度|	动作维度|	训练时间|
|---|---|---|---|
|6×6×2	|43	|12	|~2 分钟|
10×10×3|	70	30	~9 分钟|
15×10×3	|85	|45	|~11 分钟|
20×10×3	|100	|60	|~22 分钟|
30×10×3	|130	|90	|~30 分钟|

每个规模上训练了 DQN + PPO（300 集），并评估了 Greedy SPT / Round-Robin / Random 三个基线。

9 张可视化图表
|类型	|数量	|文件名|
|-|-|-|
|训练曲线	|2	|training_curves_6x6x2.png, training_curves_10x10x3.png|
|单规模方法对比	|5	|method_comparison_{6x6x2,10x10x3,15x10x3,20x10x3,30x10x3}.png|
|跨规模对比	|2	|dataset_comparison_{makespan,fatigue}.png

核心发现
Greedy SPT 在所有 5 个规模上 Makespan 保持绝对领先（100% 胜率）
PPO 优于 DQN：在大规模（30×10×3）上 PPO 比 DQN 优 15.7%，训练速度快 3×
Round-Robin 在 5 个规模中的 4 个上 Fatigue 最低
DQN 随规模退化严重：Greedy 倍率从 1.06× 上升到 1.68×
完整报告见 result_XIWEI.md，内含所有数据表格、图表和规模效应分析。