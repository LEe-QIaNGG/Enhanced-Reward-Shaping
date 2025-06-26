# SAC算法在Atari环境中的强化学习实验

这个项目使用Stable Baselines3的SAC（Soft Actor-Critic）算法在Gymnasium的Atari游戏环境中进行强化学习实验。

## 项目特点

- 使用SAC算法进行连续动作空间的强化学习
- 支持Atari游戏环境（如Breakout、Pong等）
- 包含完整的训练、评估和测试流程
- 自动保存最佳模型和检查点
- 支持TensorBoard可视化
- 包含学习率调度和自动熵调整
- 完整的配置管理系统
- **集成Weights & Biases (wandb) 实验跟踪**

## 环境要求

- Python 3.8+
- PyTorch
- Stable Baselines3
- Gymnasium (包含Atari支持)
- Weights & Biases (可选，用于实验跟踪)

## 快速开始

### 1. 安装依赖

我使用的是python3.10, torch 2.6.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```bash
pip install -r requirements.txt
```

### 2. 安装Atari ROM文件

```bash
python -m AutoROM --accept-license
```

### 3. 设置Wandb (可选)

如果您想使用wandb进行实验跟踪：

```bash
pip install wandb
wandb login
```

### 4. 开始训练

#### 基本训练

```bash
# 使用默认配置训练（Breakout游戏，100万步）
python train.py

# 指定游戏和训练步数
python train.py --env ALE/Breakout-v5 --timesteps 100 --no-wandb
```

#### 其他选项

```bash
# 禁用wandb
python train.py --no-wandb

# 显示配置
python train.py --config

# 仅测试环境
python train.py --test
```

#### 测试环境

```bash
python test_environments.py
```

## 项目结构

```
Enhanced-Reward-Shaping/
├── train.py              # 主训练脚本
├── config.py             # 配置文件
├── test_environments.py  # 环境测试脚本
├── setup_wandb.py        # Wandb设置脚本
├── wandb_config.py       # Wandb配置文件
├── wandb_callback.py     # Wandb回调类
├── requirements.txt      # 依赖项列表
├── README.md            # 项目说明
└── logs/                # 训练日志目录（自动创建）
    └── sac_atari/
        ├── best_model/   # 最佳模型
        ├── checkpoints/  # 检查点
        ├── eval_logs/    # 评估日志
        └── tensorboard/  # TensorBoard日志
```

## 配置说明

### 环境配置

在`config.py`中可以修改以下参数：

- `env_id`: Atari游戏环境ID
- `num_envs`: 并行环境数量
- `seed`: 随机种子

### 训练配置

- `total_timesteps`: 总训练步数
- `eval_freq`: 评估频率
- `save_freq`: 模型保存频率
- `n_eval_episodes`: 评估时的episode数量

### SAC算法超参数

- `learning_rate`: 学习率
- `buffer_size`: 经验回放缓冲区大小
- `batch_size`: 批次大小
- `tau`: 目标网络软更新参数
- `gamma`: 折扣因子
- `ent_coef`: 熵系数（"auto"表示自动调整）

### Wandb配置

- `enable_wandb`: 是否启用wandb
- `project`: wandb项目名称
- `entity`: wandb用户名或团队名
- `tags`: 实验标签
- `log_model`: 是否记录模型文件

## 可用的Atari游戏

常用的Atari游戏环境ID：
- `ALE/Breakout-v5` - 打砖块
- `ALE/Pong-v5` - 乒乓球
- `ALE/SpaceInvaders-v5` - 太空侵略者
- `ALE/Qbert-v5` - Q伯特
- `ALE/BeamRider-v5` - 光束骑士
- `ALE/Enduro-v5` - 耐力赛
- `ALE/Seaquest-v5` - 海底任务
- `ALE/Asteroids-v5` - 小行星

完整列表请查看`config.py`中的`AVAILABLE_ATARI_GAMES`。

## 训练过程

1. **环境初始化**: 创建Atari游戏环境
2. **模型配置**: 设置SAC算法参数
3. **训练循环**: 执行强化学习训练
4. **定期评估**: 评估模型性能
5. **模型保存**: 保存最佳模型和检查点
6. **结果测试**: 测试训练好的模型
7. **实验跟踪**: 记录训练指标到wandb

## SAC算法特点

- **连续动作空间**: 适用于连续控制问题
- **最大熵强化学习**: 平衡探索和利用
- **自动熵调整**: 动态调整探索程度
- **软更新**: 稳定的目标网络更新

## 监控训练

### TensorBoard

```bash
tensorboard --logdir logs/sac_atari/tensorboard
```

### Weights & Biases

启用wandb后，您可以在wandb网页界面查看：
- 训练和评估奖励曲线
- 超参数配置
- 模型文件
- 实验比较
- 实时训练状态

## 命令行参数

`train.py`支持以下命令行参数：

- `--env`: 指定Atari环境ID
- `--timesteps`: 指定训练步数
- `--config`: 显示当前配置
- `--test`: 仅测试环境
- `--no-wandb`: 禁用wandb日志记录

## 实验建议

1. **快速测试**: 使用100,000步进行快速验证
2. **标准训练**: 使用1,000,000步进行完整训练
3. **长时间训练**: 使用5,000,000步获得最佳性能
4. **游戏选择**: 建议从Breakout开始，它相对简单且训练效果明显
5. **实验跟踪**: 使用wandb记录不同超参数的实验结果

## 使用示例

### 示例1：快速测试Breakout
```bash
python train.py --env ALE/Breakout-v5 --timesteps 100000
```

### 示例2：标准训练Pong
```bash
python train.py --env ALE/Pong-v5 --timesteps 1000000
```

### 示例3：长时间训练SpaceInvaders
```bash
python train.py --env ALE/SpaceInvaders-v5 --timesteps 5000000
```

### 示例4：禁用wandb训练
```bash
python train.py --env ALE/Breakout-v5 --timesteps 1000000 --no-wandb
```

## Wandb使用指南

### 1. 初始设置

```bash
# 安装wandb
pip install wandb

# 登录wandb
wandb login

# 运行设置向导
python setup_wandb.py
```

### 2. 查看实验结果

训练开始后，您可以在以下位置查看结果：
- 终端输出：实时训练状态
- Wandb网页界面：详细的实验跟踪
- TensorBoard：本地可视化

### 3. 实验比较

使用wandb可以轻松比较不同实验：
- 不同超参数的效果
- 不同游戏的性能
- 不同算法的表现

### 4. 模型管理

Wandb自动保存：
- 最佳模型文件
- 检查点模型
- 最终模型

## 故障排除

### 常见问题

1. **Atari ROM错误**
   ```bash
   python -m AutoROM --accept-license
   ```

2. **依赖安装问题**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Wandb相关问题**
   ```bash
   # 检查wandb安装
   pip install wandb
   
   # 重新登录
   wandb login
   
   # 运行设置向导
   python setup_wandb.py
   ```

4. **CUDA相关问题**
   - 确保PyTorch版本与CUDA版本匹配
   - 可以设置`device="cpu"`使用CPU训练

5. **内存不足**
   - 减少`batch_size`
   - 减少`buffer_size`
   - 使用更少的并行环境

## 性能优化

1. **使用GPU**: 确保安装了CUDA版本的PyTorch
2. **调整网络架构**: 在`config.py`中修改网络层数
3. **调整超参数**: 根据具体游戏调整学习率和批次大小
4. **使用多进程**: 增加`num_envs`进行并行训练
5. **实验跟踪**: 使用wandb记录和分析实验结果

## 许可证

本项目遵循MIT许可证。