"""
PPO算法在Atari环境中的实验配置文件
"""
import torch

# 环境配置
ENVIRONMENT_CONFIG = {
    "env_id": "ALE/Breakout-v5",  # Atari游戏环境ID
    "num_envs": 1,  # 并行环境数量
    "seed": 42,  # 随机种子
}

# 训练配置
TRAINING_CONFIG = {
    "total_timesteps": 1000000,  # 总训练步数
    "eval_freq": 10000,  # 评估频率
    "save_freq": 50000,  # 保存频率
    "n_eval_episodes": 10,  # 评估时的episode数量
    "progress_bar": True,  # 是否显示进度条
}

# PPO算法超参数
PPO_CONFIG = {
    "learning_rate": 3e-4,  # 学习率
    "n_steps": 2048,  # 每次更新的步数
    "batch_size": 64,  # 批次大小
    "n_epochs": 10,  # 每次更新的epoch数
    "gamma": 0.99,  # 折扣因子
    "gae_lambda": 0.95,  # GAE lambda参数
    "clip_range": 0.2,  # PPO裁剪范围
    "clip_range_vf": None,  # 价值函数裁剪范围
    "ent_coef": 0.01,  # 熵系数
    "vf_coef": 0.5,  # 价值函数系数
    "max_grad_norm": 0.5,  # 最大梯度范数
    "use_sde": False,  # 是否使用状态依赖探索
    "sde_sample_freq": -1,  # SDE采样频率
}

# 网络架构配置
NETWORK_CONFIG = {
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 256],  # 策略网络架构
            "vf": [256, 256]   # 价值函数网络架构
        },
        "activation_fn": torch.nn.ReLU,  # 激活函数
        "ortho_init": True,  # 是否使用正交初始化
    }
}

# 日志配置
LOGGING_CONFIG = {
    "log_dir": "logs/ppo_atari",  # 日志目录
    "tensorboard_log": "logs/ppo_atari/tensorboard",  # TensorBoard日志目录
    "verbose": 1,  # 详细程度
}

# 设备配置
DEVICE_CONFIG = {
    "device": "auto",  # 设备选择（"auto", "cpu", "cuda"）
}

# Wandb配置
WANDB_CONFIG = {
    "enable_wandb": True,  # 是否启用wandb
    "project": "ppo-atari-experiments",  # wandb项目名称
    "entity": None,  # wandb用户名或团队名（None表示使用默认）
    "tags": ["ppo", "atari", "reinforcement-learning"],  # 实验标签
    "notes": "PPO算法在Atari环境中的强化学习实验",  # 实验说明
    "save_code": True,  # 是否保存代码
    "log_model": True,  # 是否记录模型
    "log_freq": 1000,  # wandb日志记录频率
    "eval_freq": 10000,  # wandb评估频率
    "save_freq": 50000,  # wandb模型保存频率
}

# 可用的Atari游戏列表
AVAILABLE_ATARI_GAMES = [
    "ALE/Breakout-v5",
    "ALE/Pong-v5",
    "ALE/SpaceInvaders-v5", 
    "ALE/Qbert-v5",
    "ALE/BeamRider-v5",
    "ALE/Enduro-v5",
    "ALE/Seaquest-v5",
    "ALE/Asteroids-v5",
    "ALE/Atlantis-v5",
    "ALE/BattleZone-v5",
    "ALE/Bowling-v5",
    "ALE/Boxing-v5",
    "ALE/Centipede-v5",
    "ALE/ChopperCommand-v5",
    "ALE/CrazyClimber-v5",
    "ALE/DemonAttack-v5",
    "ALE/DoubleDunk-v5",
    "ALE/ElevatorAction-v5",
    "ALE/FishingDerby-v5",
    "ALE/Freeway-v5",
    "ALE/Frostbite-v5",
    "ALE/Gopher-v5",
    "ALE/Gravitar-v5",
    "ALE/Hero-v5",
    "ALE/IceHockey-v5",
    "ALE/Jamesbond-v5",
    "ALE/Kangaroo-v5",
    "ALE/Krull-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MontezumaRevenge-v5",
    "ALE/MsPacman-v5",
    "ALE/NameThisGame-v5",
    "ALE/Phoenix-v5",
    "ALE/Pitfall-v5",
    "ALE/Pooyan-v5",
    "ALE/PrivateEye-v5",
    "ALE/Riverraid-v5",
    "ALE/RoadRunner-v5",
    "ALE/Robotank-v5",
    "ALE/Skiing-v5",
    "ALE/Solaris-v5",
    "ALE/StarGunner-v5",
    "ALE/Tennis-v5",
    "ALE/TimePilot-v5",
    "ALE/Tutankham-v5",
    "ALE/UpNDown-v5",
    "ALE/Venture-v5",
    "ALE/VideoPinball-v5",
    "ALE/WizardOfWor-v5",
    "ALE/YarsRevenge-v5",
    "ALE/Zaxxon-v5"
]

def get_config():
    """
    获取完整的配置字典
    
    Returns:
        dict: 包含所有配置的字典
    """
    return {
        "environment": ENVIRONMENT_CONFIG,
        "training": TRAINING_CONFIG,
        "ppo": PPO_CONFIG,
        "network": NETWORK_CONFIG,
        "logging": LOGGING_CONFIG,
        "device": DEVICE_CONFIG,
        "wandb": WANDB_CONFIG,
        "available_games": AVAILABLE_ATARI_GAMES
    }

def print_config():
    """打印当前配置"""
    config = get_config()
    print("当前实验配置:")
    print("=" * 50)
    
    for section, params in config.items():
        if section != "available_games":
            print(f"\n{section.upper()}:")
            for key, value in params.items():
                print(f"  {key}: {value}")
    
    print(f"\n可用游戏数量: {len(AVAILABLE_ATARI_GAMES)}")

if __name__ == "__main__":
    print_config() 