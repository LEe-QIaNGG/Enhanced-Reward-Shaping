"""
Weights & Biases (wandb) 配置文件
"""

import wandb
from typing import Dict, Any, Optional

# wandb项目配置
WANDB_CONFIG = {
    "project": "ppo-atari-experiments",  # wandb项目名称
    "entity": None,  # wandb用户名或团队名（None表示使用默认）
    "tags": ["ppo", "atari", "reinforcement-learning"],  # 实验标签
    "notes": "PPO算法在Atari环境中的强化学习实验",  # 实验说明
    "save_code": True,  # 是否保存代码
    "log_model": True,  # 是否记录模型
}

def init_wandb(
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    group: Optional[str] = None
) -> wandb.run:
    """
    初始化wandb运行
    
    Args:
        config: 实验配置字典
        run_name: 运行名称
        group: 运行组名
    
    Returns:
        wandb.run: wandb运行对象
    """
    # 准备wandb配置
    wandb_config = {
        "environment": config["environment"],
        "training": config["training"],
        "ppo": config["ppo"],
        "network": config["network"],
        "device": config["device"]
    }
    
    # 初始化wandb
    run = wandb.init(
        project=WANDB_CONFIG["project"],
        entity=WANDB_CONFIG["entity"],
        config=wandb_config,
        tags=WANDB_CONFIG["tags"],
        notes=WANDB_CONFIG["notes"],
        save_code=WANDB_CONFIG["save_code"],
        name=run_name,
        group=group
    )
    
    return run

def log_training_metrics(
    step: int,
    train_reward: float,
    eval_reward: float,
    loss: Optional[float] = None,
    entropy: Optional[float] = None,
    learning_rate: Optional[float] = None
):
    """
    记录训练指标
    
    Args:
        step: 训练步数
        train_reward: 训练奖励
        eval_reward: 评估奖励
        loss: 损失值
        entropy: 熵值
        learning_rate: 学习率
    """
    log_dict = {
        "train/reward": train_reward,
        "eval/reward": eval_reward,
    }
    
    if loss is not None:
        log_dict["train/loss"] = loss
    
    if entropy is not None:
        log_dict["train/entropy"] = entropy
    
    if learning_rate is not None:
        log_dict["train/learning_rate"] = learning_rate
    
    wandb.log(log_dict, step=step)

def log_episode_metrics(
    episode: int,
    episode_reward: float,
    episode_length: int,
    episode_type: str = "train"
):
    """
    记录episode指标
    
    Args:
        episode: episode编号
        episode_reward: episode奖励
        episode_length: episode长度
        episode_type: episode类型 ("train" 或 "eval")
    """
    wandb.log({
        f"{episode_type}/episode_reward": episode_reward,
        f"{episode_type}/episode_length": episode_length,
        f"{episode_type}/episode": episode
    })

def log_model_artifact(model_path: str, model_name: str, description: str = ""):
    """
    记录模型文件
    
    Args:
        model_path: 模型文件路径
        model_name: 模型名称
        description: 模型描述
    """
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=description
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

def log_hyperparameters(hyperparams: Dict[str, Any]):
    """
    记录超参数
    
    Args:
        hyperparams: 超参数字典
    """
    wandb.config.update(hyperparams)

def log_environment_info(env_id: str, observation_space: Any, action_space: Any):
    """
    记录环境信息
    
    Args:
        env_id: 环境ID
        observation_space: 观察空间
        action_space: 动作空间
    """
    wandb.config.update({
        "env_id": env_id,
        "observation_space": str(observation_space),
        "action_space": str(action_space),
        "action_space_type": type(action_space).__name__
    })

def finish_wandb():
    """
    结束wandb运行
    """
    wandb.finish()

def create_run_name(env_id: str, algorithm: str = "SAC") -> str:
    """
    创建运行名称
    
    Args:
        env_id: 环境ID
        algorithm: 算法名称
    
    Returns:
        str: 运行名称
    """
    game_name = env_id.split('/')[-1].replace('-v5', '')
    return f"{algorithm}_{game_name}"

def create_group_name(env_id: str) -> str:
    """
    创建组名称
    
    Args:
        env_id: 环境ID
    
    Returns:
        str: 组名称
    """
    game_name = env_id.split('/')[-1].replace('-v5', '')
    return f"game_{game_name}" 