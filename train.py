import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
# os.environ["TORCH_USE_CUDA_DSA"]="TRUE"
from typing import Callable
import torch
import argparse
from config import get_config, print_config
from wandb_callback import setup_wandb_logging, WandbCallback
from wandb_config import finish_wandb, log_model_artifact

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    创建环境的工厂函数
    """
    def _init() -> gym.Env:
        env = gym.make(env_id, render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def linear_schedule(initial_value: float):
    """
    线性学习率调度器
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def validate_config(config):
    """
    验证配置参数的有效性
    """
    env_id = config["environment"]["env_id"]
    
    # 检查环境是否存在
    try:
        test_env = gym.make(env_id, render_mode=None)
        test_env.close()
        print(f"✓ 环境 {env_id} 验证成功")
    except Exception as e:
        print(f"✗ 环境 {env_id} 验证失败: {str(e)}")
        print("请确保已安装Atari ROM文件: python -m AutoROM --accept-license")
        return False
    
    # 检查其他参数
    if config["training"]["total_timesteps"] <= 0:
        print("✗ total_timesteps 必须大于0")
        return False
    
    if config["training"]["eval_freq"] <= 0:
        print("✗ eval_freq 必须大于0")
        return False
    
    print("✓ 所有配置参数验证通过")
    return True

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PPO算法在Atari环境中的训练")
    parser.add_argument("--env", type=str, help="Atari环境ID (例如: ALE/Breakout-v5)")
    parser.add_argument("--timesteps", type=int, help="总训练步数")
    parser.add_argument("--config", action="store_true", help="显示当前配置")
    parser.add_argument("--test", action="store_true", help="仅测试环境")
    parser.add_argument("--no-wandb", action="store_true", help="禁用wandb日志记录")
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config()
    
    # 如果指定了显示配置，则打印并退出
    if args.config:
        print_config()
        return
    
    # 更新配置（如果提供了命令行参数）
    if args.env:
        config["environment"]["env_id"] = args.env
    if args.timesteps:
        config["training"]["total_timesteps"] = args.timesteps
    if args.no_wandb:
        config["wandb"]["enable_wandb"] = False
    
    # 验证配置
    if not validate_config(config):
        return
    
    # 如果只是测试环境，则退出
    if args.test:
        print("环境测试完成")
        return
    
    # 提取配置参数
    env_config = config["environment"]
    training_config = config["training"]
    ppo_config = config["ppo"]
    network_config = config["network"]
    logging_config = config["logging"]
    device_config = config["device"]
    wandb_config = config["wandb"]
    
    # 创建日志目录
    os.makedirs(logging_config["log_dir"], exist_ok=True)
    
    print(f"开始训练PPO模型在{env_config['env_id']}环境中...")
    print(f"总训练步数: {training_config['total_timesteps']}")
    print(f"评估频率: {training_config['eval_freq']}")
    print(f"保存频率: {training_config['save_freq']}")
    print(f"Wandb日志记录: {'启用' if wandb_config['enable_wandb'] else '禁用'}")
    
    # 设置wandb日志记录
    wandb_callback = None
    if wandb_config['enable_wandb']:
        wandb_callback = setup_wandb_logging(
            config=config,
            env_id=env_config['env_id'],
            enable_wandb=wandb_config['enable_wandb']
        )
    
    # 创建向量化环境
    env = DummyVecEnv([make_env(env_config['env_id'], i, env_config['seed']) 
                      for i in range(env_config['num_envs'])])
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env(env_config['env_id'], 0, env_config['seed'])])
    
    # 创建回调函数列表
    callbacks = []
    
    # 添加评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{logging_config['log_dir']}/best_model",
        log_path=f"{logging_config['log_dir']}/eval_logs",
        eval_freq=training_config['eval_freq'] // env_config['num_envs'],
        n_eval_episodes=training_config['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # 添加检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config['save_freq'] // env_config['num_envs'],
        save_path=f"{logging_config['log_dir']}/checkpoints",
        name_prefix="ppo_model"
    )
    callbacks.append(checkpoint_callback)
    
    # 添加wandb回调
    if wandb_callback is not None:
        callbacks.append(wandb_callback)
    
    # 准备PPO参数
    ppo_params = {
        "policy": "CnnPolicy",  # 使用CNN策略网络处理图像输入
        "env": env,
        "verbose": logging_config['verbose'],
        "tensorboard_log": logging_config['tensorboard_log'],
        "learning_rate": linear_schedule(ppo_config['learning_rate']),
        "n_steps": ppo_config['n_steps'],
        "batch_size": ppo_config['batch_size'],
        "n_epochs": ppo_config['n_epochs'],
        "gamma": ppo_config['gamma'],
        "gae_lambda": ppo_config['gae_lambda'],
        "clip_range": ppo_config['clip_range'],
        "clip_range_vf": ppo_config['clip_range_vf'],
        "ent_coef": ppo_config['ent_coef'],
        "vf_coef": ppo_config['vf_coef'],
        "max_grad_norm": ppo_config['max_grad_norm'],
        "use_sde": ppo_config['use_sde'],
        "sde_sample_freq": ppo_config['sde_sample_freq'],
        "policy_kwargs": network_config['policy_kwargs'],
        "device": device_config['device']
    }
    
    # 初始化PPO模型
    model = PPO(**ppo_params)
    
    try:
        # 开始训练
        model.learn(
            total_timesteps=training_config['total_timesteps'],
            callback=callbacks,
            progress_bar=training_config['progress_bar']
        )
        
        # 保存最终模型
        final_model_path = f"{logging_config['log_dir']}/final_model"
        model.save(final_model_path)
        print(f"训练完成！模型已保存到 {final_model_path}")
        
        # 如果启用了wandb，记录最终模型
        if wandb_config['enable_wandb'] and wandb_config['log_model']:
            try:
                log_model_artifact(
                    model_path=final_model_path + ".zip",
                    model_name="final_ppo_model",
                    description=f"最终PPO模型，训练步数: {training_config['total_timesteps']}"
                )
                print("最终模型已记录到wandb")
            except Exception as e:
                print(f"记录最终模型到wandb时出错: {e}")
        
        # 测试训练好的模型
        print("\n开始测试训练好的模型...")
        test_model(model, env_config['env_id'])
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 保存中断时的模型
        interrupted_model_path = f"{logging_config['log_dir']}/interrupted_model"
        model.save(interrupted_model_path)
        print(f"中断时的模型已保存到 {interrupted_model_path}")
        
        # 如果启用了wandb，记录中断的模型
        if wandb_config['enable_wandb'] and wandb_config['log_model']:
            try:
                log_model_artifact(
                    model_path=interrupted_model_path + ".zip",
                    model_name="interrupted_ppo_model",
                    description=f"中断的PPO模型，训练步数: {model.num_timesteps}"
                )
            except Exception as e:
                print(f"记录中断模型到wandb时出错: {e}")
        
    except Exception as e:
        print(f"\n训练过程中发生错误: {str(e)}")
        raise
    
    finally:
        # 清理资源
        env.close()
        eval_env.close()
        
        # 结束wandb运行
        if wandb_config['enable_wandb']:
            try:
                finish_wandb()
                print("Wandb运行已结束")
            except Exception as e:
                print(f"结束wandb运行时出错: {e}")

def test_model(model, env_id, num_episodes=5):
    """
    测试训练好的模型
    
    Args:
        model: 训练好的PPO模型
        env_id: 环境ID
        num_episodes: 测试的episode数量
    """
    try:
        test_env = gym.make(env_id, render_mode="human")
        
        total_reward = 0
        episode_count = 0
        
        for episode in range(num_episodes):
            obs, _ = test_env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                episode_reward += reward
                
            total_reward += episode_reward
            episode_count += 1
            print(f"Episode {episode_count}: 奖励 = {episode_reward}")
        
        test_env.close()
        print(f"平均奖励: {total_reward / episode_count}")
        
    except Exception as e:
        print(f"测试模型时发生错误: {str(e)}")

if __name__ == "__main__":
    main()
