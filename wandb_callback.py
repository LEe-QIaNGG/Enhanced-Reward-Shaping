"""
Weights & Biases (wandb) 回调类，用于与Stable Baselines3集成
"""

import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from typing import Dict, Any, Optional
import os
from wandb_config import (
    log_training_metrics, 
    log_episode_metrics, 
    log_model_artifact,
    create_run_name,
    create_group_name
)

class WandbCallback(BaseCallback):
    """
    自定义wandb回调类，用于记录训练过程中的各种指标
    """
    
    def __init__(
        self,
        env_id: str,
        log_freq: int = 1000,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        verbose: int = 1
    ):
        """
        初始化wandb回调
        
        Args:
            env_id: 环境ID
            log_freq: 日志记录频率
            eval_freq: 评估频率
            save_freq: 保存频率
            verbose: 详细程度
        """
        super(WandbCallback, self).__init__(verbose)
        self.env_id = env_id
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.episode_count = 0
        self.total_episodes = 0
        self.best_mean_reward = -np.inf
        
    def _on_training_start(self) -> None:
        """
        训练开始时的回调
        """
        # 记录环境信息
        env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
        log_environment_info(
            self.env_id,
            env.observation_space,
            env.action_space
        )
        
        if self.verbose > 0:
            print(f"WandbCallback: 开始记录训练数据到wandb")
    
    def _on_step(self) -> bool:
        """
        每步训练时的回调
        
        Returns:
            bool: 是否继续训练
        """
        # 检查是否需要记录日志
        if self.n_calls % self.log_freq == 0:
            self._log_training_metrics()
        
        # 检查是否需要评估
        if self.n_calls % self.eval_freq == 0:
            self._log_eval_metrics()
        
        # 检查是否需要保存模型
        if self.n_calls % self.save_freq == 0:
            self._save_model()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        rollout结束时的回调
        """
        # 记录episode信息
        if hasattr(self, 'ep_info_buffer') and len(self.ep_info_buffer) > 0:
            for ep_info in self.ep_info_buffer:
                if ep_info is not None:
                    self.episode_count += 1
                    self.total_episodes += 1
                    
                    # 记录episode指标
                    log_episode_metrics(
                        episode=self.episode_count,
                        episode_reward=ep_info.get('r', 0),
                        episode_length=ep_info.get('l', 0),
                        episode_type="train"
                    )
    
    def _log_training_metrics(self):
        """
        记录训练指标
        """
        # 获取最近的训练奖励
        if hasattr(self, 'ep_info_buffer') and len(self.ep_info_buffer) > 0:
            recent_rewards = [ep_info.get('r', 0) for ep_info in self.ep_info_buffer if ep_info is not None]
            if recent_rewards:
                mean_reward = np.mean(recent_rewards)
                
                # 记录到wandb
                wandb.log({
                    "train/mean_reward": mean_reward,
                    "train/total_episodes": self.total_episodes,
                    "train/training_steps": self.n_calls
                }, step=self.n_calls)
                
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: 平均训练奖励 = {mean_reward:.2f}")
    
    def _log_eval_metrics(self):
        """
        记录评估指标
        """
        # 这里可以添加评估逻辑
        # 由于评估通常在EvalCallback中处理，这里主要记录训练相关的指标
        pass
    
    def _save_model(self):
        """
        保存模型到wandb
        """
        if hasattr(self.model, 'save'):
            model_path = f"models/sac_model_step_{self.n_calls}"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            try:
                self.model.save(model_path)
                
                # 记录模型文件
                log_model_artifact(
                    model_path=model_path + ".zip",
                    model_name=f"sac_model_step_{self.n_calls}",
                    description=f"SAC模型，训练步数: {self.n_calls}"
                )
                
                if self.verbose > 0:
                    print(f"模型已保存到wandb: {model_path}")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"保存模型时出错: {e}")

class WandbEvalCallback(BaseCallback):
    """
    用于评估的wandb回调类
    """
    
    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 10,
        eval_freq: int = 10000,
        verbose: int = 1
    ):
        """
        初始化评估回调
        
        Args:
            eval_env: 评估环境
            n_eval_episodes: 评估episode数量
            eval_freq: 评估频率
            verbose: 详细程度
        """
        super(WandbEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        """
        每步训练时的回调
        """
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_model()
        return True
    
    def _evaluate_model(self):
        """
        评估模型
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        # 记录评估指标
        wandb.log({
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
            "eval/mean_length": mean_length,
            "eval/episodes": self.n_eval_episodes
        }, step=self.n_calls)
        
        # 检查是否是最佳模型
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            wandb.log({
                "eval/best_mean_reward": self.best_mean_reward
            }, step=self.n_calls)
        
        if self.verbose > 0:
            print(f"评估结果 (步数 {self.n_calls}): 平均奖励 = {mean_reward:.2f} ± {std_reward:.2f}")

def setup_wandb_logging(
    config: Dict[str, Any],
    env_id: str,
    enable_wandb: bool = True
) -> Optional[WandbCallback]:
    """
    设置wandb日志记录
    
    Args:
        config: 实验配置
        env_id: 环境ID
        enable_wandb: 是否启用wandb
    
    Returns:
        Optional[WandbCallback]: wandb回调对象
    """
    if not enable_wandb:
        return None
    
    try:
        # 初始化wandb
        run_name = create_run_name(env_id)
        group_name = create_group_name(env_id)
        
        from wandb_config import init_wandb
        init_wandb(config, run_name=run_name, group=group_name)
        
        # 创建wandb回调
        wandb_callback = WandbCallback(
            env_id=env_id,
            log_freq=1000,
            eval_freq=10000,
            save_freq=50000,
            verbose=1
        )
        
        print(f"Wandb日志记录已启用，项目: {run_name}")
        return wandb_callback
        
    except Exception as e:
        print(f"启用wandb时出错: {e}")
        print("继续训练，但不记录到wandb")
        return None 