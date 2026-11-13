# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 允许中文文档使用中文标点。
# ruff: file-ignore[missing-trailing-period, missing-terminal-punctuation, ambiguous-unicode-character-docstring]

# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""参考 RoboParty 的 RSL-RL 5.x AMP 扩展；环境提供策略和专家历史特征。"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules.amp import AMPDiscriminator, LossType, resolve_amp_config
from rsl_rl.storage import CircularBuffer, RolloutStorage
from rsl_rl.utils import reduce_gradients_in_buckets


class PPOAMP(PPO):
    """继承标准 PPO，判别器与策略放进同一个 Adam。"""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        amp_cfg: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """初始化配置、参数和所需状态；具体维度由构造参数指定。"""
        super().__init__(actor, critic, storage, **kwargs)
        if amp_cfg is None:
            raise ValueError("PPOAMP 必须提供 AMP 配置 amp_cfg。")
        self.amp_cfg = amp_cfg
        if amp_cfg["disc_obs_buffer_size"] < storage.num_transitions_per_env:
            raise ValueError("判别器缓冲区长度 disc_obs_buffer_size 不能小于每轮采样步数 num_steps_per_env。")
        try:
            self.loss_type = LossType[amp_cfg["loss_type"]]
        except KeyError:
            raise ValueError(f"不支持的 AMP 损失类型: {amp_cfg['loss_type']}") from None
        self.amp_discriminator = AMPDiscriminator(
            disc_obs_dim=amp_cfg["disc_obs_dim"],
            disc_obs_steps=amp_cfg["disc_obs_steps"],
            obs_groups=amp_cfg["obs_groups"],
            loss_type=self.loss_type,
            device=self.device,
            **amp_cfg.get("amp_discriminator", {}),
        ).to(self.device)
        # 与旧 AMPPPO 相同：判别器和策略共用 Adam，自适应学习率一起变。
        # 默认主干权重衰减 1e-3，输出层 0.1。判别器梯度不单独裁剪。
        self.optimizer.add_param_group({
            "params": list(self.amp_discriminator.disc_trunk.parameters()),
            "weight_decay": amp_cfg.get("disc_trunk_weight_decay", 10e-4),
            "lr": self.learning_rate,
        })
        self.optimizer.add_param_group({
            "params": list(self.amp_discriminator.disc_linear.parameters()),
            "weight_decay": amp_cfg.get("disc_linear_weight_decay", 10e-2),
            "lr": self.learning_rate,
        })
        self.policy_obs_buffer = CircularBuffer(amp_cfg["disc_obs_buffer_size"], storage.num_envs, self.device)
        self.expert_obs_buffer = CircularBuffer(amp_cfg["disc_obs_buffer_size"], storage.num_envs, self.device)
        self._reward_statistics = []

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOAMP:
        """复制配置，校验策略与专家特征的形状，再通过标准 PPO 工厂创建算法。"""
        cfg = copy.deepcopy(cfg)
        cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"]["amp_cfg"]["obs_groups"] = copy.deepcopy(cfg["obs_groups"])
        return PPO.construct_algorithm(obs, env, cfg, device)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """记下步进前的策略 AMP 特征，供回合结束时重复该帧。"""
        self._policy_disc_obs = self.amp_discriminator.get_disc_obs(obs).detach()
        return super().act(obs)

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """读取策略和专家历史。结束的环境用步进前的特征顶替重置后的观测。"""
        policy_obs = self.amp_discriminator.get_disc_obs(obs)
        expert_obs = self.amp_discriminator.get_disc_demo_obs(obs)
        if dones.bool().any():
            mask = dones.to(policy_obs.device).bool().reshape(-1, 1, 1)
            policy_obs = torch.where(mask, self._policy_disc_obs.to(policy_obs.device), policy_obs)
        (self.style_rewards, self.disc_score) = self.amp_discriminator.predict_style_reward(
            policy_obs, dt=self.amp_cfg["step_dt"]
        )
        stand_mask = extras.get("amp_stand_mask")
        if stand_mask is not None:
            stand = stand_mask.to(device=self.style_rewards.device, dtype=torch.bool).reshape(-1)
            self.style_rewards = torch.where(stand, torch.zeros_like(self.style_rewards), self.style_rewards)
            move = ~stand
            extras.setdefault("log", {})["AMP/move_style_reward"] = (
                self.style_rewards * move
            ).sum().detach() / move.sum().clamp_min(1)
        self.rewards_lerp = self.amp_discriminator.lerp_reward(rewards, self.style_rewards)
        self.amp_discriminator.update_normalization(policy_obs.detach())
        self.amp_discriminator.update_normalization(expert_obs.detach())
        self.policy_obs_buffer.append(policy_obs)
        self.expert_obs_buffer.append(expert_obs)
        self._reward_statistics.append(
            torch.stack((rewards.mean(), self.style_rewards.mean(), self.rewards_lerp.mean())).detach()
        )
        log_key = "episode" if "episode" in extras else "log"
        extras[log_key] = dict(extras.get(log_key, {}))
        extras[log_key].update({
            "AMP/task_reward": rewards.mean().detach(),
            "AMP/style_reward": self.style_rewards.mean().detach(),
            "AMP/mixed_reward": self.rewards_lerp.mean().detach(),
        })
        super().process_env_step(obs, self.rewards_lerp, dones, extras)

    def update(self) -> dict[str, float]:
        """与 PPO 同一次反向传播更新判别器，返回训练指标。"""
        generators = [
            buffer.mini_batch_generator(
                self.storage.num_transitions_per_env,
                self.num_mini_batches,
                self.num_learning_epochs,
            )
            for buffer in (self.policy_obs_buffer, self.expert_obs_buffer)
        ]
        self._disc_batch_iter = zip(*generators, strict=True)
        self._disc_totals = torch.zeros(4, device=self.device)
        self._disc_updates = 0
        metrics = super().update()
        names = (
            "amp/adversarial_loss",
            "amp/expert_grad_penalty",
            "amp/policy_score",
            "amp/expert_score",
        )
        metrics.update(dict(zip(names, (self._disc_totals / max(self._disc_updates, 1)).tolist())))
        if self._reward_statistics:
            means = torch.stack(self._reward_statistics).mean(0).tolist()
            metrics.update(
                dict(
                    zip(
                        ("amp/task_reward", "amp/style_reward", "amp/mixed_reward"),
                        means,
                    )
                )
            )
            self._reward_statistics.clear()
        return metrics

    def auxiliary_loss(self) -> torch.Tensor:
        """计算与旧 AMPPPO 相同的判别器损失，并加入同一次反向传播。"""
        policy_obs, expert_obs = next(self._disc_batch_iter)
        with torch.no_grad():
            policy_obs_norm = self.amp_discriminator.normalize_disc_obs(policy_obs)
            expert_obs_norm = self.amp_discriminator.normalize_disc_obs(expert_obs)
        batch_size = policy_obs.shape[0]
        policy_score = self.amp_discriminator(policy_obs_norm.reshape(batch_size, -1))
        expert_score = self.amp_discriminator(expert_obs_norm.reshape(batch_size, -1))
        if self.loss_type == LossType.GAN:
            loss = 0.5 * (
                nn.functional.binary_cross_entropy_with_logits(policy_score, torch.zeros_like(policy_score))
                + nn.functional.binary_cross_entropy_with_logits(expert_score, torch.ones_like(expert_score))
            )
        elif self.loss_type == LossType.LSGAN:
            loss = 0.5 * (
                nn.functional.mse_loss(policy_score, -torch.ones_like(policy_score))
                + nn.functional.mse_loss(expert_score, torch.ones_like(expert_score))
            )
        else:
            loss = policy_score.mean() - expert_score.mean()
        # 梯度惩罚与分类使用相同的归一化专家特征。
        penalty = self.amp_discriminator.compute_grad_penalty(
            expert_obs_norm.reshape(batch_size, -1),
            self.amp_cfg["grad_penalty_scale"],
        )
        self._disc_totals += torch.stack((
            loss.detach(),
            penalty.detach(),
            policy_score.detach().mean(),
            expert_score.detach().mean(),
        ))
        self._disc_updates += 1
        return loss + penalty

    def train_mode(self) -> None:
        """将 PPO 和判别器切换到训练模式。"""
        super().train_mode()
        self.amp_discriminator.train()

    def eval_mode(self) -> None:
        """将 PPO 和判别器切换到评估模式，停止归一化统计更新。"""
        super().eval_mode()
        self.amp_discriminator.eval()

    def save(self) -> dict:
        """保存标准 PPO 状态和 AMP 判别器。判别器动量在共享优化器里。"""
        return dict(
            super().save(),
            amp_discriminator_state_dict=self.amp_discriminator.state_dict(),
        )

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """按加载选项恢复 PPO 和 AMP 判别器，返回是否恢复训练迭代数。"""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("amp", False):
            self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
        return load_iteration

    def broadcast_parameters(self) -> None:
        """在分布式训练中同步 PPO 和判别器参数。"""
        super().broadcast_parameters()
        state = [self.amp_discriminator.state_dict()]
        torch.distributed.broadcast_object_list(state, src=0)
        self.amp_discriminator.load_state_dict(state[0])

    def reduce_parameters(self) -> None:
        """跨进程平均策略、价值、判别器梯度。判别器梯度不裁剪。"""
        params = chain(
            self.actor.parameters(),
            self.critic.parameters(),
            self.amp_discriminator.parameters(),
        )
        if self.rnd:
            params = chain(params, self.rnd.parameters())
        reduce_gradients_in_buckets(params, self.gpu_world_size, self.grad_reduce_bucket_mb)
