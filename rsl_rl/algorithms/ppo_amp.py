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
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules.amp import AMPDiscriminator, LossType, resolve_amp_config
from rsl_rl.storage import CircularBuffer, RolloutStorage
from rsl_rl.utils import resolve_optimizer


class PPOAMP(PPO):
    """继承标准 PPO，使用独立优化器训练 AMP 动作判别器。"""

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
        self.disc_optimizer = resolve_optimizer("adam")(
            [
                {
                    "params": self.amp_discriminator.disc_trunk.parameters(),
                    "weight_decay": amp_cfg["disc_trunk_weight_decay"],
                },
                {
                    "params": self.amp_discriminator.disc_linear.parameters(),
                    "weight_decay": amp_cfg["disc_linear_weight_decay"],
                },
            ],
            lr=amp_cfg["disc_learning_rate"],
        )
        self.disc_max_grad_norm = amp_cfg.get("disc_max_grad_norm", 1.0)
        self.disc_obs_buffer = CircularBuffer(amp_cfg["disc_obs_buffer_size"], storage.num_envs, self.device)
        self.disc_demo_obs_buffer = CircularBuffer(amp_cfg["disc_obs_buffer_size"], storage.num_envs, self.device)
        self._reward_statistics = []

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOAMP:
        """复制配置，校验策略与专家特征的形状，再通过标准 PPO 工厂创建算法。"""
        cfg = copy.deepcopy(cfg)
        cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"]["amp_cfg"]["obs_groups"] = copy.deepcopy(cfg["obs_groups"])
        return PPO.construct_algorithm(obs, env, cfg, device)

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """读取策略和专家历史，用终止前观测替换自动重置状态，组合奖励并保存样本。"""
        if dones.bool().any() and "final_obs" not in extras:
            raise ValueError("AMP 自动重置需要 extras['final_obs'] 提供终止前的完整观测。")
        disc_obs = self.amp_discriminator.get_disc_obs(obs)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs)
        if "final_obs" in extras:
            final_obs = extras["final_obs"]
            mask = dones.to(disc_obs.device).bool().reshape(-1, 1, 1)
            final_disc_obs = self.amp_discriminator.get_disc_obs(final_obs).to(disc_obs.device)
            final_demo_obs = self.amp_discriminator.get_disc_demo_obs(final_obs).to(disc_demo_obs.device)
            disc_obs = torch.where(mask, final_disc_obs, disc_obs)
            disc_demo_obs = torch.where(mask, final_demo_obs, disc_demo_obs)
        (self.style_rewards, self.disc_score) = self.amp_discriminator.predict_style_reward(
            disc_obs, dt=self.amp_cfg["step_dt"]
        )
        self.rewards_lerp = self.amp_discriminator.lerp_reward(rewards, self.style_rewards)
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)
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
        """复用标准 PPO 更新，再独立更新判别器，返回训练指标。"""
        metrics = super().update()
        metrics.update(self._update_discriminator())
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

    def _update_discriminator(self) -> dict[str, float]:
        """从策略和专家缓冲区采样，优化分类损失与零中心梯度惩罚。"""
        generators = [
            buffer.mini_batch_generator(
                self.storage.num_transitions_per_env,
                self.num_mini_batches,
                self.num_learning_epochs,
            )
            for buffer in (self.disc_obs_buffer, self.disc_demo_obs_buffer)
        ]
        totals = torch.zeros(4, device=self.device)
        updates = 0
        for policy_obs, demo_obs in zip(*generators):
            with torch.no_grad():
                policy_obs_norm = self.amp_discriminator.normalize_disc_obs(policy_obs)
                demo_obs_norm = self.amp_discriminator.normalize_disc_obs(demo_obs)
            batch_size = policy_obs.shape[0]
            policy_score = self.amp_discriminator(policy_obs_norm.reshape(batch_size, -1))
            demo_score = self.amp_discriminator(demo_obs_norm.reshape(batch_size, -1))
            if self.loss_type == LossType.GAN:
                loss = 0.5 * (
                    nn.functional.binary_cross_entropy_with_logits(policy_score, torch.zeros_like(policy_score))
                    + nn.functional.binary_cross_entropy_with_logits(demo_score, torch.ones_like(demo_score))
                )
            elif self.loss_type == LossType.LSGAN:
                loss = 0.5 * (
                    nn.functional.mse_loss(policy_score, -torch.ones_like(policy_score))
                    + nn.functional.mse_loss(demo_score, torch.ones_like(demo_score))
                )
            else:
                loss = policy_score.mean() - demo_score.mean()
            penalty = self.amp_discriminator.compute_grad_penalty(
                demo_obs_norm.reshape(batch_size, -1),
                self.amp_cfg["grad_penalty_scale"],
            )
            self.disc_optimizer.zero_grad()
            (loss + penalty).backward()
            if self.is_multi_gpu:
                self._reduce_discriminator_gradients()
            nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), self.disc_max_grad_norm)
            self.disc_optimizer.step()
            self.amp_discriminator.update_normalization(policy_obs)
            totals += torch.stack((
                loss.detach(),
                penalty.detach(),
                policy_score.detach().mean(),
                demo_score.detach().mean(),
            ))
            updates += 1
        names = (
            "amp/adversarial_loss",
            "amp/expert_grad_penalty",
            "amp/policy_score",
            "amp/expert_score",
        )
        return dict(zip(names, (totals / max(updates, 1)).tolist()))

    def train_mode(self) -> None:
        """将 PPO 和判别器切换到训练模式。"""
        super().train_mode()
        self.amp_discriminator.train()

    def eval_mode(self) -> None:
        """将 PPO 和判别器切换到评估模式，停止归一化统计更新。"""
        super().eval_mode()
        self.amp_discriminator.eval()

    def save(self) -> dict:
        """保存标准 PPO 状态及 AMP 判别器和优化器状态。"""
        return dict(
            super().save(),
            amp_discriminator_state_dict=self.amp_discriminator.state_dict(),
            amp_discriminator_optimizer_state_dict=self.disc_optimizer.state_dict(),
        )

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """按加载选项恢复 PPO 和 AMP 状态，返回是否恢复训练迭代数。"""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("amp", False):
            self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            if load_cfg is None or load_cfg.get("optimizer", False):
                self.disc_optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])
        return load_iteration

    def broadcast_parameters(self) -> None:
        """在分布式训练中同步 PPO 和判别器参数。"""
        super().broadcast_parameters()
        state = [self.amp_discriminator.state_dict()]
        torch.distributed.broadcast_object_list(state, src=0)
        self.amp_discriminator.load_state_dict(state[0])

    def _reduce_discriminator_gradients(self) -> None:
        """对判别器梯度进行跨进程平均。"""
        for parameter in self.amp_discriminator.parameters():
            if parameter.grad is not None:
                torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
                parameter.grad.div_(self.gpu_world_size)
