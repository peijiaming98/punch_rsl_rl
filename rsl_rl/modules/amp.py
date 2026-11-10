# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: file-ignore[missing-trailing-period, missing-terminal-punctuation, ambiguous-unicode-character-docstring]
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP 判别器、归一化、奖励计算及形状校验。"""

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Sequence
from enum import Enum
from tensordict import TensorDict
from torch import autograd

from rsl_rl.env import VecEnv
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class LossType(Enum):
    """支持的判别器损失：交叉熵、最小二乘和 Wasserstein。"""

    GAN = 0
    LSGAN = 1
    WGAN = 2


class AMPDiscriminator(nn.Module):
    """将多帧动作特征映射为专家风格分数，不绑定机器人关节或文件格式。"""

    def __init__(
        self,
        disc_obs_dim: int,
        disc_obs_steps: int,
        obs_groups: dict,
        loss_type: LossType = LossType.LSGAN,
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "relu",
        style_reward_scale: float = 1.0,
        task_style_lerp: float = 0.0,
        device: str = "cpu",
    ) -> None:
        """初始化配置、参数和所需状态；具体维度由构造参数指定。"""
        super().__init__()
        self.input_dim = disc_obs_dim * disc_obs_steps
        self.disc_obs_dim = disc_obs_dim
        self.disc_obs_steps = disc_obs_steps
        self.obs_groups = obs_groups
        if not hidden_dims or disc_obs_dim < 1 or disc_obs_steps < 1:
            raise ValueError("判别器维度、历史长度和隐藏层不能为空或非正值。")
        if not 0 <= task_style_lerp <= 1:
            raise ValueError("任务奖励混合比例必须位于零到一之间。")
        activation = resolve_nn_activation(activation)
        assert style_reward_scale >= 0, "AMP reward scale must be non-negative."
        self.style_reward_scale = style_reward_scale
        self.task_style_lerp = task_style_lerp
        self.device = device
        self.loss_type = loss_type
        self.disc_obs_normalizer = EmpiricalNormalization(shape=self.disc_obs_dim, until=100000000.0).to(device)
        disc_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in hidden_dims:
            disc_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            disc_layers.append(activation)
            curr_in_dim = hidden_dim
        self.disc_trunk = nn.Sequential(*disc_layers)
        self.disc_linear = nn.Linear(hidden_dims[-1], 1)
        print(f"AMP Discriminator MLP: {self.disc_trunk}")
        print(f"AMP Discriminator Output Layer: {self.disc_linear}")
        if self.loss_type == LossType.WGAN:
            self.disc_output_normalizer = EmpiricalNormalization(shape=1, until=100000000.0).to(device)
        else:
            self.disc_output_normalizer = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算输入动作历史的判别器分数。"""
        h = self.disc_trunk(x)
        d = self.disc_linear(h)
        return d

    def get_disc_obs(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        """按配置拼接策略特征，形状为环境数、历史帧数、单帧特征数。"""
        disc_obs_list = []
        for obs_group in self.obs_groups["discriminator"]:
            if obs_group not in obs:
                raise ValueError(f"Observation group '{obs_group}' not found in environment observations.")
            obs_tensor = obs[obs_group]
            if obs_tensor.ndim != 3:
                raise ValueError("AMP discriminator observations must have shape (num_envs, num_steps, dim).")
            (num_envs, history_length, _) = obs_tensor.shape
            if history_length != self.disc_obs_steps:
                raise ValueError("Discriminator observation history length mismatch.")
            disc_obs_list.append(obs_tensor)
        disc_obs = torch.cat(disc_obs_list, dim=-1)
        if flatten_history_dim:
            disc_obs = disc_obs.view(num_envs, -1)
        return disc_obs

    def get_disc_demo_obs(self, obs: TensorDict, flatten_history_dim: bool = False) -> torch.Tensor:
        """按相同顺序拼接专家特征，要求与策略的维度和时序定义一致。"""
        disc_demo_obs_list = []
        for obs_group in self.obs_groups["discriminator_demonstration"]:
            if obs_group not in obs:
                raise ValueError(f"Observation group '{obs_group}' not found in environment observations.")
            obs_tensor = obs[obs_group]
            if obs_tensor.ndim != 3:
                raise ValueError("AMP discriminator observations must have shape (num_envs, num_steps, dim).")
            (num_envs, history_length, _) = obs_tensor.shape
            if history_length != self.disc_obs_steps:
                raise ValueError("Discriminator observation history length mismatch.")
            disc_demo_obs_list.append(obs_tensor)
        disc_demo_obs = torch.cat(disc_demo_obs_list, dim=-1)
        if flatten_history_dim:
            disc_demo_obs = disc_demo_obs.reshape(num_envs, -1)
        return disc_demo_obs

    def normalize_disc_obs(self, disc_obs: torch.Tensor) -> torch.Tensor:
        """按单帧特征归一化，再恢复多帧历史形状。"""
        if disc_obs.ndim != 3:
            raise ValueError("Discriminator observations must have shape (num_envs, num_steps, dim).")
        if self.disc_obs_dim != disc_obs.shape[2]:
            raise ValueError(f"Expected discriminator dimension {self.disc_obs_dim}, got {disc_obs.shape[2]}.")
        if self.disc_obs_steps != disc_obs.shape[1]:
            raise ValueError(f"Expected {self.disc_obs_steps} discriminator steps, got {disc_obs.shape[1]}.")
        disc_obs_reshaped = disc_obs.reshape(-1, self.disc_obs_dim)
        normed_disc_obs = self.disc_obs_normalizer(disc_obs_reshaped)
        # 与旧 AMPPPO 的 Normalizer 一致，归一化结果裁到 ±10。
        normed_disc_obs = torch.clamp(normed_disc_obs, -10.0, 10.0)
        normed_disc_obs = normed_disc_obs.reshape(-1, self.disc_obs_steps, self.disc_obs_dim)
        return normed_disc_obs

    def update_normalization(self, disc_obs: torch.Tensor) -> None:
        """用已经归一化的策略和专家特征更新运行统计。"""
        if disc_obs.ndim != 3:
            raise ValueError("Discriminator observations must have shape (num_envs, num_steps, dim).")
        if self.disc_obs_dim != disc_obs.shape[2]:
            raise ValueError(f"Expected discriminator dimension {self.disc_obs_dim}, got {disc_obs.shape[2]}.")
        if self.disc_obs_steps != disc_obs.shape[1]:
            raise ValueError(f"Expected {self.disc_obs_steps} discriminator steps, got {disc_obs.shape[1]}.")
        disc_obs_reshaped = disc_obs.reshape(-1, self.disc_obs_dim)
        self.disc_obs_normalizer.update(disc_obs_reshaped)

    def compute_grad_penalty(self, demo_data: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
        """约束专家样本上的输入梯度接近零，稳定判别器训练。"""
        if demo_data.ndim != 2:
            raise ValueError("Demonstration data must have shape (num_samples, flattened_dim).")
        demo_data_copy = demo_data.clone().detach().requires_grad_(True)
        disc = self.forward(demo_data_copy)
        ones = torch.ones_like(disc, device=demo_data_copy.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=demo_data_copy,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = scale * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_penalty

    def predict_style_reward(self, disc_obs: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """由判别器分数计算风格奖励，并乘控制步长及奖励缩放。"""
        if disc_obs.ndim != 3:
            raise ValueError("Discriminator observations must have shape (num_envs, num_steps, dim).")
        if self.disc_obs_dim != disc_obs.shape[2]:
            raise ValueError(f"Expected discriminator dimension {self.disc_obs_dim}, got {disc_obs.shape[2]}.")
        if self.disc_obs_steps != disc_obs.shape[1]:
            raise ValueError(f"Expected {self.disc_obs_steps} discriminator steps, got {disc_obs.shape[1]}.")
        was_training = self.training
        with torch.no_grad():
            self.eval()
            normed_disc_obs = self.normalize_disc_obs(disc_obs)
            normed_disc_obs = normed_disc_obs.view(-1, self.disc_obs_steps * self.disc_obs_dim)
            disc_score = self.forward(normed_disc_obs)
            rew = 0
            if self.loss_type == LossType.GAN:
                prob = 1.0 / (1.0 + torch.exp(-disc_score))
                rew = -torch.log(torch.maximum(1 - prob, torch.tensor(1e-06, device=self.device)))
            elif self.loss_type == LossType.LSGAN:
                rew = torch.clamp(1 - 1 / 4 * torch.square(disc_score - 1), min=0)
            elif self.loss_type == LossType.WGAN:
                rew = self.disc_output_normalizer(disc_score)
            else:
                raise ValueError(f"Unknown AMP loss type: {self.loss_type}. Should be 'GAN', 'LSGAN', or 'WGAN'")
            style_reward = dt * self.style_reward_scale * rew
            if was_training:
                self.train()
                if self.loss_type == LossType.WGAN:
                    self.disc_output_normalizer.update(disc_score)
        return (style_reward.squeeze(-1), disc_score.squeeze(-1))

    def lerp_reward(self, task_reward: torch.Tensor, style_reward: torch.Tensor) -> torch.Tensor:
        """按配置比例混合任务奖励和风格奖励。"""
        return self.task_style_lerp * task_reward + (1.0 - self.task_style_lerp) * style_reward


def resolve_amp_config(alg_cfg: dict, obs: TensorDict, obs_groups: dict, env: VecEnv) -> dict:
    """检查策略与专家观测组的一致性，推断特征维度、历史长度和控制步长。"""
    if "amp_cfg" in alg_cfg and alg_cfg["amp_cfg"] is not None:
        disc_obs_dim = 0
        disc_obs_steps = -1
        if "discriminator" not in obs_groups or "discriminator_demonstration" not in obs_groups:
            raise ValueError(
                "AMP configuration requires discriminator and discriminator_demonstration observation groups."
            )
        for obs_group in obs_groups["discriminator"]:
            if obs_group not in obs:
                raise ValueError(f"Observation group '{obs_group}' not found in environment observations.")
            obs_tensor = obs[obs_group]
            if obs_tensor.ndim != 3:
                raise ValueError("AMP discriminator observations must have shape (num_envs, num_steps, dim).")
            if disc_obs_steps == -1:
                disc_obs_steps = obs_tensor.shape[1]
            elif disc_obs_steps != obs_tensor.shape[1]:
                raise ValueError("All AMP discriminator observation groups must have the same history length.")
            disc_obs_dim += obs_tensor.shape[-1]
        disc_demo_obs_dim = 0
        for obs_group in obs_groups["discriminator_demonstration"]:
            if obs_group not in obs:
                raise ValueError(f"Observation group '{obs_group}' not found in environment observations.")
            obs_tensor = obs[obs_group]
            if obs_tensor.ndim != 3:
                raise ValueError("AMP demonstration observations must have shape (num_envs, num_steps, dim).")
            if disc_obs_steps != obs_tensor.shape[1]:
                raise ValueError("All AMP discriminator observation groups must have the same history length.")
            disc_demo_obs_dim += obs_tensor.shape[-1]
        if disc_demo_obs_dim != disc_obs_dim:
            raise ValueError("Demonstration and agent discriminator dimensions must match.")
        alg_cfg["amp_cfg"]["disc_obs_steps"] = disc_obs_steps
        alg_cfg["amp_cfg"]["disc_obs_dim"] = disc_obs_dim
        alg_cfg["amp_cfg"]["step_dt"] = env.unwrapped.step_dt
        if not disc_obs_dim or disc_obs_steps < 1:
            raise ValueError("AMP 观测组不能为空。")
        if env.unwrapped.step_dt <= 0:
            raise ValueError("AMP 控制步长必须为正值。")
        return alg_cfg
    else:
        raise ValueError("AMP configuration is missing or None.")
