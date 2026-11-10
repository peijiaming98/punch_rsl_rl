# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: file-ignore[missing-trailing-period, missing-terminal-punctuation, ambiguous-unicode-character-docstring]
"""验证 AMP 在当前 PPO 接口下的更新、终止观测和检查点恢复。"""

import copy
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import PPOAMP
from rsl_rl.modules.amp import AMPDiscriminator, LossType
from rsl_rl.storage import RolloutStorage
from tests.algorithms.test_ppo import _make_actor, _make_critic


def build_amp() -> tuple[PPOAMP, TensorDict]:
    """创建小规模 CPU 算法实例，不依赖仿真或真实动作文件。"""
    obs = TensorDict(
        {"policy": torch.randn(4, 8), "disc": torch.randn(4, 2, 3), "demo": torch.randn(4, 2, 3)}, batch_size=[4]
    )
    groups = {
        "actor": ["policy"],
        "critic": ["policy"],
        "discriminator": ["disc"],
        "discriminator_demonstration": ["demo"],
    }
    amp_cfg = dict(
        disc_obs_buffer_size=4,
        loss_type="LSGAN",
        disc_obs_dim=3,
        disc_obs_steps=2,
        obs_groups=groups,
        disc_trunk_weight_decay=0.0,
        disc_linear_weight_decay=0.0,
        disc_learning_rate=1e-3,
        grad_penalty_scale=10.0,
        step_dt=0.02,
        amp_discriminator=dict(hidden_dims=[16], task_style_lerp=0.8),
    )
    alg = PPOAMP(
        _make_actor(obs, groups),
        _make_critic(obs, groups),
        RolloutStorage("rl", 4, 2, obs, [4]),
        amp_cfg=amp_cfg,
        num_learning_epochs=1,
        num_mini_batches=2,
    )
    return alg, obs


def test_amp_update_and_checkpoint() -> None:
    """验证 PPO 与判别器均更新，归一化及优化器能随检查点恢复。"""
    alg, obs = build_amp()
    before = copy.deepcopy(alg.amp_discriminator.state_dict())
    actor_before = copy.deepcopy(alg.actor.state_dict())
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs, torch.ones(4), torch.zeros(4), {})
        alg.compute_returns(obs)
    metrics = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(v, before[k]) for k, v in alg.amp_discriminator.state_dict().items())
    assert any(not torch.equal(v, actor_before[k]) for k, v in alg.actor.state_dict().items())
    state = copy.deepcopy(alg.save())
    restored, _ = build_amp()
    assert restored.load(state, None, True)
    for k, v in restored.amp_discriminator.state_dict().items():
        torch.testing.assert_close(v, state["amp_discriminator_state_dict"][k])
    assert restored.disc_optimizer.state_dict()["state"]


def test_terminal_observation_required_and_used() -> None:
    """拒绝缺失终止状态的自动重置，提供终止状态后仅替换结束环境。"""
    alg, obs = build_amp()
    dones = torch.tensor([1, 0, 0, 0])
    with torch.no_grad():
        alg.act(obs)
        with pytest.raises(ValueError, match="final_obs"):
            alg.process_env_step(obs, torch.ones(4), dones, {})
        final = obs.clone()
        final["disc"] = torch.full_like(obs["disc"], 7)
        alg.process_env_step(obs, torch.ones(4), dones, {"final_obs": final})
    stored = alg.policy_obs_buffer.buffer[:, -1]
    torch.testing.assert_close(stored[0], final["disc"][0])
    torch.testing.assert_close(stored[1:], obs["disc"][1:])


@pytest.mark.parametrize("loss", list(LossType))
def test_discriminator_rewards_and_gradient(loss: LossType) -> None:
    """三类损失均能计算有限奖励与可反向传播的梯度惩罚。"""
    disc = AMPDiscriminator(3, 2, {}, loss_type=loss, hidden_dims=[8])
    x = torch.randn(4, 2, 3)
    reward, score = disc.predict_style_reward(x, 0.02)
    assert reward.shape == score.shape == (4,)
    assert torch.isfinite(reward).all()
    penalty = disc.compute_grad_penalty(x.flatten(1))
    penalty.backward()
    assert torch.isfinite(penalty)


def test_amp_factory_and_invalid_shapes() -> None:
    """标准工厂保留调用者配置，并拒绝策略与专家的历史维度不一致。"""
    from types import SimpleNamespace

    from rsl_rl.modules.amp import resolve_amp_config

    alg, obs = build_amp()
    groups = alg.amp_cfg["obs_groups"]
    cfg = dict(
        algorithm=dict(
            class_name="rsl_rl.algorithms.ppo_amp:PPOAMP",
            amp_cfg=alg.amp_cfg,
            num_learning_epochs=1,
            num_mini_batches=2,
        ),
        actor=dict(
            class_name="MLPModel",
            hidden_dims=[8],
            activation="elu",
            distribution_cfg=dict(class_name="GaussianDistribution", init_std=1.0, std_type="scalar"),
        ),
        critic=dict(class_name="MLPModel", hidden_dims=[8], activation="elu"),
        obs_groups=groups,
        multi_gpu=None,
        num_steps_per_env=2,
    )
    env = SimpleNamespace(num_envs=4, num_actions=4, unwrapped=SimpleNamespace(step_dt=0.02))
    before = copy.deepcopy(cfg)
    result = PPOAMP.construct_algorithm(obs, env, cfg, "cpu")
    assert isinstance(result, PPOAMP)
    assert cfg == before
    bad = obs.clone()
    bad["demo"] = torch.randn(4, 3, 3)
    with pytest.raises(ValueError, match="history length"):
        resolve_amp_config(copy.deepcopy(cfg["algorithm"]), bad, groups, env)


def test_amp_history_after_wraparound() -> None:
    """缓冲区回绕后只读取仍有效的历史，采样不使用未写入槽位。"""
    from rsl_rl.storage import CircularBuffer

    buffer = CircularBuffer(2, 2, "cpu")
    for value in range(4):
        buffer.append(torch.full((2, 1), float(value)))
    torch.testing.assert_close(buffer[torch.tensor([99, 0])], torch.tensor([[2.0], [3.0]]))
    batch = next(buffer.mini_batch_generator(2, 1, 1))
    assert set(batch.flatten().tolist()) == {2.0, 3.0}
