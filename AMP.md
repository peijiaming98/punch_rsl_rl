# AMP 算法扩展

参考本地 RoboParty 的 `PPOAMP` 实现，继承当前 RSL-RL 的标准 PPO。

## 文件职责

- `rsl_rl/algorithms/ppo_amp.py`：算法工厂、奖励混合、判别器更新、保存恢复和分布式同步。
- `rsl_rl/modules/amp.py`：判别器、运行归一化、梯度惩罚和特征校验。
- `rsl_rl/storage/circular_buffer.py`：固定容量的多环境历史样本缓冲区。

算法不读取动作文件，不包含机器人关节或连杆名称。
连杆位置等特征由环境计算，专家和策略必须采用相同关节顺序、坐标系与历史采样间隔。

## 接口约定

`obs_groups` 除标准 `actor` 和 `critic` 外，需要提供：

```python
obs_groups = {
    "actor": ["actor"],
    "critic": ["critic"],
    "discriminator": ["disc"],
    "discriminator_demonstration": ["disc_demo"],
}
```

判别器每个观测组形状为 `[环境数, 历史帧数, 单帧特征数]`，不接受展平的历史。
`PPOAMP.construct_algorithm` 根据样例观测推断维度，并从 `env.unwrapped.step_dt` 获取控制步长。
环境发生终止或超时时，`extras["final_obs"]` 必须提供重置前的完整批次观测；
算法只替换其中结束的环境，不使用重置后的状态代替动作末帧。
缺少这一接口时明确报错，后续接入 mjlab 任务时需要由环境适配层提供。

算法配置示例：

```python
algorithm = {
    "class_name": "rsl_rl.algorithms.ppo_amp:PPOAMP",
    "amp_cfg": {
        "loss_type": "LSGAN",
        "disc_obs_buffer_size": 100,
        "disc_learning_rate": 1e-3,
        "disc_trunk_weight_decay": 1e-4,
        "disc_linear_weight_decay": 1e-2,
        "grad_penalty_scale": 10.0,
        "disc_max_grad_norm": 1.0,
        "amp_discriminator": {
            "hidden_dims": [1024, 512, 256],
            "activation": "relu",
            "style_reward_scale": 1.0,
            "task_style_lerp": 0.8,
        },
    },
}
```

以上为接口示例，并非已经验证的 Pi Plus 最优参数。
缓冲区容量以控制步数计，必须覆盖一轮 rollout；按环境数成倍占用内存。
每轮判别器从已有缓冲区采样一轮 rollout 等量的数据，支持多个训练 epoch。

最小二乘风格奖励为：

```text
style = dt × scale × max(0, 1 − (score − 1)² / 4)
mixed = lerp × task + (1 − lerp) × style
```

任务奖励应采用与控制步长一致的尺度，不能直接照搬未乘 dt 的旧版系数。
同时支持 GAN 和 WGAN 损失；WGAN 使用判别分数的运行归一化。
判别器输入运行统计沿用参考实现，仅使用策略特征更新。

## 范围和验证

保留 RoboParty 的授权信息，去除与 AMP 无关的混合专家网络统计。
标准 PPO 更新、策略对称增强和日志循环继续由上游实现提供。
检查点包含判别器归一化统计及优化器状态，不保存经验缓冲区；恢复后重新采样。
旧版 Pi Plus 的 ActorCritic 检查点不保证与当前分离式 actor、critic 模型兼容。

CPU 测试覆盖标准工厂、PPO 与判别器更新、终止观测替换、损失计算、缓冲区和保存恢复。
分布式接口已经实现，但尚未进行多 GPU 验证。
本次不修改或注册 `src/tasks/amp/` 中的任务。
