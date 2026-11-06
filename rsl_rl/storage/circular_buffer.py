# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: D400, D415, RUF002
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""多环境 AMP 样本循环缓冲区。"""

import torch
from collections.abc import Iterator, Sequence


class CircularBuffer:
    """保存多环境的固定容量历史张量，供判别器采样使用。"""

    def __init__(self, max_len: int, batch_size: int, device: str) -> None:
        """初始化配置、参数和所需状态；具体维度由构造参数指定。"""
        if max_len < 1 or batch_size < 1:
            raise ValueError(f"The buffer size should be greater than zero. However, it is set to {max_len}!")
        self._batch_size = batch_size
        self._device = device
        self._ALL_INDICES = torch.arange(batch_size, device=device)
        self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._pointer: int = -1
        self._buffer: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        """返回每次写入的环境数量。"""
        return self._batch_size

    @property
    def device(self) -> str:
        """返回缓冲区所在设备。"""
        return self._device

    @property
    def max_length(self) -> int:
        """返回缓冲区的最大历史长度。"""
        return int(self._max_len[0].item())

    @property
    def current_length(self) -> torch.Tensor:
        """返回每个环境已写入的有效历史长度。"""
        return torch.minimum(self._num_pushes, self._max_len)

    @property
    def buffer(self) -> torch.Tensor:
        """返回按时间排列的完整历史副本，最近样本位于末尾。"""
        if self._buffer is None:
            raise RuntimeError("缓冲区尚未写入数据。")
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
        return torch.transpose(buf, dim0=0, dim1=1)

    def reset(self, batch_ids: Sequence[int] | None = None) -> None:
        """清空指定环境的历史状态和有效写入计数。"""
        if batch_ids is None:
            batch_ids = slice(None)
        self._num_pushes[batch_ids] = 0
        if self._buffer is not None:
            self._buffer[:, batch_ids, :] = 0.0

    def append(self, data: torch.Tensor) -> None:
        """写入一批环境样本；首次写入时用该样本填充历史。"""
        if data.shape[0] != self.batch_size:
            raise ValueError(f"The input data has '{data.shape[0]}' batch size while expecting '{self.batch_size}'")
        data = data.to(self._device)
        if self._buffer is None:
            self._pointer = -1
            self._buffer = torch.empty((self.max_length, *data.shape), dtype=data.dtype, device=self._device)
        self._pointer = (self._pointer + 1) % self.max_length
        self._buffer[self._pointer] = data
        is_first_push = self._num_pushes == 0
        if torch.any(is_first_push):
            self._buffer[:, is_first_push] = data[is_first_push]
        self._num_pushes += 1

    def __getitem__(self, key: torch.Tensor) -> torch.Tensor:
        """按每个环境的历史延迟索引读取，超出有效历史时返回最早样本。"""
        if len(key) != self.batch_size:
            raise ValueError(f"The argument 'key' has length {key.shape[0]}, while expecting {self.batch_size}")
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")
        if (key < 0).any():
            raise ValueError("历史延迟不能为负值。")
        valid_keys = torch.minimum(key, self.current_length - 1)
        index_in_buffer = torch.remainder(self._pointer - valid_keys, self.max_length)
        return self._buffer[index_in_buffer, self._ALL_INDICES]

    def mini_batch_generator(
        self, fetch_length: int, num_mini_batches: int, num_epochs: int = 8
    ) -> Iterator[torch.Tensor]:
        """从已写入历史随机生成小批量，要求样本数能被批次数整除。"""
        if torch.any(self._num_pushes == 0) or self._buffer is None:
            raise RuntimeError("Attempting to retrieve data on an empty circular buffer. Please append data first.")
        min_current_length = torch.min(self.current_length).item()
        if fetch_length > min_current_length:
            raise ValueError(f"Fetch length {fetch_length} exceeds minimum current length {min_current_length}.")
        if fetch_length < 1 or num_mini_batches < 1 or num_epochs < 1:
            raise ValueError("采样长度、批次数和轮数必须为正值。")
        epoch_batch_size = self.batch_size * fetch_length
        mini_batch_size = epoch_batch_size // num_mini_batches
        if epoch_batch_size % num_mini_batches != 0:
            raise ValueError(
                f"Epoch batch size {epoch_batch_size} is not divisible by number of mini-batches {num_mini_batches}."
            )
        total_combinations = self.current_length[0] * self.batch_size
        linear_indices = torch.randperm(total_combinations, device=self.device)[:epoch_batch_size]
        indices_0 = linear_indices // self.batch_size
        indices_1 = linear_indices % self.batch_size
        for epoch in range(num_epochs):
            indices = torch.randperm(epoch_batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                mini_indices_0 = indices_0[indices[start:end]]
                mini_indices_1 = indices_1[indices[start:end]]
                yield self._buffer[mini_indices_0, mini_indices_1]
