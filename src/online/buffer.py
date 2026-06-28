from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RingView:
    x: np.ndarray
    t0: int
    t1: int


class RingBuffer:
    def __init__(self, capacity_samples: int, n_channels: int, dtype=np.float32):
        self.capacity_samples = int(capacity_samples)
        self.n_channels = int(n_channels)
        self.dtype = dtype
        self._buf = np.zeros((self.capacity_samples, self.n_channels), dtype=dtype)
        self._head = 0
        self._size = 0
        self.global_t0_of_buffer_start = 0
        self.global_t1_exclusive = 0

    def __len__(self) -> int:
        return self._size

    def has(self, n_samples: int) -> bool:
        return self._size >= int(n_samples)

    def push(self, x_new: np.ndarray) -> None:
        if x_new.size == 0:
            return
        x_new = np.asarray(x_new, dtype=self.dtype)
        if x_new.ndim != 2 or x_new.shape[1] != self.n_channels:
            raise ValueError(f"expected [N, {self.n_channels}], got {x_new.shape}")

        n = x_new.shape[0]
        if n >= self.capacity_samples:
            x_new = x_new[-self.capacity_samples :]
            n = x_new.shape[0]
            self._head = 0
            self._size = 0
            self.global_t0_of_buffer_start = self.global_t1_exclusive - self.capacity_samples
            if self.global_t0_of_buffer_start < 0:
                self.global_t0_of_buffer_start = 0

        tail = (self._head + self._size) % self.capacity_samples
        first = min(n, self.capacity_samples - tail)
        self._buf[tail : tail + first] = x_new[:first]
        remain = n - first
        if remain > 0:
            self._buf[:remain] = x_new[first:]

        if self._size + n <= self.capacity_samples:
            self._size += n
        else:
            overflow = self._size + n - self.capacity_samples
            self._size = self.capacity_samples
            self._head = (self._head + overflow) % self.capacity_samples
            self.global_t0_of_buffer_start += overflow
        self.global_t1_exclusive += n
        if self._size < self.capacity_samples:
            self.global_t0_of_buffer_start = self.global_t1_exclusive - self._size

    def _as_contiguous(self) -> np.ndarray:
        if self._size == 0:
            return np.zeros((0, self.n_channels), dtype=self.dtype)
        if self._head + self._size <= self.capacity_samples:
            return self._buf[self._head : self._head + self._size].copy()
        first = self.capacity_samples - self._head
        return np.vstack([self._buf[self._head :], self._buf[: self._size - first]]).astype(self.dtype, copy=False)

    def get_last(self, n_samples: int) -> RingView:
        n = int(n_samples)
        if n > self._size:
            raise ValueError(f"not enough samples: need {n}, have {self._size}")
        t1 = self.global_t1_exclusive
        return self.get_range(t1 - n, t1)

    def get_range(self, global_start: int, global_stop: int) -> RingView:
        gs = int(global_start)
        ge = int(global_stop)
        if ge <= gs:
            return RingView(x=np.zeros((0, self.n_channels), dtype=self.dtype), t0=gs, t1=gs)
        if gs < self.global_t0_of_buffer_start or ge > self.global_t1_exclusive:
            raise ValueError(
                f"range [{gs}, {ge}) outside buffer [{self.global_t0_of_buffer_start}, {self.global_t1_exclusive})"
            )

        n = ge - gs
        rel0 = gs - self.global_t0_of_buffer_start
        start = (self._head + rel0) % self.capacity_samples
        stop = start + n

        # Fast path: requested range is physically contiguous in ring storage.
        if stop <= self.capacity_samples:
            return RingView(x=self._buf[start:stop], t0=gs, t1=ge)

        # Wrap-around path: only copy the requested slice, not the full ring.
        first = self.capacity_samples - start
        out = np.empty((n, self.n_channels), dtype=self.dtype)
        out[:first] = self._buf[start:]
        out[first:] = self._buf[: n - first]
        return RingView(x=out, t0=gs, t1=ge)
