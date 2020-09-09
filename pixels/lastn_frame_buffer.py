from collections import deque
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np


class LastNFrameBuffer:
    def __init__(self, state_size: Sequence[int], n: int = 4):
        self.n = n
        self.state_size = state_size
        self.buffer = deque(maxlen=n)

    def add(self, frame: np.array) -> None:
        frame = np.expand_dims(frame, axis=0)
        frame = np.transpose(frame, (0, 4, 1, 2, 3))[:, :, :, :, :]
        if len(self.buffer) == 0:
            for _ in range(self.n):
                self.buffer.append(np.zeros(frame.shape))
        self.buffer.append(frame)

    def get_last(self) -> np.array:
        result = np.zeros((
            1,
            self.state_size[1],
            self.n,
            self.state_size[3],
            self.state_size[4],
        ))
        for i, frame in enumerate(self.buffer):
            result[0, :, i, :, :] = frame[0, :, 0, :, :]
        return result
