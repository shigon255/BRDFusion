import logging
import time
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


class OpTimer:
    def __init__(self, enabled: bool = False, use_cuda_sync: bool = True, prefix: str = ""):
        self.enabled = enabled
        self.use_cuda_sync = use_cuda_sync and torch.cuda.is_available()
        self.prefix = prefix
        self._starts = {}
        self._times = {}
        self._order = []

    def _sync(self) -> None:
        if self.use_cuda_sync:
            torch.cuda.synchronize()

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        if name not in self._times:
            self._order.append(name)
            self._times[name] = 0.0
        self._sync()
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        start = self._starts.pop(name, None)
        if start is None:
            return
        self._sync()
        self._times[name] += time.perf_counter() - start

    def add_time(self, name: str, seconds: float) -> None:
        if not self.enabled:
            return
        if name not in self._times:
            self._order.append(name)
            self._times[name] = 0.0
        self._times[name] += seconds

    @contextmanager
    def time_block(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def log(self, header: str = "") -> None:
        if not self.enabled or not self._order:
            return
        parts = [f"{name}={self._times[name] * 1000.0:.2f}ms" for name in self._order]
        prefix = f"{self.prefix}{header}".strip()
        message = f"{prefix} | {' '.join(parts)}" if prefix else " ".join(parts)
        root_logger = logging.getLogger()
        if logger.hasHandlers() or root_logger.hasHandlers():
            logger.info("%s", message)
        else:
            print(message)
        self._starts.clear()
        self._times.clear()
        self._order.clear()
