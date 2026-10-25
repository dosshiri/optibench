"""共通ユーティリティ: パスと時間計測."""
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下
INST = os.path.join(ROOT, "instances")


class Timer:
    def __init__(self, limit):
        self.t0 = time.perf_counter()
        self.limit = limit

    def elapsed(self):
        return time.perf_counter() - self.t0

    def left(self):
        return self.limit - self.elapsed()
