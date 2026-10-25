"""共通ユーティリティ: インスタンス読込・既知最適値・時間計測."""
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下
INST = os.path.join(ROOT, "instances")

# TSPLIB 公式の最適値 (TSPLIB95 solutions)
TSP_OPT = {"berlin52": 7542, "kroA100": 21282, "ch150": 6528, "a280": 2579,
           "pcb442": 50778, "rat783": 8806}
ATSP_OPT = {"ftv33": 1286, "ftv70": 1950, "kro124p": 36230, "ft70": 38673,
            "rbg323": 1326, "rbg443": 2720}


def load_tsplib_matrix(problem, name):
    """TSPLIB を読み込み、整数距離行列 (0-index) を返す."""
    import tsplib95
    ext = "tsp" if problem == "tsp" else "atsp"
    p = tsplib95.load(os.path.join(INST, ext, f"{name}.{ext}"))
    nodes = list(p.get_nodes())
    n = len(nodes)
    D = np.zeros((n, n), dtype=np.int64)
    for a, i in enumerate(nodes):
        for b, j in enumerate(nodes):
            if a != b:
                D[a, b] = p.get_weight(i, j)
    return D


def tour_cost(D, tour):
    t = np.asarray(tour)
    return int(D[t, np.roll(t, -1)].sum())


def load_jssp(name):
    """JSPLIB (OR-Library 形式) を読み込む. jobs[j] = [(machine, duration), ...]"""
    with open(os.path.join(INST, "jssp", name)) as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    nj, nm = map(int, lines[0].split())
    jobs = []
    for l in lines[1:1 + nj]:
        v = list(map(int, l.split()))
        jobs.append([(v[2 * k], v[2 * k + 1]) for k in range(nm)])
    return jobs, nm


def jssp_ref(name):
    with open(os.path.join(INST, "jssp", "instances.json")) as f:
        for x in json.load(f):
            if x["name"] == name:
                if x.get("optimum"):
                    return x["optimum"], x["optimum"]
                return x["bounds"]["lower"], x["bounds"]["upper"]
    return None, None


class Timer:
    def __init__(self, limit):
        self.t0 = time.perf_counter()
        self.limit = limit

    def elapsed(self):
        return time.perf_counter() - self.t0

    def left(self):
        return self.limit - self.elapsed()
