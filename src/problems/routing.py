"""巡回セールスマン問題（TSP: 対称, ATSP: 非対称）. インスタンスは TSPLIB95.
解法に渡すデータは整数距離行列 D (0-index), 解は都市の巡回順のリスト."""
import os

import numpy as np

from common import INST
from registry import Problem, register_problem

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


def evaluate(D, tour):
    """全都市をちょうど 1 回ずつ訪れる巡回路なら総距離、そうでなければ None."""
    if sorted(tour) != list(range(D.shape[0])):
        return None
    return tour_cost(D, tour)


register_problem(Problem(
    name="tsp", label="TSP（対称巡回セールスマン）", unit="距離", ref_label="TSPLIB 公表最適値",
    instance_file="tsp/{}.tsp", instances=["berlin52", "kroA100", "ch150", "a280", "pcb442", "rat783"],
    load=lambda name: load_tsplib_matrix("tsp", name), reference=TSP_OPT.get, evaluate=evaluate,
    order=10, packages=["tsplib95"]))

register_problem(Problem(
    name="atsp", label="ATSP（非対称巡回セールスマン）", unit="距離", ref_label="TSPLIB 公表最適値",
    instance_file="atsp/{}.atsp", instances=["ftv33", "ftv70", "kro124p", "ft70", "rbg323", "rbg443"],
    load=lambda name: load_tsplib_matrix("atsp", name), reference=ATSP_OPT.get, evaluate=evaluate,
    order=20, packages=["tsplib95"]))
