"""ATSP-TW の解法: PyVRP（ハイブリッド遺伝的探索 HGS）. pyvrp が入っていなければこのファイルだけ読み込みを飛ばす."""
import pyvrp  # noqa: F401  読み込めなければ登録しない
import numpy as np

from registry import solver
from solvers.atsptw import _result


@solver("PyVRP-HGS", problems="atsptw", seeds=3, packages=["pyvrp"],
        description=("配送計画ライブラリ PyVRP のハイブリッド遺伝的探索（HGS）。"
                     "解の集団を交叉させ、生まれた子を局所探索で磨く遺伝的アルゴリズムで、Vidal の HGS を基にした最先端の配送計画ソルバー。"
                     "車両 1 台・時間枠付きの配送計画として解く。"))
def pyvrp_hgs(inst, tl, seed):
    """車両 1 台・容量なしの時間枠付き配送計画として解く. 距離と所要時間はどちらも移動時間行列."""
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    D, r, d = inst["D"], inst["r"], inst["d"]
    n = D.shape[0]
    m = Model()
    locs = [m.add_location(0, 0) for _ in range(n)]
    depot = m.add_depot(locs[0], tw_early=int(r[0]), tw_late=int(d[0]))
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot, tw_early=int(r[0]), tw_late=int(d[0]))
    for i in range(1, n):
        m.add_client(locs[i], tw_early=int(r[i]), tw_late=int(d[i]))
    for i in range(n):
        for j in range(n):
            if i != j:
                m.add_edge(locs[i], locs[j], distance=int(D[i, j]), duration=int(D[i, j]))
    res = m.solve(stop=MaxRuntime(tl), seed=seed, display=False)
    if not res.is_feasible() or not res.best.routes():
        return dict(obj=None, lb=None, solution=None)
    # 経路の各活動のうち顧客の番号（顧客だけの 0 始まり）を都市番号に直す
    visits = [a.idx + 1 for a in res.best.routes()[0] if a.is_client()]
    return _result(inst, [0] + visits)
