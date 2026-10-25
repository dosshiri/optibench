"""ATSP の解法: PyVRP（ハイブリッド遺伝的探索 HGS）. pyvrp が入っていなければこのファイルだけ読み込みを飛ばす."""
import pyvrp  # noqa: F401  読み込めなければ登録しない

from problems.routing import tour_cost
from registry import solver


@solver("PyVRP-HGS", problems="atsp", seeds=3, packages=["pyvrp"],
        description=("配送計画ライブラリ PyVRP のハイブリッド遺伝的探索（HGS）。"
                     "解の集団を交叉させ、生まれた子を局所探索で磨く遺伝的アルゴリズムで、Vidal の HGS を基にした最先端の配送計画ソルバー。"
                     "車両 1 台・容量なしの配送計画として、非対称の距離をそのまま使って解く。"))
def pyvrp_hgs(D, tl, seed):
    """車両 1 台・容量なしの配送計画として解く. 都市 0 をデポ、残りを顧客にし、非対称の距離をそのまま辺に載せる."""
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    n = D.shape[0]
    m = Model()
    locs = [m.add_location(0, 0) for _ in range(n)]
    depot = m.add_depot(locs[0])
    m.add_vehicle_type(1, start_depot=depot, end_depot=depot)
    for i in range(1, n):
        m.add_client(locs[i])
    for i in range(n):
        for j in range(n):
            if i != j:
                m.add_edge(locs[i], locs[j], distance=int(D[i, j]))
    res = m.solve(stop=MaxRuntime(tl), seed=seed, display=False)
    if not res.is_feasible() or not res.best.routes():
        return dict(obj=None, lb=None, solution=None)
    # 経路の各活動のうち顧客の番号（顧客だけの 0 始まり）を都市番号に直す
    tour = [0] + [a.idx + 1 for a in res.best.routes()[0] if a.is_client()]
    return dict(obj=tour_cost(D, tour), lb=None, solution=tour)
