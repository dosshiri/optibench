"""VRPTW の解法: PyVRP（ハイブリッド遺伝的探索 HGS）. pyvrp が入っていなければこのファイルだけ読み込みを飛ばす."""
import pyvrp  # noqa: F401  読み込めなければ登録しない

from registry import solver
from solvers.vrptw import _result


@solver("PyVRP-HGS", problems="vrptw", seeds=3, packages=["pyvrp"],
        description=("配送計画ライブラリ PyVRP のハイブリッド遺伝的探索（HGS）。"
                     "解の集団を交叉させ、生まれた子を局所探索で磨く遺伝的アルゴリズムで、Vidal の HGS を基にした最先端の配送計画ソルバー。"
                     "容量と時間枠をそのまま PyVRP のモデルに載せ、総距離を最小化する。"))
def pyvrp_hgs(inst, tl, seed):
    """距離・移動時間は 10 倍の整数のまま渡す（DIMACS の規約どおりの整数化なので、元の距離と食い違わない）."""
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    D, xy, e, l, s, q = inst["D"], inst["xy"], inst["e"], inst["l"], inst["s"], inst["q"]
    n = D.shape[0]
    m = Model()
    locs = [m.add_location(int(x), int(y)) for x, y in xy]
    depot = m.add_depot(locs[0], tw_early=int(e[0]), tw_late=int(l[0]))
    m.add_vehicle_type(inst["K"], capacity=[inst["Q"]], start_depot=depot, end_depot=depot,
                       tw_early=int(e[0]), tw_late=int(l[0]))
    for i in range(1, n):
        m.add_client(locs[i], delivery=[int(q[i])], service_duration=int(s[i]), tw_early=int(e[i]), tw_late=int(l[i]))
    for i in range(n):
        for j in range(n):
            if i != j:
                m.add_edge(locs[i], locs[j], distance=int(D[i, j]), duration=int(D[i, j]))
    res = m.solve(stop=MaxRuntime(tl), seed=seed, display=False)
    if not res.is_feasible():
        return dict(obj=None, lb=None, solution=None)
    # 経路の各活動のうち顧客の番号（顧客だけの 0 始まり）を地点番号に直す
    routes = [[a.idx + 1 for a in r if a.is_client()] for r in res.best.routes()]
    return _result(inst, routes)
