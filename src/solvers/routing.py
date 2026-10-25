"""TSP / ATSP の解法群. 各関数は (D, time_limit, seed) -> dict(obj, lb, solution=巡回順)."""
import numpy as np
from numba import njit

from common import Timer
from problems.routing import tour_cost
from registry import solver

ROUTING = ["tsp", "atsp"]


# ---------------------------------------------------------------- 構築法・局所探索
@njit(cache=True)
def _nearest_neighbor(D, start):
    n = D.shape[0]
    used = np.zeros(n, np.bool_)
    tour = np.empty(n, np.int64)
    tour[0] = start
    used[start] = True
    for k in range(1, n):
        c = tour[k - 1]
        best, bj = 1 << 62, -1
        for j in range(n):
            if not used[j] and D[c, j] < best:
                best, bj = D[c, j], j
        tour[k] = bj
        used[bj] = True
    return tour


@njit(cache=True)
def _or_opt_delta(D, t, i, L, j):
    """区間 t[i..i+L-1] を t[j] と t[j+1] の間へ (向きを保って) 移す差分. j は区間外."""
    n = t.shape[0]
    a = t[(i - 1) % n]
    s = t[i]
    e = t[(i + L - 1) % n]
    b = t[(i + L) % n]
    p = t[j]
    q = t[(j + 1) % n]
    return D[a, b] + D[p, s] + D[e, q] - D[a, s] - D[e, b] - D[p, q]


@njit(cache=True)
def _apply_or_opt(t, i, L, j):
    n = t.shape[0]
    seg = np.empty(L, np.int64)
    for k in range(L):
        seg[k] = t[(i + k) % n]
    rest = np.empty(n - L, np.int64)
    m = 0
    pos_q = -1
    for k in range(n):
        idx = (i + L + k) % n
        if m < n - L:
            rest[m] = t[idx]
            if idx == j:
                pos_q = m
            m += 1
    out = np.empty(n, np.int64)
    m = 0
    for k in range(n - L):
        out[m] = rest[k]
        m += 1
        if k == pos_q:
            for r in range(L):
                out[m] = seg[r]
                m += 1
    return out


@njit(cache=True)
def _two_opt_delta(D, t, i, j):
    n = t.shape[0]
    a, b = t[i], t[(i + 1) % n]
    c, d = t[j], t[(j + 1) % n]
    return D[a, c] + D[b, d] - D[a, b] - D[c, d]


@njit(cache=True)
def _reverse(t, i, j):
    # t[i+1..j] を反転 (i<j)
    lo, hi = i + 1, j
    while lo < hi:
        t[lo], t[hi] = t[hi], t[lo]
        lo += 1
        hi -= 1


@njit(cache=True)
def _local_search(D, t, symmetric):
    """最良改善ではなく先着改善の 2-opt (対称のみ) + Or-opt (長さ1-3)."""
    n = t.shape[0]
    improved = True
    while improved:
        improved = False
        if symmetric:
            for i in range(n - 1):
                for j in range(i + 2, n if i > 0 else n - 1):
                    if _two_opt_delta(D, t, i, j) < 0:
                        _reverse(t, i, j)
                        improved = True
        for L in range(1, 4):
            for i in range(n):
                for j in range(n):
                    # j が区間内・直前でないこと
                    off = (j - i) % n
                    if off < L or (j + 1) % n == i:
                        continue
                    if _or_opt_delta(D, t, i, L, j) < 0:
                        t[:] = _apply_or_opt(t, i, L, j)
                        improved = True
                        break
    return t


@solver("nn+ls", problems=ROUTING,
        description=("最近傍法（いちばん近い未訪問の都市へ順に進む）で巡回路を作り、2-opt（対称のときだけ）と Or-opt（1〜3 都市の区間の付け替え）で改善できなくなるまで局所探索する自作手法。"
                     "始点を変えながら制限時間まで繰り返し、最良の巡回路を返す。"
                     "実装が簡単で、他の手法を測る物差しになる。"))
def nn_local_search(D, tl, seed):
    sym = bool((D == D.T).all())
    tm = Timer(tl)
    best, bt = None, None
    rng = np.random.default_rng(seed)
    starts = [0] + list(rng.permutation(D.shape[0]))
    last = 0.0
    for s in starts:  # 時間内で複数始点を試す (多始点局所探索)
        t0 = tm.elapsed()
        t = _local_search(D, _nearest_neighbor(D, s), sym)
        c = tour_cost(D, t)
        if best is None or c < best:
            best, bt = c, t.copy()
        last = tm.elapsed() - t0
        if tm.left() < last:
            break
    return dict(obj=best, lb=None, solution=bt.tolist())


# ---------------------------------------------------------------- 焼きなまし法
@njit(cache=True)
def _sa_chunk(D, t, cur, best_t, best, T, iters, symmetric, seed):
    np.random.seed(seed)
    n = t.shape[0]
    for _ in range(iters):
        if symmetric and np.random.random() < 0.5:
            i = np.random.randint(0, n - 1)
            j = np.random.randint(i + 2, n + 1)
            if j >= n or (i == 0 and j == n - 1):
                continue
            d = _two_opt_delta(D, t, i, j)
            if d <= 0 or np.random.random() < np.exp(-d / T):
                _reverse(t, i, j)
                cur += d
        else:
            L = np.random.randint(1, 4)
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            off = (j - i) % n
            if off < L or (j + 1) % n == i:
                continue
            d = _or_opt_delta(D, t, i, L, j)
            if d <= 0 or np.random.random() < np.exp(-d / T):
                t[:] = _apply_or_opt(t, i, L, j)
                cur += d
        if cur < best:
            best = cur
            best_t[:] = t
    return cur, best


@solver("SA", problems=ROUTING, seeds=3,
        description=("焼きなまし法（Simulated Annealing）。"
                     "2-opt や Or-opt のランダムな変更を、悪くなる場合も「温度」に応じた確率で受け入れ、温度を徐々に下げることで局所最適から抜け出す自作手法。"
                     "最後に局所探索で仕上げる。"))
def simulated_annealing(D, tl, seed):
    sym = bool((D == D.T).all())
    tm = Timer(tl)
    rng = np.random.default_rng(seed)
    t = _nearest_neighbor(D, int(rng.integers(D.shape[0])))
    cur = tour_cost(D, t)
    best_t, best = t.copy(), cur
    # 初期温度: 平均辺長の 50%, 終温度: 0.3% (berlin52/kroA100/ch150 で予備調整)
    mean_edge = cur / D.shape[0]
    T0, T1 = 0.5 * mean_edge, 0.003 * mean_edge
    k = 0
    while tm.left() > 0:
        frac = tm.elapsed() / tl
        T = T0 * (T1 / T0) ** frac
        cur, best = _sa_chunk(D, t, cur, best_t, best, T, 20000, sym, seed * 100003 + k)
        k += 1
    best_t = _local_search(D, best_t, sym)
    return dict(obj=tour_cost(D, best_t), lb=None, solution=best_t.tolist())


# ---------------------------------------------------------------- LKH (elkai)
@solver("LKH(elkai)", problems=ROUTING, packages=["elkai"],
        description=("Lin–Kernighan 法を発展させた Helsgaun の LKH を elkai パッケージ経由で呼ぶ。"
                     "複数の辺をまとめて入れ替える強力な局所探索で、TSP の近似解法では最強クラス。"
                     "ここでは時間ではなく試行回数（runs=10）で止めている。"))
def lkh(D, tl, seed):
    import elkai
    # elkai は時間制限を持たないので runs を固定 (LKH 既定の試行回数相当)
    m = elkai.DistanceMatrix(D.tolist())
    tour = m.solve_tsp(runs=10)[:-1]
    return dict(obj=tour_cost(D, tour), lb=None, solution=tour)


# ---------------------------------------------------------------- 誘導局所探索 (OR-Tools Routing)
@solver("ORTools-GLS", problems=ROUTING, packages=["ortools"],
        description=("Google OR-Tools の配送計画ソルバー（Routing）。"
                     "安い弧から順につないで初期解を作り、誘導局所探索（GLS: 局所最適に何度も現れる辺にペナルティを課して別の解へ誘導する）で改善する。"
                     "時間枠・容量・複数車両などの業務制約を足せる汎用性が強み。"))
def ortools_gls(D, tl, seed):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = D.shape[0]
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    r = pywrapcp.RoutingModel(mgr)
    cb = r.RegisterTransitMatrix(D.tolist())
    r.SetArcCostEvaluatorOfAllVehicles(cb)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(tl * 1000))
    sol = r.SolveWithParameters(p)
    idx, tour = r.Start(0), []
    while not r.IsEnd(idx):
        tour.append(mgr.IndexToNode(idx))
        idx = sol.Value(r.NextVar(idx))
    return dict(obj=tour_cost(D, tour), lb=None, solution=tour)


# ---------------------------------------------------------------- 制約プログラミング (CP-SAT, AddCircuit 制約)
@solver("CP-SAT", problems=ROUTING, parallel=True, packages=["ortools"],
        description=("OR-Tools の制約プログラミングソルバー CP-SAT。"
                     "各弧を使うかどうかを 0-1 変数にし、AddCircuit 制約で「全都市を 1 本の巡回路で回る」ことを表して総距離を最小化する。"
                     "下界も出すので、時間内に最適性を証明できることもある。"))
def cpsat_circuit(D, tl, seed, threads=1):
    from ortools.sat.python import cp_model
    n = D.shape[0]
    m = cp_model.CpModel()
    arcs, x = [], {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = m.NewBoolVar("")
                arcs.append((i, j, x[i, j]))
    m.AddCircuit(arcs)
    m.Minimize(sum(int(D[i, j]) * v for (i, j), v in x.items()))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = threads
    s.parameters.random_seed = seed
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(obj=None, lb=s.BestObjectiveBound(), solution=None)
    nxt = {i: j for (i, j), v in x.items() if s.Value(v)}
    tour, c = [0], nxt[0]
    while c != 0:
        tour.append(c)
        c = nxt[c]
    return dict(obj=tour_cost(D, tour), lb=int(np.ceil(s.BestObjectiveBound() - 1e-6)), solution=tour)


# ---------------------------------------------------------------- MIP (HiGHS, DFJ 逐次カット)
def _components(n, succ_pairs):
    adj = [[] for _ in range(n)]
    for i, j in succ_pairs:
        adj[i].append(j)
        adj[j].append(i)
    seen, comps = [False] * n, []
    for s in range(n):
        if seen[s]:
            continue
        st, comp = [s], []
        seen[s] = True
        while st:
            u = st.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    st.append(v)
        comps.append(comp)
    return comps


@solver("MIP-HiGHS", problems=ROUTING, parallel=True, packages=["highspy"],
        description=("混合整数計画（MIP）をオープンソースのソルバー HiGHS で解く。"
                     "まず各都市の次数の制約だけで解き、小さな輪（部分巡回路）が出るたびにそれを禁じる制約（DFJ）を足して解き直す。"
                     "最適性を証明できるが、時間切れになると巡回路を返せない。"))
def mip_dfj(D, tl, seed, threads=1):
    """対称: 無向辺 x_e, 次数2. 非対称: 有向弧, 入出次数1. 部分巡回路を見つけるたびに除去制約を追加して再求解."""
    import highspy
    tm = Timer(tl)
    n = D.shape[0]
    sym = bool((D == D.T).all())
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", threads)
    h.setOptionValue("random_seed", seed)
    inf = highspy.kHighsInf
    if sym:
        E = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        E = [(i, j) for i in range(n) for j in range(n) if i != j]
    eidx = {e: k for k, e in enumerate(E)}
    m = len(E)
    h.addVars(m, np.zeros(m), np.ones(m))
    h.changeColsCost(m, np.arange(m, dtype=np.int32), np.array([D[i, j] for i, j in E], dtype=float))
    h.changeColsIntegrality(m, np.arange(m, dtype=np.int32),
                            np.array([highspy.HighsVarType.kInteger] * m))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    if sym:
        for v in range(n):
            idx = [eidx[(min(v, u), max(v, u))] for u in range(n) if u != v]
            h.addRow(2, 2, len(idx), np.array(idx, dtype=np.int32), np.ones(len(idx)))
    else:
        for v in range(n):
            out = [eidx[(v, u)] for u in range(n) if u != v]
            inn = [eidx[(u, v)] for u in range(n) if u != v]
            h.addRow(1, 1, n - 1, np.array(out, dtype=np.int32), np.ones(n - 1))
            h.addRow(1, 1, n - 1, np.array(inn, dtype=np.int32), np.ones(n - 1))
    lb, it = None, 0
    while True:
        left = tm.left()
        if left <= 1:
            break
        h.setOptionValue("time_limit", float(left))
        h.run()
        info = h.getInfo()
        st = h.getModelStatus()
        if info.primal_solution_status != 2:  # 実行可能解なし
            lb = info.mip_dual_bound if np.isfinite(info.mip_dual_bound) else lb
            break
        val = np.array(h.getSolution().col_value)
        chosen = [E[k] for k in range(m) if val[k] > 0.5]
        comps = _components(n, chosen)
        if st == highspy.HighsModelStatus.kOptimal:
            lb = int(round(info.objective_function_value))
        else:
            lb = max(lb or 0, int(np.ceil(info.mip_dual_bound - 1e-6)))
        if len(comps) == 1:
            tour, prev, c = [0], None, 0
            nb = {}
            for i, j in chosen:
                nb.setdefault(i, []).append(j)
                if sym:
                    nb.setdefault(j, []).append(i)
            while True:
                nxt = [u for u in nb[c] if u != prev] if sym else nb[c]
                prev, c = c, nxt[0]
                if c == 0:
                    break
                tour.append(c)
            if st == highspy.HighsModelStatus.kOptimal:
                return dict(obj=tour_cost(D, tour), lb=lb, solution=tour, iters=it)
            return dict(obj=tour_cost(D, tour), lb=lb, solution=tour, iters=it)
        if st != highspy.HighsModelStatus.kOptimal:
            break
        for comp in comps:  # 各成分 S について sum_{i,j in S} x_ij <= |S|-1
            S = set(comp)
            if sym:
                idx = [eidx[(i, j)] for i in comp for j in comp if i < j]
            else:
                idx = [eidx[(i, j)] for i in comp for j in comp if i != j]
            h.addRow(-inf, len(S) - 1, len(idx), np.array(idx, dtype=np.int32), np.ones(len(idx)))
        it += 1
    return dict(obj=None, lb=lb, solution=None, iters=it)

