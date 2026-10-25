"""ATSP-TW の解法群. 各関数は (inst, time_limit, seed) -> dict(obj, lb, solution=デポから始まる訪問順).
inst は problems/atsptw.py の説明どおり dict(D=移動時間行列, r=時間枠の開始, d=時間枠の終了).
時間枠を守れる巡回路が見つからなかったときは obj=None を返す."""
import os
import tempfile

import numpy as np
from numba import njit

from common import Timer
from problems.atsptw import schedule
from registry import solver

W_LATE = 1_000_000  # 局所探索の評価値 = 遅れの合計 × W_LATE + 総移動時間（遅れの解消を最優先）


def _result(inst, tour, lb=None, **extra):
    """訪問順を返り値の dict にする. 時間枠を守れていなければ obj=None."""
    tour = [int(x) for x in tour]
    cost, late = schedule(inst, tour)
    if late > 0:
        return dict(obj=None, lb=lb, solution=None, late=late, **extra)
    return dict(obj=cost, lb=lb, solution=tour, **extra)


def _usable_arcs(D, r, d):
    """時間枠の上で通りうる弧 (i, j). 最も早く i を出ても d_j に間に合わない弧は除く."""
    n = D.shape[0]
    return [(i, j) for i in range(n) for j in range(n) if i != j and r[i] + D[i, j] <= d[j]]


# ---------------------------------------------------------------- 局所探索（遅れの合計と総移動時間を辞書式に最小化）
@njit(cache=True)
def _prefix(D, r, d, t, T, C, LT):
    """位置 k までの開始時刻 T[k]・移動時間の累計 C[k]・遅れの累計 LT[k] を求め、巡回路全体の評価値を返す.
    添字 n はデポへ戻ったところ（配列の長さは n+1）."""
    n = t.shape[0]
    T[0], C[0], LT[0] = r[t[0]], 0, 0
    for k in range(1, n + 1):
        a, b = t[k - 1], t[k % n]
        T[k] = max(T[k - 1] + D[a, b], r[b])
        C[k] = C[k - 1] + D[a, b]
        LT[k] = LT[k - 1] + max(0, T[k] - d[b])
    return LT[n] * W_LATE + C[n]


@njit(cache=True)
def _eval_move(D, r, d, t, p, a0, b0, a1, b1, a2, b2, q, T, C, LT, cap):
    """位置 p から先を t[a0..b0], t[a1..b1], t[a2..b2], t[q..] の順に並べ替えた巡回路の評価値.
    元の巡回路と同じ位置・同じ開始時刻に戻ったら、そこから先は元と同じなので計算を打ち切る.
    途中で cap 以上になったら cap を返す."""
    n = t.shape[0]
    tm, cost, late = T[p - 1], C[p - 1], LT[p - 1]
    prev = t[p - 1]
    for seg in range(3):
        a, b = (a0, b0) if seg == 0 else ((a1, b1) if seg == 1 else (a2, b2))
        for k in range(a, b + 1):
            v = t[k]
            tm = max(tm + D[prev, v], r[v])
            cost += D[prev, v]
            late += max(0, tm - d[v])
            if late * W_LATE + cost >= cap:
                return cap
            prev = v
    for k in range(q, n + 1):
        v = t[k % n]
        tm = max(tm + D[prev, v], r[v])
        cost += D[prev, v]
        late += max(0, tm - d[v])
        if tm == T[k]:
            return (late + LT[n] - LT[k]) * W_LATE + cost + C[n] - C[k]
        if late * W_LATE + cost >= cap:
            return cap
        prev = v
    return late * W_LATE + cost


@njit(cache=True)
def _or_opt(t, i, L, j, out):
    """区間 t[i..i+L-1] を位置 j の直後へ（向きを保って）移した巡回路を out に書く. j は区間とその直前以外."""
    n = t.shape[0]
    m = 0
    for k in range(n):
        if i <= k < i + L:
            continue
        out[m] = t[k]
        m += 1
        if k == j:
            for q in range(L):
                out[m] = t[i + q]
                m += 1


@njit(cache=True)
def _local_search(D, r, d, t):
    """先着改善の Or-opt（長さ 1〜3 の区間の移動）と 2 都市の交換. デポは先頭に固定."""
    n = t.shape[0]
    T = np.empty(n + 1, np.int64)
    C = np.empty(n + 1, np.int64)
    LT = np.empty(n + 1, np.int64)
    c = np.empty(n, np.int64)
    cur = _prefix(D, r, d, t, T, C, LT)
    improved = True
    while improved:
        improved = False
        for L in range(1, 4):
            for i in range(1, n - L + 1):
                for j in range(n):
                    if i - 1 <= j < i + L:
                        continue
                    if j >= i + L:  # 後ろへ移す: t[i+L..j], 区間, t[j+1..]
                        s = _eval_move(D, r, d, t, i, i + L, j, i, i + L - 1, 0, -1, j + 1, T, C, LT, cur)
                    else:  # 前へ移す: t[..j], 区間, t[j+1..i-1], t[i+L..]
                        s = _eval_move(D, r, d, t, j + 1, i, i + L - 1, j + 1, i - 1, 0, -1, i + L,
                                       T, C, LT, cur)
                    if s < cur:
                        _or_opt(t, i, L, j, c)
                        t[:] = c
                        cur = _prefix(D, r, d, t, T, C, LT)
                        improved = True
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                s = _eval_move(D, r, d, t, i, j, j, i + 1, j - 1, i, i, j + 1, T, C, LT, cur)
                if s < cur:
                    t[i], t[j] = t[j], t[i]
                    cur = _prefix(D, r, d, t, T, C, LT)
                    improved = True
    return cur


def _edd_tour(r, d):
    """デポのあとに時間枠の終了が早い順（同じなら開始が早い順）に並べた巡回路."""
    rest = sorted(range(1, len(d)), key=lambda i: (d[i], r[i]))
    return np.array([0] + rest, dtype=np.int64)


@solver("EDD+LS", problems="atsptw",
        description=("時間枠の締め切りが早い順（Earliest Due Date）に並べた巡回路から始め、Or-opt（1〜3 地点の付け替え）と 2 地点の交換で、まず遅れの合計を、次に総移動時間を減らす局所探索。"
                     "決定的で一瞬で終わる、実務でまず試す素朴な方法。"))
def edd_local_search(inst, tl, seed):
    """締め切り順に並べてから局所探索. 決定的で速い、実務でまず試す素朴な方法."""
    D, r, d = inst["D"], inst["r"], inst["d"]
    t = _edd_tour(r, d)
    _local_search(D, r, d, t)
    return _result(inst, t)


# ---------------------------------------------------------------- 一般化変近傍探索（da Silva & Urrutia 2010 にならう）
@njit(cache=True)
def _vns_chunk(D, r, d, t, cur, best_t, best, k, kmax, iters, seed):
    """揺さぶり（ランダムな 1 都市移動を k 回）→ 局所探索 → 改善なら受理して k=1、だめなら k+1."""
    np.random.seed(seed)
    n = t.shape[0]
    x = np.empty(n, np.int64)
    c = np.empty(n, np.int64)
    for _ in range(iters):
        x[:] = t
        for _s in range(k):
            i = np.random.randint(1, n)
            j = np.random.randint(0, n - 1)
            if i - 1 <= j < i + 1:
                continue
            _or_opt(x, i, 1, j, c)
            x[:] = c
        s = _local_search(D, r, d, x)
        if s < cur:
            t[:] = x
            cur = s
            k = 1
            if s < best:
                best = s
                best_t[:] = x
        else:
            k = k + 1 if k < kmax else 1
    return cur, best, k


@solver("GVNS", problems="atsptw", seeds=3,
        description=("一般化変近傍探索（General Variable Neighborhood Search）。"
                     "いまの解をランダムに k 回崩してから局所探索をかけ、良くなれば受け入れて k=1 に戻し、だめなら k を増やしてより大きく崩す。"
                     "da Silva & Urrutia (2010) にならった自作版。"))
def gvns(inst, tl, seed):
    D, r, d = inst["D"], inst["r"], inst["d"]
    tm = Timer(tl)
    t = _edd_tour(r, d)
    cur = _local_search(D, r, d, t)
    best_t, best = t.copy(), cur
    it, k = 0, 1
    while tm.left() > 0:
        t0 = tm.elapsed()
        cur, best, k = _vns_chunk(D, r, d, t, cur, best_t, best, k, 8, 1, seed * 100003 + it)
        it += 1
        if tm.left() < tm.elapsed() - t0:  # 次の 1 回が制限時間を超えそうなら止める
            break
    return _result(inst, best_t, iters=it)


# ---------------------------------------------------------------- LKH-3（elkai 同梱版. TYPE: TSPTW）
@solver("LKH-3(elkai)", problems="atsptw", seeds=3, packages=["elkai"],
        description=("LKH の後継で、制約付きの巡回路問題を扱える LKH-3 を時間枠付き巡回セールスマン問題（TSPTW）として呼ぶ。"
                     "時間枠の違反をペナルティとして扱いながら LK 型の局所探索を行い、制限時間いっぱい試行を繰り返す。"))
def lkh3(inst, tl, seed):
    """elkai に同梱の LKH-3.0.8 を TSPTW として呼ぶ. 1 run の試行回数は既定（都市数）のまま、
    制限時間いっぱいまで run を繰り返す（runs=10 で止めると、時間枠を守れる解が出ないまま終わることがあるため）."""
    from elkai import _elkai
    D, r, d = inst["D"], inst["r"], inst["d"]
    n = D.shape[0]
    lines = [f"TYPE : TSPTW", f"DIMENSION : {n}", "EDGE_WEIGHT_TYPE : EXPLICIT",
             "EDGE_WEIGHT_FORMAT : FULL_MATRIX", "EDGE_WEIGHT_SECTION"]
    lines += [" ".join(map(str, row)) for row in D.tolist()]
    lines += ["TIME_WINDOW_SECTION"] + [f"{i + 1} {r[i]} {d[i]}" for i in range(n)]
    lines += ["DEPOT_SECTION", "1", "-1", "EOF", ""]
    # elkai は問題を標準入力（:stdin:）から読むと時間枠を読めないので、一時ファイルを渡す.
    # LKH はパラメータ中の空白でファイル名を区切るので、カレントディレクトリに置いて相対パスで渡す
    fd, path = tempfile.mkstemp(prefix="_lkh3_", suffix=".tsptw", dir=".")
    path = os.path.basename(path)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines))
        params = (f"PROBLEM_FILE = {path}\nRUNS = 1000000\nSEED = {seed}\n"
                  f"TIME_LIMIT = {tl:.1f}\nTOTAL_TIME_LIMIT = {tl:.1f}\n")
        tour = _elkai.solve_problem(params, "")
    finally:
        os.remove(path)
    if not tour:  # elkai は時間枠を守れる巡回路が見つからないと空のリストを返す
        return dict(obj=None, lb=None, solution=None)
    tour = [x - 1 for x in tour]
    s = tour.index(0)
    return _result(inst, tour[s:] + tour[:s])


# ---------------------------------------------------------------- OR-Tools Routing（時間の次元 + 誘導局所探索）
@solver("ORTools-GLS", problems="atsptw", packages=["ortools"],
        description=("OR-Tools の配送計画ソルバー（Routing）に「時刻」の次元を加えて、各地点の到着時刻を時間枠に収める。"
                     "挿入法で初期解を作り、誘導局所探索（GLS）で改善する。"))
def ortools_gls(inst, tl, seed):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    D, r, d = inst["D"], inst["r"], inst["d"]
    n = D.shape[0]
    mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    rt = pywrapcp.RoutingModel(mgr)
    cb = rt.RegisterTransitMatrix(D.tolist())
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    horizon = int(d.max())
    rt.AddDimension(cb, horizon, horizon, False, "time")  # 待ち（スラック）を許す
    time = rt.GetDimensionOrDie("time")
    for i in range(1, n):
        time.CumulVar(mgr.NodeToIndex(i)).SetRange(int(r[i]), int(d[i]))
    time.CumulVar(rt.Start(0)).SetRange(int(r[0]), int(r[0]))
    time.CumulVar(rt.End(0)).SetRange(int(r[0]), int(d[0]))
    p = pywrapcp.DefaultRoutingSearchParameters()
    # TSP/ATSP で使う PATH_CHEAPEST_ARC は時間枠が厳しいと 19 都市でも初期解を作れないので、挿入法にする
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(tl * 1000))
    sol = rt.SolveWithParameters(p)
    if sol is None:
        return dict(obj=None, lb=None, solution=None)
    idx, tour = rt.Start(0), []
    while not rt.IsEnd(idx):
        tour.append(mgr.IndexToNode(idx))
        idx = sol.Value(rt.NextVar(idx))
    return _result(inst, tour)


# ---------------------------------------------------------------- 制約プログラミング（CP-SAT, AddCircuit + 時刻変数）
@solver("CP-SAT", problems="atsptw", threads=[1, 4], packages=["ortools"],
        description=("CP-SAT で、弧の 0-1 変数と AddCircuit 制約で巡回路を表し、各地点の開始時刻の整数変数に「弧を使うなら次の地点は移動時間ぶん後に始まる」条件を付けたモデル。"
                     "解と同時に下界を出し、最適性を証明できる。"))
def cpsat(inst, tl, seed, threads=1):
    from ortools.sat.python import cp_model
    D, r, d = inst["D"], inst["r"], inst["d"]
    n = D.shape[0]
    m = cp_model.CpModel()
    T = [None] + [m.NewIntVar(int(r[i]), int(d[i]), f"T{i}") for i in range(1, n)]
    x = {}
    for i, j in _usable_arcs(D, r, d):
        v = x[i, j] = m.NewBoolVar("")
        if i == 0:
            m.Add(T[j] >= int(r[0] + D[0, j])).OnlyEnforceIf(v)
        elif j == 0:
            m.Add(T[i] + int(D[i, 0]) <= int(d[0])).OnlyEnforceIf(v)
        else:
            m.Add(T[j] >= T[i] + int(D[i, j])).OnlyEnforceIf(v)
    m.AddCircuit([(i, j, v) for (i, j), v in x.items()])
    m.Minimize(sum(int(D[i, j]) * v for (i, j), v in x.items()))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = threads
    s.parameters.random_seed = seed
    st = s.Solve(m)
    lb = int(np.ceil(s.BestObjectiveBound() - 1e-6)) if st != cp_model.INFEASIBLE else None
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(obj=None, lb=lb, solution=None)
    nxt = {i: j for (i, j), v in x.items() if s.Value(v)}
    tour, c = [0], nxt[0]
    while c != 0:
        tour.append(c)
        c = nxt[c]
    return _result(inst, tour, lb=lb, optimal=st == cp_model.OPTIMAL)


# ---------------------------------------------------------------- MIP（HiGHS, 時刻変数の big-M 定式化）
@solver("MIP-HiGHS", problems="atsptw", threads=[1, 4], packages=["highspy"],
        description=("弧の 0-1 変数と各地点の開始時刻の連続変数を、big-M 制約（弧を使うときだけ時刻の順序を強制する）で結んだ定式化を HiGHS で解く。"
                     "時刻の制約が部分巡回路も防ぐ。"
                     "big-M は緩和が弱く、大きな問題は苦手。"))
def mip_bigm(inst, tl, seed, threads=1):
    """x_ij（弧を通るか）と T_i（都市 i の開始時刻）. 入次数・出次数 1 と
    T_j >= T_i + D_ij - M_ij (1 - x_ij)（M_ij = d_i + D_ij - r_j）. 時刻の制約が部分巡回路も除く."""
    import highspy
    D, r, d = inst["D"], inst["r"], inst["d"]
    n = D.shape[0]
    arcs = _usable_arcs(D, r, d)
    m = len(arcs)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", threads)
    h.setOptionValue("random_seed", seed)
    h.setOptionValue("time_limit", float(tl))
    inf = highspy.kHighsInf
    # 列: x_0..x_{m-1}（0-1）, T_1..T_{n-1}（連続）
    h.addVars(m, np.zeros(m), np.ones(m))
    h.changeColsIntegrality(m, np.arange(m, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * m))
    h.changeColsCost(m, np.arange(m, dtype=np.int32), np.array([D[i, j] for i, j in arcs], dtype=float))
    h.addVars(n - 1, r[1:].astype(float), d[1:].astype(float))
    tcol = lambda i: m + i - 1
    outs, ins = [[] for _ in range(n)], [[] for _ in range(n)]
    for k, (i, j) in enumerate(arcs):
        outs[i].append(k)
        ins[j].append(k)
    for v in range(n):
        for idx in (outs[v], ins[v]):
            h.addRow(1, 1, len(idx), np.array(idx, dtype=np.int32), np.ones(len(idx)))
    for k, (i, j) in enumerate(arcs):
        if i == 0:  # T_j >= r_j + (r_0 + D_0j - r_j) x_0j
            a = r[0] + D[0, j] - r[j]
            if a > 0:
                h.addRow(r[j], inf, 2, np.array([tcol(j), k], dtype=np.int32), np.array([1.0, -a]))
        elif j == 0:  # T_i + D_i0 <= d_0 + M (1 - x_i0),  M = d_i + D_i0 - d_0
            M = d[i] + D[i, 0] - d[0]
            if M > 0:
                h.addRow(-inf, d[0] - D[i, 0] + M, 2, np.array([tcol(i), k], dtype=np.int32), np.array([1.0, M]))
        else:  # T_j - T_i - M x_ij >= D_ij - M
            M = d[i] + D[i, j] - r[j]
            if M > 0:
                h.addRow(D[i, j] - M, inf, 3, np.array([tcol(j), tcol(i), k], dtype=np.int32),
                         np.array([1.0, -1.0, -float(M)]))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    h.run()
    info = h.getInfo()
    st = h.getModelStatus()
    lb = int(np.ceil(info.mip_dual_bound - 1e-6)) if np.isfinite(info.mip_dual_bound) else None
    if info.primal_solution_status != 2:
        return dict(obj=None, lb=lb, solution=None)
    val = np.array(h.getSolution().col_value)
    nxt = {i: j for k, (i, j) in enumerate(arcs) if val[k] > 0.5}
    tour, c = [0], nxt[0]
    while c != 0 and len(tour) <= n:
        tour.append(c)
        c = nxt[c]
    return _result(inst, tour, lb=lb, optimal=st == highspy.HighsModelStatus.kOptimal)
