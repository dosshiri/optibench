"""VRPTW の解法群. 各関数は (inst, time_limit, seed) -> dict(obj, lb, solution=経路のリスト).
inst は problems/vrptw.py の説明どおりの dict（距離・時刻は 10 倍の整数）. 目的関数値は総距離を 10 で割った値.
台数・容量・時間枠を守れる解が見つからなかったときは obj=None を返す."""
import os
import tempfile

import numpy as np
from numba import njit

from common import Timer
from problems.vrptw import evaluate
from registry import solver


def _result(inst, routes, lb=None, **extra):
    """経路のリストを返り値の dict にする. 制約を守れていなければ obj=None. lb は 10 倍の整数で受け取る."""
    routes = [[int(v) for v in r] for r in routes if len(r)]
    lb = None if lb is None else round(lb / 10, 1)
    obj = evaluate(inst, routes)
    if obj is None:
        return dict(obj=None, lb=lb, solution=None, **extra)
    return dict(obj=obj, lb=lb, solution=routes, **extra)


def _usable_arcs(inst):
    """時間枠と容量の上で通りうる弧 (i, j). 最も早く i で作業を始めても l_j に間に合わない弧と、
    2 顧客の需要の和が容量を超える弧は除く."""
    D, e, l, s, q, Q = inst["D"], inst["e"], inst["l"], inst["s"], inst["q"], inst["Q"]
    n = D.shape[0]
    return [(i, j) for i in range(n) for j in range(n)
            if i != j and e[i] + s[i] + D[i, j] <= l[j] and q[i] + q[j] <= Q]


# ---------------------------------------------------------------- 経路の表現と挿入（自作手法の共通部品, numba）
# path[r, 0..ln[r]+1] が経路 r（両端はデポ 0）. S は各位置の作業開始時刻, Lt は「そこから先を守れる最も遅い開始時刻」,
# Ld は先頭からの積載量の累計. 挿入の可否は前後の S と Lt だけで O(1) で判定できる.
@njit(cache=True)
def _update(D, e, l, s, q, path, ln, S, Lt, Ld, r):
    """経路 r の S・Lt・Ld を計算し、総距離を返す."""
    m = ln[r]
    S[r, 0] = e[0]
    Ld[r, 0] = 0
    dist = 0
    for k in range(1, m + 2):
        a, b = path[r, k - 1], path[r, k]
        S[r, k] = max(S[r, k - 1] + s[a] + D[a, b], e[b])
        Ld[r, k] = Ld[r, k - 1] + q[b]
        dist += D[a, b]
    Lt[r, m + 1] = l[0]
    for k in range(m, -1, -1):
        a, b = path[r, k], path[r, k + 1]
        Lt[r, k] = min(l[a], Lt[r, k + 1] - D[a, b] - s[a])
    return dist


@njit(cache=True)
def _insert_cost(D, e, l, s, q, Q, path, ln, S, Lt, Ld, r, k, u):
    """顧客 u を経路 r の位置 k-1 と k の間に入れたときの距離の増分. 入れられなければ -1."""
    if Ld[r, ln[r] + 1] + q[u] > Q:
        return -1
    a, b = path[r, k - 1], path[r, k]
    t = max(S[r, k - 1] + s[a] + D[a, u], e[u])
    if t > l[u]:
        return -1
    if max(t + s[u] + D[u, b], e[b]) > Lt[r, k]:
        return -1
    return D[a, u] + D[u, b] - D[a, b]


@njit(cache=True)
def _insert(path, ln, r, k, u):
    m = ln[r]
    for p in range(m + 1, k - 1, -1):
        path[r, p + 1] = path[r, p]
    path[r, k] = u
    ln[r] = m + 1


@njit(cache=True)
def _remove(path, ln, r, k):
    m = ln[r]
    for p in range(k, m + 1):
        path[r, p] = path[r, p + 1]
    ln[r] = m - 1


@njit(cache=True)
def _compact(D, e, l, s, q, path, ln, S, Lt, Ld, rc, nr):
    """空になった経路を詰めて、経路数を返す."""
    w = 0
    for r in range(nr):
        if ln[r] > 0:
            if w != r:
                path[w, :] = path[r, :]
                ln[w] = ln[r]
            w += 1
    for r in range(w):
        rc[r] = _update(D, e, l, s, q, path, ln, S, Lt, Ld, r)
    return w


@njit(cache=True)
def _recreate(D, e, l, s, q, Q, K, path, ln, S, Lt, Ld, rc, nr, order, blink):
    """order の顧客を 1 人ずつ、距離の増分が最小の位置へ挿入する（貪欲挿入）. 各位置を確率 blink で見送る.
    どこにも入らなければ新しい経路を開く. 台数の上限を超えたら -1 を返す."""
    for u in order:
        best, br, bk = 1 << 60, -1, -1
        for r in range(nr):
            for k in range(1, ln[r] + 2):
                if blink > 0 and np.random.random() < blink:
                    continue
                c = _insert_cost(D, e, l, s, q, Q, path, ln, S, Lt, Ld, r, k, u)
                if c >= 0 and c < best:
                    best, br, bk = c, r, k
        if br < 0:
            if nr >= K:
                return -1
            br, bk = nr, 1
            ln[br] = 0
            path[br, 0] = 0
            path[br, 1] = 0
            nr += 1
        _insert(path, ln, br, bk, u)
        rc[br] = _update(D, e, l, s, q, path, ln, S, Lt, Ld, br)
    return nr


# ---------------------------------------------------------------- 局所探索（Relocate と 2-opt*）
@njit(cache=True)
def _relocate(D, e, l, s, q, Q, path, ln, S, Lt, Ld, rc, nr):
    """顧客を 1 人ずつ抜き出し、より距離の短くなる位置（別の経路を含む）へ移す. 改善したら True."""
    n = D.shape[0]
    improved = False
    for u in range(1, n):
        r0, k0 = -1, -1
        for r in range(nr):
            for k in range(1, ln[r] + 1):
                if path[r, k] == u:
                    r0, k0 = r, k
        if r0 < 0:
            continue
        a, b = path[r0, k0 - 1], path[r0, k0 + 1]
        gain = D[a, u] + D[u, b] - D[a, b]
        old = rc[r0]
        _remove(path, ln, r0, k0)
        rc[r0] = _update(D, e, l, s, q, path, ln, S, Lt, Ld, r0)
        best, br, bk = gain, r0, k0
        for r in range(nr):
            if ln[r] == 0 and r != r0:
                continue
            for k in range(1, ln[r] + 2):
                c = _insert_cost(D, e, l, s, q, Q, path, ln, S, Lt, Ld, r, k, u)
                if c >= 0 and c < best:
                    best, br, bk = c, r, k
        if br == r0 and bk == k0:
            _insert(path, ln, r0, k0, u)
            rc[r0] = old
            _update(D, e, l, s, q, path, ln, S, Lt, Ld, r0)
        else:
            _insert(path, ln, br, bk, u)
            rc[br] = _update(D, e, l, s, q, path, ln, S, Lt, Ld, br)
            improved = True
    return improved


@njit(cache=True)
def _two_opt_star(D, e, l, s, q, Q, path, ln, S, Lt, Ld, rc, nr, tmp):
    """2 本の経路の後半どうしを入れ替える（a の位置 i の後ろと b の位置 j の後ろを交換）. 改善したら True."""
    improved = False
    for ra in range(nr):
        for rb in range(ra + 1, nr):
            done = False
            for i in range(0, ln[ra] + 1):
                if done:
                    break
                x, x2 = path[ra, i], path[ra, i + 1]
                for j in range(0, ln[rb] + 1):
                    y, y2 = path[rb, j], path[rb, j + 1]
                    delta = D[x, y2] + D[y, x2] - D[x, x2] - D[y, y2]
                    if delta >= 0:
                        continue
                    # 容量: a の前半 + b の後半, b の前半 + a の後半
                    if Ld[ra, i] + Ld[rb, ln[rb] + 1] - Ld[rb, j] > Q:
                        continue
                    if Ld[rb, j] + Ld[ra, ln[ra] + 1] - Ld[ra, i] > Q:
                        continue
                    # 時間枠: つなぎ目の開始時刻が、後半の Lt を超えないか
                    if max(S[ra, i] + s[x] + D[x, y2], e[y2]) > Lt[rb, j + 1]:
                        continue
                    if max(S[rb, j] + s[y] + D[y, x2], e[x2]) > Lt[ra, i + 1]:
                        continue
                    # 交換する. tmp に a の後半を退避
                    la, lb = ln[ra], ln[rb]
                    na = la + 2 - (i + 1)
                    for p in range(na):
                        tmp[p] = path[ra, i + 1 + p]
                    for p in range(j + 1, lb + 2):
                        path[ra, i + 1 + p - (j + 1)] = path[rb, p]
                    for p in range(na):
                        path[rb, j + 1 + p] = tmp[p]
                    ln[ra] = i + (lb + 1 - j) - 1
                    ln[rb] = j + na - 1
                    rc[ra] = _update(D, e, l, s, q, path, ln, S, Lt, Ld, ra)
                    rc[rb] = _update(D, e, l, s, q, path, ln, S, Lt, Ld, rb)
                    improved = done = True
                    break
    return improved


@njit(cache=True)
def _local_search(D, e, l, s, q, Q, path, ln, S, Lt, Ld, rc, nr):
    tmp = np.empty(path.shape[1], np.int64)
    while True:
        a = _relocate(D, e, l, s, q, Q, path, ln, S, Lt, Ld, rc, nr)
        nr = _compact(D, e, l, s, q, path, ln, S, Lt, Ld, rc, nr)
        b = _two_opt_star(D, e, l, s, q, Q, path, ln, S, Lt, Ld, rc, nr, tmp)
        nr = _compact(D, e, l, s, q, path, ln, S, Lt, Ld, rc, nr)
        if not (a or b):
            return nr


def _arrays(inst):
    n = inst["D"].shape[0]
    shape = (n, n + 2)  # 経路は最大 n-1 本, 1 本は最大 n+1 地点
    return (np.zeros(shape, np.int64), np.zeros(n, np.int64), np.zeros(shape, np.int64),
            np.zeros(shape, np.int64), np.zeros(shape, np.int64), np.zeros(n, np.int64))


def _routes(path, ln, nr):
    return [path[r, 1:ln[r] + 1].tolist() for r in range(nr)]


def _args(inst):
    return inst["D"], inst["e"], inst["l"], inst["s"], inst["q"], inst["Q"]


@solver("Insertion+LS", problems="vrptw",
        description=("デポから遠い顧客から順に、距離の増え方が最も小さく時間枠と容量を守れる位置へ挿入して経路を作る構築法（入らなければ新しい車両を出す）。"
                     "続けて、顧客 1 人の移動（Relocate）と 2 本の経路の後半の交換（2-opt*）で、改善できなくなるまで局所探索する。"
                     "決定的で一瞬で終わる、実務でまず試す素朴な方法。"))
def insertion_ls(inst, tl, seed):
    D, e, l, s, q, Q = _args(inst)
    path, ln, S, Lt, Ld, rc = _arrays(inst)
    order = np.argsort(-D[0, 1:], kind="stable") + 1
    nr = _recreate(D, e, l, s, q, Q, D.shape[0] - 1, path, ln, S, Lt, Ld, rc, 0, order, 0.0)
    nr = _local_search(D, e, l, s, q, Q, path, ln, S, Lt, Ld, rc, nr)
    return _result(inst, _routes(path, ln, nr), n_routes=nr)


# ---------------------------------------------------------------- SISR（Christiaens & Vanden Berghe 2020 にならう）
@njit(cache=True)
def _copy(path, ln, S, Lt, Ld, rc, path2, ln2, S2, Lt2, Ld2, rc2, nr):
    path2[:nr] = path[:nr]
    ln2[:nr] = ln[:nr]
    S2[:nr] = S[:nr]
    Lt2[:nr] = Lt[:nr]
    Ld2[:nr] = Ld[:nr]
    rc2[:nr] = rc[:nr]


@njit(cache=True)
def _sisr_chunk(D, e, l, s, q, Q, K, adj, path, ln, S, Lt, Ld, rc, nr, cur, best, bp, bln, bnr,
                T0, T1, frac0, frac1, iters, seed, cbar, lmax):
    """崩す（近くの顧客の連なりを数本の経路から抜く）→ 直す（貪欲挿入, 見送り付き）→ 焼きなましの基準で受理, を iters 回.
    温度は制限時間の経過割合 frac0..frac1 に合わせて T0 から T1 へ指数的に下げる."""
    np.random.seed(seed)
    n = D.shape[0]
    cp, cln, cS, cLt, cLd, crc = path.copy(), ln.copy(), S.copy(), Lt.copy(), Ld.copy(), rc.copy()
    removed = np.empty(n, np.int64)
    ruined = np.zeros(n, np.bool_)
    where = np.empty(n, np.int64)
    for it in range(iters):
        frac = frac0 + (frac1 - frac0) * it / iters
        T = T0 * (T1 / T0) ** frac
        _copy(path, ln, S, Lt, Ld, rc, cp, cln, cS, cLt, cLd, crc, nr)
        cnr = nr
        # 崩す
        for r in range(cnr):
            for k in range(1, cln[r] + 1):
                where[cp[r, k]] = r
        tot = 0
        for r in range(cnr):
            tot += cln[r]
        ls_max = min(lmax, tot / max(cnr, 1))
        ks_max = 4.0 * cbar / (1.0 + ls_max) - 1.0
        ks = int(np.random.random() * ks_max) + 1
        nrem = 0
        ruined[:] = False
        seed_c = np.random.randint(1, n)
        for c in adj[seed_c]:
            if nrem >= n - 1 or ks <= 0:
                break
            r = where[c]
            if r < 0 or ruined[r]:
                continue
            m = cln[r]
            lt = int(np.random.random() * min(m, ls_max)) + 1
            pc = 0
            for k in range(1, m + 1):
                if cp[r, k] == c:
                    pc = k
            # c を含む長さ lt の連なり（開始位置をランダムに）
            lo = max(1, pc - lt + 1)
            hi = min(pc, m - lt + 1)
            st = lo + np.random.randint(0, hi - lo + 1)
            for _ in range(lt):
                v = cp[r, st]
                removed[nrem] = v
                nrem += 1
                where[v] = -1
                _remove(cp, cln, r, st)
            ruined[r] = True
            ks -= 1
        for r in range(cnr):
            if ruined[r]:
                crc[r] = _update(D, e, l, s, q, cp, cln, cS, cLt, cLd, r)
        cnr = _compact(D, e, l, s, q, cp, cln, cS, cLt, cLd, crc, cnr)
        # 直す: 並べ方をランダム（4）/ 需要の大きい順（4）/ デポから遠い順（2）/ 近い順（1）から選ぶ
        rem = removed[:nrem].copy()
        np.random.shuffle(rem)
        x = np.random.random() * 11
        if x < 4:
            pass
        elif x < 8:
            rem = rem[np.argsort(-q[rem], kind="mergesort")]
        elif x < 10:
            rem = rem[np.argsort(-D[0, rem], kind="mergesort")]
        else:
            rem = rem[np.argsort(D[0, rem], kind="mergesort")]
        cnr = _recreate(D, e, l, s, q, Q, K, cp, cln, cS, cLt, cLd, crc, cnr, rem, 0.01)
        if cnr < 0:
            continue
        new = 0
        for r in range(cnr):
            new += crc[r]
        if new < cur - T * np.log(np.random.random()):
            _copy(cp, cln, cS, cLt, cLd, crc, path, ln, S, Lt, Ld, rc, cnr)
            nr, cur = cnr, new
            if new < best and nr <= K:
                best = new
                bp[:nr] = path[:nr]
                bln[:nr] = ln[:nr]
                bnr = nr
    return nr, cur, best, bnr


@solver("SISR", problems="vrptw", seeds=3,
        description=("焼きなまし法の受理基準を使う「壊して直す」型の探索（ruin and recreate）で、Christiaens & Vanden Berghe (2020) の SISR にならった自作版。"
                     "近くにいる顧客の連なりを数本の経路からまとめて抜き、抜いた顧客を見送りを交えた貪欲挿入で入れ直す。"
                     "温度は制限時間に合わせて 100 から 1 まで下げる（元論文の分割した連なりの抜き方と、台数を減らす手順は省いた）。"))
def sisr(inst, tl, seed):
    D, e, l, s, q, Q = _args(inst)
    K = inst["K"]
    n = D.shape[0]
    tm = Timer(tl)
    path, ln, S, Lt, Ld, rc = _arrays(inst)
    np.random.seed(seed)
    order = np.argsort(-D[0, 1:], kind="stable") + 1
    nr = _recreate(D, e, l, s, q, Q, n - 1, path, ln, S, Lt, Ld, rc, 0, order, 0.0)
    cur = int(rc[:nr].sum())
    bp, bln, bnr, best = path.copy(), ln.copy(), nr, cur if nr <= K else 1 << 60
    # 各顧客から近い順の顧客の一覧（先頭は自分自身. デポは除く）
    adj = np.array([row[row != 0] for row in np.argsort(D, axis=1, kind="stable")], dtype=np.int64)
    it, chunk = 0, 50
    while tm.left() > 0:
        t0 = tm.elapsed()
        f0 = min(1.0, t0 / tl)
        f1 = min(1.0, f0 + 0.01)
        # 温度は元論文の 100 → 1（距離の単位）を、10 倍の整数距離に合わせて 1000 → 10 にする
        res = _sisr_chunk(D, e, l, s, q, Q, K, adj, path, ln, S, Lt, Ld, rc, nr, cur, best,
                          bp, bln, bnr, 1000.0, 10.0, f0, f1, chunk, seed * 1000003 + it, 10.0, 10.0)
        nr, cur, best, bnr = res
        it += chunk
        dt = tm.elapsed() - t0
        if dt < 0.05:
            chunk *= 2
        if tm.left() < dt:  # 次の 1 回が制限時間を超えそうなら止める
            break
    if best >= 1 << 60:
        return dict(obj=None, lb=None, solution=None, iters=it)
    return _result(inst, _routes(bp, bln, bnr), iters=it)


# ---------------------------------------------------------------- LKH-3（elkai 同梱版. TYPE: CVRPTW）
@solver("LKH-3(elkai)", problems="vrptw", seeds=3, packages=["elkai"],
        description=("LKH の後継で、制約付きの巡回路問題を扱える LKH-3 を、容量と時間枠付きの配送計画（CVRPTW）として呼ぶ。"
                     "車両の台数ぶんデポを複製した 1 本の巡回路として扱い、容量・時間枠の違反をペナルティにして LK 型の局所探索を行う。"
                     "制限時間いっぱい試行を繰り返す。"))
def lkh3(inst, tl, seed):
    from elkai import _elkai
    D, e, l, s, q = inst["D"], inst["e"], inst["l"], inst["s"], inst["q"]
    n, K = D.shape[0], inst["K"]
    lines = ["TYPE : CVRPTW", f"DIMENSION : {n}", f"VEHICLES : {K}", f"CAPACITY : {inst['Q']}",
             "EDGE_WEIGHT_TYPE : EXPLICIT", "EDGE_WEIGHT_FORMAT : FULL_MATRIX", "EDGE_WEIGHT_SECTION"]
    lines += [" ".join(map(str, row)) for row in D.tolist()]
    lines += ["DEMAND_SECTION"] + [f"{i + 1} {q[i]}" for i in range(n)]
    lines += ["TIME_WINDOW_SECTION"] + [f"{i + 1} {e[i]} {l[i]}" for i in range(n)]
    lines += ["SERVICE_TIME_SECTION"] + [f"{i + 1} {s[i]}" for i in range(n)]
    lines += ["DEPOT_SECTION", "1", "-1", "EOF", ""]
    # elkai は標準入力（:stdin:）からは時間枠を読めないので一時ファイルを渡す（ATSP-TW と同じ. 相対パスで渡す）
    fd, path = tempfile.mkstemp(prefix="_lkh3_", suffix=".vrptw", dir=".")
    path = os.path.basename(path)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines))
        # MTSP_MIN_SIZE = 0: 使わない車両（空の経路）を許し、台数は LKH-3 に任せる
        params = (f"PROBLEM_FILE = {path}\nRUNS = 1000000\nSEED = {seed}\nMTSP_MIN_SIZE = 0\n"
                  f"TIME_LIMIT = {tl:.1f}\nTOTAL_TIME_LIMIT = {tl:.1f}\n")
        tour = _elkai.solve_problem(params, "")
    finally:
        os.remove(path)
    if not tour:
        return dict(obj=None, lb=None, solution=None)
    # 巡回路の中の n 以上の番号（1 始まりで n+1 以上）は複製したデポ. デポで区切って経路にする
    routes, cur = [], []
    for v in tour:
        v -= 1
        if v == 0 or v >= n:
            routes.append(cur)
            cur = []
        else:
            cur.append(v)
    routes.append(cur)
    return _result(inst, routes)


# ---------------------------------------------------------------- OR-Tools Routing（容量・時間の次元 + 誘導局所探索）
@solver("ORTools-GLS", problems="vrptw", packages=["ortools"],
        description=("OR-Tools の配送計画ソルバー（Routing）に、積載量と時刻の 2 つの次元を加えて容量と時間枠を表す。"
                     "挿入法で初期解を作り、誘導局所探索（GLS）で改善する。"))
def ortools_gls(inst, tl, seed):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    D, e, l, s, q = inst["D"], inst["e"], inst["l"], inst["s"], inst["q"]
    n, K = D.shape[0], inst["K"]
    mgr = pywrapcp.RoutingIndexManager(n, K, 0)
    rt = pywrapcp.RoutingModel(mgr)
    dist = rt.RegisterTransitMatrix(D.tolist())
    rt.SetArcCostEvaluatorOfAllVehicles(dist)
    rt.AddDimensionWithVehicleCapacity(rt.RegisterUnaryTransitVector(q.tolist()), 0, [inst["Q"]] * K, True, "load")
    T = (D + s[:, None]).tolist()  # i の作業時間 + i から j への移動時間
    horizon = int(l[0])
    rt.AddDimension(rt.RegisterTransitMatrix(T), horizon, horizon, False, "time")
    time = rt.GetDimensionOrDie("time")
    for i in range(1, n):
        time.CumulVar(mgr.NodeToIndex(i)).SetRange(int(e[i]), int(l[i]))
    for v in range(K):
        time.CumulVar(rt.Start(v)).SetRange(int(e[0]), int(l[0]))
        time.CumulVar(rt.End(v)).SetRange(int(e[0]), int(l[0]))
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.FromMilliseconds(int(tl * 1000))
    sol = rt.SolveWithParameters(p)
    if sol is None:
        return dict(obj=None, lb=None, solution=None)
    routes = []
    for v in range(K):
        idx, r = sol.Value(rt.NextVar(rt.Start(v))), []
        while not rt.IsEnd(idx):
            r.append(mgr.IndexToNode(idx))
            idx = sol.Value(rt.NextVar(idx))
        routes.append(r)
    return _result(inst, routes)


def _routes_from_arcs(n, used):
    """使われた弧 (i, j) から経路を組み立てる. デポ 0 から出る弧ごとに 1 本."""
    nxt = {}
    starts = []
    for i, j in used:
        if i == 0:
            starts.append(j)
        else:
            nxt[i] = j
    routes = []
    for c in starts:
        r = []
        while c != 0 and len(r) < n:
            r.append(c)
            c = nxt.get(c, 0)
        routes.append(r)
    return routes


# ---------------------------------------------------------------- 制約プログラミング（CP-SAT, AddMultipleCircuit + 時刻・積載量の変数）
@solver("CP-SAT", problems="vrptw", threads=[1, 4], packages=["ortools"],
        description=("CP-SAT で、弧の 0-1 変数と AddMultipleCircuit 制約（デポを何度も通ってよい巡回路）で経路を表し、各顧客の作業開始時刻と積載量の整数変数に「弧を使うなら、次の顧客は作業時間と移動時間のぶん後に始まり、積載量は需要のぶん増える」条件を付けたモデル。"
                     "解と同時に下界を出し、最適性を証明できる。"))
def cpsat(inst, tl, seed, threads=1):
    from ortools.sat.python import cp_model
    D, e, l, s, q, Q, K = inst["D"], inst["e"], inst["l"], inst["s"], inst["q"], inst["Q"], inst["K"]
    n = D.shape[0]
    m = cp_model.CpModel()
    T = [None] + [m.NewIntVar(int(e[i]), int(l[i]), f"T{i}") for i in range(1, n)]
    L = [None] + [m.NewIntVar(int(q[i]), Q, f"L{i}") for i in range(1, n)]
    x = {}
    for i, j in _usable_arcs(inst):
        v = x[i, j] = m.NewBoolVar("")
        if i == 0:
            m.Add(T[j] >= int(e[0] + D[0, j])).OnlyEnforceIf(v)
            m.Add(L[j] == int(q[j])).OnlyEnforceIf(v)
        elif j == 0:
            m.Add(T[i] + int(s[i] + D[i, 0]) <= int(l[0])).OnlyEnforceIf(v)
        else:
            m.Add(T[j] >= T[i] + int(s[i] + D[i, j])).OnlyEnforceIf(v)
            m.Add(L[j] == L[i] + int(q[j])).OnlyEnforceIf(v)
    m.AddMultipleCircuit([(i, j, v) for (i, j), v in x.items()])
    m.Add(sum(v for (i, j), v in x.items() if i == 0) <= K)
    m.Minimize(sum(int(D[i, j]) * v for (i, j), v in x.items()))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tl
    sv.parameters.num_workers = threads
    sv.parameters.random_seed = seed
    st = sv.Solve(m)
    lb = int(np.ceil(sv.BestObjectiveBound() - 1e-6)) if st != cp_model.INFEASIBLE else None
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(obj=None, lb=lb, solution=None)
    routes = _routes_from_arcs(n, [a for a, v in x.items() if sv.Value(v)])
    return _result(inst, routes, lb=lb, optimal=st == cp_model.OPTIMAL)


# ---------------------------------------------------------------- MIP（HiGHS, 時刻・積載量変数の big-M 定式化）
@solver("MIP-HiGHS", problems="vrptw", threads=[1, 4], packages=["highspy"],
        description=("弧の 0-1 変数と、各顧客の作業開始時刻・積載量の連続変数を big-M 制約（弧を使うときだけ時刻と積載量の順序を強制する）で結んだ 2 添字の定式化を HiGHS で解く。"
                     "時刻の制約が部分巡回路も防ぐ。"
                     "big-M は緩和が弱く、専用の厳密解法（列生成）に比べて扱える規模がずっと小さい。"))
def mip_bigm(inst, tl, seed, threads=1):
    """x_ij（弧を通るか）, T_i（作業開始時刻）, L_i（i を出るときの積載量）. 顧客の入次数・出次数 1、デポから出る台数 K 以下、
    T_j >= T_i + s_i + D_ij - M_ij (1 - x_ij), L_j >= L_i + q_j - Q (1 - x_ij)."""
    import highspy
    D, e, l, s, q, Q, K = inst["D"], inst["e"], inst["l"], inst["s"], inst["q"], inst["Q"], inst["K"]
    n = D.shape[0]
    arcs = _usable_arcs(inst)
    m = len(arcs)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", threads)
    h.setOptionValue("random_seed", seed)
    h.setOptionValue("time_limit", float(tl))
    inf = highspy.kHighsInf
    # 列: x_0..x_{m-1}（0-1）, T_1..T_{n-1}, L_1..L_{n-1}（連続）
    h.addVars(m, np.zeros(m), np.ones(m))
    h.changeColsIntegrality(m, np.arange(m, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * m))
    h.changeColsCost(m, np.arange(m, dtype=np.int32), np.array([D[i, j] for i, j in arcs], dtype=float))
    h.addVars(n - 1, e[1:].astype(float), l[1:].astype(float))
    h.addVars(n - 1, q[1:].astype(float), np.full(n - 1, float(Q)))
    tcol = lambda i: m + i - 1
    lcol = lambda i: m + n - 1 + i - 1
    outs, ins = [[] for _ in range(n)], [[] for _ in range(n)]
    for k, (i, j) in enumerate(arcs):
        outs[i].append(k)
        ins[j].append(k)
    for v in range(1, n):
        for idx in (outs[v], ins[v]):
            h.addRow(1, 1, len(idx), np.array(idx, dtype=np.int32), np.ones(len(idx)))
    h.addRow(0, K, len(outs[0]), np.array(outs[0], dtype=np.int32), np.ones(len(outs[0])))
    h.addRow(0, 0, len(outs[0]) + len(ins[0]), np.array(outs[0] + ins[0], dtype=np.int32),
             np.array([1.0] * len(outs[0]) + [-1.0] * len(ins[0])))
    for k, (i, j) in enumerate(arcs):
        if i == 0:  # T_j >= e_j + (e_0 + D_0j - e_j) x_0j
            a = e[0] + D[0, j] - e[j]
            if a > 0:
                h.addRow(e[j], inf, 2, np.array([tcol(j), k], dtype=np.int32), np.array([1.0, -a]))
        elif j == 0:  # T_i + s_i + D_i0 <= l_0 + M (1 - x_i0)
            M = l[i] + s[i] + D[i, 0] - l[0]
            if M > 0:
                h.addRow(-inf, l[0] - s[i] - D[i, 0] + M, 2, np.array([tcol(i), k], dtype=np.int32),
                         np.array([1.0, M]))
        else:
            M = l[i] + s[i] + D[i, j] - e[j]
            if M > 0:  # T_j - T_i - M x_ij >= s_i + D_ij - M
                h.addRow(s[i] + D[i, j] - M, inf, 3, np.array([tcol(j), tcol(i), k], dtype=np.int32),
                         np.array([1.0, -1.0, -float(M)]))
            # L_j - L_i - Q x_ij >= q_j - Q
            h.addRow(q[j] - Q, inf, 3, np.array([lcol(j), lcol(i), k], dtype=np.int32),
                     np.array([1.0, -1.0, -float(Q)]))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    h.run()
    info = h.getInfo()
    st = h.getModelStatus()
    lb = int(np.ceil(info.mip_dual_bound - 1e-6)) if np.isfinite(info.mip_dual_bound) else None
    if info.primal_solution_status != 2:
        return dict(obj=None, lb=lb, solution=None)
    val = np.array(h.getSolution().col_value)
    routes = _routes_from_arcs(n, [a for k, a in enumerate(arcs) if val[k] > 0.5])
    return _result(inst, routes, lb=lb, optimal=st == highspy.HighsModelStatus.kOptimal)
