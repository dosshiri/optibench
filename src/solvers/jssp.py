"""JSSP の解法群. 各関数は ((jobs, nm), time_limit, seed) -> dict(obj, lb)."""
import numpy as np
from numba import njit

from common import Timer
from registry import solver


def _arrays(jobs, nm):
    nj = len(jobs)
    M = np.array([[o[0] for o in j] for j in jobs], dtype=np.int64)
    P = np.array([[o[1] for o in j] for j in jobs], dtype=np.int64)
    return nj, nm, M, P


# ---------------------------------------------------------------- ディスパッチングルール (Giffler-Thompson)
@njit(cache=True)
def _giffler_thompson(M, P, prio_kind, keys):
    """能動スケジュールを生成. prio_kind: 0=SPT, 1=MWKR, 2=keys (乱数キー, 小さいほど優先)."""
    nj, nm = M.shape
    nxt = np.zeros(nj, np.int64)
    jr = np.zeros(nj, np.int64)  # ジョブの準備時刻
    mr = np.zeros(nm, np.int64)  # 機械の準備時刻
    rem = P.sum(axis=1).copy()
    seq = np.empty((nm, nj), np.int64)
    cnt = np.zeros(nm, np.int64)
    for _ in range(nj * nm):
        # 最早完了時刻が最小の操作を探す
        best_c, bm = 1 << 62, -1
        for j in range(nj):
            if nxt[j] < nm:
                k = nxt[j]
                c = max(jr[j], mr[M[j, k]]) + P[j, k]
                if c < best_c:
                    best_c, bm = c, M[j, k]
        # 衝突集合から優先度で選ぶ
        bj, bp = -1, 1e30
        for j in range(nj):
            if nxt[j] < nm and M[j, nxt[j]] == bm:
                k = nxt[j]
                s = max(jr[j], mr[bm])
                if s < best_c:
                    if prio_kind == 0:
                        pr = P[j, k]
                    elif prio_kind == 1:
                        pr = -rem[j]
                    else:
                        pr = keys[j * nm + k]
                    if pr < bp:
                        bp, bj = pr, j
        k = nxt[bj]
        s = max(jr[bj], mr[bm])
        jr[bj] = s + P[bj, k]
        mr[bm] = s + P[bj, k]
        rem[bj] -= P[bj, k]
        nxt[bj] += 1
        seq[bm, cnt[bm]] = bj
        cnt[bm] += 1
    return mr.max(), seq


@solver("SPT", problems="jssp")
def spt(data, tl, seed):
    nj, nm, M, P = _arrays(*data)
    c, _ = _giffler_thompson(M, P, 0, np.zeros(1))
    return dict(obj=int(c), lb=None)


@solver("MWKR", problems="jssp")
def mwkr(data, tl, seed):
    nj, nm, M, P = _arrays(*data)
    c, _ = _giffler_thompson(M, P, 1, np.zeros(1))
    return dict(obj=int(c), lb=None)


# ---------------------------------------------------------------- 遺伝的アルゴリズム (乱数キー + GT 復号, BRKGA 型)
@solver("GA", problems="jssp", seeds=3)
def ga(data, tl, seed):
    nj, nm, M, P = _arrays(*data)
    tm = Timer(tl)
    rng = np.random.default_rng(seed)
    N, n = 100, nj * nm
    ne, nmut = 20, 15  # エリート数, 突然変異体数
    pop = rng.random((N, n))
    fit = np.array([_giffler_thompson(M, P, 2, x)[0] for x in pop])
    while tm.left() > 0:
        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        child = np.empty((N - ne - nmut, n))
        for c in range(len(child)):
            a = pop[rng.integers(ne)]
            b = pop[rng.integers(ne, N)]
            mask = rng.random(n) < 0.7  # エリート側を 70% で継承
            child[c] = np.where(mask, a, b)
        mut = rng.random((nmut, n))
        new = np.vstack([child, mut])
        nf = np.array([_giffler_thompson(M, P, 2, x)[0] for x in new])
        pop = np.vstack([pop[:ne], new])
        fit = np.concatenate([fit[:ne], nf])
    return dict(obj=int(fit.min()), lb=None)


# ---------------------------------------------------------------- タブー探索 (N5 近傍, Nowicki & Smutnicki 1996)
@njit(cache=True)
def _evaluate(M, P, seq):
    """機械順序 seq から先頭時刻と臨界路を求める. 閉路なら -1."""
    nj, nm = M.shape
    N = nj * nm
    pos_in_job = np.empty((nj, nm), np.int64)  # 機械 m 上のジョブ j の工程番号
    for j in range(nj):
        for k in range(nm):
            pos_in_job[j, M[j, k]] = k
    mpred = -np.ones(N, np.int64)
    msucc = -np.ones(N, np.int64)
    for m in range(nm):
        for r in range(nj - 1):
            a = seq[m, r] * nm + pos_in_job[seq[m, r], m]
            b = seq[m, r + 1] * nm + pos_in_job[seq[m, r + 1], m]
            mpred[b] = a
            msucc[a] = b
    indeg = np.zeros(N, np.int64)
    for v in range(N):
        if v % nm > 0:
            indeg[v] += 1
        if mpred[v] >= 0:
            indeg[v] += 1
    head = np.zeros(N, np.int64)
    via = -np.ones(N, np.int64)
    stack = np.empty(N, np.int64)
    top = 0
    for v in range(N):
        if indeg[v] == 0:
            stack[top] = v
            top += 1
    done = 0
    while top > 0:
        top -= 1
        v = stack[top]
        done += 1
        fin = head[v] + P[v // nm, v % nm]
        for w in (v + 1 if v % nm < nm - 1 else -1, msucc[v]):
            if w >= 0:
                if fin > head[w]:
                    head[w] = fin
                    via[w] = v
                indeg[w] -= 1
                if indeg[w] == 0:
                    stack[top] = w
                    top += 1
    if done < N:
        return -1, np.empty(0, np.int64)
    cmax, last = 0, 0
    for v in range(N):
        f = head[v] + P[v // nm, v % nm]
        if f > cmax:
            cmax, last = f, v
    path = []
    v = last
    while v >= 0:
        path.append(v)
        v = via[v]
    out = np.empty(len(path), np.int64)
    for i in range(len(path)):
        out[i] = path[len(path) - 1 - i]
    return cmax, out


@njit(cache=True)
def _n5_moves(M, seq, path):
    """臨界ブロックの先頭2つ/末尾2つの交換 (N5). 戻り値: (機械, 順序位置) の配列."""
    nj, nm = M.shape
    moves = []
    L = path.shape[0]
    i = 0
    blocks = []
    while i < L:
        j = i
        mach = M[path[i] // nm, path[i] % nm]
        while j + 1 < L and M[path[j + 1] // nm, path[j + 1] % nm] == mach:
            j += 1
        if j > i:
            blocks.append((i, j, mach))
        i = j + 1
    nb = len(blocks)
    for b in range(nb):
        i, j, mach = blocks[b]
        ja = path[i] // nm
        r0 = 0
        for r in range(nj):
            if seq[mach, r] == ja:
                r0 = r
        r1 = r0 + (j - i)
        first_ok = b > 0 or nb == 1   # 先頭ブロック以外: 先頭2つを交換
        last_ok = b < nb - 1 or nb == 1  # 末尾ブロック以外: 末尾2つを交換
        if first_ok:
            moves.append((mach, r0))
        if last_ok and not (first_ok and r1 - 1 == r0):
            moves.append((mach, r1 - 1))
    return moves


@njit(cache=True)
def _tabu(M, P, seq, iters, seed, tenure_lo, tenure_hi):
    np.random.seed(seed)
    nj, nm = M.shape
    cur, path = _evaluate(M, P, seq)
    best, best_seq = cur, seq.copy()
    tabu = np.zeros((nj, nj), np.int64)  # (a,b) の交換を禁止する期限
    stall = 0
    for it in range(1, iters + 1):
        moves = _n5_moves(M, seq, path)
        if len(moves) == 0:
            break
        bv, bmv, bt = 1 << 62, -1, False
        for k in range(len(moves)):
            m, r = moves[k]
            a, b = seq[m, r], seq[m, r + 1]
            seq[m, r], seq[m, r + 1] = b, a
            v, _ = _evaluate(M, P, seq)
            seq[m, r], seq[m, r + 1] = a, b
            if v < 0:
                continue
            is_tabu = tabu[b, a] > it
            if (not is_tabu or v < best) and v < bv:
                bv, bmv = v, k
        if bmv < 0:  # 全て禁止: ランダムに一つ選ぶ
            bmv = np.random.randint(0, len(moves))
        m, r = moves[bmv]
        a, b = seq[m, r], seq[m, r + 1]
        seq[m, r], seq[m, r + 1] = b, a
        tabu[a, b] = it + np.random.randint(tenure_lo, tenure_hi + 1)
        cur, path = _evaluate(M, P, seq)
        if cur < best:
            best, best_seq, stall = cur, seq.copy(), 0
        else:
            stall += 1
        if stall > 2000:  # 停滞したら最良解へ戻して摂動
            seq[:] = best_seq
            for _ in range(5):
                mm = np.random.randint(0, nm)
                rr = np.random.randint(0, nj - 1)
                seq[mm, rr], seq[mm, rr + 1] = seq[mm, rr + 1], seq[mm, rr]
            cur, path = _evaluate(M, P, seq)
            if cur < 0:
                seq[:] = best_seq
                cur, path = _evaluate(M, P, seq)
            stall = 0
    return best, best_seq


@solver("Tabu-N5", problems="jssp", seeds=3)
def tabu_n5(data, tl, seed):
    nj, nm, M, P = _arrays(*data)
    tm = Timer(tl)
    c, seq = _giffler_thompson(M, P, 1, np.zeros(1))
    best = int(c)
    n = nj * nm
    lo, hi = max(2, int(np.sqrt(n) / 2)), max(4, int(np.sqrt(n)))
    k = 0
    while tm.left() > 0:
        b, s = _tabu(M, P, seq, 500, seed * 7919 + k, lo, hi)
        k += 1
        if b < best:
            best = int(b)
        seq = s
    return dict(obj=best, lb=None)


# ---------------------------------------------------------------- CP-SAT
def _cpsat(data, tl, seed, workers):
    from ortools.sat.python import cp_model
    jobs, nm = data
    H = sum(p for j in jobs for _, p in j)
    m = cp_model.CpModel()
    ends, per_m = [], [[] for _ in range(nm)]
    for jid, job in enumerate(jobs):
        prev = None
        for k, (mach, p) in enumerate(job):
            s = m.NewIntVar(0, H, "")
            e = m.NewIntVar(0, H, "")
            per_m[mach].append(m.NewIntervalVar(s, p, e, ""))
            if prev is not None:
                m.Add(s >= prev)
            prev = e
        ends.append(prev)
    for iv in per_m:
        m.AddNoOverlap(iv)
    mk = m.NewIntVar(0, H, "")
    m.AddMaxEquality(mk, ends)
    m.Minimize(mk)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = tl
    sol.parameters.num_workers = workers
    sol.parameters.random_seed = seed
    st = sol.Solve(m)
    ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return dict(obj=int(sol.ObjectiveValue()) if ok else None, lb=int(sol.BestObjectiveBound()),
                optimal=st == cp_model.OPTIMAL)


@solver("CP-SAT", problems="jssp", packages=["ortools"])
def cpsat1(data, tl, seed):
    return _cpsat(data, tl, seed, 1)


@solver("CP-SAT-4w", problems="jssp", threads=4, packages=["ortools"])
def cpsat4(data, tl, seed):
    return _cpsat(data, tl, seed, 4)


# ---------------------------------------------------------------- MIP (Manne の離接定式化, HiGHS)
@solver("MIP-HiGHS", problems="jssp", packages=["highspy"])
def mip_disjunctive(data, tl, seed):
    import highspy
    jobs, nm = data
    H = float(sum(p for j in jobs for _, p in j))
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("random_seed", seed)
    h.setOptionValue("time_limit", float(tl))
    inf = highspy.kHighsInf
    S = {}
    for j, job in enumerate(jobs):
        for k in range(nm):
            S[j, k] = h.addVariable(lb=0, ub=H)
    Cmax = h.addVariable(lb=0, ub=H, obj=1)
    ops_on = [[] for _ in range(nm)]
    for j, job in enumerate(jobs):
        for k, (mach, p) in enumerate(job):
            ops_on[mach].append((j, k, p))
            if k > 0:
                h.addConstr(S[j, k] - S[j, k - 1] >= job[k - 1][1])
        h.addConstr(Cmax - S[j, nm - 1] >= job[nm - 1][1])
    for ops in ops_on:
        for a in range(len(ops)):
            for b in range(a + 1, len(ops)):
                (j1, k1, p1), (j2, k2, p2) = ops[a], ops[b]
                y = h.addBinary()
                # y=1: j1 が先, y=0: j2 が先
                h.addConstr(S[j1, k1] + p1 - S[j2, k2] - H * (1 - y) <= 0)
                h.addConstr(S[j2, k2] + p2 - S[j1, k1] - H * y <= 0)
    h.minimize(Cmax)
    info = h.getInfo()
    ok = info.primal_solution_status == 2
    return dict(obj=int(round(info.objective_function_value)) if ok else None,
                lb=int(np.ceil(info.mip_dual_bound - 1e-6)) if np.isfinite(info.mip_dual_bound) else None,
                optimal=h.getModelStatus() == highspy.HighsModelStatus.kOptimal)

