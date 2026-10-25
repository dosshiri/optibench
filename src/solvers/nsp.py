"""NSP の解法群. 各関数は (inst, time_limit, seed) -> dict(obj, lb, hard, solution=勤務表)."""
import numpy as np
from numba import njit

from common import Timer
from problems.nsp import HARD, cover_cost, emp_cost, evaluate, to_arrays
from registry import solver

_ARGS = ("length", "forbid", "max_type", "min_minutes", "max_minutes", "max_consec", "min_consec",
         "min_off", "max_weekends", "dayoff", "pref", "on_pen")


@njit(cache=True)
def _all_costs(R, length, forbid, max_type, min_minutes, max_minutes, max_consec, min_consec,
               min_off, max_weekends, dayoff, pref, on_pen):
    E = R.shape[0]
    ec = np.zeros(E, np.int64)
    for e in range(E):
        h, s = emp_cost(R[e], e, length, forbid, max_type, min_minutes, max_minutes, max_consec,
                        min_consec, min_off, max_weekends, dayoff, pref, on_pen)
        ec[e] = HARD * h + s
    return ec


@njit(cache=True)
def _cell_delta_cover(cnt, cover, d, old, new, w_under, w_over):
    dc = 0
    if old >= 0:
        x = cnt[d, old] - cover[d, old]
        dc += (w_under if x <= 0 else -w_over)  # 1人減る
    if new >= 0:
        x = cnt[d, new] - cover[d, new] - (1 if old == new else 0)
        dc += (-w_under if x < 0 else w_over)  # 1人増える
    return dc


@njit(cache=True)
def _greedy(R, cnt, ec, cover, w_under, w_over, length, forbid, max_type, min_minutes, max_minutes,
            max_consec, min_consec, min_off, max_weekends, dayoff, pref, on_pen):
    """1マス変更の最急降下 (空の勤務表から)."""
    E, H = R.shape
    S = cover.shape[1]
    while True:
        best, be, bd, bs, bnew = 0, -1, -1, -1, 0
        for e in range(E):
            for d in range(H):
                old = R[e, d]
                for s in range(-1, S):
                    if s == old:
                        continue
                    R[e, d] = s
                    h, so = emp_cost(R[e], e, length, forbid, max_type, min_minutes, max_minutes,
                                     max_consec, min_consec, min_off, max_weekends, dayoff, pref, on_pen)
                    R[e, d] = old
                    nc = HARD * h + so
                    delta = nc - ec[e] + _cell_delta_cover(cnt, cover, d, old, s, w_under, w_over)
                    if delta < best:
                        best, be, bd, bs, bnew = delta, e, d, s, nc
        if be < 0:
            break
        old = R[be, bd]
        if old >= 0:
            cnt[bd, old] -= 1
        if bs >= 0:
            cnt[bd, bs] += 1
        R[be, bd] = bs
        ec[be] = bnew


@njit(cache=True)
def _sa_chunk(R, cnt, ec, best_R, best, T, iters, seed, cover, w_under, w_over, length, forbid,
              max_type, min_minutes, max_minutes, max_consec, min_consec, min_off, max_weekends,
              dayoff, pref, on_pen):
    np.random.seed(seed)
    E, H = R.shape
    S = cover.shape[1]
    cur = ec.sum() + cover_cost(cnt, cover, w_under, w_over)
    for _ in range(iters):
        if np.random.random() < 0.5:  # 1マス変更
            e = np.random.randint(0, E)
            d = np.random.randint(0, H)
            old = R[e, d]
            s = np.random.randint(-1, S)
            if s == old:
                continue
            R[e, d] = s
            h, so = emp_cost(R[e], e, length, forbid, max_type, min_minutes, max_minutes,
                             max_consec, min_consec, min_off, max_weekends, dayoff, pref, on_pen)
            nc = HARD * h + so
            delta = nc - ec[e] + _cell_delta_cover(cnt, cover, d, old, s, w_under, w_over)
            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                ec[e] = nc
                if old >= 0:
                    cnt[d, old] -= 1
                if s >= 0:
                    cnt[d, s] += 1
                cur += delta
            else:
                R[e, d] = old
        else:  # 2人の職員の連続区間 (1-4日) を交換 (必要人数は不変)
            e1 = np.random.randint(0, E)
            e2 = np.random.randint(0, E)
            if e1 == e2:
                continue
            L = np.random.randint(1, 5)
            d0 = np.random.randint(0, H - L + 1)
            same = True
            for d in range(d0, d0 + L):
                if R[e1, d] != R[e2, d]:
                    same = False
            if same:
                continue
            for d in range(d0, d0 + L):
                R[e1, d], R[e2, d] = R[e2, d], R[e1, d]
            h1, s1 = emp_cost(R[e1], e1, length, forbid, max_type, min_minutes, max_minutes,
                              max_consec, min_consec, min_off, max_weekends, dayoff, pref, on_pen)
            h2, s2 = emp_cost(R[e2], e2, length, forbid, max_type, min_minutes, max_minutes,
                              max_consec, min_consec, min_off, max_weekends, dayoff, pref, on_pen)
            n1, n2 = HARD * h1 + s1, HARD * h2 + s2
            delta = n1 + n2 - ec[e1] - ec[e2]
            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                ec[e1], ec[e2] = n1, n2
                cur += delta
            else:
                for d in range(d0, d0 + L):
                    R[e1, d], R[e2, d] = R[e2, d], R[e1, d]
        if cur < best:
            best = cur
            best_R[:] = R
    return best


def _greedy_start(inst):
    A = to_arrays(inst)
    R = -np.ones((A["E"], A["H"]), np.int64)
    cnt = np.zeros((A["H"], A["S"]), np.int64)
    args = tuple(A[k] for k in _ARGS)
    ec = _all_costs(R, *args)
    _greedy(R, cnt, ec, A["cover"], A["w_under"], A["w_over"], *args)
    return A, R, cnt, ec, args


@solver("Greedy", problems="nsp",
        description=("空の勤務表から、1 マス（ある職員のある日の勤務）を変えたときにペナルティが最も減る変更を、減らなくなるまで繰り返す最急降下法。"
                     "決定的で速い。"))
def greedy(inst, tl, seed):
    _, R, _, _, _ = _greedy_start(inst)
    h, obj = evaluate(inst, R)
    return dict(obj=obj if h == 0 else None, hard=h, lb=None, solution=R.tolist())


@solver("SA", problems="nsp", seeds=3,
        description=("焼きなまし法。"
                     "Greedy の勤務表から始め、1 マスの変更と、2 人の職員の連続 1〜4 日分の勤務の入れ替えをランダムに試し、悪くなる変更も温度に応じた確率で受け入れる。"))
def simulated_annealing(inst, tl, seed):
    tm = Timer(tl)
    A, R, cnt, ec, args = _greedy_start(inst)
    h0, o0 = evaluate(inst, R)
    best = HARD * h0 + o0
    best_R = R.copy()
    T0, T1 = 50.0, 0.5
    k = 0
    while tm.left() > 0:
        T = T0 * (T1 / T0) ** (tm.elapsed() / tl)
        best = _sa_chunk(R, cnt, ec, best_R, best, T, 20000, seed * 100003 + k, A["cover"],
                         A["w_under"], A["w_over"], *args)
        k += 1
    h, obj = evaluate(inst, best_R)
    return dict(obj=obj if h == 0 else None, hard=h, lb=None, solution=best_R.tolist())


# ---------------------------------------------------------------- 厳密モデル (CP-SAT / HiGHS 共通の線形定式化)
def _build(inst, add_bin, add_int, add_le, add_ge, add_eq):
    """x[e,d,s] の 0-1 線形モデルを構築. 目的関数 (定数項, [(係数, 変数)]) を返す."""
    A = to_arrays(inst)
    E, H, S = A["E"], A["H"], A["S"]
    x = {(e, d, s): add_bin(ub=0 if A["dayoff"][e, d] else 1) for e in range(E) for d in range(H)
         for s in range(S)}
    obj, const = [], 0
    for e in range(E):
        w = [[(1, x[e, d, s]) for s in range(S)] for d in range(H)]
        for d in range(H):
            add_le(w[d], 1)
            for a in range(S):
                for b in range(S):
                    if d + 1 < H and A["forbid"][a, b]:
                        add_le([(1, x[e, d, a]), (1, x[e, d + 1, b])], 1)
                if A["pref"][e, d, a]:
                    obj.append((int(A["pref"][e, d, a]), x[e, d, a]))
                if A["on_pen"][e, d, a]:
                    const += int(A["on_pen"][e, d, a])
                    obj.append((-int(A["on_pen"][e, d, a]), x[e, d, a]))
        for s in range(S):
            add_le([(1, x[e, d, s]) for d in range(H)], int(A["max_type"][e, s]))
        mins = [(int(A["length"][s]), x[e, d, s]) for d in range(H) for s in range(S)]
        add_le(mins, int(A["max_minutes"][e]))
        add_ge(mins, int(A["min_minutes"][e]))
        c = int(A["max_consec"][e])
        for d in range(H - c):
            add_le([t for k in range(d, d + c + 1) for t in w[k]], c)
        for l in range(1, int(A["min_consec"][e])):  # 休-勤 l 日-休 を禁止
            for d in range(1, H - l):
                terms = [(-1, v) for _, v in w[d - 1]] + [t for k in range(d, d + l) for t in w[k]] + \
                        [(-1, v) for _, v in w[d + l]]
                add_le(terms, l - 1)
        for l in range(1, int(A["min_off"][e])):  # 勤-休 l 日-勤 を禁止
            for d in range(1, H - l):
                terms = list(w[d - 1]) + [(-1, v) for k in range(d, d + l) for _, v in w[k]] + list(w[d + l])
                add_le(terms, 1)
        ks = []
        for w0 in range(5, H, 7):
            kv = add_bin(ub=1)
            for dd in (w0, w0 + 1):
                if dd < H:
                    add_ge([(1, kv)] + [(-1, v) for _, v in w[dd]], 0)
            ks.append((1, kv))
        add_le(ks, int(A["max_weekends"][e]))
    for d in range(H):
        for s in range(S):
            over, under = add_int(E), add_int(int(A["cover"][d, s]))
            add_eq([(1, x[e, d, s]) for e in range(E)] + [(-1, over), (1, under)], int(A["cover"][d, s]))
            obj += [(A["w_over"], over), (A["w_under"], under)]
    return x, obj, const, (E, H, S)


def _cpsat(inst, tl, seed, workers):
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    lin = lambda t: sum(c * v for c, v in t)
    x, obj, const, (E, H, S) = _build(
        inst, lambda ub: m.NewIntVar(0, ub, ""), lambda ub: m.NewIntVar(0, ub, ""),
        lambda t, b: m.Add(lin(t) <= b), lambda t, b: m.Add(lin(t) >= b), lambda t, b: m.Add(lin(t) == b))
    m.Minimize(lin(obj) + const)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = workers
    s.parameters.random_seed = seed
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(obj=None, lb=int(s.BestObjectiveBound()), hard=None)
    R = -np.ones((E, H), np.int64)
    for (e, d, k), v in x.items():
        if s.Value(v):
            R[e, d] = k
    h, o = evaluate(inst, R)
    return dict(obj=o, lb=int(np.ceil(s.BestObjectiveBound() - 1e-6)), hard=h,
                optimal=st == cp_model.OPTIMAL, solution=R.tolist())


@solver("CP-SAT", problems="nsp", threads=[1, 4], packages=["ortools"],
        description=("「職員 e が日 d に勤務 s に入るか」の 0-1 変数で、連続勤務の上限・禁止される勤務の並び・必要人数などを線形制約にしたモデルを CP-SAT で解く。"
                     "モデルは MIP-HiGHS と同じ。"))
def cpsat(inst, tl, seed, threads=1):
    return _cpsat(inst, tl, seed, threads)


@solver("MIP-HiGHS", problems="nsp", parallel=True, packages=["highspy"],
        description=("CP-SAT と同じ 0-1 線形モデルを MIP ソルバー HiGHS で解く。"
                     "LP 緩和による下界を使って、最適性を証明することもできる。"))
def mip(inst, tl, seed, threads=1):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", threads)
    h.setOptionValue("random_seed", seed)
    h.setOptionValue("time_limit", float(tl))
    lin = lambda t: sum(c * v for c, v in t)
    x, obj, const, (E, H, S) = _build(
        inst, lambda ub: h.addVariable(lb=0, ub=ub, type=highspy.HighsVarType.kInteger),
        lambda ub: h.addVariable(lb=0, ub=ub, type=highspy.HighsVarType.kInteger),
        lambda t, b: h.addConstr(lin(t) <= b), lambda t, b: h.addConstr(lin(t) >= b),
        lambda t, b: h.addConstr(lin(t) == b))
    h.minimize(lin(obj) + const)
    info = h.getInfo()
    lb = int(np.ceil(info.mip_dual_bound - 1e-6)) if np.isfinite(info.mip_dual_bound) else None
    if info.primal_solution_status != 2:
        return dict(obj=None, lb=lb, hard=None)
    val = h.getSolution().col_value
    R = -np.ones((E, H), np.int64)
    for (e, d, k), v in x.items():
        if val[v.index] > 0.5:
            R[e, d] = k
    hd, o = evaluate(inst, R)
    return dict(obj=o, lb=lb, hard=hd, optimal=h.getModelStatus() == highspy.HighsModelStatus.kOptimal,
                solution=R.tolist())

