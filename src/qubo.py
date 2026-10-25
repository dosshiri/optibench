"""量子・疑似量子アニーリング系の解法で使う QUBO（0/1 変数の二次式の最小化）の定式化と、
アニーラを制限時間まで繰り返し呼ぶ共通の枠組み.

アニーラ本体（OpenJij, dwave-samplers など）は solvers/qubo_*.py で、
sample(Q, seed) -> 0/1 の解のリスト を渡して anneal_tsp / anneal_jssp を呼ぶ.
QUBO は {(i, j): 係数}（i <= j, 対角は 1 次の項）の dict で表す."""
import time
from collections import defaultdict

import numpy as np

from common import Timer
from problems.routing import tour_cost

QUBO_PROBLEMS = ["tsp", "tsp_small", "atsp", "jssp"]
NUM_READS = 10  # アニーラ 1 回の呼び出しで得る解の数

# QUBO の変数がこれより多いインスタンスは解かない. 疑似量子アニーリングでは、10 解を求める 1 回の呼び出しが
# 70 都市（4761 変数）で約 20 秒かかり、100 都市（9801 変数）では制限時間 60 秒にほぼ届くため
MAX_VARS = 5000


def _onehot(Q, idx, A):
    """変数 idx のうちちょうど 1 つが 1 になる制約 A (Σx - 1)^2 を Q に足す（定数項は省く）."""
    for a, i in enumerate(idx):
        Q[i, i] -= A
        for j in idx[a + 1:]:
            Q[min(i, j), max(i, j)] += 2 * A


# ---------------------------------------------------------------- TSP / ATSP
def tsp_qubo(D, A):
    """位置符号化の TSP: x[i, t] = 都市 i を t 番目に訪れる. 都市 0 を 0 番目に固定して (n-1)^2 変数にする.
    目的は巡回路の長さ、制約は「各都市を 1 回」「各位置に 1 都市」で、違反 1 つにつき罰金 A."""
    n = D.shape[0]
    m = n - 1
    var = lambda i, t: (i - 1) * m + (t - 1)  # 都市 1..n-1, 位置 1..n-1
    Q = defaultdict(float)
    for i in range(1, n):
        _onehot(Q, [var(i, t) for t in range(1, n)], A)
    for t in range(1, n):
        _onehot(Q, [var(i, t) for i in range(1, n)], A)
    for i in range(1, n):
        Q[var(i, 1), var(i, 1)] += D[0, i]  # 都市 0 → 1 番目
        Q[var(i, m), var(i, m)] += D[i, 0]  # 最後 → 都市 0
        for j in range(1, n):
            if i != j:
                for t in range(1, m):  # t 番目に i、t+1 番目に j
                    a, b = var(i, t), var(j, t + 1)
                    Q[min(a, b), max(a, b)] += D[i, j]
    return dict(Q)


def tsp_decode(n, x):
    """0/1 の解を巡回路に直す. 置換行列になっていなければ None."""
    X = np.asarray(x, dtype=np.int64).reshape(n - 1, n - 1)
    if (X.sum(axis=0) != 1).any() or (X.sum(axis=1) != 1).any():
        return None
    order = np.argmax(X, axis=0)  # 各位置の都市
    return [0] + [int(i) + 1 for i in order]


def anneal_tsp(D, tl, seed, sample):
    """TSP / ATSP を QUBO にしてアニーラ sample を制限時間まで繰り返し呼び、最良の巡回路を返す.
    罰金 A は最大の距離から始め、1 回の呼び出しで実行可能解が 1 つも出なければ 1.5 倍、
    半分を超えれば 0.8 倍にする（罰金は小さいほど距離の差が効くが、小さすぎると制約を破る）."""
    tm = Timer(tl)
    n = D.shape[0]
    nvar = (n - 1) ** 2
    if nvar > MAX_VARS:
        return dict(obj=None, lb=None, qubo_vars=nvar, skipped=f"QUBO の変数が {nvar} 個で上限 {MAX_VARS} を超える")
    A = float(D.max())
    Q = tsp_qubo(D, A)
    best, best_tour, calls, samples, feas = None, None, 0, 0, 0
    last = 0.0
    while tm.left() > last * 1.2 or calls == 0:
        t0 = time.perf_counter()
        xs = sample(Q, seed * 100000 + calls)  # シードごとに乱数の種が重ならないようにする
        last = time.perf_counter() - t0
        calls += 1
        ok = 0
        for x in xs:
            samples += 1
            tour = tsp_decode(n, x)
            if tour is None:
                continue
            ok += 1
            c = tour_cost(D, tour)
            if best is None or c < best:
                best, best_tour = c, tour
        # 罰金が強すぎると距離の差が埋もれ、弱すぎると制約を破るので、実行可能解の割合で調整する
        feas += ok
        if ok == 0:
            A *= 1.5
        elif ok > len(xs) / 2:
            A *= 0.8
        else:
            continue
        Q = tsp_qubo(D, A)
    return dict(obj=best, lb=None, solution=best_tour, qubo_vars=nvar, calls=calls, samples=samples,
                feasible_samples=feas, penalty=A)


# ---------------------------------------------------------------- JSSP
def jssp_horizon(jobs, nm):
    """メイクスパンの下界（最長のジョブ・最も負荷の高い機械）と、貪欲法（最早開始）で作った上界."""
    lb = max(max(sum(p for _, p in j) for j in jobs), max(sum(p for j in jobs for m, p in j if m == k)
                                                           for k in range(nm)))
    jr = [0] * len(jobs)
    mr = [0] * nm
    nxt = [0] * len(jobs)
    for _ in range(sum(len(j) for j in jobs)):
        j = min((j for j in range(len(jobs)) if nxt[j] < len(jobs[j])),
                key=lambda j: max(jr[j], mr[jobs[j][nxt[j]][0]]))
        m, p = jobs[j][nxt[j]]
        s = max(jr[j], mr[m])
        jr[j] = mr[m] = s + p
        nxt[j] += 1
    return lb, max(jr)


def jssp_qubo(jobs, nm, T, A=1.0):
    """時間添字の JSSP（Venturelli ら 2015）: x[o, t] = 工程 o を時刻 t に始める.
    メイクスパン T 以内に収まる開始時刻だけを変数にし（各工程の前後の工程の処理時間で範囲を絞る）、
    「各工程を 1 回だけ始める」「ジョブ内の順序」「機械の重なりなし」を罰金 A で入れる（目的関数はなく、
    メイクスパン T の実行可能解を探す判定問題）. 戻り値は (Q, 変数一覧 [(工程, 時刻)], 工程一覧)."""
    ops = [(j, k, m, p) for j, job in enumerate(jobs) for k, (m, p) in enumerate(job)]
    oid = {(j, k): o for o, (j, k, _, _) in enumerate(ops)}
    var, vid = [], {}
    for o, (j, k, m, p) in enumerate(ops):
        head = sum(q for _, q in jobs[j][:k])
        tail = sum(q for _, q in jobs[j][k:])
        for t in range(head, T - tail + 1):
            vid[o, t] = len(var)
            var.append((o, t))
    by_op = defaultdict(list)
    for (o, t), v in vid.items():
        by_op[o].append((t, v))
    Q = defaultdict(float)

    def add(a, b):
        Q[min(a, b), max(a, b)] += A

    for o in range(len(ops)):
        _onehot(Q, [v for _, v in by_op[o]], A)
    for j, job in enumerate(jobs):  # ジョブ内の順序: 後の工程は前の工程が終わってから
        for k in range(len(job) - 1):
            o, o2, p = oid[j, k], oid[j, k + 1], job[k][1]
            for t, a in by_op[o]:
                for t2, b in by_op[o2]:
                    if t2 < t + p:
                        add(a, b)
    on = defaultdict(list)
    for o, (_, _, m, _) in enumerate(ops):
        on[m].append(o)
    for m, os_ in on.items():  # 同じ機械の工程は重ならない
        for x, o in enumerate(os_):
            for o2 in os_[x + 1:]:
                p, p2 = ops[o][3], ops[o2][3]
                for t, a in by_op[o]:
                    for t2, b in by_op[o2]:
                        if t < t2 + p2 and t2 < t + p:
                            add(a, b)
    return dict(Q), var, ops


def jssp_qubo_size(jobs, T):
    """メイクスパン T の時間添字 QUBO の変数の数."""
    n = 0
    for job in jobs:
        total = sum(p for _, p in job)
        n += len(job) * max(0, T - total + 1)
    return n


def jssp_decode(jobs, nm, var, ops, x):
    """0/1 の解を開始時刻に直し、すべての制約を満たせばメイクスパン、そうでなければ None."""
    start = {}
    for v, b in enumerate(x):
        if b:
            o, t = var[v]
            if o in start:
                return None
            start[o] = t
    if len(start) != len(ops):
        return None
    oid = {(j, k): o for o, (j, k, _, _) in enumerate(ops)}
    for j, job in enumerate(jobs):
        for k in range(len(job) - 1):
            if start[oid[j, k + 1]] < start[oid[j, k]] + job[k][1]:
                return None
    iv = sorted((ops[o][2], start[o], start[o] + ops[o][3]) for o in start)
    for (m, s, e), (m2, s2, _) in zip(iv, iv[1:]):
        if m == m2 and s2 < e:
            return None
    return max(e for _, _, e in iv)


def anneal_jssp(data, tl, seed, sample):
    """JSSP を時間添字の QUBO にして、メイクスパン T を二分探索しながらアニーラ sample を呼ぶ.
    T の初期範囲は下界と貪欲法の上界. T で実行可能解が出れば上側をそのメイクスパンに、
    出なければ下側を T+1 に狭める（出なかった T が実行不能とは限らないので、最適性の証明にはならない）.
    範囲が尽きたら、解が見つかっていない最小の T で制限時間まで試し続ける."""
    tm = Timer(tl)
    jobs, nm = data
    lo, hi = jssp_horizon(jobs, nm)
    nvar = jssp_qubo_size(jobs, hi)
    if nvar > MAX_VARS:
        return dict(obj=None, lb=None, qubo_vars=nvar, skipped=f"QUBO の変数が {nvar} 個で上限 {MAX_VARS} を超える")
    lb = lo
    best, calls, samples, feas, tried = None, 0, 0, 0, []
    last = 0.0
    T = hi
    while (best is None or best > lb) and (tm.left() > last * 1.2 or calls == 0):
        Q, var, ops = jssp_qubo(jobs, nm, T)
        t0 = time.perf_counter()
        xs = sample(Q, seed * 100000 + calls)  # シードごとに乱数の種が重ならないようにする
        last = time.perf_counter() - t0
        calls += 1
        ms = None
        for x in xs:
            samples += 1
            c = jssp_decode(jobs, nm, var, ops, x)
            if c is not None:
                feas += 1
                ms = c if ms is None else min(ms, c)
        tried.append([T, ms])
        if ms is not None:
            best = ms if best is None else min(best, ms)
            hi = best - 1
        else:
            lo = T + 1
        if lo > hi:  # 探索範囲が尽きたら、まだ解が見つかっていない最小の T で時間まで繰り返す
            lo = hi
        T = (lo + hi) // 2
    return dict(obj=best, lb=None, qubo_vars=nvar, calls=calls, samples=samples, feasible_samples=feas,
                horizons=tried)


def samples(res, nvar):
    """OpenJij / dimod の結果を、変数番号順の 0/1 配列のリストに直す."""
    order = np.asarray(list(res.variables))
    out = []
    for row in res.record.sample:
        x = np.zeros(nvar, np.int8)
        x[order] = row
        out.append(x)
    return out


def normalized(sample):
    """係数の絶対値の最大が 1 になるよう QUBO を割ってからアニーラに渡す（解は変わらない）.
    OpenJij-SQA の逆温度などの既定値は、係数がこの程度の大きさであることを前提にしている."""
    def f(Q, seed):
        s = max(abs(v) for v in Q.values()) or 1.0
        return sample({k: v / s for k, v in Q.items()}, seed)
    return f


def run(data, tl, seed, sample):
    """問題に応じて anneal_jssp か anneal_tsp を呼ぶ. QUBO は正規化してからアニーラに渡す."""
    sample = normalized(sample)
    if isinstance(data, tuple):  # JSSP は (jobs, 機械数)、TSP / ATSP は距離行列
        return anneal_jssp(data, tl, seed, sample)
    return anneal_tsp(data, tl, seed, sample)
