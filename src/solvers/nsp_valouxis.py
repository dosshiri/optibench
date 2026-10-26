"""NSP の 2 段階解法 (Valouxis et al. 2012 の考え方の再実装).

Valouxis, Gogos, Goulas, Alefragis, Housos (2012) "A systematic two phase approach for the nurse
rostering problem", EJOR 219(2), 425-433. INRC-2010 の全トラックで 1 位.
論文の公開コードは無いため、論文の要約にある骨組みを HiGHS で組み直した:
  第 1 段階: 勤務種別を区別せず「誰がどの日に働くか（日ごとの勤務量）」を整数計画で決める
  第 2 段階: 働く日を固定したまま、具体的な勤務種別を整数計画で割り当てる
  局所最適化: 数人の職員 × 数日、または全職員 × 数日の部分勤務表だけを自由にした整数計画を解き直し、制限時間まで繰り返す
"""
import numpy as np

from common import Timer
from problems.nsp import evaluate, to_arrays
from registry import solver
from solvers.nsp import _build


def _highs(threads, seed):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", threads)
    h.setOptionValue("random_seed", seed)
    return h


def _phase1(inst, A, tl, threads, seed):
    """勤務種別をまとめた集約モデル. y[e,d] = 1 なら職員 e は日 d に働く. 戻り値 y (E×H) または None."""
    import highspy
    E, H, S = A["E"], A["H"], A["S"]
    h = _highs(threads, seed)
    h.setOptionValue("time_limit", float(max(tl, 0.5)))
    I = highspy.HighsVarType.kInteger
    lin = lambda t: sum(c * v for c, v in t)
    y = [[h.addVariable(lb=0, ub=0 if A["dayoff"][e, d] else 1, type=I) for d in range(H)] for e in range(E)]
    obj, const, ns = [], 0, []
    for e in range(E):
        # 勤務種別ごとの回数 n[e,s]（どの日に入るかは第 2 段階で決める）
        n = [h.addVariable(lb=0, ub=int(A["max_type"][e, s]), type=I) for s in range(S)]
        ns.append(n)
        h.addConstr(lin([(1, v) for v in n]) == lin([(1, v) for v in y[e]]))
        mins = lin([(int(A["length"][s]), n[s]) for s in range(S)])
        h.addConstr(mins <= int(A["max_minutes"][e]))
        h.addConstr(mins >= int(A["min_minutes"][e]))
        c = int(A["max_consec"][e])
        for d in range(H - c):
            h.addConstr(lin([(1, y[e][k]) for k in range(d, d + c + 1)]) <= c)
        for l in range(1, int(A["min_consec"][e])):
            for d in range(1, H - l):
                h.addConstr(lin([(-1, y[e][d - 1])] + [(1, y[e][k]) for k in range(d, d + l)]
                                + [(-1, y[e][d + l])]) <= l - 1)
        for l in range(1, int(A["min_off"][e])):
            for d in range(1, H - l):
                h.addConstr(lin([(1, y[e][d - 1])] + [(-1, y[e][k]) for k in range(d, d + l)]
                                + [(1, y[e][d + l])]) <= 1)
        ks = []
        for w0 in range(5, H, 7):
            kv = h.addVariable(lb=0, ub=1, type=I)
            for dd in (w0, w0 + 1):
                if dd < H:
                    h.addConstr(kv >= y[e][dd])
            ks.append((1, kv))
        h.addConstr(lin(ks) <= int(A["max_weekends"][e]))
        for d in range(H):
            on = int(A["on_pen"][e, d].sum())  # 勤務希望は休むと必ず違反になる
            if on:
                const += on
                obj.append((-on, y[e][d]))
            if S == 1 and A["pref"][e, d, 0]:
                obj.append((int(A["pref"][e, d, 0]), y[e][d]))
        # 遅番・夜勤の翌日の早番禁止などは第 2 段階に任せる
    # 必要人数は日ごとの合計で見る（勤務種別ごとの内訳は第 2 段階）
    for d in range(H):
        need = int(A["cover"][d].sum())
        over, under = h.addVariable(lb=0, ub=E, type=I), h.addVariable(lb=0, ub=need, type=I)
        h.addConstr(lin([(1, y[e][d]) for e in range(E)]) - over + under == need)
        obj += [(A["w_over"], over), (A["w_under"], under)]
        for s in range(S):  # 夜勤不可の職員などを考え、勤務種別 s に入れる職員だけで必要人数を満たせるかも見る
            need = int(A["cover"][d, s])
            if need:
                us = h.addVariable(lb=0, ub=need, type=I)
                h.addConstr(lin([(1, y[e][d]) for e in range(E) if A["max_type"][e, s] > 0]) + us >= need)
                obj.append((A["w_under"], us))
    for s in range(S):  # 期間を通した勤務種別ごとの延べ人数も必要人数に合わせる（夜勤で総勤務時間を稼ぐ解を避ける）
        need = int(A["cover"][:, s].sum())
        over, under = h.addVariable(lb=0, ub=E * H, type=I), h.addVariable(lb=0, ub=need, type=I)
        h.addConstr(lin([(1, ns[e][s]) for e in range(E)]) - over + under == need)
        obj += [(A["w_over"], over), (A["w_under"], under)]
    h.minimize(lin(obj) + const)
    if h.getInfo().primal_solution_status != 2:
        return None
    val = h.getSolution().col_value
    return np.array([[val[y[e][d].index] > 0.5 for d in range(H)] for e in range(E)])


class _Full:
    """元問題の 0-1 線形モデル（MIP-HiGHS と同じ）. 変数の上下限と「1 日 1 勤務」の行の下限を
    書き換えて、第 2 段階と局所最適化の部分問題を同じモデルの上で解く."""

    def __init__(self, inst, threads, seed):
        import highspy
        self.hs = highspy
        h = self.h = _highs(threads, seed)
        lin = lambda t: sum(c * v for c, v in t)
        day_rows = {}

        def add_le(t, b):
            c = h.addConstr(lin(t) <= b)
            if b == 1 and all(k == 1 for k, _ in t):
                day_rows.setdefault(frozenset(v.index for _, v in t), c.index)

        I = highspy.HighsVarType.kInteger
        self.x, obj, const, (E, H, S) = _build(
            inst, lambda ub: h.addVariable(lb=0, ub=ub, type=I), lambda ub: h.addVariable(lb=0, ub=ub, type=I),
            add_le, lambda t, b: h.addConstr(lin(t) >= b), lambda t, b: h.addConstr(lin(t) == b))
        h.setObjective(lin(obj) + const, highspy.ObjSense.kMinimize)  # minimize() は解き始めてしまう
        self.E, self.H, self.S = E, H, S
        self.col = np.array([[[self.x[e, d, s].index for s in range(S)] for d in range(H)] for e in range(E)])
        self.ub0 = np.array(h.getLp().col_upper_)
        self.row = np.array([[day_rows[frozenset(self.col[e, d].tolist())] for d in range(H)] for e in range(E)])
        self.ncol = h.getNumCol()

    def _set_cols(self, idx, lo, up):
        if len(idx):
            self.h.changeColsBounds(len(idx), np.asarray(idx, np.int32), np.asarray(lo, float), np.asarray(up, float))

    def _set_rows(self, rows, lo):
        if len(rows):
            rows = np.asarray(rows, np.int32)
            self.h.changeRowsBounds(len(rows), rows, np.full(len(rows), float(lo)), np.ones(len(rows)))

    def reset(self):
        allx = self.col.ravel()
        self._set_cols(allx, np.zeros(len(allx)), self.ub0[allx])
        self._set_rows(self.row.ravel(), 0)

    def solve(self, tl, hint=None):
        h = self.h
        h.setOptionValue("time_limit", float(max(tl, 0.1)))
        if hint is not None:
            h.setSolution(_sol(self.hs, hint))
        h.run()
        if h.getInfo().primal_solution_status != 2:
            return None, False
        val = np.array(h.getSolution().col_value)
        return val, h.getModelStatus() == self.hs.HighsModelStatus.kOptimal

    def roster(self, val):
        R = -np.ones((self.E, self.H), np.int64)
        on = val[self.col] > 0.5
        e, d, s = np.nonzero(on)
        R[e, d] = s
        return R

    def phase2(self, work):
        """働く日 work[e,d] を固定: 働かない日は全勤務種別を 0、働く日は「1 日 1 勤務」の行を = 1 にする."""
        off = self.col[~work].ravel()
        self._set_cols(off, np.zeros(len(off)), np.zeros(len(off)))
        self._set_rows(self.row[work], 1)

    def neighborhood(self, R, free):
        """free[e,d] が True のマスだけ自由にし、それ以外は今の勤務表 R に固定する."""
        fixed = ~free
        idx = self.col[fixed]  # (n, S)
        cur = R[fixed]
        val = (np.arange(self.S)[None, :] == cur[:, None]).astype(float)
        self._set_cols(idx.ravel(), val.ravel(), val.ravel())
        fr = self.col[free].ravel()
        self._set_cols(fr, np.zeros(len(fr)), self.ub0[fr])


def _sol(hs, val):
    s = hs.HighsSolution()
    s.col_value = list(val)
    s.value_valid = True
    return s


def _two_phase(inst, tl, seed, threads):
    tm = Timer(tl)
    rng = np.random.default_rng(seed)
    A = to_arrays(inst)
    E, H = A["E"], A["H"]
    work = _phase1(inst, A, min(0.25 * tl, tm.left() - 1), threads, seed)
    M = _Full(inst, threads, seed)
    val = None
    if work is not None:
        M.phase2(work)
        val, _ = M.solve(min(0.25 * tl, tm.left() - 0.5))
        if val is None:  # 働く日の固定が厳しすぎたら、働く日に休みを入れることだけ許す
            M._set_rows(M.row.ravel(), 0)
            val, _ = M.solve(min(0.15 * tl, tm.left() - 0.5))
    M.reset()
    if val is None:  # 第 1・2 段階で解が無ければ元のモデルをそのまま解く
        val, _ = M.solve(tm.left() - 0.5)
        if val is None:
            return dict(obj=None, lb=None, hard=None, phase1=work is not None, iterations=0)
    R = M.roster(val)
    best = evaluate(inst, R)[1]
    p2 = best
    # 局所最適化: 部分勤務表を解き直す. 近傍は 2 種類を交互に使う
    #   職員方向: k 人の職員 × w 日（職員どうしの勤務の入れ替え）
    #   日方向:   全職員 × 連続 t 日（その日の勤務種別ごとの人数合わせ）
    # 速く最適まで解ければ近傍を広げ、時間切れなら狭める
    k, t, it, improved = min(3, E), min(2, H), 0, 0
    while tm.left() > 1.0:
        free = np.zeros((E, H), bool)
        by_day = it % 2 == 1
        if by_day:
            d0 = int(rng.integers(0, H - t + 1))
            free[:, d0:d0 + t] = True
        else:
            w = int(rng.integers(min(7, H), H + 1))
            d0 = int(rng.integers(0, H - w + 1))
            emps = rng.choice(E, size=min(k, E), replace=False)
            free[np.ix_(emps, np.arange(d0, d0 + w))] = True
        M.neighborhood(R, free)
        nv, opt = M.solve(min(3.0, tm.left() - 0.5), hint=val)
        it += 1
        if nv is not None:
            R2 = M.roster(nv)
            hd, o = evaluate(inst, R2)
            if hd == 0 and o <= best:
                improved += o < best
                R, val, best = R2, nv, o
        if by_day:
            t = min(H, t + 1) if opt else max(1, t - 1)
        else:
            k = min(E, k + 1) if opt else max(2, k - 1)
    hd, o = evaluate(inst, R)
    return dict(obj=o if hd == 0 else None, lb=None, hard=hd, phase2_obj=p2, iterations=it,
                improved=int(improved), solution=R.tolist())


@solver("Valouxis-2phase", problems="nsp", seeds=3, threads=[1, 4], packages=["highspy"],
        description=("Valouxis ら（2012）が INRC-2010 で 1 位になった 2 段階の整数計画法を、論文の骨組みに沿って HiGHS で組み直したもの。"
                     "第 1 段階で勤務種別を区別せずに職員ごとの働く日を決め、第 2 段階で働く日を固定したまま勤務種別を割り当てる。"
                     "その後、数人の職員の数日分、または全職員の数日分だけを自由にした整数計画を解き直す局所最適化を、制限時間まで繰り返す。"))
def valouxis(inst, tl, seed, threads=1):
    return _two_phase(inst, tl, seed, threads)
