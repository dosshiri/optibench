"""ナーススケジューリング問題 (NSP) のモデル・インスタンス生成・評価.

Curtois & Qu (2014) "Computational results on new staff scheduling benchmark instances"
の制約体系 (schedulingbenchmarks.org の Shift Scheduling Benchmark) に倣った合成インスタンス.
  ハード制約: 1日1勤務, 禁止勤務遷移, 勤務種別ごとの上限回数, 総勤務時間の上下限,
              最大連続勤務日数, 最小連続勤務日数, 最小連続休日数, 最大勤務週末数, 指定休日
  ソフト制約: 勤務希望 (on) / 勤務回避希望 (off), 必要人数の不足 (重み100/人) と超過 (重み1/人)
解法に渡すデータはインスタンスの dict, 解は R[e][d] = 勤務種別 index または -1（休み）.
インスタンスの再生成: python src/make_instances.py nsp
"""
import json
import os

import numpy as np
from numba import njit

from common import INST
from registry import Problem, register_problem

HARD = 10000  # 局所探索でハード制約違反に課すペナルティ
NSP_DIR = os.path.join(INST, "nsp")

SIZES = {  # name: (従業員数, 日数, 勤務種別数, seed)
    "nsp01": (8, 14, 1, 1),
    "nsp02": (14, 14, 2, 2),
    "nsp03": (20, 28, 2, 3),
    "nsp04": (30, 28, 3, 4),
    "nsp05": (40, 28, 3, 5),
    "nsp06": (50, 42, 4, 6),
}


def generate(name):
    E, H, S, seed = SIZES[name]
    rng = np.random.default_rng(seed)
    shift_names = ["E", "D", "L", "N"][:S] if S < 4 else ["E", "D", "L", "N"]
    if S == 1:
        shift_names = ["D"]
    elif S == 2:
        shift_names = ["E", "L"]
    elif S == 3:
        shift_names = ["E", "L", "N"]
    length = {"E": 480, "D": 480, "L": 480, "N": 600}
    # 休息時間が足りない遷移を禁止
    forbid_pairs = {("L", "E"), ("N", "E"), ("N", "D"), ("N", "L"), ("L", "D")}
    forbid = [[1 if (a, b) in forbid_pairs else 0 for b in shift_names] for a in shift_names]
    weeks = H // 7
    emps = []
    while len(emps) < E:
        full = rng.random() < 0.75  # 常勤 75%, 非常勤 25%
        days = 5 if full else 3
        maxmin = days * weeks * 480 + 480
        minmin = max(0, maxmin - 4 * 480)
        maxtype = [weeks * 5 if s != "N" else weeks * 2 for s in shift_names]
        if rng.random() < 0.3 and "N" in shift_names:
            maxtype[shift_names.index("N")] = 0  # 夜勤不可の職員
        off = sorted(rng.choice(H, size=weeks, replace=False).tolist())
        on_req, off_req = [], []
        for _ in range(weeks):
            d = int(rng.integers(H))
            if d not in off:
                on_req.append([d, int(rng.integers(S)), int(rng.integers(1, 4))])
            d = int(rng.integers(H))
            off_req.append([d, int(rng.integers(S)), int(rng.integers(1, 4))])
        emps.append(dict(max_type=maxtype, min_minutes=minmin, max_minutes=maxmin, max_consec=5,
                         min_consec=2, min_off=2, max_weekends=max(1, (2 * weeks + 2) // 3), days_off=off,
                         on_req=on_req, off_req=off_req))
        if not _employee_feasible(emps[-1], H, shift_names, [length[s] for s in shift_names], forbid):
            emps.pop()  # 単独で実行不能な職員条件は引き直す
    # 必要人数: 供給能力の約 80% を日・勤務種別にばらつかせて配分
    cap = sum((5 if e["max_minutes"] > 4 * weeks * 480 else 3) * weeks for e in emps)
    base = 0.8 * cap / (H * S)
    cover = np.maximum(0, np.round(base * rng.uniform(0.6, 1.4, size=(H, S)))).astype(int)
    for d in range(H):
        if d % 7 in (5, 6):
            cover[d] = np.maximum(0, cover[d] - 1)
    inst = dict(name=name, horizon=H, shifts=shift_names, length=[length[s] for s in shift_names],
                forbid=forbid, employees=emps, cover=cover.tolist(), w_under=100, w_over=1)
    os.makedirs(NSP_DIR, exist_ok=True)
    with open(os.path.join(NSP_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(inst, f, indent=1)
    return inst


def _employee_feasible(emp, H, shifts, length, forbid):
    from solvers.nsp import _cpsat
    sub = dict(name="chk", horizon=H, shifts=shifts, length=length, forbid=forbid, employees=[emp],
               cover=[[0] * len(shifts)] * H, w_under=100, w_over=1)
    return _cpsat(sub, 10, 1, 1)["obj"] is not None


def load_instance(name):
    with open(os.path.join(NSP_DIR, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 配列表現
def to_arrays(inst):
    E, H, S = len(inst["employees"]), inst["horizon"], len(inst["shifts"])
    A = dict(E=E, H=H, S=S, length=np.array(inst["length"], np.int64),
             forbid=np.array(inst["forbid"], np.int64), cover=np.array(inst["cover"], np.int64),
             w_under=inst["w_under"], w_over=inst["w_over"])
    A["max_type"] = np.array([e["max_type"] for e in inst["employees"]], np.int64)
    for k in ("min_minutes", "max_minutes", "max_consec", "min_consec", "min_off", "max_weekends"):
        A[k] = np.array([e[k] for e in inst["employees"]], np.int64)
    dayoff = np.zeros((E, H), np.int64)
    pref = np.zeros((E, H, S), np.int64)  # 割当時に加算されるペナルティ (on 希望は未割当で加算)
    on_pen = np.zeros((E, H, S), np.int64)
    for i, e in enumerate(inst["employees"]):
        for d in e["days_off"]:
            dayoff[i, d] = 1
        for d, s, w in e["off_req"]:
            pref[i, d, s] += w
        for d, s, w in e["on_req"]:
            on_pen[i, d, s] += w
    A["dayoff"], A["pref"], A["on_pen"] = dayoff, pref, on_pen
    return A


@njit(cache=True)
def emp_cost(row, e, length, forbid, max_type, min_minutes, max_minutes, max_consec, min_consec,
             min_off, max_weekends, dayoff, pref, on_pen):
    """従業員 e の勤務行 row (-1=休み) のコスト. 戻り値 (ハード違反量, ソフトペナルティ)."""
    H = row.shape[0]
    S = length.shape[0]
    hard = 0
    soft = 0
    cnt = np.zeros(S, np.int64)
    minutes = 0
    for d in range(H):
        s = row[d]
        for t in range(S):
            if on_pen[e, d, t] > 0 and s != t:
                soft += on_pen[e, d, t]
        if s >= 0:
            cnt[s] += 1
            minutes += length[s]
            soft += pref[e, d, s]
            if dayoff[e, d]:
                hard += 1
            if d + 1 < H and row[d + 1] >= 0 and forbid[s, row[d + 1]]:
                hard += 1
    for t in range(S):
        if cnt[t] > max_type[e, t]:
            hard += cnt[t] - max_type[e, t]
    if minutes > max_minutes[e]:
        hard += (minutes - max_minutes[e] + 479) // 480
    if minutes < min_minutes[e]:
        hard += (min_minutes[e] - minutes + 479) // 480
    # 連続勤務・連続休日 (期間の端に接する連は最小制約の対象外)
    d = 0
    while d < H:
        w = row[d] >= 0
        k = d
        while k < H and (row[k] >= 0) == w:
            k += 1
        L = k - d
        if w:
            if L > max_consec[e]:
                hard += L - max_consec[e]
            if L < min_consec[e] and d > 0 and k < H:
                hard += min_consec[e] - L
        else:
            if L < min_off[e] and d > 0 and k < H:
                hard += min_off[e] - L
        d = k
    wk = 0
    for w0 in range(5, H, 7):
        if row[w0] >= 0 or (w0 + 1 < H and row[w0 + 1] >= 0):
            wk += 1
    if wk > max_weekends[e]:
        hard += wk - max_weekends[e]
    return hard, soft


@njit(cache=True)
def cover_cost(cnt, cover, w_under, w_over):
    c = 0
    for d in range(cover.shape[0]):
        for s in range(cover.shape[1]):
            x = cnt[d, s] - cover[d, s]
            c += -x * w_under if x < 0 else x * w_over
    return c


def evaluate(inst, R):
    """R[e, d] = 勤務種別 index または -1. 戻り値 (ハード違反量, 目的関数値)."""
    A = to_arrays(inst)
    hard = soft = 0
    for e in range(A["E"]):
        h, s = emp_cost(R[e], e, A["length"], A["forbid"], A["max_type"], A["min_minutes"],
                        A["max_minutes"], A["max_consec"], A["min_consec"], A["min_off"],
                        A["max_weekends"], A["dayoff"], A["pref"], A["on_pen"])
        hard += h
        soft += s
    cnt = np.zeros((A["H"], A["S"]), np.int64)
    for e in range(A["E"]):
        for d in range(A["H"]):
            if R[e, d] >= 0:
                cnt[d, R[e, d]] += 1
    return int(hard), int(soft + cover_cost(cnt, A["cover"], A["w_under"], A["w_over"]))


def check(inst, R):
    """解の検証用: 実行可能（ハード制約違反なし）なら目的関数値、そうでなければ None."""
    hard, obj = evaluate(inst, np.asarray(R, np.int64))
    return obj if hard == 0 else None


def generate_all():
    for n in SIZES:
        inst = generate(n)
        print(n, len(inst["employees"]), inst["horizon"], inst["shifts"], np.array(inst["cover"]).sum())


SHIFT_NAME = {"E": "早番", "D": "日勤", "L": "遅番", "N": "夜勤"}


def instance_info(name):
    inst = load_instance(name)
    E, H, S = len(inst["employees"]), inst["horizon"], len(inst["shifts"])
    req = sum(len(e["on_req"]) + len(e["off_req"]) for e in inst["employees"])
    return {"職員数": E, "日数": H, "勤務種別": "・".join(SHIFT_NAME.get(x, x) for x in inst["shifts"]),
            "0-1 変数": E * H * S, "必要人数（延べ）": int(np.sum(inst["cover"])), "勤務の希望": req}


register_problem(Problem(
    name="nsp", label="ナーススケジューリング", unit="ペナルティ",
    ref_label="全 run・全手法中の最良値（合成インスタンス、最適値未知）", instance_file="nsp/{}.json",
    instances=lambda: sorted(f[:-5] for f in os.listdir(NSP_DIR) if f.endswith(".json")),
    load=load_instance, evaluate=check, generate=generate_all, display="value", order=40,
    discussion="nsp.html", instance_info=instance_info,
    setting=dict(
        brief="職員ごとに毎日の勤務か休みを決めて勤務表を作る。労務上の規則を守りつつ、人数の過不足と希望の違反を最小にする。",
        summary=("職員ごとに、計画期間の毎日の勤務（早番・遅番・夜勤などの勤務種別のどれか、または休み）を決めて勤務表を作る。"
                 "労務上の規則（ハード制約）を必ず守ったうえで、日ごとの必要人数をなるべく満たし、職員の希望をなるべくかなえる。"
                 "病棟の看護師のほか、コールセンターや工場、小売店のシフト作成に現れる。"),
        variables="x_eds = 1 は「職員 e が日 d に勤務種別 s で働く」ことを表す 0-1 変数。どの勤務種別も 0 の日は休み。",
        constraints=[
            "1 日に入る勤務は 1 つまで。",
            "休息が足りない勤務の並び（遅番や夜勤の翌日の早番など）を禁止する。",
            "勤務種別ごとの回数の上限（夜勤不可の職員は夜勤 0 回）。",
            "期間中の総勤務時間の下限と上限。",
            "連続勤務は 2〜5 日、連続休日は 2 日以上（期間の端にかかる連続は除く）。",
            "土日のどちらかに勤務する週末の数の上限。",
            "職員ごとの指定休日には勤務しない。",
        ],
        objective=("ペナルティの合計を最小化する。必要人数に対する不足は 1 人につき 100、超過は 1 人につき 1。"
                   "勤務の希望がかなわないとき、または避けたい勤務に入れたときは、希望ごとの重み（1〜3）を加える。"),
        note="上の制約はすべてハード制約で、1 つでも破った勤務表は実行不能（解なし）として扱う。",
        instances=("Curtois & Qu (2014) の Shift Scheduling Benchmark（schedulingbenchmarks.org）と同じ制約体系を持つ合成インスタンス 6 問で、"
                   "8〜50 人、14〜42 日、勤務種別 1〜4 種類。職員の約 75% が常勤（週 5 日相当）、残りが非常勤（週 3 日相当）で、"
                   "必要人数は職員が働ける量の約 8 割を日と勤務種別にばらつかせて配分した（週末は 1 人ずつ減らす）。"
                   "python src/make_instances.py nsp で同じものを再生成できる。最適値が分からないため、参照値は全 run・全手法中の最良値。"),
        scale="合成 6 問（8〜50 人 × 14〜42 日、勤務種別 1〜4）")))
