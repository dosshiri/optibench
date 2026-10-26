"""時間枠付き配送計画問題（VRPTW）. インスタンスは Solomon（100 顧客と、その先頭 25・50 顧客）と Gehring & Homberger（200 顧客）.

デポ（地点 0）から容量 Q の車両を最大 K 台出し、各顧客 i をちょうど 1 台が 1 回訪れて需要 q_i を配る.
顧客 i での作業は時間枠 [e_i, l_i] の中で始め（早く着いたら e_i まで待つ）、s_i かけて行う.
車両はデポの時間枠 [e_0, l_0] の中で出発し、戻る. 目的関数は総移動距離（車両台数は問わない）.

距離と移動時間は DIMACS 2021 VRPTW 部門の規約どおり、ユークリッド距離を小数第 1 位で切り捨てた値で、
既知最良値・最適値もこの規約の値（CVRPLIB の一覧）. 内部では 10 倍した整数で扱い、目的関数値は 10 で割って小数第 1 位で返す.

解法に渡すデータは dict（時間・距離はすべて 10 倍した整数）:
  D   距離 = 移動時間の行列 (n×n, 地点 0 がデポ, 対称)
  xy  座標 (n×2, 元の単位)
  q   需要 (長さ n, q_0 = 0),  Q  車両の容量,  K  車両の上限台数
  e   時間枠の開始,  l  時間枠の終了,  s  作業時間 (s_0 = 0)
解は経路のリスト. 各経路はデポを除いた顧客の訪問順（例: [[3, 1, 7], [2, 5]]）.
"""
import numpy as np

from registry import Problem, register_problem

# 既知最良値（CVRPLIB / PyVRP/Instances の一覧. 距離を小数第 1 位で切り捨てる規約、目的は総距離）.
# Solomon の 100 顧客 56 件は全件で最適性が証明済み（Pecin ら 2017）. 25・50 顧客版は Kohl ら（1999）以来の
# 最適値で、本リポジトリの CP-SAT / MIP でも証明を確かめた. 200 顧客は Pecin ら（2017）が 60 件中 51 件を解いた
BKS = {"R101.25": 617.1, "R201.25": 463.3, "R101.50": 1044.0, "R201.50": 791.9,
       "C101": 827.3, "R101": 1637.7, "R112": 948.6, "RC101": 1619.8, "R201": 1143.2, "RC208": 776.1,
       "R1_2_1": 4667.2, "RC1_2_1": 3516.9}


def load(name):
    """PyVRP 版の VRPLIB 形式（NODE_COORD / DEMAND / TIME_WINDOW の各 SECTION）を読む."""
    head, sec, rows = {}, None, {}
    with open(PROBLEM.instance_path(name), encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s == "EOF":
                continue
            if s.endswith("_SECTION"):
                sec = s
                rows[sec] = []
            elif sec is None:
                k, v = s.split(":", 1)
                head[k.strip()] = v.strip()
            else:
                rows[sec].append([int(x) for x in s.split()])
    n = int(head["DIMENSION"])
    xy = np.array([r[1:3] for r in rows["NODE_COORD_SECTION"][:n]], dtype=np.int64)
    q = np.array([r[1] for r in rows["DEMAND_SECTION"][:n]], dtype=np.int64)
    tw = np.array([r[1:3] for r in rows["TIME_WINDOW_SECTION"][:n]], dtype=np.int64) * 10
    diff = xy[:, None, :] - xy[None, :, :]
    # 浮動小数の誤差で 1 つ下に切り捨てないよう、整数の二乗和から 10 倍の距離の整数部分を求める
    sq = 100 * (diff ** 2).sum(axis=2)
    D = np.floor(np.sqrt(sq.astype(float))).astype(np.int64)
    D += ((D + 1) ** 2 <= sq)
    D -= (D ** 2 > sq)
    s = np.full(n, int(head["SERVICE_TIME"]) * 10, dtype=np.int64)
    s[0] = 0
    return dict(D=D, xy=xy, q=q, Q=int(head["CAPACITY"]), K=int(head["VEHICLES"]),
                e=tw[:, 0].copy(), l=tw[:, 1].copy(), s=s)


def route_check(inst, route):
    """1 本の経路の (距離, 時間枠の遅れの合計, 容量の超過量). 距離と遅れは 10 倍の整数."""
    D, e, l, s = inst["D"], inst["e"], inst["l"], inst["s"]
    t, late, dist, prev = int(e[0]), 0, 0, 0
    for v in list(route) + [0]:
        dist += int(D[prev, v])
        t = max(t + int(s[prev]) + int(D[prev, v]), int(e[v]))
        late += max(0, t - int(l[v]))
        prev = v
    return dist, late, max(0, int(inst["q"][list(route)].sum()) - inst["Q"]) if len(route) else 0


def evaluate(inst, routes):
    """全顧客をちょうど 1 回ずつ訪れ、台数・容量・時間枠を守る経路の集まりなら総距離（小数第 1 位）、そうでなければ None."""
    routes = [[int(v) for v in r] for r in routes if len(r)]
    n = inst["D"].shape[0]
    if len(routes) > inst["K"] or sorted(v for r in routes for v in r) != list(range(1, n)):
        return None
    total = 0
    for r in routes:
        dist, late, over = route_check(inst, r)
        if late or over:
            return None
        total += dist
    return round(total / 10, 1)


PROBLEM = register_problem(Problem(
    name="vrptw", label="VRPTW（時間枠付き配送計画）", unit="総距離",
    ref_label="既知最良値（Solomon 100 顧客以下は証明済みの最適値）", instance_file="vrptw/{}.vrp",
    instances=list(BKS), load=load, reference=BKS.get, evaluate=evaluate, order=27))
