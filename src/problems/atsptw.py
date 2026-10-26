"""時間枠付き非対称巡回セールスマン問題（ATSP-TW）. インスタンスは Ascheuer の rbg 系（AFG セット）.

デポ（都市 0）を時刻 r_0 に出発し、各都市 i をちょうど 1 回、時間枠 [r_i, d_i] の中で訪れてデポに戻る.
r_i より早く着いたら r_i まで待つ（待ち時間は費用にならない）. d_i より遅れて着くのは実行不能.
目的関数は通った弧の移動時間の合計（作業時間は移動時間行列に含まれている）.

解法に渡すデータは dict:
  D  整数の移動時間行列 (n×n, 0-index, 都市 0 がデポ, 非対称)
  r  各都市の時間枠の開始 (長さ n)
  d  各都市の時間枠の終了 (長さ n). デポの d_0 は戻りの締め切り
解は都市 0 から始まる訪問順のリスト（最後にデポへ戻る弧は含めない）.
"""
import numpy as np

from registry import Problem, register_problem

# 最適値（López-Ibáñez の TSPTW インスタンス集の一覧. 9 問とも最適性が証明済み.
# Ascheuer, Fischetti & Grötschel 2001; Dash et al. 2012 ほか）
OPT = {"rbg019a": 1262, "rbg027a": 5091, "rbg041a": 2598, "rbg055a": 3761, "rbg067a": 4625,
       "rbg092a": 7158, "rbg125a": 7936, "rbg172a": 10950, "rbg233": 14992}


def load(name):
    """Ascheuer 形式（1 行目に都市数 n、続く n 行に移動時間行列、続く n 行に時間枠）を読む."""
    with open(PROBLEM.instance_path(name), encoding="utf-8") as f:
        lines = [l.split() for l in f if l.strip() and not l.lstrip().startswith("#")]
    n = int(lines[0][0])
    D = np.array([[int(x) for x in row] for row in lines[1:n + 1]], dtype=np.int64)
    tw = np.array([[int(x) for x in row[:2]] for row in lines[n + 1:2 * n + 1]], dtype=np.int64)
    np.fill_diagonal(D, 0)
    return dict(D=D, r=tw[:, 0].copy(), d=tw[:, 1].copy())


def schedule(inst, tour):
    """訪問順 tour の (総移動時間, 時間枠の超過量の合計). 各都市への到着が遅れた分を超過量として足す."""
    D, r, d = inst["D"], inst["r"], inst["d"]
    t, late, cost = int(r[0]), 0, 0
    for a, b in zip(tour, list(tour[1:]) + [tour[0]]):
        cost += int(D[a, b])
        t = max(t + int(D[a, b]), int(r[b]))
        late += max(0, t - int(d[b]))
    return cost, late


def evaluate(inst, tour):
    """デポから始まり全都市を 1 回ずつ訪れ、すべての時間枠を守る巡回路なら総移動時間、そうでなければ None."""
    tour = [int(x) for x in tour]
    if not tour or tour[0] != 0 or sorted(tour) != list(range(inst["D"].shape[0])):
        return None
    cost, late = schedule(inst, tour)
    return cost if late == 0 else None


def instance_info(name):
    inst = load(name)
    w = (inst["d"] - inst["r"])[1:]
    return {"都市数（デポを含む）": len(inst["r"]), "時間枠の幅（中央値）": int(np.median(w)),
            "デポに戻る締め切り": int(inst["d"][0]), "最適値": OPT.get(name)}


PROBLEM = register_problem(Problem(
    name="atsptw", label="ATSP-TW（時間枠付き非対称巡回セールスマン）", unit="移動時間",
    ref_label="公表最適値（AFG セット）", instance_file="atsptw/{}.tw", instances=list(OPT),
    load=load, reference=OPT.get, evaluate=evaluate, order=25, discussion="atsptw.html",
    instance_info=lambda name: instance_info(name),
    setting=dict(
        brief="各地点を時間枠の中で 1 回ずつ訪れてデポに戻る巡回路のうち、移動時間の合計が最小のものを求める。移動時間は非対称。",
        summary=("デポ（都市 0）を出発し、各都市をちょうど 1 回ずつ、それぞれに決められた時間枠 [r_i, d_i] の中で訪れてデポに戻る"
                 "巡回路のうち、移動時間の合計が最小のものを求める。移動時間は向きによって異なってよい。"
                 "時間指定のある配送や、作業ごとに開始可能時刻と締め切りがある作業順序の決定に現れる。"),
        variables=("都市を回る順序と、各都市での作業開始時刻 t_i。0-1 変数で書くと、x_ij = 1 は「都市 i の次に都市 j へ行く」ことを表す。"),
        constraints=[
            "各都市をちょうど 1 回訪れ、全都市が 1 本の巡回路でつながる（部分巡回路を作らない）。",
            "時間枠: r_i ≤ t_i ≤ d_i。r_i より早く着いたら r_i まで待つ（待ち時間は費用にならない）。d_i より遅れて着くのは実行不能。",
            "時刻のつじつま: 都市 i の次に都市 j へ行くなら t_j ≥ t_i + c_ij（c_ij は i での作業時間を含む移動時間）。",
            "デポに戻る時刻は、デポの締め切り d_0 までに収める。",
        ],
        objective="通った弧の移動時間の合計 Σ c_ij x_ij を最小化する（待ち時間は含めない）。",
        instances=("Ascheuer が自動倉庫のスタッカークレーンの作業順序から作った rbg 系（AFG セット）の 9 問で、デポを含め"
                   "20〜232 都市。López-Ibáñez の TSPTW インスタンス集で公表されている最適値（9 問とも最適性が証明済み）を参照値にした。"
                   "時間枠の幅はほとんどの都市で 600（rbg019a は 300、rbg027a は約 1,700）で、"
                   "デポに戻る締め切りまでの計画期間（約 6,000〜53,000）に比べてかなり狭い。"),
        scale="AFG セット 9 問（20〜232 都市）")))
