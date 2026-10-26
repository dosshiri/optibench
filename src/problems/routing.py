"""巡回セールスマン問題（TSP: 対称, ATSP: 非対称）. インスタンスは TSPLIB95.
tsp_small は、QUBO（量子・疑似量子アニーリング）で扱える数十都市までの TSP を全手法で比べるための小規模セット.
解法に渡すデータは整数距離行列 D (0-index), 解は都市の巡回順のリスト."""
import os

import numpy as np

from common import INST
from registry import Problem, register_problem

# TSPLIB 公式の最適値 (TSPLIB95 solutions)
TSP_OPT = {"berlin52": 7542, "kroA100": 21282, "ch150": 6528, "a280": 2579,
           "pcb442": 50778, "rat783": 8806}
TSP_SMALL_OPT = {"burma14": 3323, "ulysses16": 6859, "ulysses22": 7013, "bayg29": 1610, "att48": 10628,
                 "eil51": 426, "berlin52": 7542, "st70": 675}
ATSP_OPT = {"ftv33": 1286, "ftv70": 1950, "kro124p": 36230, "ft70": 38673,
            "rbg323": 1326, "rbg443": 2720}


def load_tsplib_matrix(problem, name):
    """TSPLIB を読み込み、整数距離行列 (0-index) を返す."""
    import tsplib95
    ext = "atsp" if problem == "atsp" else "tsp"
    p = tsplib95.load(os.path.join(INST, problem, f"{name}.{ext}"))
    nodes = list(p.get_nodes())
    n = len(nodes)
    D = np.zeros((n, n), dtype=np.int64)
    for a, i in enumerate(nodes):
        for b, j in enumerate(nodes):
            if a != b:
                D[a, b] = p.get_weight(i, j)
    return D


def tour_cost(D, tour):
    t = np.asarray(tour)
    return int(D[t, np.roll(t, -1)].sum())


def evaluate(D, tour):
    """全都市をちょうど 1 回ずつ訪れる巡回路なら総距離、そうでなければ None."""
    if sorted(tour) != list(range(D.shape[0])):
        return None
    return tour_cost(D, tour)


def tsplib_header(problem, name):
    """TSPLIB ファイルの見出し（DIMENSION, EDGE_WEIGHT_TYPE など）を読む. 距離行列は作らない."""
    ext = "atsp" if problem == "atsp" else "tsp"
    head = {}
    with open(os.path.join(INST, problem, f"{name}.{ext}"), encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                break
            k, v = line.split(":", 1)
            head[k.strip()] = v.strip()
    return head


# 距離の与え方（TSPLIB の EDGE_WEIGHT_TYPE）
WEIGHT_TYPE = {"EUC_2D": "平面座標のユークリッド距離", "GEO": "緯度経度から求めた地球上の距離",
               "ATT": "平面座標の疑似ユークリッド距離", "EXPLICIT": "距離行列で直接与える"}

# インスタンスの内容（TSPLIB の COMMENT を日本語にしたもの）
CONTENT = {
    "berlin52": "ベルリン市内の 52 地点（Grötschel）",
    "kroA100": "平面上にランダムに置いた 100 都市（Krolak, Felts & Nelson）",
    "ch150": "平面上の 150 都市（Churritz）",
    "a280": "プリント基板の穴あけ順序（Ludwig）",
    "pcb442": "プリント基板の穴あけ順序（Grötschel, Jünger & Reinelt）",
    "rat783": "格子点を少しずつずらした 783 点（Pulleyblank）",
    "burma14": "ミャンマーの 14 都市（Zaw Win）",
    "ulysses16": "『オデュッセイア』でオデュッセウスがたどった 16 地点（Grötschel & Padberg）",
    "ulysses22": "『オデュッセイア』でオデュッセウスがたどった 22 地点（Grötschel & Padberg）",
    "bayg29": "バイエルン州の 29 都市（Grötschel, Jünger & Reinelt）",
    "att48": "アメリカ本土 48 州の州都（Padberg & Rinaldi）",
    "eil51": "平面上の 51 都市（Christofides & Eilon）",
    "st70": "平面上の 70 都市（Smith & Thompson）",
    "ftv33": "都市部での医薬品配送（Fischetti, Toth & Vigo）",
    "ftv70": "都市部での医薬品配送（Fischetti, Toth & Vigo）",
    "kro124p": "kroA100 の距離を一部変えて非対称にしたもの（Fischetti & Toth）",
    "ft70": "非対称の 70 都市（Fischetti & Toth）",
    "rbg323": "自動倉庫のスタッカークレーンの作業順序（Ascheuer）",
    "rbg443": "自動倉庫のスタッカークレーンの作業順序（Ascheuer）",
}


def instance_info(problem, opt, name):
    head = tsplib_header(problem, name)
    n = int(head["DIMENSION"])
    info = {"都市数": n}
    if problem != "atsp":
        info["距離"] = WEIGHT_TYPE.get(head.get("EDGE_WEIGHT_TYPE"), head.get("EDGE_WEIGHT_TYPE", ""))
    info["内容"] = CONTENT.get(name, head.get("COMMENT", ""))
    if problem == "tsp_small":
        info["QUBO の変数"] = (n - 1) ** 2
    info["最適値"] = opt.get(name)
    return info


TSP_VARIABLES = "都市を回る順序（巡回路）。0-1 変数で書くと、x_ij = 1 は「都市 i の次に都市 j へ行く」ことを表す。"
TSP_CONSTRAINTS = [
    "各都市にちょうど 1 回ずつ入り、ちょうど 1 回ずつ出る（各都市をちょうど 1 回訪れる）。",
    "全都市が 1 本の巡回路でつながる。一部の都市だけで閉じた小さな輪（部分巡回路）を作らない。",
]
TSP_OBJECTIVE = "巡回路の総距離 Σ d_ij x_ij を最小化する。"

register_problem(Problem(
    name="tsp", label="TSP（対称巡回セールスマン）", unit="距離", ref_label="TSPLIB 公表最適値",
    instance_file="tsp/{}.tsp", instances=["berlin52", "kroA100", "ch150", "a280", "pcb442", "rat783"],
    load=lambda name: load_tsplib_matrix("tsp", name), reference=TSP_OPT.get, evaluate=evaluate,
    order=10, packages=["tsplib95"], discussion="tsp.html",
    instance_info=lambda name: instance_info("tsp", TSP_OPT, name),
    setting=dict(
        brief="全都市を 1 回ずつ回って出発点に戻る巡回路のうち、総距離が最短のものを求める。距離は行きと帰りで等しい。",
        summary=("n 個の都市と都市間の距離が与えられたとき、すべての都市をちょうど 1 回ずつ訪れて出発点に戻る巡回路のうち、"
                 "総距離が最短のものを求める。距離は向きによらない（i から j へ行く距離と j から i へ行く距離が等しい）。"
                 "配送や巡回の経路、プリント基板の穴あけ順序などに現れる、組合せ最適化の代表的な問題。"),
        variables=TSP_VARIABLES, constraints=TSP_CONSTRAINTS, objective=TSP_OBJECTIVE,
        instances=("TSPLIB95 から 52〜783 都市の 6 問を選んだ。どれも平面上の座標で与えられ、距離は 2 点間のユークリッド距離を"
                   "整数に丸めた値。参照値は公表されている最適値で、ギャップはこれとの差で測る。"),
        scale="TSPLIB 6 問（52〜783 都市）")))

register_problem(Problem(
    name="atsp", label="ATSP（非対称巡回セールスマン）", unit="距離", ref_label="TSPLIB 公表最適値",
    instance_file="atsp/{}.atsp", instances=["ftv33", "ftv70", "kro124p", "ft70", "rbg323", "rbg443"],
    load=lambda name: load_tsplib_matrix("atsp", name), reference=ATSP_OPT.get, evaluate=evaluate,
    order=20, packages=["tsplib95"], discussion="atsp.html",
    instance_info=lambda name: instance_info("atsp", ATSP_OPT, name),
    setting=dict(
        brief="TSP と同じく全都市を 1 回ずつ回る最短の巡回路を求めるが、距離が行きと帰りで異なる。",
        summary=("TSP と同じく、すべての都市をちょうど 1 回ずつ訪れて出発点に戻る最短の巡回路を求める。"
                 "ただし距離が向きによって異なってよい（i から j と j から i で距離が違う）。"
                 "一方通行や上り下りのある移動のほか、前の品種によって段取り替えの時間が変わる生産順序の決定などに現れる。"),
        variables="都市を回る順序（巡回路）。0-1 変数で書くと、x_ij = 1 は「都市 i の次に都市 j へ行く」ことを表す（向きつき）。",
        constraints=TSP_CONSTRAINTS, objective=TSP_OBJECTIVE,
        instances=("TSPLIB95 から 34〜443 都市の 6 問を選んだ。距離はすべて行列で直接与えられる。"
                   "名前の数字と都市数がずれるものがある（ftv33 は 34 都市、kro124p は 100 都市）。"
                   "rbg323・rbg443 は自動倉庫のスタッカークレーンの作業順序から作られた実務由来の問題。"
                   "参照値は公表されている最適値。"),
        scale="TSPLIB 6 問（34〜443 都市）")))

register_problem(Problem(
    name="tsp_small", label="TSP 小規模（QUBO 比較用）", unit="距離", ref_label="TSPLIB 公表最適値",
    instance_file="tsp_small/{}.tsp",
    instances=["burma14", "ulysses16", "ulysses22", "bayg29", "att48", "eil51", "berlin52", "st70"],
    load=lambda name: load_tsplib_matrix("tsp_small", name), reference=TSP_SMALL_OPT.get, evaluate=evaluate,
    order=15, packages=["tsplib95"], discussion="qubo.html",
    instance_info=lambda name: instance_info("tsp_small", TSP_SMALL_OPT, name),
    setting=dict(
        brief="TSP と同じ問題を、量子・疑似量子アニーリングで扱える 14〜70 都市に絞って全手法で比べる。",
        summary=("問題は TSP（対称巡回セールスマン）と同じ。量子・疑似量子アニーリング（QUBO 系の解法）は扱える変数の数に限りがあるため、"
                 "数十都市までのインスタンスに絞り、すべての手法を同じ土俵で比べる。"),
        variables=TSP_VARIABLES, constraints=TSP_CONSTRAINTS, objective=TSP_OBJECTIVE,
        note=("QUBO 系の解法では、x_it = 1 を「都市 i を t 番目に訪れる」とする位置の 0/1 変数を使い、都市 0 を先頭に固定して "
              "(n − 1)² 個の変数にする。「各都市を 1 回」「各位置に 1 都市」の制約は、破ると目的関数に罰金が加わる形で表す。"
              "変数が 5,000 個を超える問題（72 都市以上）は解かずに「解なし」とする。"),
        instances=("TSPLIB95 から 14〜70 都市の 8 問を選んだ。距離の与え方は問題によって異なり、緯度経度から求めた地球上の距離"
                   "（burma14・ulysses16・ulysses22）、距離行列（bayg29）、疑似ユークリッド距離（att48）、平面のユークリッド距離"
                   "（eil51・berlin52・st70）が混ざる。参照値は公表されている最適値。"),
        scale="TSPLIB 8 問（14〜70 都市）")))
