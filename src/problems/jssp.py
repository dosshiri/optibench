"""ジョブショップスケジューリング問題（JSSP, メイクスパン最小化）. インスタンスは JSPLIB（OR-Library 形式）.
解法に渡すデータは (jobs, 機械数). jobs[j] = [(機械, 処理時間), ...]（工程順）."""
import json
import os

from common import INST
from registry import Problem, register_problem

# JSPLIB の instances.json は ta11/ta21 を未解決扱いだが、その後最適性が証明された文献値を使う
JSSP_BKS_OVERRIDE = {"ta11": 1357, "ta21": 1642}


def load_jssp(name):
    """JSPLIB (OR-Library 形式) を読み込む. jobs[j] = [(machine, duration), ...]"""
    with open(os.path.join(INST, "jssp", name), encoding="utf-8") as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    nj, nm = map(int, lines[0].split())
    jobs = []
    for l in lines[1:1 + nj]:
        v = list(map(int, l.split()))
        jobs.append([(v[2 * k], v[2 * k + 1]) for k in range(nm)])
    return jobs, nm


def jssp_ref(name):
    with open(os.path.join(INST, "jssp", "instances.json"), encoding="utf-8") as f:
        for x in json.load(f):
            if x["name"] == name:
                if x.get("optimum"):
                    return x["optimum"], x["optimum"]
                return x["bounds"]["lower"], x["bounds"]["upper"]
    return None, None


def reference(name):
    return JSSP_BKS_OVERRIDE.get(name) or jssp_ref(name)[1]


# インスタンスの出典
SOURCE = {"ft06": "Fisher & Thompson (1963)", "ft10": "Fisher & Thompson (1963)",
          "la16": "Lawrence (1984)", "la21": "Lawrence (1984)", "abz5": "Adams, Balas & Zawack (1988)",
          "orb01": "Applegate & Cook (1991)", "swv01": "Storer, Wu & Vaccari (1992)",
          "ta01": "Taillard (1993)", "ta11": "Taillard (1993)", "ta21": "Taillard (1993)"}


def instance_info(name):
    jobs, nm = load_jssp(name)
    return {"ジョブ数 × 機械数": f"{len(jobs)} × {nm}", "工程数": sum(len(j) for j in jobs),
            "出典": SOURCE.get(name, ""), "最適値": reference(name)}


register_problem(Problem(
    name="jssp", label="JSSP（ジョブショップ）", unit="メイクスパン",
    ref_label="既知最適値（JSPLIB, ta11/ta21 は文献値）", instance_file="jssp/{}",
    instances=["ft06", "ft10", "la16", "la21", "abz5", "orb01", "ta01", "swv01", "ta11", "ta21"],
    load=load_jssp, reference=reference, order=30, discussion="jssp.html", instance_info=instance_info,
    setting=dict(
        brief="各ジョブが決まった順序で機械を回るとき、全工程の開始時刻を決めて、最後の工程が終わる時刻（メイクスパン）を最小にする。",
        summary=("n 個のジョブを m 台の機械で加工する。各ジョブは決まった順序で機械を 1 台ずつ回り（回る順序はジョブごとに異なる）、"
                 "各工程の処理時間は決まっている。すべての工程の開始時刻を決めて、最後の工程が終わる時刻（メイクスパン）を最小にする。"
                 "多品種少量生産の工場で、機械ごとの作業の順番を決める問題の基本形。"),
        variables=("各工程 o の開始時刻 s_o（整数）。同じ機械で処理する工程の順番を決めることと同じで、"
                   "MIP では同じ機械の 2 工程のどちらを先にするかを表す 0-1 変数も使う。"),
        constraints=[
            "ジョブ内の順序: 同じジョブの前の工程が終わるまで、次の工程を始められない（s_次 ≥ s_前 + p_前、p は処理時間）。",
            "機械の容量: 1 台の機械は同時に 1 つの工程しか処理できない（同じ機械の工程の処理時間帯が重ならない）。",
            "工程は始めたら途中で中断しない。",
        ],
        objective="メイクスパン C_max = max_o (s_o + p_o) を最小化する。",
        instances=("JSPLIB（OR-Library 形式）から 6×6〜20×20 の 10 問を選んだ。各ジョブはすべての機械を 1 回ずつ通るので、"
                   "工程数はジョブ数 × 機械数になる。1963 年の Fisher & Thompson の古典問題から、難しいことで知られる "
                   "Storer らや Taillard の問題まで含む。参照値は既知の最適値（ta11・ta21 は JSPLIB では未解決扱いだが、"
                   "その後に最適性が証明された文献値）。"),
        scale="JSPLIB 10 問（6×6〜20×20、36〜400 工程）")))
