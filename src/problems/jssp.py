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


register_problem(Problem(
    name="jssp", label="JSSP（ジョブショップ）", unit="メイクスパン",
    ref_label="既知最適値（JSPLIB, ta11/ta21 は文献値）", instance_file="jssp/{}",
    instances=["ft06", "ft10", "la16", "la21", "abz5", "orb01", "ta01", "swv01", "ta11", "ta21"],
    load=load_jssp, reference=reference, order=30))
