"""run の results.jsonl を集計し、results/runs/<run_id>/summary.csv と Markdown 表を出力する.
python analyze.py [run_id ...]   （省略時は問題ごとに最新の run）"""
import json
import os
import sys

import numpy as np
import pandas as pd

import runmeta
from common import ATSP_OPT, TSP_OPT, jssp_ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下
# JSPLIB の instances.json は ta11/ta21 を未解決扱いだが、その後最適性が証明された文献値を使う
JSSP_BKS_OVERRIDE = {"ta11": 1357, "ta21": 1642}


def load(run_id):
    p = os.path.join(runmeta.run_dir(run_id), "results.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["method", "instance", "seed", "obj", "time", "time_limit"])
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows)
    # 再実行があれば最後の記録を採用
    return df.drop_duplicates(["method", "instance", "seed"], keep="last")


def nsp_best(inst, sha256):
    """NSP の参照値: 同じ内容のインスタンスを解いた全 run・全手法の最良値（run 間でギャップを比べられるように）."""
    objs = [load(m["run_id"]).query("instance == @inst").obj for m in runmeta.list_runs("nsp")
            if m["instance_sha256"].get(inst) == sha256]
    return pd.concat(objs).dropna().min() if objs else np.nan


def reference(problem, df, meta=None):
    ref = {}
    for inst in df.instance.unique():
        if problem == "tsp":
            ref[inst] = TSP_OPT[inst]
        elif problem == "atsp":
            ref[inst] = ATSP_OPT[inst]
        elif problem == "jssp":
            ref[inst] = JSSP_BKS_OVERRIDE.get(inst) or jssp_ref(inst)[1]
        elif meta is not None:  # NSP: 既知最適値なし
            ref[inst] = nsp_best(inst, meta["instance_sha256"][inst])
        else:
            ref[inst] = df[df.instance == inst].obj.dropna().min()
    return ref


def summarize(run_id):
    meta = runmeta.read_meta(run_id)
    problem = meta["problem"]
    df = load(run_id)
    ref = reference(problem, df, meta)
    df["ref"] = df.instance.map(ref)
    df["gap"] = (df.obj - df.ref) / df.ref * 100
    df["failed"] = df.obj.isna().astype(int)
    if "lb" not in df:
        df["lb"] = np.nan
    s = df.groupby(["instance", "method"], as_index=False).agg(
        best=("obj", "min"), mean_obj=("obj", "mean"), mean_gap=("gap", "mean"), best_gap=("gap", "min"), lb=("lb", "max"),
        time=("time", "mean"), runs=("seed", "count"), fails=("failed", "sum"))
    s["ref"] = s.instance.map(ref)
    s.to_csv(os.path.join(runmeta.run_dir(run_id), "summary.csv"), index=False)
    return df, s


def gap_table(s, order):
    t = s.pivot(index="instance", columns="method", values="mean_gap")
    t = t.reindex(order)
    t.loc["平均"] = t.mean()
    return t


def fmt(v):
    return "—" if pd.isna(v) else f"{v:.2f}"


def to_md(t):
    cols = list(t.columns)
    lines = ["| インスタンス | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in t.iterrows():
        lines.append(f"| {idx} | " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    from run_bench import PLAN
    runs = sys.argv[1:] or [r for r in (runmeta.latest_run(p) for p in PLAN) if r]
    for run_id in runs:
        meta = runmeta.read_meta(run_id)
        prob = meta["problem"]
        df, s = summarize(run_id)
        if not len(s):
            continue
        order = meta["conditions"]["instances"]
        env = meta["environment"]
        print(f"\n## {prob} 平均ギャップ(%)  run={run_id}")
        print(f"{env['cpu']} / {env['cpu_count']} CPU / {env['memory_gb']} GB / {env['os']}, "
              f"制限時間 {meta['conditions']['time_limit']:g} 秒, commit {(meta['git']['commit'] or '?')[:7]}"
              + (" (未コミット変更あり)" if meta["git"]["dirty"] else ""))
        print(to_md(gap_table(s, order)))
        err = df[df.get("error").notna()] if "error" in df else []
        if len(err):
            print("errors:", err[["method", "instance", "error"]].to_string())
