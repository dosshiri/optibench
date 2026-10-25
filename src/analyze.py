"""run の results.jsonl を集計し、results/runs/<run_id>/summary.csv と Markdown 表を出力する.
python analyze.py [run_id ...]   （省略時は問題ごとに最新の run）"""
import json
import os
import sys

import numpy as np
import pandas as pd

import registry
import runmeta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下


def load(run_id):
    p = os.path.join(runmeta.run_dir(run_id), "results.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["method", "instance", "seed", "obj", "time", "time_limit"])
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows)
    # 再実行があれば最後の記録を採用
    return df.drop_duplicates(["method", "instance", "seed"], keep="last")


def best_known(problem, inst, sha256):
    """既知の参照値が無い問題の参照値: 同じ内容のインスタンスを解いた全 run・全手法の最良値
    （run 間でギャップを比べられるように）."""
    objs = [load(m["run_id"]).query("instance == @inst").obj for m in runmeta.list_runs(problem)
            if m["instance_sha256"].get(inst) == sha256]
    if not objs:
        return np.nan
    v = pd.concat(objs).dropna()
    return v.max() if registry.get_problem(problem).sense == "max" else v.min()


def reference(problem, df, meta=None):
    p = registry.get_problem(problem)
    ref = {}
    for inst in df.instance.unique():
        r = p.reference(inst)
        if r is not None:
            ref[inst] = r
        elif meta is not None:
            ref[inst] = best_known(problem, inst, meta["instance_sha256"][inst])
        else:
            v = df[df.instance == inst].obj.dropna()
            ref[inst] = v.max() if p.sense == "max" else v.min()
    return ref


def summarize(run_id):
    meta = runmeta.read_meta(run_id)
    problem = meta["problem"]
    df = load(run_id)
    ref = reference(problem, df, meta)
    df["ref"] = df.instance.map(ref)
    # ギャップは参照値より悪い方向を正にとる（最大化問題では符号を反転）
    sgn = -1 if registry.get_problem(problem).sense == "max" else 1
    df["gap"] = sgn * (df.obj - df.ref) / df.ref * 100
    df["failed"] = df.obj.isna().astype(int)
    if "lb" not in df:
        df["lb"] = np.nan
    s = df.groupby(["instance", "method"], as_index=False).agg(
        best=("obj", "max" if sgn < 0 else "min"), mean_obj=("obj", "mean"), mean_gap=("gap", "mean"), best_gap=("gap", "min"), lb=("lb", "min" if sgn < 0 else "max"),
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
    runs = sys.argv[1:] or [r for r in (runmeta.latest_run(p.name) for p in registry.problems()) if r]
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
