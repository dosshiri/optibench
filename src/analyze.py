"""results/*.jsonl を集計し、results/summary_<problem>.csv と Markdown 表を出力する."""
import json
import os

import numpy as np
import pandas as pd

from common import ATSP_OPT, TSP_OPT, jssp_ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下
# JSPLIB の instances.json は ta11/ta21 を未解決扱いだが、その後最適性が証明された文献値を使う
JSSP_BKS_OVERRIDE = {"ta11": 1357, "ta21": 1642}


def load(problem):
    p = os.path.join(ROOT, "results", f"{problem}.jsonl")
    rows = [json.loads(l) for l in open(p)]
    df = pd.DataFrame(rows)
    # 再実行があれば最後の記録を採用
    return df.drop_duplicates(["method", "instance", "seed"], keep="last")


def reference(problem, df):
    ref = {}
    for inst in df.instance.unique():
        if problem == "tsp":
            ref[inst] = TSP_OPT[inst]
        elif problem == "atsp":
            ref[inst] = ATSP_OPT[inst]
        elif problem == "jssp":
            ref[inst] = JSSP_BKS_OVERRIDE.get(inst) or jssp_ref(inst)[1]
        else:  # NSP: 全手法の最良値 (既知最適値なし)
            ref[inst] = df[df.instance == inst].obj.dropna().min()
    return ref


def summarize(problem):
    df = load(problem)
    ref = reference(problem, df)
    df["ref"] = df.instance.map(ref)
    df["gap"] = (df.obj - df.ref) / df.ref * 100
    df["failed"] = df.obj.isna().astype(int)
    if "lb" not in df:
        df["lb"] = np.nan
    s = df.groupby(["instance", "method"], as_index=False).agg(
        best=("obj", "min"), mean_gap=("gap", "mean"), best_gap=("gap", "min"), lb=("lb", "max"),
        time=("time", "mean"), runs=("seed", "count"), fails=("failed", "sum"))
    s["ref"] = s.instance.map(ref)
    s.to_csv(os.path.join(ROOT, "results", f"summary_{problem}.csv"), index=False)
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
    for prob in ["tsp", "atsp", "jssp", "nsp"]:
        if not os.path.exists(os.path.join(ROOT, "results", f"{prob}.jsonl")):
            continue
        df, s = summarize(prob)
        order = PLAN[prob][0] or sorted(df.instance.unique())
        print(f"\n## {prob} 平均ギャップ(%)")
        print(to_md(gap_table(s, order)))
        err = df[df.get("error").notna()] if "error" in df else []
        if len(err):
            print("errors:", err[["method", "instance", "error"]].to_string())
