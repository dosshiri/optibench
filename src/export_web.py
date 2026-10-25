"""集計結果をビュー用データに書き出す.
docs/data.json と docs/data.js（benchmark.html をダブルクリックで開いても読めるようにするため）を書き、
GitHub 上で読める results/README.md と results/figures/*.svg も更新する.
results/runs/ の全 run を含め、問題ごとに最新の run を既定の表示にする.
python export_web.py [出力先 data.json]"""
import datetime
import json
import os
import sys

import pandas as pd

import export_report
import registry
import runmeta
from analyze import summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下

def solver_names(conditions):
    """手法名 -> 解法名. 記録が無い古い run は、手法名の末尾の "-<スレッド数>w" を外して求める."""
    out = {}
    for m, t in conditions["threads"].items():
        suffix = f"-{t}w"
        out[m] = m[:-len(suffix)] if t > 1 and m.endswith(suffix) else m
    out.update(conditions.get("solvers", {}))
    return out


def run_summary(meta):
    """ビューに載せる run のメタデータ."""
    c, e = meta["conditions"], meta["environment"]
    return dict(run_id=meta["run_id"], problem=meta["problem"], started_at=meta["started_at"],
                finished_at=meta.get("finished_at"), status=meta.get("status"), note=meta.get("note", ""),
                migrated=meta.get("migrated", False), environment=e, git=meta["git"],
                time_limit=c["time_limit"], workers=c["workers"], threads=c["threads"], solvers=solver_names(c),
                seeds={m: len(s) for m, s in c["methods"].items()}, instance_sha256=meta["instance_sha256"],
                additions=meta.get("additions", []))


def hidden_methods(meta):
    """結果ビューに載せない手法名（解法に hidden の理由があるもの）."""
    solvers = registry.solvers(meta["problem"])
    names = solver_names(meta["conditions"])
    return {m for m, n in names.items() if n in solvers and solvers[n].hidden}


def drop_methods(summary, hide):
    """run のメタデータから、載せない手法を除く."""
    out = dict(summary)
    for k in ("threads", "solvers", "seeds"):
        out[k] = {m: v for m, v in summary[k].items() if m not in hide}
    adds = [dict(a, methods=[m for m in a["methods"] if m not in hide]) for a in summary["additions"]]
    out["additions"] = [a for a in adds if a["methods"]]
    return out


def run_cells(run_id, hide=()):
    df, s = summarize(run_id)
    cells = []
    for r in s.itertuples():
        if r.method in hide:
            continue
        cells.append(dict(instance=r.instance, method=r.method,
                          mean_gap=None if pd.isna(r.mean_gap) else round(float(r.mean_gap), 3),
                          best=None if pd.isna(r.best) else float(r.best),
                          mean_obj=None if pd.isna(r.mean_obj) else round(float(r.mean_obj), 2),
                          ref=None if pd.isna(r.ref) else float(r.ref),
                          lb=None if pd.isna(r.lb) else float(r.lb),
                          time=round(float(r.time), 2), runs=int(r.runs), fails=int(r.fails)))
    return cells, {c["method"] for c in cells}, set(s.instance)


def main(out):
    data = dict(generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                problems={}, runs={})
    for p in registry.problems():
        prob = p.name
        ids = []
        for meta in runmeta.list_runs(prob):
            hide = hidden_methods(meta)
            cells, methods, insts = run_cells(meta["run_id"], hide)
            if not cells:
                continue
            c = meta["conditions"]
            data["runs"][meta["run_id"]] = dict(drop_methods(run_summary(meta), hide), cells=cells,
                                                time_limit=c["time_limit"],
                                                instances=[i for i in c["instances"] if i in insts],
                                                methods=[m for m in c["methods"] if m in methods])
            ids.append(meta["run_id"])
        if not ids:
            continue
        # run は既定で表示する最新の run（results/README.md もこれを使う）、runs は古い順の一覧
        info = dict(label=p.label, ref=p.ref_label, unit=p.unit,
                    about={s.name: s.description for s in registry.solvers(prob).values()
                           if s.description and not s.hidden})
        hidden = {s.name: s.hidden for s in registry.solvers(prob).values() if s.hidden}
        if hidden:
            info["hidden"] = hidden  # 載せない手法と理由（結果は results/runs/ に残っている）
        if p.display != "gap":
            info["display"] = p.display
        if p.sense != "min":
            info["sense"] = p.sense
        data["problems"][prob] = dict(info, run=ids[-1], runs=ids)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    # file:// で開くと fetch がブロックされるため、同じ内容をスクリプトとしても書き出す
    with open(os.path.splitext(out)[0] + ".js", "w", encoding="utf-8") as f:
        f.write("window.OPTIBENCH_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n")
    export_report.main(data)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "data.json"))
