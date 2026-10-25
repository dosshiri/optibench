"""集計結果をビュー用データに書き出す.
docs/data.json と docs/data.js（index.html をダブルクリックで開いても読めるようにするため）を書き、
GitHub 上で読める results/README.md と results/figures/*.svg も更新する.
results/runs/ の全 run を含め、問題ごとに最新の run を既定の表示にする.
python export_web.py [出力先 data.json]"""
import datetime
import json
import os
import sys

import pandas as pd

import export_report
import runmeta
from analyze import summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下

META = {
    "tsp": dict(label="TSP（対称巡回セールスマン）", ref="TSPLIB 公表最適値", unit="距離"),
    "atsp": dict(label="ATSP（非対称巡回セールスマン）", ref="TSPLIB 公表最適値", unit="距離"),
    "jssp": dict(label="JSSP（ジョブショップ）", ref="既知最適値（JSPLIB, ta11/ta21 は文献値）", unit="メイクスパン"),
    "nsp": dict(label="ナーススケジューリング", ref="全 run・全手法中の最良値（合成インスタンス、最適値未知）", unit="ペナルティ",
                display="value"),
}


def run_summary(meta):
    """ビューに載せる run のメタデータ."""
    c, e = meta["conditions"], meta["environment"]
    return dict(run_id=meta["run_id"], problem=meta["problem"], started_at=meta["started_at"],
                finished_at=meta.get("finished_at"), status=meta.get("status"), note=meta.get("note", ""),
                migrated=meta.get("migrated", False), environment=e, git=meta["git"],
                time_limit=c["time_limit"], workers=c["workers"], threads=c["threads"],
                seeds={m: len(s) for m, s in c["methods"].items()}, instance_sha256=meta["instance_sha256"])


def run_cells(run_id):
    df, s = summarize(run_id)
    cells = []
    for r in s.itertuples():
        cells.append(dict(instance=r.instance, method=r.method,
                          mean_gap=None if pd.isna(r.mean_gap) else round(float(r.mean_gap), 3),
                          best=None if pd.isna(r.best) else float(r.best),
                          mean_obj=None if pd.isna(r.mean_obj) else round(float(r.mean_obj), 2),
                          ref=None if pd.isna(r.ref) else float(r.ref),
                          lb=None if pd.isna(r.lb) else float(r.lb),
                          time=round(float(r.time), 2), runs=int(r.runs), fails=int(r.fails)))
    return cells, set(s.method), set(s.instance)


def main(out):
    data = dict(generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                problems={}, runs={})
    for prob in ["tsp", "atsp", "jssp", "nsp"]:
        ids = []
        for meta in runmeta.list_runs(prob):
            cells, methods, insts = run_cells(meta["run_id"])
            if not cells:
                continue
            c = meta["conditions"]
            data["runs"][meta["run_id"]] = dict(run_summary(meta), cells=cells, time_limit=c["time_limit"],
                                                instances=[i for i in c["instances"] if i in insts],
                                                methods=[m for m in c["methods"] if m in methods])
            ids.append(meta["run_id"])
        if not ids:
            continue
        # run は既定で表示する最新の run（results/README.md もこれを使う）、runs は古い順の一覧
        data["problems"][prob] = dict(META[prob], run=ids[-1], runs=ids)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    # file:// で開くと fetch がブロックされるため、同じ内容をスクリプトとしても書き出す
    with open(os.path.splitext(out)[0] + ".js", "w", encoding="utf-8") as f:
        f.write("window.OPTIBENCH_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n")
    export_report.main(data)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "data.json"))
