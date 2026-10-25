"""集計結果を GitHub Pages のビュー用 JSON (docs/data.json) に書き出す.
python export_web.py <出力先 data.json>"""
import datetime
import json
import os
import platform
import sys

import pandas as pd

from analyze import summarize
from run_bench import PLAN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下

META = {
    "tsp": dict(label="TSP（対称巡回セールスマン）", ref="TSPLIB 公表最適値", unit="距離"),
    "atsp": dict(label="ATSP（非対称巡回セールスマン）", ref="TSPLIB 公表最適値", unit="距離"),
    "jssp": dict(label="JSSP（ジョブショップ）", ref="既知最適値（JSPLIB, ta11/ta21 は文献値）", unit="メイクスパン"),
    "nsp": dict(label="ナーススケジューリング", ref="全手法中の最良値（合成インスタンス、最適値未知）", unit="ペナルティ",
                display="value"),
}


def main(out):
    data = dict(generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                env=dict(python=platform.python_version(), machine=platform.machine()), problems={})
    for prob in ["tsp", "atsp", "jssp", "nsp"]:
        if not os.path.exists(os.path.join(ROOT, "results", f"{prob}.jsonl")):
            continue
        df, s = summarize(prob)
        insts = PLAN[prob][0] or sorted(df.instance.unique())
        methods = [m for m in PLAN[prob][1] if m in set(s.method)]
        tl = float(df.time_limit.max())
        cells = []
        for r in s.itertuples():
            cells.append(dict(instance=r.instance, method=r.method,
                              mean_gap=None if pd.isna(r.mean_gap) else round(float(r.mean_gap), 3),
                              best=None if pd.isna(r.best) else float(r.best),
                              mean_obj=None if pd.isna(r.mean_obj) else round(float(r.mean_obj), 2),
                              ref=None if pd.isna(r.ref) else float(r.ref),
                              lb=None if pd.isna(r.lb) else float(r.lb),
                              time=round(float(r.time), 2), runs=int(r.runs), fails=int(r.fails)))
        data["problems"][prob] = dict(META[prob], instances=insts, methods=methods, time_limit=tl,
                                      cells=cells)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "data.json"))
