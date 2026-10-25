"""ベンチマーク一括実行: python run_bench.py <problem> <time_limit> [--workers 4] [--note "..."]
実行ごとに results/runs/<run_id>/ を作り、run.json（実行日時・環境・条件・git コミット）と
results.jsonl（生データ）を保存する. --run <run_id> で中断した run を再開する（完了済みジョブはスキップ）."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import runmeta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下

PLAN = {
    "tsp": (["berlin52", "kroA100", "ch150", "a280", "pcb442", "rat783"],
            {"nn+ls": [1], "SA": [1, 2, 3], "LKH(elkai)": [1], "ORTools-GLS": [1],
             "CP-SAT": [1], "MIP-HiGHS": [1]}),
    "atsp": (["ftv33", "ftv70", "kro124p", "ft70", "rbg323", "rbg443"],
             {"nn+ls": [1], "SA": [1, 2, 3], "LKH(elkai)": [1], "ORTools-GLS": [1],
              "CP-SAT": [1], "MIP-HiGHS": [1]}),
    "jssp": (["ft06", "ft10", "la16", "la21", "abz5", "orb01", "ta01", "swv01", "ta11", "ta21"],
             {"SPT": [1], "MWKR": [1], "Tabu-N5": [1, 2, 3], "GA": [1, 2, 3],
              "CP-SAT": [1], "CP-SAT-4w": [1], "MIP-HiGHS": [1]}),
    "nsp": (None, {"Greedy": [1], "SA": [1, 2, 3], "CP-SAT": [1], "CP-SAT-4w": [1],
                   "MIP-HiGHS": [1]}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("time_limit", type=float)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="カンマ区切りの手法名")
    ap.add_argument("--run", default=None, help="再開する run_id（省略時は新しい run を作る）")
    ap.add_argument("--machine", default=None, help="実行環境の名前（省略時はホスト名）。run_id にも使う")
    ap.add_argument("--note", default="", help="run に付けるメモ")
    a = ap.parse_args()
    insts, methods = PLAN[a.problem]
    if insts is None:
        insts = sorted(f[:-5] for f in os.listdir(os.path.join(ROOT, "instances", "nsp")) if f.endswith(".json"))
    methods = {m: seeds for m, seeds in methods.items() if a.only is None or m in a.only.split(",")}

    if a.run:
        meta = runmeta.read_meta(a.run)
        if meta["problem"] != a.problem or meta["conditions"]["time_limit"] != a.time_limit:
            sys.exit(f"run {a.run} は {meta['problem']} / {meta['conditions']['time_limit']} 秒の run です")
        if meta["git"]["commit"] != runmeta.git_info()["commit"]:
            print("警告: run 作成時と git コミットが異なります", file=sys.stderr)
        if meta["instance_sha256"] != runmeta.instance_hashes(a.problem, meta["conditions"]["instances"]):
            print("警告: run 作成時とインスタンスファイルの内容が異なります", file=sys.stderr)
        insts = meta["conditions"]["instances"]
        methods = {m: s for m, s in meta["conditions"]["methods"].items() if a.only is None or m in a.only.split(",")}
        meta.update(status="running", finished_at=None)
        meta.setdefault("resumed_at", []).append(runmeta.now())
    else:
        env = runmeta.environment(a.machine)
        meta = dict(run_id=runmeta.new_run_id(a.problem, env["machine"]), problem=a.problem,
                    started_at=runmeta.now(), finished_at=None, status="running", note=a.note,
                    command=" ".join([os.path.basename(sys.executable)] + sys.argv),
                    environment=env, git=runmeta.git_info(),
                    conditions=dict(time_limit=a.time_limit, workers=a.workers,
                                    threads={m: 4 if m.endswith("-4w") else 1 for m in methods},
                                    instances=insts, methods=methods),
                    instance_sha256=runmeta.instance_hashes(a.problem, insts))
        if meta["git"]["dirty"]:
            print("警告: 未コミットの変更があります:", ", ".join(meta["git"]["dirty_files"]), file=sys.stderr)
    runmeta.write_meta(meta)
    print("run:", meta["run_id"], flush=True)

    out = os.path.join(runmeta.run_dir(meta["run_id"]), "results.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["method"], r["instance"], r["seed"]))
    jobs = [(m, i, s) for i in insts for m, seeds in methods.items() for s in seeds if (m, i, s) not in done]
    # 4 ワーカーを使う手法は単独で走らせる
    single = [j for j in jobs if not j[0].endswith("-4w")]
    multi = [j for j in jobs if j[0].endswith("-4w")]

    def run(job):
        m, i, s = job
        env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        try:
            p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "run_one.py"), a.problem, m, i, str(s),
                                str(a.time_limit)], capture_output=True, text=True, encoding="utf-8",
                               timeout=a.time_limit * 3 + 120, env=env)
            line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else None
            r = json.loads(line) if line else dict(error=p.stderr[-500:])
        except subprocess.TimeoutExpired:
            r = dict(error="timeout")
        r.update(problem=a.problem, method=m, instance=i, seed=s, time_limit=a.time_limit)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        print(m, i, s, r.get("obj"), r.get("lb"), round(r.get("time", -1), 1), r.get("error", ""), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, single))
    for j in multi:
        run(j)
    meta.update(finished_at=runmeta.now(), status="complete")
    runmeta.write_meta(meta)


if __name__ == "__main__":
    main()
