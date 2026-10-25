"""ベンチマーク一括実行: python run_bench.py <problem> <time_limit> [--workers 4] [--only 手法,...] [--note "..."]
問題・手法・既定のインスタンスとシード数は src/problems/, src/solvers/ の登録内容（registry.py）から決まる.
実行ごとに results/runs/<run_id>/ を作り、run.json（実行日時・環境・条件・git コミット）と
results.jsonl（生データ）を保存する. --run <run_id> で中断した run を再開する（完了済みジョブはスキップ）.
--run <run_id> --only 手法 で既存の run に手法を後から追加する（--run latest はこの実行環境で作った最新の run）.
再開・追加は、実行環境（CPU・メモリ・OS・Python・パッケージのバージョン）とインスタンスが run 作成時と同じときだけできる."""
import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import registry
import runmeta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下


def plan_methods(problem, only=None, threads=None):
    """実行する手法 {手法名: dict(solver, threads, seeds)} を決める.
    only: カンマ区切りの解法名・手法名, threads: カンマ区切りのスレッド数（どちらも None なら既定）."""
    solvers = registry.solvers(problem)
    req = [int(t) for t in threads.split(",")] if threads else None
    plan = {}

    def add(s, t):
        plan[registry.label(s.name, t)] = dict(solver=s.name, threads=t, seeds=list(range(1, s.seeds + 1)))

    names = only.split(",") if only else list(solvers)
    for name in names:
        if name in solvers:
            s = solvers[name]
            for t in (s.threads if req is None else [t for t in req if s.parallel or t == 1]):
                add(s, t)
        else:
            add(*registry.resolve(problem, name))  # "CP-SAT-8w" のような手法名. 未登録ならここで止める
    if not plan:
        sys.exit("実行する手法がありません（--only / --threads の指定を確認してください）")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("time_limit", type=float)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None, help="カンマ区切りの解法名または手法名（例: CP-SAT, CP-SAT-8w）")
    ap.add_argument("--threads", default=None,
                    help="カンマ区切りのスレッド数（例: 1,4）. 並列実行できる解法をそれぞれのスレッド数で実行する。"
                         "1 を含めない場合、1 スレッド専用の解法は実行しない（省略時は解法ごとの既定）")
    ap.add_argument("--instances", default=None, help="カンマ区切りのインスタンス名（省略時は問題の既定の一覧）")
    ap.add_argument("--run", default=None, help="再開・追加する run_id。latest でこの実行環境で作った最新の run"
                                                 "（省略時は新しい run を作る）")
    ap.add_argument("--machine", default=None, help="実行環境の名前（省略時はホスト名）。run_id にも使う")
    ap.add_argument("--note", default="", help="run に付けるメモ")
    a = ap.parse_args()
    problem = registry.get_problem(a.problem)
    insts = a.instances.split(",") if a.instances else problem.default_instances()
    plan = plan_methods(a.problem, a.only, a.threads)
    methods = {m: p["seeds"] for m, p in plan.items()}
    for m, p in plan.items():
        if p["threads"] > (os.cpu_count() or 1):
            print(f"警告: {m} のスレッド数 {p['threads']} が論理 CPU 数 {os.cpu_count()} を超えています", file=sys.stderr)

    if a.run:
        env = runmeta.environment(a.machine)
        if a.run == "latest":
            # この実行環境・制限時間で作った最新の run
            same = [m for m in runmeta.list_runs(a.problem) if m["conditions"]["time_limit"] == a.time_limit
                    and not runmeta.env_diff(m["environment"], env)]
            if not same:
                sys.exit(f"この実行環境で作った {a.problem} / {a.time_limit:g} 秒の run がありません")
            a.run = same[-1]["run_id"]
        meta = runmeta.read_meta(a.run)
        if meta["problem"] != a.problem or meta["conditions"]["time_limit"] != a.time_limit:
            sys.exit(f"run {a.run} は {meta['problem']} / {meta['conditions']['time_limit']} 秒の run です")
        # 結果を同じ表で比べられるよう、実行環境とインスタンスが run 作成時と同じときだけ追加・再開する
        diff = runmeta.env_diff(meta["environment"], env)
        if diff:
            sys.exit(f"run {a.run} と実行環境が異なるため、この run には追加・再開できません"
                     "（新しい run として実行してください）:\n  " + "\n  ".join(diff))
        if meta["instance_sha256"] != runmeta.instance_hashes(a.problem, meta["conditions"]["instances"]):
            sys.exit(f"run {a.run} の作成時とインスタンスファイルの内容が異なるため、追加・再開できません")
        for p, v in env["packages"].items():  # 後から入れたパッケージのバージョンも記録する
            if v is not None and meta["environment"]["packages"].get(p) is None:
                meta["environment"]["packages"][p] = v
        git = runmeta.git_info()
        insts = meta["conditions"]["instances"]
        c = meta["conditions"]
        added = []
        if a.only or a.threads:
            # 指定された手法だけを実行する。run に無い手法（新しい解法やスレッド数）は条件に追加する
            for m, p in plan.items():
                if m not in c["methods"]:
                    c["methods"][m] = p["seeds"]
                    c["threads"][m] = p["threads"]
                    c.setdefault("solvers", {})[m] = p["solver"]
                    added.append(m)
                    print(f"run に {m} を追加します", file=sys.stderr)
            methods = {m: c["methods"][m] for m in plan}
        else:
            methods = c["methods"]
        if added:
            # 追加した手法は run 作成時と別のコミットで実行することがあるので、追加ごとにコミットを記録する
            meta.setdefault("additions", []).append(dict(
                at=runmeta.now(), methods=added, git=git,
                command=" ".join([os.path.basename(sys.executable)] + sys.argv)))
            if git["dirty"]:
                print("警告: 未コミットの変更があります:", ", ".join(git["dirty_files"]), file=sys.stderr)
        elif git["commit"] != meta["git"]["commit"]:
            print("警告: run 作成時と git コミットが異なります", file=sys.stderr)
        meta.update(status="running", finished_at=None)
        meta.setdefault("resumed_at", []).append(runmeta.now())
    else:
        env = runmeta.environment(a.machine)
        meta = dict(run_id=runmeta.new_run_id(a.problem, env["machine"]), problem=a.problem,
                    started_at=runmeta.now(), finished_at=None, status="running", note=a.note,
                    command=" ".join([os.path.basename(sys.executable)] + sys.argv),
                    environment=env, git=runmeta.git_info(),
                    conditions=dict(time_limit=a.time_limit, workers=a.workers,
                                    threads={m: p["threads"] for m, p in plan.items()},
                                    solvers={m: p["solver"] for m, p in plan.items()},
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
    # 複数スレッドを使う手法は単独で走らせる
    threads = meta["conditions"]["threads"]
    single = [j for j in jobs if threads.get(j[0], 1) <= 1]
    multi = [j for j in jobs if threads.get(j[0], 1) > 1]

    write_lock = threading.Lock()

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
        # 並列ジョブが同時に追記すると行が失われることがある（Windows で確認）ため、書き込みを排他する
        with write_lock:
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
