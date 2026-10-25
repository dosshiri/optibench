"""1 ジョブ実行: python run_one.py <problem> <method> <instance> <seed> <time_limit>  -> JSON 1 行."""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import registry  # noqa: E402


def main():
    prob, method, inst, seed, tl = sys.argv[1:6]
    seed, tl = int(seed), float(tl)
    problem = registry.get_problem(prob)
    func = registry.get_solver(prob, method).func
    data = problem.load(inst)
    t0 = time.perf_counter()
    res = func(data, tl, seed)
    res["time"] = time.perf_counter() - t0
    sol = res.pop("solution", None)
    # 解法が解を返したら、申告された目的関数値を問題側の評価関数で検証する
    if sol is not None and problem.evaluate is not None and res.get("obj") is not None:
        res["verified"] = problem.evaluate(data, sol) == res["obj"]
        if not res["verified"]:
            print(f"警告: {method} の申告値 {res['obj']} が評価関数の値と一致しません", file=sys.stderr)
    res.update(problem=prob, method=method, instance=inst, seed=seed, time_limit=tl)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
