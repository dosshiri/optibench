"""1 ジョブ実行: python run_one.py <problem> <method> <instance> <seed> <time_limit>  -> JSON 1 行."""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    prob, method, inst, seed, tl = sys.argv[1:6]
    seed, tl = int(seed), float(tl)
    if prob in ("tsp", "atsp"):
        from common import load_tsplib_matrix
        from tsp_methods import METHODS
        data = load_tsplib_matrix(prob, inst)
    elif prob == "jssp":
        from common import load_jssp
        from jssp_methods import METHODS
        data = load_jssp(inst)
    elif prob == "nsp":
        from nsp_model import load_instance
        from nsp_methods import METHODS
        data = load_instance(inst)
    t0 = time.perf_counter()
    res = METHODS[method](data, tl, seed)
    res["time"] = time.perf_counter() - t0
    res.pop("tour", None)
    res.pop("schedule", None)
    res.update(problem=prob, method=method, instance=inst, seed=seed, time_limit=tl)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
