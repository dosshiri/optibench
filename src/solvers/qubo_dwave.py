"""QUBO にした問題をアニーリングで解く解法: D-Wave の dwave-samplers（旧 neal）の焼きなまし.
dwave-samplers が入っていなければこのファイルだけ読み込みを飛ばす. QUBO の定式化と繰り返しの枠組みは src/qubo.py."""
from dwave.samplers import SimulatedAnnealingSampler

from qubo import NUM_READS, QUBO_PROBLEMS, run, samples
from registry import solver


def _sample(Q, seed):
    nvar = 1 + max(max(k) for k in Q)
    return samples(SimulatedAnnealingSampler().sample_qubo(Q, num_reads=NUM_READS, seed=seed), nvar)


@solver("DWave-SA", problems=QUBO_PROBLEMS, seeds=3, packages=["dwave-samplers"],
        description=("D-Wave 社が公開している QUBO 向け焼きなまし法（dwave-samplers、旧 neal）。"
                     "量子アニーラ実機の前に問題の定式化を確かめる用途で広く使われる、古典計算機上の基準となるソルバー。"
                     "OpenJij と同じ QUBO を既定のスケジュール（1000 スイープ）で 10 解ずつ解き、制限時間まで繰り返す。"))
def dwave_sa(data, tl, seed):
    return run(data, tl, seed, _sample)
