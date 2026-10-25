"""QUBO にした問題をアニーリングで解く解法: OpenJij（疑似量子アニーリング SQA と焼きなまし SA）.
openjij が入っていなければこのファイルだけ読み込みを飛ばす. QUBO の定式化と繰り返しの枠組みは src/qubo.py."""
import openjij as oj

from qubo import NUM_READS, QUBO_PROBLEMS, run, samples
from registry import solver


def _sampler(make, **params):
    def sample(Q, seed):
        nvar = 1 + max(max(k) for k in Q)
        return samples(make().sample_qubo(Q, num_reads=NUM_READS, seed=seed, **params), nvar)
    return sample


@solver("OpenJij-SQA", problems=QUBO_PROBLEMS, seeds=3, packages=["openjij"],
        description=("疑似量子アニーリング（SQA）。量子アニーリングの量子揺らぎを、解のコピー（Trotter スライス）を並べて"
                     "互いに引き合わせる形で古典計算機上に再現する、OpenJij の実装。"
                     "問題を QUBO（0/1 変数の二次式）に直し、1000 スイープ、Trotter 数 4、逆温度 20 で"
                     "10 解ずつ求める呼び出しを制限時間まで繰り返して、制約を満たす最良の解を採る。"))
def openjij_sqa(data, tl, seed):
    # 逆温度 β の既定値 5 では、正規化した QUBO で制約を満たす解がほとんど出ない（bayg29 で 5/10/20/50 を試して 20 にした）
    return run(data, tl, seed, _sampler(oj.SQASampler, beta=20.0))


@solver("OpenJij-SA", problems=QUBO_PROBLEMS, seeds=3, packages=["openjij"],
        description=("QUBO に対する焼きなまし法（SA）を OpenJij で実行する。"
                     "変数を 1 つずつ反転させる単純な近傍で、温度を下げながらエネルギー（目的関数と罰金の和）を最小化する。"
                     "OpenJij-SQA と同じ QUBO・同じ繰り返し方で、量子揺らぎの有無だけが違う比較対象。"))
def openjij_sa(data, tl, seed):
    return run(data, tl, seed, _sampler(oj.SASampler))
