"""QUBO にした問題をアニーリングで解く解法: Fixstars Amplify AE（GPU 上の疑似量子アニーリング、クラウドサービス）.
実行には amplify パッケージと、無料登録で得られる API トークン（環境変数 AMPLIFY_TOKEN）が要る.
解説をビューに出せるよう、パッケージやトークンが無くても登録だけはする."""
import datetime
import os

import numpy as np

from qubo import QUBO_PROBLEMS, run
from registry import solver

CALL_SECONDS = 10  # 1 回の呼び出しで Amplify AE に与える計算時間（秒）


def _sample(Q, seed):
    """Q を上三角の行列にして Amplify AE に送り、返ってきた解を 0/1 配列のリストにする.
    Amplify AE は乱数の種を受け付けないので seed は使わない."""
    import amplify
    n = 1 + max(max(k) for k in Q)
    W = np.zeros((n, n))
    for (i, j), v in Q.items():
        W[i, j] += v
    m = amplify.VariableGenerator().matrix("Binary", n)
    m.quadratic = W
    client = amplify.FixstarsClient()
    client.token = os.environ["AMPLIFY_TOKEN"]
    client.parameters.timeout = datetime.timedelta(seconds=CALL_SECONDS)
    res = amplify.solve(amplify.Model(m), client)
    x = m.variable_array
    return [np.asarray(x.evaluate(s.values), dtype=np.int8) for s in res.solutions]


# 無償プランの利用規約では、動作確認以外の目的（研究開発業務など）での利用に事前の同意が要るため、
# 公開してよいか確認できるまで、結果は results/runs/ に残し、結果ビューと結果ページには載せない
HIDDEN = ("Fixstars Amplify の無償プランの利用規約（第 5 条第 2 項、第 6 条第 1 項第 1 号）では、"
          "動作確認以外の目的での利用に事前の同意が要るとされており、結果を公開してよいか確認できていないため")


@solver("Amplify-AE", problems=QUBO_PROBLEMS, packages=["amplify"], hidden=HIDDEN,
        description=("Fixstars 社のクラウドサービス Amplify AE。GPU 上で多数の解を並列に更新する疑似量子アニーリング系のソルバーで、"
                     "大規模な QUBO も扱える。"
                     "OpenJij と同じ QUBO を 1 回 10 秒の計算時間で送る呼び出しを制限時間まで繰り返す（通信の時間も制限時間に含む）。"))
def amplify_ae(data, tl, seed):
    if not os.environ.get("AMPLIFY_TOKEN"):
        raise RuntimeError("環境変数 AMPLIFY_TOKEN に Amplify の API トークンを設定してください")
    return run(data, tl, seed, _sample)
