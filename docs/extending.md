# 問題・解法の追加方法

問題種別も解法も、`src/problems/` と `src/solvers/` に Python ファイルを置くだけで登録されます。
実行（`run_bench.py`）・集計（`analyze.py`）・結果ページとビュー（`export_web.py`）は登録内容を読むだけなので、既存のファイルを編集する必要はありません。

```
src/
  registry.py          登録簿（Problem / Solver の定義と、プラグインの読み込み）
  problems/            問題種別. 1 ファイルで 1 つ以上の問題を登録する
    routing.py           tsp, atsp
    atsptw.py            atsptw
    vrptw.py             vrptw
    jssp.py              jssp
    nsp.py               nsp（合成インスタンスの生成を含む）
  solvers/             解法. 1 ファイルに好きな数の解法を登録できる
    routing.py           TSP/ATSP の解法
    atsptw.py            ATSP-TW の解法（任意の依存を使う解法は atsptw_pyvrp.py のように別ファイル）
    vrptw.py             VRPTW の解法（PyVRP 版は vrptw_pyvrp.py）
    jssp.py              JSSP の解法
    nsp.py               NSP の解法
instances/<問題>/      インスタンスファイル
```

登録されている問題と解法は `python src/registry.py` で一覧できます。`_` で始まるファイルは読み込まれません（作業中のファイルの退避などに使えます）。
依存パッケージが入っていないなどの理由で読み込めないファイルは、警告を出して飛ばします（他の問題・解法はそのまま動きます）。

## 解法を追加する

既存の問題に解法を足す場合は、`src/solvers/` に新しいファイルを作るか、既存のファイルに関数を追加して `@solver` を付けます。

```python
# src/solvers/my_tsp.py
from common import Timer
from problems.routing import tour_cost
from registry import solver


@solver("2opt-random", problems=["tsp", "atsp"], seeds=3,
        description=("ランダムな巡回路から始め、2-opt で改善できなくなるまで局所探索する。"
                     "始点を変えて制限時間まで繰り返す。"))
def two_opt_random(D, tl, seed):
    tm = Timer(tl)            # 制限時間 tl 秒を守る
    ...
    return dict(obj=tour_cost(D, tour), lb=None, solution=tour)
```

### 関数の約束

- 引数は `(データ, 制限時間[秒], seed)`。データの中身は問題ごとに決まっています（各 `src/problems/*.py` の冒頭の説明を参照）。
- 戻り値は dict。
  - `obj`: 目的関数値。制限時間内に実行可能解が得られなければ `None`
  - `lb`: 双対境界（最小化なら下界、最大化なら上界）。無ければ `None`
  - `solution`: 解そのもの（任意）。返すと、問題側の評価関数で `obj` を計算し直して検証し、結果に `verified: true/false` を記録します（解自体は保存しません）
  - そのほかのキー（反復回数、最適性を証明したかなど）は結果の JSON にそのまま残ります
- 実行時間はフレームワークが計ります。制限時間を守るのは解法の責任です（`common.Timer` が使えます）。

### `@solver` の引数

| 引数 | 内容 |
|---|---|
| `name` | 手法名（結果・グラフに出る名前。問題ごとに一意） |
| `problems` | 対象の問題名。文字列またはリスト |
| `seeds` | 既定で回すシード数（確率的手法は 3 など。既定 1） |
| `threads` | 既定で実行するスレッド数の一覧（既定 `1`）。`[1, 4]` なら 1 スレッドと 4 スレッドの両方を実行します |
| `parallel` | スレッド数を変えて実行できるか。`threads` に 1 以外を含めると自動で `True`。既定は 1 スレッドだけにしつつ `--threads` で並列も試せるようにしたい場合は `parallel=True` とします |
| `packages` | `run.json` にバージョンを記録したい追加パッケージ（例: `["pyvrp"]`）。新しい依存は `requirements.txt` にも追加してください |
| `description` | 手法の簡単な解説（日本語 2〜3 文）。どんな考え方の手法か、この実装の特徴（初期解・近傍・定式化など）を書きます。結果ビューの「手法の解説」と `results/README.md` に表示されます。書き忘れると `python src/registry.py` の一覧に「解説なし」と出ます |

追加したら、短い制限時間で動作を確かめてから本番の run を回します。

```bash
python src/run_one.py tsp 2opt-random berlin52 1 5                 # 1 ジョブだけ実行して JSON を表示
python src/run_bench.py tsp 60 --only 2opt-random --note "2opt-random を追加"
```

`--only` で走らせた新しい run には、その手法だけが含まれます。

### 既存の run に手法を追加する

全手法を回し直さなくても、実行環境が同じなら既存の run に後から手法だけを追加できます。追加した手法は、run 作成時からある手法と同じ表・グラフに並びます。

```bash
python src/run_bench.py tsp 60 --run latest --only 2opt-random     # この実行環境で作った最新の run に追加
python src/run_bench.py tsp 60 --run <run_id> --only 2opt-random   # run を指定して追加
```

- 追加できるのは、実行環境（CPU、論理 CPU 数、メモリ、OS、Python、記録済みパッケージのバージョン）、制限時間、インスタンスファイルの内容が run 作成時と同じときだけです。違う場合は違いを表示して止まるので、新しい run として実行してください。マシン名は比べません
- run 作成時に記録が無かったパッケージ（追加した解法が使う新しいパッケージなど）は、違いとせずにバージョンを追記します
- 追加した手法、日時、そのときの git コミットは `run.json` の `additions` に記録され、ビューの「実行環境と条件」にも「後から追加」として表示されます
- `--run latest` は、この実行環境・制限時間で作った最新の run を選びます

### スレッド数を変えて実行する

並列実行できる解法は、関数に `threads` 引数を持たせて、そのスレッド数でソルバーを動かします。

```python
@solver("CP-SAT", problems="jssp", threads=[1, 4], packages=["ortools"])
def cpsat(data, tl, seed, threads=1):
    ...
    solver.parameters.num_workers = threads
```

- 結果の手法名は、1 スレッドなら解法名のまま（`CP-SAT`）、それ以外は `CP-SAT-4w` のように `-<スレッド数>w` が付きます。`run.json` の `conditions.threads`（手法名 → スレッド数）と `conditions.solvers`（手法名 → 解法名）、結果の各行の `solver`・`threads` にも記録されます
- 2 スレッド以上のジョブは、他のジョブと同時に走らせず単独で実行します
- `run_bench.py --threads 1,2,4,8` で、並列実行できる解法を指定のスレッド数すべてで実行します（1 スレッド専用の解法は、一覧に 1 があるときだけ実行）
- `--only CP-SAT-8w` のように手法名でスレッド数を指定することもできます
- 既存の run に後からスレッド条件を足すには `python src/run_bench.py jssp 60 --run <run_id> --only CP-SAT --threads 8` とします。run に無い手法（`CP-SAT-8w`）が条件に追加され、その分だけ実行されます
- ビューでは、スレッド数で表示を絞り込めるほか、同じ解法を複数のスレッド数で実行した run では「スレッド数による違い」の表が出ます（`results/README.md` にも同じ表が出ます）

## 問題種別を追加する

1. `instances/<問題名>/` にインスタンスファイルを置く
2. `src/problems/<名前>.py` で `Problem` を登録する
3. `src/solvers/<名前>.py` で解法を登録する

例として、0-1 ナップサック問題（最大化）を追加する場合です。

```python
# src/problems/knapsack.py
"""0-1 ナップサック問題. データはインスタンスの dict（capacity, weights, values）, 解は選ぶ品物の番号のリスト."""
import json

from registry import Problem, register_problem


def load(name):
    with open(PROBLEM.instance_path(name), encoding="utf-8") as f:
        return json.load(f)


def evaluate(inst, chosen):
    """実行可能なら目的関数値、そうでなければ None."""
    if sum(inst["weights"][i] for i in chosen) > inst["capacity"]:
        return None
    return sum(inst["values"][i] for i in chosen)


PROBLEM = register_problem(Problem(
    name="knapsack", label="0-1 ナップサック", unit="価値", sense="max", ref_label="全 run・全手法中の最良値",
    instance_file="knapsack/{}.json", instances=["k30"], load=load, evaluate=evaluate, display="value"))
```

```python
# src/solvers/knapsack.py
from registry import solver


@solver("Greedy", problems="knapsack")
def greedy(inst, tl, seed):
    w, v, cap = inst["weights"], inst["values"], inst["capacity"]
    chosen, used = [], 0
    for i in sorted(range(len(w)), key=lambda i: -v[i] / w[i]):
        if used + w[i] <= cap:
            chosen.append(i)
            used += w[i]
    return dict(obj=sum(v[i] for i in chosen), lb=None, solution=chosen)


@solver("CP-SAT", problems="knapsack", packages=["ortools"])
def cpsat(inst, tl, seed):
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    x = [m.NewBoolVar("") for _ in inst["weights"]]
    m.Add(sum(wi * xi for wi, xi in zip(inst["weights"], x)) <= inst["capacity"])
    m.Maximize(sum(vi * xi for vi, xi in zip(inst["values"], x)))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tl
    s.parameters.num_workers = 1
    s.parameters.random_seed = seed
    s.Solve(m)
    chosen = [i for i, xi in enumerate(x) if s.Value(xi)]
    return dict(obj=int(s.ObjectiveValue()), lb=int(s.BestObjectiveBound()), solution=chosen)
```

これで `python src/run_bench.py knapsack 60` → `python src/export_web.py` とすると、結果ページとビューに「0-1 ナップサック」のタブが増えます。

### `Problem` の項目

| 項目 | 必須 | 内容 |
|---|:-:|---|
| `name` | ✓ | 問題の識別子（コマンド引数・run_id・結果ファイルに使う。英小文字推奨） |
| `label` | ✓ | 表示名 |
| `unit` | ✓ | 目的関数値の単位（例: 距離、メイクスパン） |
| `ref_label` | ✓ | 参照値の説明（結果ページとビューに表示） |
| `instance_file` | ✓ | `instances/` からのファイルパス。`{}` にインスタンス名が入る。`run.json` に記録する SHA-256 もこのファイルから計算する |
| `instances` | ✓ | 既定で解くインスタンスの一覧。リストか、一覧を返す関数（ディレクトリ内の全ファイルなど）。`run_bench.py --instances a,b` で一時的に変えられる |
| `load` | ✓ | `load(インスタンス名)` → 解法に渡すデータ |
| `reference` | | `reference(インスタンス名)` → 参照値（既知最適値など）。`None` を返す（または省略する）と、同じ内容のインスタンスを解いた全 run・全手法の最良値を参照値にする |
| `evaluate` | | `evaluate(データ, 解)` → 目的関数値（実行不能なら `None`）。解法が `solution` を返したときの検証に使う |
| `generate` | | 合成インスタンスを（再）生成する関数。`python src/make_instances.py <問題名>` で呼ばれる |
| `sense` | | `"min"`（既定）または `"max"`。ギャップは常に「参照値より悪い方向が正」になる |
| `display` | | `"gap"`（既定。参照値に対するギャップで比較）または `"value"`（参照値が最適値でないとき。値と平均順位で比較） |
| `order` | | ビューのタブなどの並び順（小さいほど先。既定 100） |
| `packages` | | `run.json` にバージョンを記録したい追加パッケージ |
| `setting` | | 問題設定の説明（日本語の dict）。結果ビューと `results/README.md` の各問題の先頭、考察の問題ページの「問題設定とインスタンス」、ポータルの「扱う問題」に表示する（下記） |
| `instance_info` | | `instance_info(インスタンス名)` → `{列名: 値}`。インスタンスごとの規模（都市数、ジョブ数 × 機械数など）の表に使う。数値の列は右寄せで表示する |
| `discussion` | | 考察ページのファイル名（`docs/discussion/` から、例: `"tsp.html"`）。問題設定の節をこのページに書き出す |

1 つのファイルで複数の問題を登録しても構いません（`problems/routing.py` は TSP と ATSP を登録しています）。

### 問題設定とインスタンスの説明

`setting` には次のキーを書きます。初めて読む人が、何を決めてどんな制約のもとで何を最小（最大）にする問題か、どんな規模のインスタンスで測ったかが分かるようにします。

| キー | 内容 |
|---|---|
| `brief` | 何を決めて何を最小にするかの 1 文（ポータルの「扱う問題」に出す） |
| `summary` | 問題の説明。どんな問題か、実務のどこに現れるか |
| `variables` | 決めるもの（決定変数） |
| `constraints` | 制約のリスト（1 項目 1 文） |
| `objective` | 目的関数 |
| `note` | 定式化の補足（省略可。例: QUBO 系での変数の取り方） |
| `instances` | 使ったインスタンスの説明（出典・規模・性質・参照値の意味） |
| `scale` | インスタンスの規模の 1 文（例: 「TSPLIB 6 問（52〜783 都市）」。ポータルとビューに出す） |

`python src/export_web.py` を実行すると、`docs/data.json` に入って結果ビューに出るほか、`results/README.md`、考察の問題ページ（`discussion` に書いたページの目印のコメントの間。初回は「比べた手法」の直前に入る）、ポータルの「扱う問題」の表が書き換わります。これらのページの該当部分は直接編集しないでください。

## 文献レビューと考察のページを追加する

文献レビュー（`docs/literature/`）と考察（`docs/discussion/`）は、問題ごとに 1 ファイルです。新しい問題を追加したら、両方に 1 ページずつ足します。

1. 既存の問題のページ（例: `docs/discussion/jssp.html`、`docs/literature/jssp.html`）を `<問題名>.html` として複製し、中身を書き換える。スタイル（`style.css`）とリンクの切り替え（`common.js`）は各フォルダで共有しているので、変更は不要です。
2. 考察のトップ（`docs/discussion/index.html`）の「手法選びの早見表」と「問題別の考察」に、新しい問題を 1 行・1 項目ずつ足す。
3. 文献レビューのトップ（`docs/literature/index.html`）の「問題ごとの要点」に 1 項目足す。
4. ポータル（`docs/index.html`）の「手法選びの早見表」に 1 行足す。

考察の数値は手で書き起こしているので、ベンチマークを再実行したときは該当する問題のページを更新してください。

## 中立性のための注意

- 新しい解法も、既存の手法と同じ制限時間・スレッド数の条件で比べてください。複数スレッドを使う解法は、自前で手法名に `-4w` などを付けず、`threads` 引数で受け取るようにしてください（スレッド数が正しく記録され、スレッド別の比較表にも載ります）。
- 解法が解を返せる場合は `solution` を返し、問題側には `evaluate` を用意してください。ソルバーの自己申告値ではなく共通の評価関数で確かめられます（現在 TSP/ATSP/ATSP-TW/VRPTW/NSP は対応済み、JSSP は未対応）。
- コードを変えたらコミットしてから run を回してください（`run.json` に記録されるコミットで結果を再現できるようにするため）。
