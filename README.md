# optibench

組合せ最適化ベンチマーク。実務で使われる代表的な組合せ最適化問題について、数理計画・制約プログラミング・メタヒューリスティクスなどの解法を、同じ条件で手元比較するためのリポジトリです。

- 対象問題: TSP（対称巡回セールスマン）、ATSP（非対称）、ATSP-TW（時間枠付き非対称）、JSSP（ジョブショップスケジューリング）、ナーススケジューリング（NSP）
- 入口: `docs/index.html`（文献調査・結果・考察の 3 ページへのリンク集。ダブルクリックで開きます）
- 文献調査: `docs/literature/`（既存研究における解法比較のレビュー。`index.html` が全体の要点、`<問題>.html` が問題ごとのページ）
- 考察: `docs/discussion/`（問題ごとの手法の選び方と、その根拠・限界・次の一手。`index.html` が早見表、`<問題>.html` が問題ごとのページ）
- 結果: [`results/README.md`](results/README.md)（GitHub 上でそのまま読めるグラフと表）、`docs/benchmark.html`（手元でダブルクリックすると開く対話型ビュー）

## 構成

問題種別と解法はプラグイン方式で、`src/problems/` と `src/solvers/` にファイルを追加するだけで増やせます（既存コードの変更は不要）。手順は [`docs/extending.md`](docs/extending.md) を参照してください。

```
src/            実行スクリプト
  registry.py     問題・解法の登録簿（python src/registry.py で登録内容を一覧）
  problems/       問題種別（インスタンス読込・参照値・解の評価・インスタンス生成）
    routing.py      TSP / ATSP
    atsptw.py       ATSP-TW（時間枠の評価を含む）
    jssp.py         JSSP
    nsp.py          NSP のモデル・評価関数・合成インスタンス生成
  solvers/        解法（@solver で登録）
    routing.py      TSP / ATSP の解法
    atsptw.py       ATSP-TW の解法（PyVRP 版は atsptw_pyvrp.py）
    jssp.py         JSSP の解法
    nsp.py          NSP の解法
  common.py       共通ユーティリティ（パス・時間計測）
  run_one.py      1 ジョブ実行（JSON 1 行を出力。解法が解を返したら目的関数値を評価関数で検証）
  run_bench.py    一括実行（1 回の実行を 1 つの run として results/runs/<run_id>/ に保存）
  runmeta.py      run のメタデータ（実行環境・条件・git コミット）の収集と読み書き
  analyze.py      集計（results/runs/<run_id>/summary.csv）
  export_web.py   ビュー用データ（docs/data.json, docs/data.js）と結果ページの書き出し
  export_report.py 結果ページ（results/README.md, results/figures/*.svg）の生成
  make_instances.py 合成インスタンスの（再）生成
instances/      ベンチマークインスタンス
results/        run ごとの実行結果（runs/）と結果ページ（README.md と figures/）
docs/           入口（index.html）、対話型のグラフビュー（benchmark.html）、文献調査（literature/）、考察（discussion/）、追加手順（extending.md）
```

## インスタンス

| 問題 | インスタンス | 出典 | 参照値 |
|---|---|---|---|
| TSP | berlin52, kroA100, ch150, a280, pcb442, rat783 | TSPLIB95 | 公表最適値 |
| ATSP | ftv33, ftv70, kro124p, ft70, rbg323, rbg443 | TSPLIB95 | 公表最適値 |
| ATSP-TW | rbg019a, rbg027a, rbg041a, rbg055a, rbg067a, rbg092a, rbg125a, rbg172a, rbg233 | Ascheuer の AFG セット（López-Ibáñez の一覧） | 公表最適値（9 問とも証明済み） |
| JSSP | ft06, ft10, la16, la21, abz5, orb01, ta01, swv01, ta11, ta21 | JSPLIB（OR-Library 形式） | 既知最適値（ta11=1357, ta21=1642 は文献値） |
| NSP | nsp01–nsp06（8〜50 人, 14〜42 日, 1〜4 勤務種別） | 合成（`make_instances.py` で再生成可能） | 全 run・全手法中の最良値 |

NSP は Curtois & Qu (2014) の Shift Scheduling Benchmark（schedulingbenchmarks.org）と同じ制約体系を持つ合成インスタンスです。ハード制約は 1 日 1 勤務、禁止勤務遷移、勤務種別ごとの上限回数、総勤務時間の上下限、最大・最小連続勤務日数、最小連続休日数、最大勤務週末数、指定休日。ソフト制約は勤務希望・回避希望と、必要人数の不足（重み 100/人）・超過（重み 1/人）です。公式インスタンスへの差し替えは今後の課題です。

## 解法

| 問題 | 手法 | 区分 |
|---|---|---|
| TSP/ATSP | `nn+ls` 多始点の最近傍法 + 2-opt/Or-opt 局所探索 | 構築法 + 局所探索（自作, numba） |
| | `SA` 焼きなまし法（2-opt / Or-opt 近傍） | メタヒューリスティクス（自作, numba） |
| | `LKH(elkai)` Lin-Kernighan-Helsgaun（elkai, runs=10） | 専用ヒューリスティクス |
| | `ORTools-GLS` OR-Tools Routing + 誘導局所探索 | メタヒューリスティクス |
| | `CP-SAT` AddCircuit 制約 | 制約プログラミング |
| | `MIP-HiGHS` DFJ 定式化 + 部分巡回路除去制約の逐次追加 | 数理計画（厳密解法） |
| ATSP-TW | `EDD+LS` 締め切り順 + Or-opt/2 点交換の局所探索 | 構築法 + 局所探索（自作, numba） |
| | `GVNS` 一般化変近傍探索（遅れの合計を最優先に評価） | メタヒューリスティクス（自作, numba） |
| | `LKH-3(elkai)` elkai 同梱の LKH-3.0.8 を TSPTW として実行 | 専用ヒューリスティクス |
| | `PyVRP-HGS` PyVRP（車両 1 台の時間枠付き配送計画として解く） | メタヒューリスティクス（HGS） |
| | `ORTools-GLS` OR-Tools Routing + 時間の次元 + 誘導局所探索 | メタヒューリスティクス |
| | `CP-SAT`, `CP-SAT-4w` AddCircuit + 時刻変数（1 / 4 ワーカー） | 制約プログラミング |
| | `MIP-HiGHS`, `MIP-HiGHS-4w` 時刻変数の big-M 定式化（1 / 4 スレッド） | 数理計画 |
| JSSP | `SPT`, `MWKR` Giffler-Thompson 法 + 優先規則 | ディスパッチングルール |
| | `GA` 乱数キー GA（BRKGA 型, 能動スケジュール復号） | メタヒューリスティクス（自作） |
| | `Tabu-N5` タブー探索（Nowicki & Smutnicki の N5 近傍） | メタヒューリスティクス（自作, numba） |
| | `CP-SAT`, `CP-SAT-4w` 区間変数 + NoOverlap（1 / 4 ワーカー） | 制約プログラミング |
| | `MIP-HiGHS` Manne の離接定式化（big-M） | 数理計画 |
| NSP | `Greedy` 空の勤務表からの最急降下 | 構築法 |
| | `SA` 焼きなまし法（1 マス変更 + 職員間の区間交換） | メタヒューリスティクス（自作, numba） |
| | `CP-SAT`, `CP-SAT-4w` 0-1 線形モデル（1 / 4 ワーカー） | 制約プログラミング |
| | `MIP-HiGHS` 同じ 0-1 線形モデル | 数理計画 |

## 実行条件

- 制限時間 60 秒（`LKH(elkai)` は時間制限を持たないため runs=10 固定、構築法は収束まで）
- 全手法 1 スレッド（`-4w` のみ 4 スレッド、単独実行）。1 スレッド手法は 4 ジョブ並列で実行。手法名の `-<n>w` は同じ解法を n スレッドで実行したもの（`--threads` で任意のスレッド数を追加できます。[`docs/extending.md`](docs/extending.md#スレッド数を変えて実行する)）
- 確率的手法（`SA`, `GA`, `Tabu-N5`）は 3 シードの平均
- 指標: 参照値に対するギャップ (%) = (得られた値 − 参照値) / 参照値 × 100
- 実行環境: 4 vCPU / 16 GB のクラウドコンテナ、Python 3.13、OR-Tools 9.15、HiGHS 1.15（run ごとの詳細は各 `run.json`）

## 初回結果のサマリ（2026-10-25, 制限時間 60 秒）

数値は参照値に対する平均ギャップ (%)。詳細はビュー（`docs/`）と `results/runs/*-cloud-*/summary.csv` を参照してください。

| 問題 | 1 位 | 主な結果 |
|---|---|---|
| TSP（6 問） | `LKH(elkai)` 0.00% | LKH は全問で最適値に平均 8.5 秒で到達。`MIP-HiGHS` は a280 まで最適性を証明したが、pcb442 以上は 60 秒で巡回路が得られず。自作 `SA` 0.42%、`nn+ls` 0.83%、`ORTools-GLS` 0.97%。`CP-SAT`（1 ワーカー）は 15.4% と苦手 |
| ATSP（6 問） | `LKH(elkai)` 0.00% | `MIP-HiGHS` は 124 都市まで最適性を証明、rbg323/443 は時間切れ。`SA` 0.04%、`ORTools-GLS` 0.47%、`nn+ls` 3.03%。`CP-SAT` は rbg323 で最適性を証明する一方、rbg443 で 104% と不安定 |
| JSSP（10 問） | `CP-SAT-4w` 0.55% | `CP-SAT` は 1 ワーカーでも 1.68%（10 問中 5 問で最適性を証明、4 ワーカーで 7 問）。自作 `Tabu-N5` 2.00% が肉薄。`GA` 10.7%、`MIP-HiGHS`（big-M）26.6%、ディスパッチ則 29〜55% |
| NSP（合成 6 問） | `MIP-HiGHS`（5 問で最良） | `MIP-HiGHS` は nsp01〜05 で最良値（3 問で最適性を証明）だが、最大の nsp06 は解なし。`CP-SAT-4w` は全問で解を得て平均 2.7%。`CP-SAT`（1 ワーカー）は大規模 2 問で解なし。`SA` は全問で実行可能解を得るが 0〜84% 劣る。`Greedy` は 4 問でハード制約違反 |

傾向としては、ルーティング系（TSP/ATSP）では専用ヒューリスティクス（LKH）が圧倒的で、厳密解法は数百ノードが 60 秒での限界でした。スケジューリング系（JSSP）では CP-SAT が最も強く、よく設計された近傍のタブー探索が続きます。勤務表（NSP）のように 0-1 線形で素直に書けて LP 緩和が強い問題では MIP が強い一方、大規模化すると CP-SAT の並列探索が安定します。

### 注意点

- 決定的な手法は 1 回、確率的手法は 3 シードの結果です。順位の僅差は誤差の範囲と考えてください。
- `CP-SAT` は本来マルチワーカー前提のソルバーです。1 ワーカーの結果は公平性のための参考値で、実用上は `-4w` を見てください。
- `LKH(elkai)` は時間制限ではなく試行回数（runs=10）で止めています。
- NSP は合成インスタンスで、参照値は全 run・全手法中の最良値です（最適値ではありません）。

## 再現方法

```bash
python -m venv .venv && . .venv/bin/activate   # Windows (PowerShell) は .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/make_instances.py nsp      # NSP インスタンスを再生成（同一 seed で同じものが出ます）
python src/run_bench.py tsp 60       # tsp / atsp / jssp / nsp
python src/analyze.py                # 問題ごとに最新の run の集計表を表示・CSV 出力（run_id を渡すとその run）
python src/export_web.py             # docs/data.json・data.js と results/README.md・figures を更新
```

`--only` で手法を、`--instances` でインスタンスを絞れます。`--threads 1,4,8` で並列実行できる解法（CP-SAT, MIP-HiGHS）をそれぞれのスレッド数で実行し、`--run <run_id> --threads 8` で既存の run に後からスレッド条件を追加できます。`--note "メモ"` で run にメモを、`--machine 名前` で実行環境の名前（省略時はホスト名）を付けられます。

## 実験結果の履歴管理

`run_bench.py` を 1 回実行するごとに、1 つの run として `results/runs/<run_id>/` に保存されます。run_id は `<開始日時 UTC>-<マシン名>-<問題>`（例: `20261025T050501Z-cloud-tsp`）です。

```
results/runs/<run_id>/
  run.json       メタデータ
  results.jsonl  1 ジョブ 1 行の生データ
  summary.csv    集計（analyze.py / export_web.py が書き出す）
```

`run.json` には次の情報が入ります。

| キー | 内容 |
|---|---|
| `started_at`, `finished_at`, `status` | 実行開始・終了日時（UTC）と状態（`running` / `complete`） |
| `environment` | マシン名、CPU 型番、論理 CPU 数、メモリ、OS、Python とソルバー各パッケージのバージョン |
| `conditions` | 制限時間、同時実行ジョブ数、手法ごとのスレッド数（`threads`）と元の解法名（`solvers`）、インスタンス一覧、手法ごとのシード |
| `git` | 実行時のコミットハッシュ・ブランチ、`src/`・`instances/`・`requirements.txt` に未コミットの変更があったか（`dirty`）とそのファイル |
| `instance_sha256` | 各インスタンスファイルの SHA-256（同じ名前で中身が変わっていないかの確認用） |
| `command`, `note` | 実行したコマンドと任意のメモ |
| `additions` | 後から追加した手法ごとの日時・手法名・git コミット・コマンド（追加したときだけ） |

結果を正しく紐づけるため、コードを変えたらコミットしてから実行してください（未コミットのまま実行すると警告が出て、`dirty: true` が記録されます）。途中で止まった run は `python src/run_bench.py <problem> <time_limit> --run <run_id>` で再開でき、記録済みのジョブはスキップされます。実行環境が同じなら、`python src/run_bench.py <problem> <time_limit> --run latest --only <手法>` で既存の run に手法を後から追加できます（追加した手法とそのときのコミットは `run.json` の `additions` に記録。詳しくは [docs/extending.md](docs/extending.md)）。再開・追加は、実行環境やインスタンスが run 作成時と異なると止まります。

`results/README.md` と `docs/` のビューは、問題ごとに最新の run を既定で表示します。ビューでは run を切り替えられ、「比較対象」に別の run を選ぶと、実行環境・条件の違いと手法別・インスタンス別の差が表示されます。NSP の参照値は、同じ内容のインスタンスを解いた全 run・全手法の最良値です。

初回の結果（クラウドコンテナで実行）は `*-cloud-*` の 4 つの run として移行済みです。移行前は開始・終了日時を記録していなかったため、結果をコミットした時刻を入れています。

## 結果の閲覧

- GitHub 上: [`results/README.md`](results/README.md) に問題ごとの棒グラフ・ヒートマップ（SVG）と数値表があります。リポジトリが非公開のままでも見られます。
- 手元: `docs/index.html`（入口）をダブルクリックすると、対話型の結果ビュー（`docs/benchmark.html`、マウスオーバーで詳細が出ます）、文献調査（`docs/literature/index.html`）、考察（`docs/discussion/index.html`）へ進めます（サーバー不要）。
- GitHub Pages を使える場合は、Settings → Pages でブランチ `main`・フォルダ `/docs` を選ぶと入口のポータル（`docs/index.html`）から同じページを公開できます。
