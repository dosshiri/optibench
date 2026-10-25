# optibench

組合せ最適化ベンチマーク。実務で使われる代表的な組合せ最適化問題について、数理計画・制約プログラミング・メタヒューリスティクスなどの解法を、同じ条件で手元比較するためのリポジトリです。

- 対象問題: TSP（対称巡回セールスマン）、ATSP（非対称）、JSSP（ジョブショップスケジューリング）、ナーススケジューリング（NSP）
- 文献調査: `docs/literature/`（既存研究における解法比較のレビュー）
- 結果: [`results/README.md`](results/README.md)（GitHub 上でそのまま読めるグラフと表）、`docs/index.html`（手元でダブルクリックすると開く対話型ビュー）

## 構成

```
src/            実行スクリプト
  common.py       インスタンス読込・既知最適値
  tsp_methods.py  TSP/ATSP の解法
  jssp_methods.py JSSP の解法
  nsp_model.py    NSP のモデル・評価関数・合成インスタンス生成
  nsp_methods.py  NSP の解法
  run_one.py      1 ジョブ実行（JSON 1 行を出力）
  run_bench.py    一括実行（results/<problem>.jsonl に追記、完了済みはスキップ）
  analyze.py      集計（results/summary_<problem>.csv）
  export_web.py   ビュー用データ（docs/data.json, docs/data.js）と結果ページの書き出し
  export_report.py 結果ページ（results/README.md, results/figures/*.svg）の生成
instances/      ベンチマークインスタンス
results/        実行結果の生データ（JSONL）・集計 CSV・結果ページ（README.md と figures/）
docs/           対話型のグラフビュー（index.html）
```

## インスタンス

| 問題 | インスタンス | 出典 | 参照値 |
|---|---|---|---|
| TSP | berlin52, kroA100, ch150, a280, pcb442, rat783 | TSPLIB95 | 公表最適値 |
| ATSP | ftv33, ftv70, kro124p, ft70, rbg323, rbg443 | TSPLIB95 | 公表最適値 |
| JSSP | ft06, ft10, la16, la21, abz5, orb01, ta01, swv01, ta11, ta21 | JSPLIB（OR-Library 形式） | 既知最適値（ta11=1357, ta21=1642 は文献値） |
| NSP | nsp01–nsp06（8〜50 人, 14〜42 日, 1〜4 勤務種別） | 合成（`nsp_model.py` で再生成可能） | 全手法中の最良値 |

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
- 全手法 1 スレッド（`-4w` のみ 4 スレッド、単独実行）。1 スレッド手法は 4 ジョブ並列で実行
- 確率的手法（`SA`, `GA`, `Tabu-N5`）は 3 シードの平均
- 指標: 参照値に対するギャップ (%) = (得られた値 − 参照値) / 参照値 × 100
- 実行環境: 4 vCPU / 16 GB のクラウドコンテナ、Python 3.13、OR-Tools 9.15、HiGHS 1.15

## 初回結果のサマリ（2026-10-25, 制限時間 60 秒）

数値は参照値に対する平均ギャップ (%)。詳細はビュー（`docs/`）と `results/summary_*.csv` を参照してください。

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
- NSP は合成インスタンスで、参照値は全手法中の最良値です（最適値ではありません）。

## 再現方法

```bash
python -m venv .venv && . .venv/bin/activate   # Windows (PowerShell) は .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/nsp_model.py              # NSP インスタンスを再生成（同一 seed で同じものが出ます）
python src/run_bench.py tsp 60       # tsp / atsp / jssp / nsp
python src/analyze.py                # 集計表を表示・CSV 出力
python src/export_web.py             # docs/data.json・data.js と results/README.md・figures を更新
```

`results/<problem>.jsonl` に記録済みのジョブはスキップされるため、手法やインスタンスを追加して同じコマンドを実行すれば差分だけ実行されます。`--only` で手法を絞れます。

## 結果の閲覧

- GitHub 上: [`results/README.md`](results/README.md) に問題ごとの棒グラフ・ヒートマップ（SVG）と数値表があります。リポジトリが非公開のままでも見られます。
- 手元: `docs/index.html` をダブルクリックすると、マウスオーバーで詳細が出る対話型ビューが開きます（サーバー不要）。
- GitHub Pages を使える場合は、Settings → Pages でブランチ `main`・フォルダ `/docs` を選ぶと同じビューを公開できます。
