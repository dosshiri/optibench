# optibench

組合せ最適化ベンチマーク。実務で使われる代表的な組合せ最適化問題について、数理計画・制約プログラミング・メタヒューリスティクスなどの解法を、同じ条件で手元比較するためのリポジトリです。

- 対象問題: TSP（対称巡回セールスマン）、ATSP（非対称）、JSSP（ジョブショップスケジューリング）、ナーススケジューリング（NSP）
- 結果ビュー: `docs/index.html`（GitHub Pages で公開すると、手法別ギャップ・ヒートマップ・数値表をブラウザで閲覧できます）

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
  export_web.py   ビュー用 JSON（docs/data.json）の書き出し
instances/      ベンチマークインスタンス
results/        実行結果の生データ（JSONL）と集計 CSV
docs/           GitHub Pages 用のグラフビュー
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

## 再現方法

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python src/nsp_model.py              # NSP インスタンスを再生成（同一 seed で同じものが出ます）
python src/run_bench.py tsp 60       # tsp / atsp / jssp / nsp
python src/analyze.py                # 集計表を表示・CSV 出力
python src/export_web.py             # docs/data.json を更新
```

`results/<problem>.jsonl` に記録済みのジョブはスキップされるため、手法やインスタンスを追加して同じコマンドを実行すれば差分だけ実行されます。`--only` で手法を絞れます。

## GitHub Pages での閲覧

Settings → Pages で「Deploy from a branch」、ブランチ `main`、フォルダ `/docs` を選ぶと、`https://<owner>.github.io/optibench/` でビューが開けます。ローカルでは `python -m http.server -d docs` で確認できます。
