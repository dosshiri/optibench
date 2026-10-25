"""問題と解法の登録簿.

src/problems/*.py と src/solvers/*.py を読み込むと、各ファイルが register_problem() と @solver で
自分自身を登録する. 実行・集計・ビュー書き出しはすべてこの登録内容を参照するので、
新しい問題や解法はファイルを 1 つ追加するだけで使えるようになる（手順は docs/extending.md）.

python registry.py   登録されている問題と解法の一覧を表示する
"""
import glob
import importlib
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)  # リポジトリ直下
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@dataclass
class Problem:
    """問題種別の定義.

    name           問題の識別子（run_id・コマンド引数・結果ファイルに使う。英小文字推奨）
    label          表示名
    unit           目的関数値の単位（例: 距離, メイクスパン）
    ref_label      参照値の説明（ビューと結果ページに表示）
    instance_file  instances/ からのインスタンスファイルの相対パス. {} にインスタンス名が入る
    instances      既定で解くインスタンスの一覧. リスト、または一覧を返す関数
    load           load(インスタンス名) -> 解法に渡すデータ
    reference      reference(インスタンス名) -> 参照値（既知最適値など）. None を返すと、
                   同じ内容のインスタンスを解いた全 run・全手法の最良値を参照値にする
    evaluate       evaluate(データ, 解) -> 目的関数値（実行不能なら None）. 解法が解を返したとき、
                   申告された目的関数値をこれで検証する. 省略可
    generate       インスタンスを（再）生成する関数（合成インスタンスの場合）. 省略可
    sense          "min"（最小化）または "max"（最大化）
    display        "gap"（参照値に対するギャップで比較）または "value"（参照値が最適値でないので
                   値と順位で比較）
    order          ビューのタブなどに並べる順（小さいほど先）
    packages       run.json にバージョンを記録する追加パッケージ
    """
    name: str
    label: str
    unit: str
    ref_label: str
    instance_file: str
    instances: object
    load: Callable
    reference: Callable = lambda name: None
    evaluate: Optional[Callable] = None
    generate: Optional[Callable] = None
    sense: str = "min"
    display: str = "gap"
    order: int = 100
    packages: list = field(default_factory=list)

    def instance_path(self, name):
        return os.path.join(ROOT, "instances", self.instance_file.format(name))

    def default_instances(self):
        return list(self.instances()) if callable(self.instances) else list(self.instances)


@dataclass
class Solver:
    """解法の定義. func(データ, 制限時間[秒], seed) -> dict(obj=目的関数値 or None, lb=下界 or None, ...).
    並列実行できる解法（parallel=True）は func(データ, 制限時間, seed, threads=スレッド数) で呼ばれる.

    seeds     既定で回すシード数（確率的手法は 3 など）
    threads   既定で回すスレッド数の一覧（例: [1, 4]）. run_bench.py --threads で変えられる
    parallel  スレッド数を変えて実行できるか（threads=1 以外を指定すると自動で True）
    packages  run.json にバージョンを記録する追加パッケージ

    結果の手法名（ラベル）は、1 スレッドなら name、それ以外は "name-4w" のようにスレッド数を付ける.
    """
    name: str
    problem: str
    func: Callable
    seeds: int = 1
    threads: list = field(default_factory=lambda: [1])
    parallel: bool = False
    packages: list = field(default_factory=list)

    def run(self, data, tl, seed, threads=1):
        if self.parallel:
            return self.func(data, tl, seed, threads=threads)
        if threads != 1:
            raise ValueError(f"{self.name} は 1 スレッドでしか実行できません")
        return self.func(data, tl, seed)


def label(name, threads):
    """結果に記録する手法名. 1 スレッドは解法名のまま、それ以外は "-<スレッド数>w" を付ける."""
    return name if threads == 1 else f"{name}-{threads}w"


PROBLEMS = {}
SOLVERS = {}  # problem -> {name: Solver}（登録順）
_loaded = False


def register_problem(p):
    if p.name in PROBLEMS:
        raise ValueError(f"問題 {p.name} は登録済みです")
    PROBLEMS[p.name] = p
    return p


def solver(name, problems, seeds=1, threads=1, parallel=None, packages=()):
    """解法を登録するデコレータ.
    例: @solver("SA", problems=["tsp", "atsp"], seeds=3)
        @solver("CP-SAT", problems="jssp", threads=[1, 4])   # 1 スレッドと 4 スレッドで実行"""
    threads = [threads] if isinstance(threads, int) else list(threads)
    if parallel is None:
        parallel = threads != [1]
    def deco(func):
        for prob in ([problems] if isinstance(problems, str) else problems):
            d = SOLVERS.setdefault(prob, {})
            if name in d:
                raise ValueError(f"{prob} の解法 {name} は登録済みです")
            d[name] = Solver(name, prob, func, seeds, threads, parallel, list(packages))
        return func
    return deco


def load_plugins():
    """src/problems/ と src/solvers/ の全モジュールを読み込む（_ で始まるファイルは除く）.
    依存パッケージが無いなどで読み込めないモジュールは警告を出して飛ばす."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    for pkg in ("problems", "solvers"):
        for path in sorted(glob.glob(os.path.join(SRC, pkg, "*.py"))):
            mod = os.path.splitext(os.path.basename(path))[0]
            if mod.startswith("_"):
                continue
            try:
                importlib.import_module(f"{pkg}.{mod}")
            except Exception:
                print(f"警告: {pkg}/{mod}.py を読み込めませんでした（スキップします）\n"
                      + traceback.format_exc(limit=1), file=sys.stderr)


def problems():
    """登録されている問題を表示順に返す."""
    load_plugins()
    return sorted(PROBLEMS.values(), key=lambda p: (p.order, p.name))


def get_problem(name):
    load_plugins()
    if name not in PROBLEMS:
        sys.exit(f"問題 {name} は登録されていません（登録済み: {', '.join(p.name for p in problems())}）")
    return PROBLEMS[name]


def solvers(problem):
    """問題に登録されている解法 {name: Solver} を登録順に返す."""
    load_plugins()
    return SOLVERS.get(problem, {})


def resolve(problem, method):
    """手法名（ラベル）から (Solver, スレッド数) を返す. 例: "CP-SAT-4w" -> (CP-SAT, 4)"""
    s = solvers(problem)
    if method in s:
        return s[method], 1
    m = re.fullmatch(r"(.+)-(\d+)w", method)
    if m and m.group(1) in s and s[m.group(1)].parallel:
        return s[m.group(1)], int(m.group(2))
    sys.exit(f"{problem} の手法 {method} は登録されていません（登録済み: {', '.join(s)}）")


def packages():
    """run.json にバージョンを記録する、問題・解法が宣言した追加パッケージ."""
    load_plugins()
    out = [x for p in PROBLEMS.values() for x in p.packages]
    out += [x for d in SOLVERS.values() for s in d.values() for x in s.packages]
    return list(dict.fromkeys(out))


def main():
    for p in problems():
        print(f"{p.name}: {p.label}（{len(p.default_instances())} インスタンス, 参照値: {p.ref_label}）")
        for s in solvers(p.name).values():
            extra = [f"{s.seeds} シード"] * (s.seeds > 1)
            if s.parallel:
                extra.append("並列可, 既定 " + "/".join(map(str, s.threads)) + " スレッド")
            print(f"  {s.name}" + (f"（{', '.join(extra)}）" if extra else ""))
    orphan = sorted(set(SOLVERS) - set(PROBLEMS))
    if orphan:
        print("警告: 問題が登録されていない解法があります:", ", ".join(orphan))


if __name__ == "__main__":
    # プラグインは "registry" として import するので、スクリプト実行時もそちらの登録簿を使う
    import registry
    registry.main()
