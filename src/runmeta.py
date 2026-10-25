"""run（1 回のベンチマーク実行）の保存先とメタデータ.

results/runs/<run_id>/
  run.json       メタデータ（実行日時・実行環境・実行条件・git コミット・インスタンスのハッシュ）
  results.jsonl  1 ジョブ 1 行の生データ
  summary.csv    集計（analyze.py が書き出す）
"""
import datetime
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys

import registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下
RUNS = os.path.join(ROOT, "results", "runs")
# 実行環境として必ずバージョンを記録するパッケージ（問題・解法が宣言したパッケージも追加で記録する）
PACKAGES = ["ortools", "highspy", "elkai", "numba", "numpy", "pandas", "tsplib95"]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_model():
    if sys.platform.startswith("linux"):
        try:
            for l in open("/proc/cpuinfo", encoding="utf-8"):
                if l.startswith("model name"):
                    return l.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        s = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if s:
            return s
    return platform.processor() or platform.machine()


def memory_gb():
    try:
        if sys.platform.startswith("linux"):
            for l in open("/proc/meminfo", encoding="utf-8"):
                if l.startswith("MemTotal"):
                    return round(int(l.split()[1]) / 1024 ** 2, 1)
        elif sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 1024 ** 3, 1)
        elif sys.platform == "darwin":
            return round(int(_cmd(["sysctl", "-n", "hw.memsize"])) / 1024 ** 3, 1)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in dict.fromkeys(PACKAGES + registry.packages()):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def environment(machine=None):
    return dict(machine=machine or socket.gethostname(), cpu=cpu_model(), cpu_count=os.cpu_count(),
                memory_gb=memory_gb(), os=platform.platform(), python=platform.python_version(),
                packages=package_versions())


def git_info():
    commit = _cmd(["git", "rev-parse", "HEAD"])
    # 計測結果に影響するファイル（コード・インスタンス・依存）の未コミット変更
    dirty = _cmd(["git", "diff", "--name-only", "HEAD", "--", "src", "instances", "requirements.txt"]).splitlines()
    return dict(commit=commit or None, branch=_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or None,
                dirty=bool(dirty), dirty_files=dirty)


def instance_hashes(problem, instances):
    out = {}
    p = registry.get_problem(problem)
    for i in instances:
        with open(p.instance_path(i), "rb") as f:
            out[i] = hashlib.sha256(f.read()).hexdigest()
    return out


def new_run_id(problem, machine):
    t = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{t}-{re.sub(r'[^A-Za-z0-9_.-]+', '-', machine)}-{problem}"


def run_dir(run_id):
    return os.path.join(RUNS, run_id)


def read_meta(run_id):
    with open(os.path.join(run_dir(run_id), "run.json"), encoding="utf-8") as f:
        return json.load(f)


def write_meta(meta):
    d = run_dir(meta["run_id"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")


def list_runs(problem=None):
    """run.json を持つ run を開始日時順に返す."""
    if not os.path.isdir(RUNS):
        return []
    metas = [read_meta(r) for r in os.listdir(RUNS) if os.path.exists(os.path.join(RUNS, r, "run.json"))]
    metas = [m for m in metas if problem is None or m["problem"] == problem]
    return sorted(metas, key=lambda m: (m["started_at"], m["run_id"]))


def latest_run(problem):
    runs = list_runs(problem)
    return runs[-1]["run_id"] if runs else None
