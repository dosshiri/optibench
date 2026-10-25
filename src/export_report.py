"""ビュー用データ (docs/data.json) から、GitHub 上でそのまま読める結果ページを書き出す.
results/README.md（表）と results/figures/*.svg（棒グラフ・ヒートマップ）を生成する.
python export_report.py [data.json]   （通常は export_web.py から呼ばれる）"""
import json
import os
import sys
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # リポジトリ直下
OUT_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# docs/index.html と同じ配色・段階（GitHub のダーク表示でも読めるよう背景は白で固定）
BINS = [0.001, 0.5, 1, 2, 5, 10, 20]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BIN_NAMES = ["最良値", "<0.5%", "<1%", "<2%", "<5%", "<10%", "≥10%"]
BG, TEXT, TEXT2, GRID, AXIS, BAR, MISSING = "#ffffff", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#2a78d6", "#f0efec"
FONT = "-apple-system, 'Segoe UI', 'Hiragino Sans', 'Yu Gothic UI', Meiryo, sans-serif"


def bin_of(g):
    if g <= BINS[0]:
        return 0
    for i in range(1, len(BINS)):
        if g < BINS[i]:
            return i
    return 6


def fmt(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def method_stats(P):
    """index.html の methodStats と同じ集計・並び順."""
    out = []
    for m in P["methods"]:
        cs = [c for c in P["cells"] if c["method"] == m]
        ok = [c for c in cs if c["mean_gap"] is not None]
        mean = sum(c["mean_gap"] for c in ok) / len(ok) if ok else None
        ranks = []
        for inst in P["instances"]:
            mine = next((c for c in cs if c["instance"] == inst), None)
            if mine is None or mine["mean_obj"] is None:
                ranks.append(len(P["methods"]))
                continue
            others = [c for c in P["cells"] if c["instance"] == inst and c["mean_obj"] is not None]
            ranks.append(1 + sum(c["mean_obj"] < mine["mean_obj"] - 1e-9 for c in others))
        out.append(dict(method=m, mean=mean, rank=sum(ranks) / len(ranks), solved=len(ok),
                        total=len(P["instances"]), nopt=sum(c["mean_gap"] <= 1e-9 for c in ok),
                        time=sum(c["time"] for c in cs) / len(cs) if cs else 0.0))
    if P.get("display") == "value":
        out.sort(key=lambda s: s["rank"])
    else:
        out.sort(key=lambda s: (-s["solved"], s["mean"] if s["mean"] is not None else 1e9))
    return out


def cell(P, inst, m):
    return next((c for c in P["cells"] if c["instance"] == inst and c["method"] == m), None)


def text_width(s):
    """12px 文字列のおおよその幅（全角 12px・半角 7px）."""
    return sum(12 if ord(ch) > 0x2000 else 7 for ch in s)


def svg_doc(W, H, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}" font-size="12">\n<rect width="{W}" height="{H}" fill="{BG}"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def text(x, y, s, anchor="start", fill=TEXT, weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}"{w}>{escape(str(s))}</text>'


def bars_svg(P, st):
    by_rank = P.get("display") == "value"
    W, row_h, left, right, top = 720, 30, 110, 190, 36
    H = top + len(st) * row_h + 26
    vmax = len(P["methods"]) if by_rank else max([1.0] + [s["mean"] or 0 for s in st]) * 1.05
    x = lambda v: left + (W - left - right) * v / vmax
    title = f'{P["label"]}: ' + ("手法別の平均順位（小さいほど良い）" if by_rank else "手法別の平均ギャップ (%)")
    body = [text(8, 20, title, weight="600")]
    for i in range(5):
        v = vmax * i / 4
        xx = x(v)
        body.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{H - 20}" stroke="{AXIS if i == 0 else GRID}"/>')
        lab = f"{v:.1f}位" if by_rank else (f"{v:.1f}%" if v < 10 else f"{v:.0f}%")
        body.append(text(xx, H - 5, lab, "middle", TEXT2))
    for i, s in enumerate(st):
        y = top + i * row_h
        body.append(text(left - 8, y + row_h / 2 + 4, s["method"], "end"))
        val = s["rank"] if by_rank else s["mean"]
        w = 0 if val is None else max(2, x(val) - left)
        body.append(f'<rect x="{left}" y="{y + 7}" width="{w:.1f}" height="{row_h - 14}" rx="4" fill="{BAR}"/>')
        lab = f'{fmt(s["rank"], 1)}位' if by_rank else ("解なし" if s["mean"] is None else f'{fmt(s["mean"])}%')
        if s["solved"] < s["total"]:
            lab += f'（解なし{s["total"] - s["solved"]}問）'
        body.append(text(left + w + 6, y + row_h / 2 + 4, lab, fill=TEXT2))
    return svg_doc(W, H, body)


def heat_svg(P, order):
    by_value = P.get("display") == "value"
    cw, ch, left, top = 96, 30, 96, 64
    legend = list(zip(BIN_NAMES, SEQ)) + [("解なし", MISSING)]
    legend_w = 86 + sum(30 + text_width(n) for n, _ in legend)
    W = max(left + cw * len(order) + 8, legend_w + 8)
    H = top + ch * len(P["instances"]) + 40
    title = f'{P["label"]}: ' + (f'インスタンス × 手法の{P["unit"]}（色は最良値からの差）' if by_value
                                 else "インスタンス × 手法のギャップ (%)")
    body = [text(8, 20, title, weight="600")]
    for j, m in enumerate(order):
        body.append(text(left + j * cw + cw / 2, top - 10, m, "middle"))
    for i, inst in enumerate(P["instances"]):
        y0 = top + i * ch
        body.append(text(left - 8, y0 + ch / 2 + 4, inst, "end"))
        for j, m in enumerate(order):
            c = cell(P, inst, m)
            x0 = left + j * cw
            fill, ink, txt = MISSING, TEXT2, "×"
            if c and c["mean_gap"] is not None:
                b = bin_of(c["mean_gap"])
                fill, ink = SEQ[b], ("#ffffff" if b >= 4 else TEXT)
                if by_value:
                    txt = str(round(c["mean_obj"]))
                else:
                    g = c["mean_gap"]
                    txt = "0" if g <= 1e-9 else fmt(g, 2 if g < 10 else 1)
            body.append(f'<rect x="{x0 + 1}" y="{y0 + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="{fill}"/>')
            body.append(text(x0 + cw / 2, y0 + ch / 2 + 4, txt, "middle", ink))
    # 凡例
    ly, lx = H - 16, 8
    body.append(text(lx, ly + 4, "最良値との差:" if by_value else "ギャップ:", fill=TEXT2))
    lx += 86 if by_value else 60
    for name, color in legend:
        body.append(f'<rect x="{lx}" y="{ly - 6}" width="12" height="12" rx="2" fill="{color}"/>')
        body.append(text(lx + 16, ly + 4, name, fill=TEXT2))
        lx += 30 + text_width(name)
    return svg_doc(W, H, body)


def md_tables(P, st):
    by_value = P.get("display") == "value"
    order = [s["method"] for s in st]
    lines = ["| 手法 | " + ("平均順位 | 平均ペナルティ差 (%)" if by_value else "平均ギャップ (%) | 平均順位")
             + " | 解あり | 最良値到達 | 平均時間 (秒) |", "|---|---:|---:|---:|---:|---:|"]
    for s in st:
        a, b = (fmt(s["rank"], 1), fmt(s["mean"])) if by_value else (fmt(s["mean"]), fmt(s["rank"], 1))
        lines.append(f'| {s["method"]} | {a} | {b} | {s["solved"]}/{s["total"]} | {s["nopt"]} | {fmt(s["time"], 1)} |')
    lines += ["", ("インスタンス別の" + P["unit"] + "の平均値" if by_value else "インスタンス別の平均ギャップ (%)")
              + "。太字は各インスタンスの最良、× は制限時間内に実行可能解なし。", ""]
    lines += ["| インスタンス | " + " | ".join(order) + " |", "|---" + "|---:" * len(order) + "|"]
    for inst in P["instances"]:
        cs = {m: cell(P, inst, m) for m in order}
        vals = {m: (c["mean_obj"] if by_value else c["mean_gap"]) for m, c in cs.items()
                if c and c["mean_gap"] is not None}
        best = min(vals.values()) if vals else None
        row = []
        for m in order:
            if m not in vals:
                row.append("×")
                continue
            v = vals[m]
            s = str(round(v)) if by_value else ("0" if v <= 1e-9 else fmt(v))
            row.append(f"**{s}**" if abs(v - best) <= 1e-9 else s)
        lines.append(f"| {inst} | " + " | ".join(row) + " |")
    return lines


def main(data):
    os.makedirs(FIG_DIR, exist_ok=True)
    md = ["# ベンチマーク結果", "",
          "このページは `python src/export_web.py` で自動生成しています（手で編集しないでください）。"
          "マウスオーバーで詳細が見られる対話型のビューは、手元で `docs/index.html` をダブルクリックすると開きます。", "",
          f'データ生成: {data["generated"]}（Python {data["env"]["python"]}, {data["env"]["machine"]}）', ""]
    md += ["- " + " / ".join(f'[{P["label"]}](#{k})' for k, P in data["problems"].items()), ""]
    for k, P in data["problems"].items():
        st = method_stats(P)
        order = [s["method"] for s in st]
        for name, svg in ((f"{k}_bars.svg", bars_svg(P, st)), (f"{k}_heat.svg", heat_svg(P, order))):
            with open(os.path.join(FIG_DIR, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(svg)
        md += [f'<a id="{k}"></a>', "", f'## {P["label"]}', "",
               f'参照値: {P["ref"]}。制限時間 {P["time_limit"]:g} 秒、全手法 1 スレッド（-4w は 4 スレッド）。'
               "確率的手法は複数シードの平均。", "",
               f"![{P['label']} 手法別](figures/{k}_bars.svg)", "",
               f"![{P['label']} ヒートマップ](figures/{k}_heat.svg)", ""]
        md += md_tables(P, st) + [""]
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "data.json")
    with open(p, encoding="utf-8") as f:
        main(json.load(f))
