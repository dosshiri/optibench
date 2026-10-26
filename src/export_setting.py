"""問題設定とインスタンスの説明を、各ページに書き出す.

説明の中身は src/problems/*.py の register_problem(Problem(..., setting=..., instance_info=...)) に書く.
export_web.py から呼ばれ、次の場所を更新する.
  docs/data.json    problems.<問題>.setting と instance_table（結果ビューが表示する）
  results/README.md 問題ごとの「問題設定」の節（export_report.py が書く）
  docs/discussion/  考察の問題ページの「問題設定とインスタンス」の節（目印のコメントの間を置き換える）
  docs/index.html   ポータルの「扱う問題」の表（同上）
"""
import html
import os
import re

import registry

ROOT = registry.ROOT
BEGIN = "<!-- 問題設定とインスタンス: python src/export_web.py が src/problems/ の setting と instance_info から書き出す（この間は直接編集しない） -->"
END = "<!-- /問題設定とインスタンス -->"
PORTAL_BEGIN = "<!-- 扱う問題: python src/export_web.py が src/problems/ の setting から書き出す（この間は直接編集しない） -->"
PORTAL_END = "<!-- /扱う問題 -->"
DISCUSSION_WEB = "https://claude.ai/artifact/UHqkvC8dU3UuCXUqZ5AvEN"  # 考察の Claude ページ


def short_label(p):
    """表示名の括弧書きを省いた問題名（例: "TSP（対称巡回セールスマン）" -> "TSP"）."""
    return p.label.split("（")[0]


def instance_table(p):
    """インスタンスごとの規模などの表. {"columns": [...], "rows": [[インスタンス名, 値...], ...]}"""
    if not p.instance_info:
        return None
    infos = [(name, p.instance_info(name)) for name in p.default_instances()]
    columns = list(dict.fromkeys(k for _, info in infos for k in info))
    return dict(columns=columns, rows=[[name] + [info.get(k) for k in columns] for name, info in infos])


def problem_info(p):
    """data.json の problems.<問題> に足す項目."""
    out = {}
    if p.setting:
        out["setting"] = p.setting
    table = instance_table(p)
    if table:
        out["instance_table"] = table
    return out


def cell_text(v):
    if v is None:
        return "—"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------- HTML（考察ページ）
def setting_html(p, heading):
    s, e = p.setting, html.escape
    parts = ["  " + BEGIN, f'  <h3 id="setting">{e(heading)}</h3>', f'  <p>{e(s["summary"])}</p>', '  <dl class="setting">',
             f'    <dt>決めるもの</dt><dd>{e(s["variables"])}</dd>',
             '    <dt>制約</dt><dd><ul class="pts">' + "".join(f"<li>{e(c)}</li>" for c in s["constraints"]) + "</ul></dd>",
             f'    <dt>目的関数</dt><dd>{e(s["objective"])}</dd>']
    if s.get("note"):
        parts.append(f'    <dt>補足</dt><dd>{e(s["note"])}</dd>')
    parts += ["  </dl>", f'  <p>{e(s["instances"])}</p>']
    t = instance_table(p)
    if t:
        nums = [all(is_num(r[i + 1]) or r[i + 1] is None for r in t["rows"]) for i in range(len(t["columns"]))]
        head = "<th>インスタンス</th>" + "".join(
            f'<th class="num">{e(c)}</th>' if n else f"<th>{e(c)}</th>" for c, n in zip(t["columns"], nums))
        rows = ["<tr><td><code>" + e(r[0]) + "</code></td>" + "".join(
            f'<td class="num">{e(cell_text(v))}</td>' if n else f"<td>{e(cell_text(v))}</td>"
            for v, n in zip(r[1:], nums)) + "</tr>" for r in t["rows"]]
        parts += ['  <div class="frame">', "  <table>", f"    <thead><tr>{head}</tr></thead>", "    <tbody>"]
        parts += [f"      {r}" for r in rows]
        parts += ["    </tbody>", "  </table>", "  </div>"]
    parts.append("  " + END)
    return "\n".join(parts)


def portal_html(probs):
    e = html.escape
    rows = []
    for p in probs:
        link = f"discussion/{p.discussion}#setting"
        rows.append(f'      <tr><td>{e(short_label(p))}</td><td>{e(p.setting["brief"])}</td><td>{e(p.setting["scale"])}</td>'
                    f'<td><a href="{link}" data-web="{DISCUSSION_WEB}">考察</a></td></tr>')
    return "\n".join([
        "  " + PORTAL_BEGIN,
        '  <div class="frame">', "  <table>",
        "    <thead><tr><th>問題</th><th>何を決めて何を最小にするか</th><th>インスタンス</th><th>詳しい説明</th></tr></thead>",
        "    <tbody>", *rows, "    </tbody>", "  </table>", "  </div>",
        "  " + PORTAL_END])


def replace_block(path, begin, end, block, anchor=None):
    """path の begin〜end の間を block に置き換える. 目印が無ければ anchor（正規表現）の直前に入れる."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    pat = re.compile(r"[ \t]*" + re.escape(begin) + r".*?" + re.escape(end), re.S)
    if pat.search(text):
        new = pat.sub(lambda m: block, text, count=1)
    elif anchor:
        m = re.search(anchor, text)
        if not m:
            print(f"警告: {os.path.relpath(path, ROOT)} に問題設定を入れる場所が見つかりません", flush=True)
            return
        new = text[:m.start()] + block + "\n\n" + text[m.start():]
    else:
        return
    if new != text:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)


def write_pages():
    probs = [p for p in registry.problems() if p.setting]
    for p in probs:
        if not p.discussion:
            continue
        path = os.path.join(ROOT, "docs", "discussion", p.discussion)
        if not os.path.exists(path):
            continue
        # 1 つの考察ページで複数の問題を扱うときは見出しに問題名を入れる（例: 量子・疑似量子アニーリングの章の TSP 小規模）
        own = os.path.splitext(p.discussion)[0] == p.name
        heading = "問題設定とインスタンス" if own else f"問題設定とインスタンス（{short_label(p)}）"
        replace_block(path, BEGIN, END, setting_html(p, heading), anchor=r'[ \t]*<h3 id="methods">')
    portal = os.path.join(ROOT, "docs", "index.html")
    if os.path.exists(portal):
        replace_block(portal, PORTAL_BEGIN, PORTAL_END, portal_html(probs))


# ---------------------------------------------------------------- Markdown（results/README.md）
def md_lines(P):
    """data.json の問題の項目から、results/README.md の「問題設定」の節を作る."""
    s, t = P.get("setting"), P.get("instance_table")
    if not s:
        return []
    lines = ["### 問題設定", "", s["summary"], "",
             f'- **決めるもの**: {s["variables"]}',
             "- **制約**:", *[f"  - {c}" for c in s["constraints"]],
             f'- **目的関数**: {s["objective"]}']
    if s.get("note"):
        lines.append(f'- **補足**: {s["note"]}')
    lines += ["", "### インスタンス", "", s["instances"], ""]
    if t:
        nums = [all(is_num(r[i + 1]) or r[i + 1] is None for r in t["rows"]) for i in range(len(t["columns"]))]
        lines += ["| インスタンス | " + " | ".join(t["columns"]) + " |",
                  "|---|" + "|".join("---:" if n else "---" for n in nums) + "|"]
        lines += ["| " + " | ".join([r[0]] + [cell_text(v).replace("|", "\\|") for v in r[1:]]) + " |" for r in t["rows"]]
        lines.append("")
    return lines
