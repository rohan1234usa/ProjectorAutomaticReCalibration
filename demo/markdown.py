"""The small part of Markdown the project's documents use, turned into HTML for the demo pages.

docs/findings.md and the CLAUDE.md tables use headings, paragraphs, bold and italic, code spans,
links, tables, and bulleted or numbered lists nested by indentation, whose items continue on
indented lines (also after a blank line). That is all this handles; there is no dependency to
install offline. Text is escaped before any markup is added, so a stray ``<`` in a document stays
text.
"""

from __future__ import annotations

import html
import re

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_EM = re.compile(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])")


def slug(text: str) -> str:
    """A heading's anchor: lower case, words joined by hyphens."""
    words = re.findall(r"[a-z0-9]+", re.sub(r"<[^>]+>", "", text).lower())
    return "-".join(words)[:80] or "section"


def _href(url: str) -> str | None:
    """A link target made safe for an attribute: web, mail and relative links only."""
    url = html.unescape(url)  # the text around it was escaped already: undo that once
    scheme = re.match(r"([A-Za-z][A-Za-z0-9+.-]*):", url)
    if scheme and scheme.group(1).lower() not in ("http", "https", "mailto"):
        return None  # e.g. javascript: never becomes a link
    return html.escape(url, quote=True)


def _link(m: re.Match) -> str:
    href = _href(m.group(2))
    return f'<a href="{href}">{m.group(1)}</a>' if href else m.group(1)


def inline(text: str) -> str:
    """Inline markup: `code`, **bold**, *italic*, [links](url); everything else escaped.

    Code spans are set aside first and put back last, so bold or italic around a code span
    still pairs up, and nothing inside a code span is read as markup.
    """
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(f"<code>{html.escape(m.group(1), quote=False)}</code>")
        return f"\x00{len(spans) - 1}\x00"

    s = html.escape(re.sub(r"`([^`]+)`", stash, text), quote=False)
    s = _LINK.sub(_link, s)
    s = _BOLD.sub(r"<strong>\1</strong>", s)
    s = _EM.sub(r"<em>\1</em>", s)
    return re.sub("\x00(\\d+)\x00", lambda m: spans[int(m.group(1))], s)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _starts_block(line: str) -> bool:
    s = line.lstrip()
    return bool(_HEADING.match(s) or _ITEM.match(line) or s.startswith("|") or s.startswith("```"))


def table_rows(lines: list[str]) -> list[list[str]]:
    """Cells of each table row, the separator row dropped (raw Markdown, not yet converted)."""
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def _table(lines: list[str]) -> str:
    rows = table_rows(lines)
    head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>" for row in rows[1:])
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _list(lines: list[str], i: int) -> tuple[str, int]:
    first = _ITEM.match(lines[i])
    base, ordered = len(first.group(1)), first.group(2)[0].isdigit()
    items = []
    while i < len(lines):
        m = _ITEM.match(lines[i])
        if not m or len(m.group(1)) != base or m.group(2)[0].isdigit() != ordered:
            break
        content = base + len(m.group(2)) + 1
        body, i = [m.group(3)], i + 1
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                j = i
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and _indent(lines[j]) > base:
                    body += [""] * (j - i)  # the item goes on after a blank line
                    i = j
                    continue
                nxt = _ITEM.match(lines[j]) if j < len(lines) else None
                if nxt and len(nxt.group(1)) == base and nxt.group(2)[0].isdigit() == ordered:
                    i = j  # a loose list: the next item follows a blank line
                break
            if _indent(line) <= base:
                break
            body.append(line[min(_indent(line), content):])
            i += 1
        items.append(_item(body))
    tag = "ol" if ordered else "ul"
    return f"<{tag}>" + "".join(f"<li>{item}</li>" for item in items) + f"</{tag}>", i


def _item(body: list[str]) -> str:
    """A list item: a lone first paragraph is not wrapped in <p> (a tight list)."""
    parts = _blocks(body)
    if parts and parts[0][0] == "p" and sum(kind == "p" for kind, _ in parts) == 1:
        parts[0] = ("text", parts[0][1][3:-4])
    return "".join(h for _, h in parts)


def _blocks(lines: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
        elif m := _HEADING.match(stripped):
            level, text = len(m.group(1)), m.group(2)
            out.append(("h", f'<h{level} id="{slug(text)}">{inline(text)}</h{level}>'))
            i += 1
        elif stripped.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            code = html.escape("\n".join(lines[i + 1 : j]), quote=False)
            out.append(("pre", f"<pre><code>{code}</code></pre>"))
            i = j + 1
        elif stripped.startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            out.append(("table", _table(lines[i:j])))
            i = j
        elif _ITEM.match(line):
            text, i = _list(lines, i)
            out.append(("list", text))
        else:
            para = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not _starts_block(lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(("p", f"<p>{inline(' '.join(para))}</p>"))
    return out


def to_html(text: str) -> str:
    """Convert a Markdown document (the subset above) to HTML."""
    return "\n".join(h for _, h in _blocks(text.splitlines()))
