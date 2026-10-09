"""Enforce the competition README limits: <= 1000 words and <= 2 figures/tables in total.

Words are counted over the whole file (markdown syntax stripped from links/images; code blocks
included, to stay conservative). Figures = markdown/HTML images; tables = markdown tables.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_WORDS = 1000
MAX_FIGS_TABLES = 2


def count(text: str) -> tuple[int, int, int]:
    images = len(re.findall(r"!\[[^\]]*\]\([^)]*\)", text)) + len(re.findall(r"<img\b", text))
    lines = text.splitlines()
    tables = sum(
        1
        for i in range(1, len(lines))
        if re.match(r"^\s*\|?\s*:?-{3,}", lines[i]) and "|" in lines[i] and "|" in lines[i - 1]
    )
    plain = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    plain = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", plain)
    plain = re.sub(r"^\s*\|?\s*:?-{3,}.*$", "", plain, flags=re.M)
    words = len(re.findall(r"[A-Za-z0-9][\w'’.\-/]*", plain))
    return words, images, tables


def main(path: str = "README.md") -> int:
    words, images, tables = count(Path(path).read_text())
    ok = words <= MAX_WORDS and images + tables <= MAX_FIGS_TABLES
    print(
        f"{path}: {words} words (max {MAX_WORDS}), {images} figures + {tables} tables "
        f"(max {MAX_FIGS_TABLES}) -> {'OK' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
