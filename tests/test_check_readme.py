import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_readme", Path(__file__).parents[1] / "scripts" / "check_readme.py"
)
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def test_counts_words_figures_tables():
    text = "# Title\n\nOne two three.\n\n![fig](a.png)\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    words, images, tables = cr.count(text)
    assert images == 1 and tables == 1
    assert words == 1 + 3 + 1 + 4  # title, sentence, alt text, table cells


def test_repo_readme_within_limits():
    assert cr.main(str(Path(__file__).parents[1] / "README.md")) == 0
