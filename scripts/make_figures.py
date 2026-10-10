"""README Figure 1 and Table 1, regenerated from the pipeline outputs.

Figure 1 (docs/figure1.png): worked example (pitch at loss / +3 s / +6 s with the team's own
reference block, and its D(t) trace) next to the confirmed result H3 (which shape component is
still wrong 6 s after a loss, league-wide, with 95% match-bootstrap intervals).
Table 1 (docs/table1.md): the four confirmatory tests plus the key secondary result.

Needs outputs/confirmatory/*.csv from scripts/confirmatory.py.
Usage: python scripts/make_figures.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import reorg
from reorg import viz
from reorg.profiles import LABELS
from reorg.reorganisation import episodes_for_match

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (2007721, 18757)  # worked example reviewed at GATE 1 (Auckland FC, slow_1)
SNAP_S = (0.0, 3.0, 6.0)


def figure1(cfg: dict, conf: Path, path: Path) -> None:
    from matplotlib.ticker import PercentFormatter

    plt = viz.plt
    res = reorg.run_league(cfg, match_ids=None)
    md = res.match(EXAMPLE[0])
    e, tr = episodes_for_match(md, res.reference, res.tau, res.params, keep_traces=True)
    row = e[e["frame_loss"] == EXAMPLE[1]].iloc[0]
    t = tr[f"{EXAMPLE[0]}_{EXAMPLE[1]}"]
    v = viz.EpisodeView(md, res.reference, int(row["losing_team_id"]), EXAMPLE[1])
    team = res.episodes.loc[res.episodes["losing_team_id"] == row["losing_team_id"], "team"].iloc[0]
    fps = res.params.fps

    fig = plt.figure(figsize=(15, 8.2))
    top = fig.add_gridspec(1, 3, left=0.04, right=0.985, top=0.85, bottom=0.50, wspace=0.08)
    bot = fig.add_gridspec(
        1, 2, left=0.06, right=0.97, top=0.40, bottom=0.10, wspace=0.58, width_ratios=[1.0, 0.95]
    )
    for j, o in enumerate(SNAP_S):
        ax = fig.add_subplot(top[0, j])
        viz.draw_pitch(ax, v.L, v.W)
        k = int(round(o * fps))
        lab = ("moment of loss" if o == 0 else f"+{o:g} s") + f"    D = {t.D[k]:.2f}"
        viz._draw_state(ax, v, v.at(k), lab)
    fig.legend(
        handles=viz.legend_handles(),
        loc="upper right",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.985, 0.93),
        labelcolor=viz.INK_2,
    )
    fig.suptitle(
        "Time to Reorganise: how far a team is from its own defensive block after losing the ball",
        x=0.05,
        ha="left",
        fontsize=15,
        color=viz.INK,
        y=0.985,
    )
    fig.text(
        0.05,
        0.925,
        f"A. {team} lose the ball high up; back in shape after {row['T_r']:.1f} s",
        fontsize=12,
        color=viz.INK,
    )

    ax = fig.add_subplot(bot[0, 0])
    tt = np.arange(len(t.D)) / fps
    ax.plot(tt, t.D, color=viz.DEF, lw=2.5)
    ax.axhline(t.tau, color=viz.INK_2, lw=1, ls="--")
    ax.text(
        0.01,
        t.tau,
        "τ: the team's own 'organised' level",
        va="bottom",
        ha="left",
        color=viz.INK_2,
        transform=ax.get_yaxis_transform(),
        fontsize=10,
    )
    ax.axvline(row["T_r"], color=viz.DEF, lw=1, ls=":")
    ax.text(
        row["T_r"],
        0.98,
        f"  T_r = {row['T_r']:.1f} s",
        transform=ax.get_xaxis_transform(),
        va="top",
        color=viz.INK_2,
        fontsize=10,
    )
    for o in SNAP_S:
        ax.axvline(o, color=viz.GRID, lw=1, zorder=0)
    ax.set_xlabel("seconds after losing the ball")
    ax.set_ylabel("D(t): distance from own block")
    ax.set_ylim(0, None)
    ax.grid(axis="y", color=viz.GRID, lw=0.6)
    ax.set_title(
        "B. The same episode as one number per tenth of a second",
        loc="left",
        fontsize=12,
        color=viz.INK,
    )

    h3 = pd.read_csv(conf / "h3.csv").sort_values("share")
    ax = fig.add_subplot(bot[0, 1])
    y = np.arange(len(h3))
    colors = [viz.DEF if i == len(h3) - 1 else viz.REF for i in range(len(h3))]
    ax.barh(y, h3["share"], color=colors, height=0.6)
    ax.errorbar(
        h3["share"],
        y,
        xerr=[h3["share"] - h3["lo"], h3["hi"] - h3["share"]],
        fmt="none",
        ecolor=viz.INK_2,
        elinewidth=1.2,
        capsize=3,
    )
    for i, r in enumerate(h3.itertuples(index=False)):
        ax.text(r.hi + 0.008, i, f"{r.share:.0%}", va="center", fontsize=10, color=viz.INK_2)
    ax.set_yticks(y, [LABELS[c] for c in h3["component"]])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, h3["hi"].max() + 0.07)
    ax.set_xlabel("share of the remaining disorganisation 6 s after the loss")
    ax.grid(axis="x", color=viz.GRID, lw=0.6)
    ax.set_title(
        "C. League-wide, what is still wrong 6 s later (95% CI)",
        loc="left",
        fontsize=12,
        color=viz.INK,
    )
    fig.text(
        0.04,
        0.012,
        "Data: SkillCorner open data, A-League 2024/25 (20 games).",
        ha="left",
        fontsize=9,
        color=viz.INK_2,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def table1(conf: Path, path: Path) -> None:
    h2 = pd.read_csv(conf / "h2.csv")
    h3 = pd.read_csv(conf / "h3.csv")
    h4 = pd.read_csv(conf / "h4.csv")
    holm = pd.read_csv(conf / "holm.csv").set_index("test")
    gen = (ROOT / "docs" / "confirmatory_generated.md").read_text()
    h1_line = next(ln for ln in gen.splitlines() if ln.startswith("- Odd/even split-half"))
    sb = float(h1_line.split("Spearman-Brown = ")[1].split(" ")[0])
    p4 = h4[(h4["regime"] == "all") & (h4["role"] == "primary")].iloc[0]
    ph = h4[(h4["regime"] == "all") & (h4["comparison"] == "M_D vs M_phase")].iloc[0]
    top = h3.iloc[0]

    def dec(k: str) -> str:
        return "**confirmed**" if bool(holm.loc[k, "reject"]) else "not confirmed"

    rows = [
        (
            "H3 Which component is still wrong at +6 s?",
            f"{LABELS[top['component']]}: {top['share']:.0%} [{top['lo']:.0%}, {top['hi']:.0%}] "
            f"of remaining disorganisation",
            f"{holm.loc['H3', 'p_holm']:.3f}",
            dec("H3"),
        ),
        (
            "H2 Still out of shape at 3 s → danger in next 12 s?",
            f"OR {h2.iloc[0]['or']:.2f} [{h2.iloc[0]['lo']:.2f}, {h2.iloc[0]['hi']:.2f}]",
            f"{holm.loc['H2', 'p_holm']:.3f}",
            dec("H2"),
        ),
        (
            "H1 Do teams differ reliably?",
            f"split-half reliability {sb:.2f} (bar: 0.5)",
            f"{holm.loc['H1', 'p_holm']:.3f}",
            dec("H1"),
        ),
        (
            "H4 Better prediction than context alone?",
            f"ΔAUC {p4['dauc']:+.3f} [{p4['dauc_lo']:+.3f}, {p4['dauc_hi']:+.3f}]; "
            f"vs SkillCorner phase label ΔAUC {ph['dauc']:+.3f}",
            f"{holm.loc['H4', 'p_holm']:.3f}",
            dec("H4"),
        ),
        (
            "Secondary: still out of shape at 6 s → danger",
            f"OR {h2.iloc[2]['or']:.2f} [{h2.iloc[2]['lo']:.2f}, {h2.iloc[2]['hi']:.2f}]",
            "n/a",
            "secondary, may be partly reverse",
        ),
    ]
    lines = [
        "<!-- generated by scripts/make_figures.py; do not edit by hand -->",
        "",
        "| Question | Estimate [95% CI] | Holm p | Result |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    a = ap.parse_args()
    cfg = reorg.load_config(a.config)
    conf = ROOT / "outputs" / "confirmatory"
    figure1(cfg, conf, ROOT / "docs" / "figure1.png")
    table1(conf, ROOT / "docs" / "table1.md")
    print((ROOT / "docs" / "table1.md").read_text())


if __name__ == "__main__":
    main()
