"""Coach-facing team profiles: "what goes wrong after we lose the ball" (descriptive).

Per team: how often it loses the ball while already organised, how fast it reorganises compared
with the league (cumulative incidence with a match-bootstrap band), and which shape components
are still late 6 s after the loss (share of D^2, team vs league). These are descriptive
summaries; with 1-7 games per team, team rankings are not yet reliable (H1, see
docs/confirmatory_generated.md), and every card says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reorg.shape import COMPONENTS
from reorg.survival import cumulative_incidence

LABELS = {
    "goal_side": "players behind the ball",
    "cx_ball": "block ahead of / behind the ball",
    "depth": "block too long / too short",
    "line_height": "back line too high / too deep",
    "width": "block too wide / too narrow",
    "nn_median": "spacing between players",
    "cy_ball": "sideways shift to the ball",
}


def _boot_cif(df: pd.DataFrame, grid: np.ndarray, n_boot: int, seed: int) -> pd.DataFrame:
    """CIF of reorganisation with a match-cluster percentile band."""
    est = cumulative_incidence(df["time_s"], df["code"], grid)
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("match_id")]
    boots = []
    for _ in range(n_boot):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots.append(cumulative_incidence(s["time_s"], s["code"], grid)["cif_1"].to_numpy())
    b = np.vstack(boots)
    est["lo"], est["hi"] = np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)
    return est


def late_shares(df: pd.DataFrame, horizon_s: float) -> pd.Series:
    """Mean share of D^2 per component among episodes still late (D > tau) at the horizon."""
    h = f"{horizon_s:g}s"
    late = df[np.isfinite(df[f"D_{h}"]) & (df[f"D_{h}"] > df["tau"])]
    s = late[[f"share_{c}_{h}" for c in COMPONENTS]].mean()
    s.index = list(COMPONENTS)
    s.attrs["n_late"] = len(late)
    return s


def boot_late_shares(
    df: pd.DataFrame, horizon_s: float, n_boot: int, seed: int
) -> tuple[pd.Series, pd.Series]:
    """Match-cluster bootstrap 95% band for :func:`late_shares`."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("match_id")]
    b = np.vstack(
        [
            late_shares(
                pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]), horizon_s
            ).to_numpy()
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
    return pd.Series(lo, index=list(COMPONENTS)), pd.Series(hi, index=list(COMPONENTS))


def team_profile(
    episodes: pd.DataFrame, team: str, horizon_s: float, grid: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Everything a team card shows. ``episodes`` = all episodes of the league (with ``team``)."""
    t = episodes[episodes["team"] == team]
    dis_t = t[t["disorganised_at_loss"]]
    dis_l = episodes[episodes["disorganised_at_loss"]]
    cif_t = _boot_cif(dis_t, grid, n_boot, seed)
    cif_l = _boot_cif(dis_l, grid, n_boot, seed)
    return {
        "team": team,
        "matches": int(t["match_id"].nunique()),
        "losses": len(t),
        "share_organised_at_loss": float(1 - t["disorganised_at_loss"].mean()),
        "league_share_organised_at_loss": float(1 - episodes["disorganised_at_loss"].mean()),
        "disorganised_losses": len(dis_t),
        "cif_team": cif_t,
        "cif_league": cif_l,
        "shares_team": late_shares(dis_t, horizon_s),
        "shares_team_ci": boot_late_shares(dis_t, horizon_s, n_boot, seed),
        "shares_league": late_shares(dis_l, horizon_s),
        "horizon_s": horizon_s,
    }


def plot_team_card(p: dict, path=None, mark_s: float = 6.0):
    """One-page card: reorganisation curve vs league, and what is still late at +6 s."""
    from reorg import viz  # applies the shared style

    plt = viz.plt
    from matplotlib.ticker import PercentFormatter

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1, 1.1], wspace=0.62, left=0.06, right=0.98, top=0.74, bottom=0.12
    )
    ct, cl = p["cif_team"], p["cif_league"]
    at = float(np.interp(mark_s, ct["t"], ct["cif_1"]))
    at_lo = float(np.interp(mark_s, ct["t"], ct["lo"]))
    at_hi = float(np.interp(mark_s, ct["t"], ct["hi"]))
    # A match-level bootstrap needs at least two matches; with one, no interval is shown.
    has_ci = p["matches"] >= 2
    ci_txt = f" [{at_lo:.0%}, {at_hi:.0%}]" if has_ci else " (1 game: no interval)"
    games = "game" if p["matches"] == 1 else "games"
    al = float(np.interp(mark_s, cl["t"], cl["cif_1"]))
    fig.suptitle(
        f"{p['team']}: what happens after losing the ball",
        x=0.06,
        ha="left",
        fontsize=14,
        color=viz.INK,
        y=0.98,
    )
    fig.text(
        0.06,
        0.895,
        f"{p['losses']} losses in {p['matches']} {games}. Already organised at the moment of "
        f"loss: {p['share_organised_at_loss']:.0%} "
        f"(league {p['league_share_organised_at_loss']:.0%}). "
        f"Back in shape within {mark_s:g} s: {at:.0%}{ci_txt} "
        f"(league {al:.0%}).",
        fontsize=10,
        color=viz.INK_2,
    )
    fig.text(
        0.06,
        0.85,
        "Descriptive. With 1-7 games per team, team rankings are not yet "
        "reliable (confirmatory test H1).",
        fontsize=9,
        color=viz.INK_2,
        style="italic",
    )

    ax = fig.add_subplot(gs[0])
    ax.fill_between(cl["t"], cl["lo"], cl["hi"], step="post", color=viz.REF, alpha=0.18, lw=0)
    ax.step(cl["t"], cl["cif_1"], where="post", color=viz.REF, lw=2, label="league")
    if has_ci:
        ax.fill_between(ct["t"], ct["lo"], ct["hi"], step="post", color=viz.DEF, alpha=0.15, lw=0)
    ax.step(ct["t"], ct["cif_1"], where="post", color=viz.DEF, lw=2, label=p["team"])
    ax.axvline(mark_s, color=viz.GRID, lw=1, zorder=0)
    ax.set_xlabel("seconds after losing the ball")
    ax.set_ylabel("share back in shape")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=viz.GRID, lw=0.6)
    ax.legend(frameon=False, loc="upper left", labelcolor=viz.INK)
    ax.set_title(
        "How fast does the team get back into shape?", loc="left", fontsize=11, color=viz.INK
    )

    ax = fig.add_subplot(gs[1])
    st, sl = p["shares_team"], p["shares_league"]
    lo, hi = p["shares_team_ci"]
    order = sl.sort_values().index
    y = np.arange(len(order))
    if has_ci:
        ax.hlines(
            y,
            lo[order],
            hi[order],
            color=viz.DEF,
            lw=2,
            alpha=0.35,
            zorder=1,
            label=f"{p['team']} 95% band",
        )
    ax.scatter(sl[order], y, s=60, color=viz.REF, zorder=2, label="league")
    ax.scatter(st[order], y, s=60, color=viz.DEF, ec=viz.SURFACE, lw=1, zorder=3, label=p["team"])
    for i, c in enumerate(order):  # label sits on the team's own dot, never on the league's
        ax.text(
            st[c], i + 0.2, f"{st[c]:.0%}", ha="center", va="bottom", fontsize=8.5, color=viz.INK_2
        )
    ax.set_yticks(y, [LABELS[c] for c in order])
    ax.set_xlabel(f"share of the remaining disorganisation at +{p['horizon_s']:g} s")
    ax.set_xlim(0, max(hi.max(), sl.max(), st.max()) + 0.06)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=viz.GRID, lw=0.6)
    ax.legend(frameon=False, loc="lower right", labelcolor=viz.INK)
    ax.set_title(
        f"What is still wrong {p['horizon_s']:g} s later? ({st.attrs['n_late']} late episodes)",
        loc="left",
        fontsize=11,
        color=viz.INK,
    )
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig
