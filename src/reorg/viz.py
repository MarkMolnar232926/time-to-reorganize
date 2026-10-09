"""Figures and animations (matplotlib only).

Pitch views are drawn in the defending (losing) team's own frame: it defends the goal on the
left (x = -L/2). Defenders are filled when detected and hollow when extrapolated, so tracking
reliability is visible in every frame. The reference block is drawn as the team's median
organised block in the current ball context: back line at its median line height, extending
by its median depth, centred laterally at the ball plus its median lateral offset, with its
median width. It is a summary envelope, not a set of player positions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from reorg.pipeline import MatchData  # noqa: E402
from reorg.reference import Reference  # noqa: E402
from reorg.shape import hull_polygon  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots 1-3, ink and surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"
DEF = "#2a78d6"  # slot 1: defending team / reorganised
ATT = "#eb6834"  # slot 2: attacking team / regain
AQUA = "#1baf7a"  # slot 3: censored
REF = "#8a8984"  # neutral: reference block
PITCH_LINE = "#b9b8b3"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


def draw_pitch(ax, L: float, W: float) -> None:
    kw = {"color": PITCH_LINE, "lw": 1.0, "zorder": 0}
    ax.plot([-L / 2, L / 2, L / 2, -L / 2, -L / 2], [-W / 2, -W / 2, W / 2, W / 2, -W / 2], **kw)
    ax.plot([0, 0], [-W / 2, W / 2], **kw)
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(9.15 * np.cos(t), 9.15 * np.sin(t), **kw)
    for s in (-1, 1):
        x0 = s * L / 2
        ax.plot([x0, x0 - s * 16.5, x0 - s * 16.5, x0], [-20.16, -20.16, 20.16, 20.16], **kw)
        ax.plot([x0, x0 - s * 5.5, x0 - s * 5.5, x0], [-9.16, -9.16, 9.16, 9.16], **kw)
    ax.set_xlim(-L / 2 - 2, L / 2 + 2)
    ax.set_ylim(-W / 2 - 2, W / 2 + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def reference_box(ref: Reference, team_id: int, ball_x: float, ball_y: float, L: float, W: float):
    """(x0, y0, depth, width) of the (ball-interpolated) reference envelope; None if unknown."""
    if not (np.isfinite(ball_x) and np.isfinite(ball_y)):
        return None
    C, _ = ref.lookup_continuous(team_id, np.array([ball_x]), np.array([ball_y]), L, W)
    c = dict(zip(ref.components, C[0], strict=True))
    x0 = -L / 2 + c["line_height"]
    yc = ball_y + c["cy_ball"]
    return x0, yc - c["width"] / 2, c["depth"], c["width"]


class EpisodeView:
    """Positions for one episode in the defending team's frame."""

    def __init__(self, md: MatchData, ref: Reference, team_id: int, frame_loss: int):
        self.md, self.ref, self.team_id = md, ref, team_id
        self.td = md.teams[team_id]
        opp = next(t for t in md.teams if t != team_id)
        self.op = md.teams[opp]
        self.i0 = int(frame_loss - md.frame[0])
        self.L, self.W = self.td.tf.pitch_length, self.td.tf.pitch_width

    def at(self, k: int) -> dict:
        i = self.i0 + k
        tf = self.td.tf
        return {
            "X": tf.X[i],
            "Y": tf.Y[i],
            "det": tf.det[i],
            "OX": -self.op.tf.X[i],
            "OY": -self.op.tf.Y[i],
            "Odet": self.op.tf.det[i],
            "bx": tf.ball_x[i],
            "by": tf.ball_y[i],
            "box": reference_box(
                self.ref, self.team_id, tf.ball_x[i], tf.ball_y[i], self.L, self.W
            ),
        }


def _draw_state(ax, v: EpisodeView, s: dict, title: str | None = None) -> list:
    arts = []
    if s["box"] is not None:
        x0, y0, d, w = s["box"]
        arts.append(
            ax.add_patch(
                Rectangle((x0, y0), d, w, fc=REF, alpha=0.15, ec=REF, lw=1.5, ls="--", zorder=1)
            )
        )
    if np.isfinite(s["X"]).all():
        hp = hull_polygon(s["X"], s["Y"])
        arts.append(
            ax.add_patch(Polygon(hp, closed=True, fc=DEF, alpha=0.12, ec=DEF, lw=1.5, zorder=2))
        )
        det = s["det"]
        arts.append(ax.scatter(s["X"][det], s["Y"][det], s=55, c=DEF, ec=SURFACE, lw=1.2, zorder=4))
        arts.append(
            ax.scatter(
                s["X"][~det], s["Y"][~det], s=55, facecolors="none", ec=DEF, lw=1.6, zorder=4
            )
        )
    if np.isfinite(s["OX"]).all():
        od = s["Odet"]
        arts.append(
            ax.scatter(
                s["OX"][od], s["OY"][od], s=40, c=ATT, ec=SURFACE, lw=1.0, zorder=3, marker="D"
            )
        )
        arts.append(
            ax.scatter(
                s["OX"][~od],
                s["OY"][~od],
                s=40,
                facecolors="none",
                ec=ATT,
                lw=1.4,
                zorder=3,
                marker="D",
            )
        )
    if np.isfinite(s["bx"]):
        arts.append(ax.scatter([s["bx"]], [s["by"]], s=30, c=INK, zorder=5))
    if title:
        arts.append(ax.set_title(title, fontsize=11, color=INK, loc="left"))
    return arts


def plot_snapshots(
    v: EpisodeView,
    offsets_s=(0.0, 3.0, 6.0),
    fps: float = 10.0,
    D=None,
    tau=None,
    path=None,
    suptitle: str | None = None,
):
    """Pitch snapshots at loss and later offsets, optionally with the D(t) trace beneath."""
    n = len(offsets_s)
    has_d = D is not None
    fig = plt.figure(figsize=(4.6 * n, 5.4 if has_d else 3.6))
    gs = fig.add_gridspec(
        2 if has_d else 1,
        n,
        height_ratios=[3.1, 1.6][: 2 if has_d else 1],
        hspace=0.25,
        wspace=0.04,
        left=0.05,
        right=0.99,
        top=0.86,
        bottom=0.1,
    )
    for j, o in enumerate(offsets_s):
        ax = fig.add_subplot(gs[0, j])
        draw_pitch(ax, v.L, v.W)
        k = int(round(o * fps))
        lab = "loss" if o == 0 else f"+{o:g} s"
        if has_d and k < len(D) and np.isfinite(D[k]):
            lab += f"   D = {D[k]:.2f}"
        _draw_state(ax, v, v.at(k), lab)
    fig.legend(
        handles=legend_handles(),
        loc="upper right",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.99, 0.995),
        labelcolor=INK_2,
    )
    if D is not None:
        ax = fig.add_subplot(gs[1, :])
        _plot_trace(ax, D, tau, fps, offsets_s)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, color=INK)
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def legend_handles() -> list:
    from matplotlib.lines import Line2D

    return [
        Line2D([], [], ls="", marker="o", mfc=DEF, mec=DEF, label="defender (detected)"),
        Line2D([], [], ls="", marker="o", mfc="none", mec=DEF, label="defender (extrapolated)"),
        Line2D([], [], ls="", marker="D", mfc=ATT, mec=ATT, label="attacker"),
        Line2D([], [], ls="", marker="o", mfc=INK, mec=INK, ms=5, label="ball"),
        Rectangle((0, 0), 1, 1, fc=REF, alpha=0.3, ec=REF, ls="--", label="own reference block"),
    ]


def _plot_trace(ax, D, tau, fps, marks=()):
    t = np.arange(len(D)) / fps
    ax.plot(t, D, color=DEF, lw=2)
    if tau is not None:
        ax.axhline(tau, color=INK_2, lw=1, ls="--")
        ax.text(
            0.995,
            tau,
            "τ (organised)",
            va="bottom",
            ha="right",
            color=INK_2,
            transform=ax.get_yaxis_transform(),
        )
    for m in marks:
        ax.axvline(m, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("seconds after loss")
    ax.set_ylabel("D(t)")
    ax.grid(axis="y", color=GRID, lw=0.6)


def plot_d_curves(
    traces: list,
    labels: list[str],
    colors: list[str],
    tau: float | None,
    fps: float,
    path=None,
    title: str | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    for tr, lab, col in zip(traces, labels, colors, strict=True):
        t = np.arange(len(tr)) / fps
        ax.plot(t, tr, color=col, lw=2, alpha=0.9, label=lab)
    if tau is not None:
        ax.axhline(tau, color=INK_2, lw=1, ls="--")
        ax.text(
            0.995,
            tau,
            "τ (organised)",
            va="bottom",
            ha="right",
            color=INK_2,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_xlabel("seconds after loss")
    ax.set_ylabel("D(t)  (RMS deviations from own block)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, labelcolor=INK)
    if title:
        ax.set_title(title, loc="left", color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_cif(cif, path=None, title: str | None = None, label_suffix: str = ""):
    """Cumulative incidence of reorganisation and of regain (competing), with bands if present."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, col, lab in ((1, DEF, "reorganised"), (2, ATT, "regained first (competing)")):
        ax.step(cif["t"], cif[f"cif_{c}"], where="post", color=col, lw=2, label=lab + label_suffix)
        if f"cif_{c}_lo" in cif:
            ax.fill_between(
                cif["t"],
                cif[f"cif_{c}_lo"],
                cif[f"cif_{c}_hi"],
                step="post",
                color=col,
                alpha=0.15,
                lw=0,
            )
        ax.text(
            cif["t"].iloc[-1],
            cif[f"cif_{c}"].iloc[-1],
            f"  {cif[f'cif_{c}'].iloc[-1]:.2f}",
            va="center",
            color=INK_2,
        )
    ax.set_xlabel("seconds after loss")
    ax.set_ylabel("cumulative incidence")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    if title:
        ax.set_title(title, loc="left", color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def animate_episode(
    v: EpisodeView,
    D: np.ndarray,
    tau: float,
    end: int,
    path: str,
    fps: float = 10.0,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    title: str = "",
    t_r: float | None = None,
) -> None:
    """MP4 animation of an episode: pitch view above, D(t) building up below."""
    pre, post = int(pre_s * fps), int(post_s * fps)
    ks = np.arange(-pre, min(end + post, len(D) - 1) + 1)
    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], top=0.85, bottom=0.08, hspace=0.3)
    ax = fig.add_subplot(gs[0])
    axd = fig.add_subplot(gs[1])
    tt = np.arange(len(D)) / fps
    axd.set_xlim(-pre_s, ks[-1] / fps)
    finite = D[np.isfinite(D)]
    axd.set_ylim(0, max(np.nanmax(finite) * 1.1 if finite.size else 3, tau * 1.5))
    axd.axhline(tau, color=INK_2, lw=1, ls="--")
    axd.text(
        0.995,
        tau,
        "τ (organised)",
        va="bottom",
        ha="right",
        color=INK_2,
        transform=axd.get_yaxis_transform(),
    )
    axd.axvline(end / fps, color=GRID, lw=1)
    axd.text(
        end / fps,
        1.0,
        " window ends",
        transform=axd.get_xaxis_transform(),
        va="top",
        fontsize=8,
        color=INK_2,
    )
    if t_r is not None:
        axd.axvline(t_r, color=DEF, lw=1, ls=":")
        axd.text(
            t_r,
            1.0,
            "T_r ",
            transform=axd.get_xaxis_transform(),
            va="top",
            ha="right",
            fontsize=8,
            color=INK_2,
        )
    axd.set_xlabel("seconds after loss")
    axd.set_ylabel("D(t)")
    axd.grid(axis="y", color=GRID, lw=0.6)
    (line,) = axd.plot([], [], color=DEF, lw=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.965),
        labelcolor=INK_2,
    )
    holder: list = []

    def update(k):
        for a in holder:
            a.remove()
        holder.clear()
        ax.cla()
        draw_pitch(ax, v.L, v.W)
        lab = f"t = {k / fps:+.1f} s"
        if 0 <= k < len(D) and np.isfinite(D[k]):
            lab += f"    D = {D[k]:.2f}"
        _draw_state(ax, v, v.at(k), lab)
        upto = max(0, min(k, len(D) - 1))
        line.set_data(tt[: upto + 1], D[: upto + 1])
        return [line]

    ani = animation.FuncAnimation(fig, update, frames=ks, blit=False)
    ani.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=100)
    plt.close(fig)
