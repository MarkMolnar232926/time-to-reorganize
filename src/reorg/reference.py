"""Team-specific, context-conditioned reference block R(c).

For each team and ball context cell c, R holds a robust centre (median) and spread
(1.4826 x MAD) of every shape component over the team's organised-defence frames. Thin cells are
shrunk toward the league-pooled cell: ``w = n / (n + k)``, ``R = w * team + (1 - w) * league``.
Spreads get a per-component floor. Unknown teams (e.g. new data) fall back to the league row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reorg.shape import COMPONENTS

LEAGUE = -1  # team_id used for league-pooled rows
MAD_TO_SD = 1.4826


def _robust(g: pd.DataFrame, comps: tuple[str, ...]) -> pd.Series:
    vals = g[list(comps)].to_numpy()
    med = np.median(vals, axis=0)
    mad = np.median(np.abs(vals - med), axis=0) * MAD_TO_SD
    out = {f"c_{k}": m for k, m in zip(comps, med, strict=True)}
    out.update({f"s_{k}": s for k, s in zip(comps, mad, strict=True)})
    out["n"] = len(g)
    return pd.Series(out)


@dataclass
class Reference:
    """Reference table indexed by (team_id, cell) with columns c_<k>, s_<k>, n, w."""

    table: pd.DataFrame
    components: tuple[str, ...]
    n_cells: int
    n_channels: int = 3
    n_depths: int = 3

    def _grids(self, team_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Centre and spread grids of shape (n_depths, n_channels, K)."""
        K = len(self.components)
        tid = team_id if team_id in self.table.index.get_level_values(0) else LEAGUE
        sub = self.table.loc[tid].reindex(range(self.n_cells))
        cen = sub[[f"c_{k}" for k in self.components]].to_numpy(float)
        spr = sub[[f"s_{k}" for k in self.components]].to_numpy(float)
        shape = (self.n_depths, self.n_channels, K)
        return cen.reshape(shape), spr.reshape(shape)

    def lookup_continuous(
        self, team_id: int, ball_x: np.ndarray, ball_y: np.ndarray, L: float, W: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Centre and spread (n_frames, K), bilinearly interpolated in ball position between
        the centres of the context cells (constant beyond the outermost cell centres).

        Avoids discontinuities in the target when the ball crosses a cell boundary.
        """
        cen, spr = self._grids(team_id)
        # Fractional cell coordinates: 0 at the centre of the first cell.
        u = np.clip((ball_x + L / 2) / (L / self.n_depths) - 0.5, 0, self.n_depths - 1)
        v = np.clip((ball_y + W / 2) / (W / self.n_channels) - 0.5, 0, self.n_channels - 1)
        ok = np.isfinite(u) & np.isfinite(v)
        u, v = np.where(ok, u, 0), np.where(ok, v, 0)
        i0 = np.minimum(np.floor(u).astype(int), self.n_depths - 2)
        j0 = np.minimum(np.floor(v).astype(int), self.n_channels - 2)
        fu, fv = (u - i0)[:, None], (v - j0)[:, None]

        def bil(g: np.ndarray) -> np.ndarray:
            return (
                (1 - fu) * (1 - fv) * g[i0, j0]
                + fu * (1 - fv) * g[i0 + 1, j0]
                + (1 - fu) * fv * g[i0, j0 + 1]
                + fu * fv * g[i0 + 1, j0 + 1]
            )

        C, Sg = bil(cen), bil(spr)
        C[~ok], Sg[~ok] = np.nan, np.nan
        return C, Sg

    def lookup(self, team_id: int, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centre and spread arrays (n_frames, K) for a team's per-frame cells (-1 -> NaN)."""
        K = len(self.components)
        cen = np.full((self.n_cells, K), np.nan)
        spr = np.full((self.n_cells, K), np.nan)
        tid = team_id if team_id in self.table.index.get_level_values(0) else LEAGUE
        sub = self.table.loc[tid]
        for c in sub.index:
            cen[c] = sub.loc[c, [f"c_{k}" for k in self.components]].to_numpy(float)
            spr[c] = sub.loc[c, [f"s_{k}" for k in self.components]].to_numpy(float)
        ok = cell >= 0
        C = np.full((len(cell), K), np.nan)
        Sg = np.full((len(cell), K), np.nan)
        C[ok], Sg[ok] = cen[cell[ok]], spr[cell[ok]]
        return C, Sg


def organised_frames(match_data: list, components: tuple[str, ...] = COMPONENTS) -> pd.DataFrame:
    """Stack organised-defence frames of all teams in all matches: S, team_id, match_id, cell."""
    rows = []
    for md in match_data:
        for team_id, td in md.teams.items():
            m = td.organised
            df = td.S.loc[m, list(components)].copy()
            df["team_id"] = team_id
            df["match_id"] = md.meta.match_id
            df["cell"] = td.cell[m]
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def build_reference(
    org: pd.DataFrame, cfg: dict, components: tuple[str, ...] = COMPONENTS
) -> Reference:
    """Build the shrunk team x cell reference from stacked organised frames."""
    rc = cfg["reference"]
    n_cells = int(rc["n_channels"] * rc["n_depths"])
    k = float(rc["shrinkage_k_frames"])
    floor = np.array([float(rc["spread_floor"][c]) for c in components])
    league = org.groupby("cell").apply(_robust, comps=components, include_groups=False)
    team = org.groupby(["team_id", "cell"]).apply(_robust, comps=components, include_groups=False)
    cc = [f"c_{c}" for c in components]
    sc = [f"s_{c}" for c in components]
    rows = []
    for (tid, cell), r in team.iterrows():
        lg = league.loc[cell]
        w = r["n"] / (r["n"] + k)
        row = {"team_id": tid, "cell": cell, "n": r["n"], "w": w}
        row.update(dict(zip(cc, w * r[cc].to_numpy() + (1 - w) * lg[cc].to_numpy(), strict=True)))
        row.update(dict(zip(sc, w * r[sc].to_numpy() + (1 - w) * lg[sc].to_numpy(), strict=True)))
        rows.append(row)
    for tid in team.index.get_level_values(0).unique():  # cells a team never visited: league
        seen = set(team.loc[tid].index)
        for cell in league.index.difference(list(seen)):
            lg = league.loc[cell]
            rows.append(
                {
                    "team_id": tid,
                    "cell": cell,
                    "n": 0,
                    "w": 0.0,
                    **dict(zip(cc, lg[cc], strict=True)),
                    **dict(zip(sc, lg[sc], strict=True)),
                }
            )
    for cell, lg in league.iterrows():
        rows.append(
            {
                "team_id": LEAGUE,
                "cell": cell,
                "n": lg["n"],
                "w": 1.0,
                **dict(zip(cc, lg[cc], strict=True)),
                **dict(zip(sc, lg[sc], strict=True)),
            }
        )
    tab = pd.DataFrame(rows)
    tab[sc] = np.maximum(tab[sc].to_numpy(float), floor)
    tab[["team_id", "cell"]] = tab[["team_id", "cell"]].astype(int)
    return Reference(
        tab.set_index(["team_id", "cell"]).sort_index(),
        tuple(components),
        n_cells,
        int(rc["n_channels"]),
        int(rc["n_depths"]),
    )


def split_half_stability(
    org: pd.DataFrame, cfg: dict, components: tuple[str, ...] = COMPONENTS
) -> pd.DataFrame:
    """Odd/even-match split per team: per component, correlation across team-cells of the
    team-minus-league centre deviation in each half (unshrunk). Only team-cells with at least
    ``split_half_min_frames`` in both halves enter."""
    min_n = int(cfg["reference"]["split_half_min_frames"])
    org = org.copy()
    order = org.groupby("team_id")["match_id"].transform(lambda s: s.rank(method="dense"))
    org["half"] = (order % 2).astype(int)
    league = org.groupby("cell")[list(components)].median()
    halves = {}
    for h in (0, 1):
        g = org[org["half"] == h].groupby(["team_id", "cell"])
        med = g[list(components)].median()
        med = med - league.reindex(med.index.get_level_values("cell")).to_numpy()
        halves[h] = med[g.size() >= min_n]
    both = halves[0].join(halves[1], lsuffix="_a", rsuffix="_b", how="inner")
    out = []
    for c in components:
        a, b = both[f"{c}_a"], both[f"{c}_b"]
        out.append(
            {
                "component": c,
                "n_team_cells": len(both),
                "r": float(np.corrcoef(a, b)[0, 1]) if len(both) > 2 else np.nan,
            }
        )
    return pd.DataFrame(out)
