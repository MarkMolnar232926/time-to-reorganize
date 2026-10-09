"""Phase 1 prototype: references, D(t), T_r for all games + face-validity material.

No outcome (danger) variables are used here (ANALYSIS_PLAN.md is not frozen yet).
Outputs go to outputs/phase1/; a generated summary is written to docs/gate1_generated.md.

Usage: python scripts/phase1_prototype.py [--config config.yaml] [--no-anim]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reorg.io import list_match_ids
from reorg.pipeline import prepare_match
from reorg.reference import build_reference, organised_frames, split_half_stability
from reorg.reliability import regime_episodes, reliability_report
from reorg.reorganisation import ReorgParams, episodes_for_match, team_taus
from reorg.survival import bootstrap_cif, cumulative_incidence
from reorg.viz import (
    ATT,
    DEF,
    EpisodeView,
    animate_episode,
    plot_cif,
    plot_d_curves,
    plot_snapshots,
)

ROOT = Path(__file__).resolve().parents[1]
GRID_STEP_S = 0.1  # evaluation grid for cumulative incidence curves
REPORT_HORIZONS_S = (3.0, 6.0, 10.0, 15.0)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    df = df.round(digits)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
        for r in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--no-anim", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = ROOT / "outputs" / "phase1"
    (out / "anim").mkdir(parents=True, exist_ok=True)
    root, cache = str(ROOT / cfg["data"]["root"]), str(ROOT / cfg["data"]["cache"])
    P = ReorgParams.from_config(cfg)
    pc = cfg["prototype"]
    rng = np.random.default_rng(pc["seed"])

    mds = [prepare_match(root, m, cfg, cache) for m in list_match_ids(root)]
    names = {}
    for md in mds:
        names[md.meta.home_team_id] = md.meta.home_team_name
        names[md.meta.away_team_id] = md.meta.away_team_name
    org = organised_frames(mds)
    ref = build_reference(org, cfg)
    ref.table.to_csv(out / "reference.csv")
    sh = split_half_stability(org, cfg)
    taus = team_taus(mds, ref, P)

    eps_all, eps_det, traces = [], [], {}
    thr = float(cfg["reliability"]["frame_min_detected"])
    for md in mds:
        e, tr = episodes_for_match(md, ref, taus, P, keep_traces=md.meta.match_id == pc["match_id"])
        eps_all.append(e)
        traces.update(tr)
        eps_det.append(episodes_for_match(md, ref, taus, P, min_frame_det=thr)[0])
    eps = pd.concat(eps_all, ignore_index=True)
    epd = pd.concat(eps_det, ignore_index=True)
    eps["team"] = eps["losing_team_id"].map(names)
    eps.to_csv(out / "episodes.csv", index=False)

    # Funnel.
    tracked = eps[np.isfinite(eps["D0"])]
    needs = tracked[tracked["D0"] > tracked["tau"]]
    funnel = pd.DataFrame(
        [
            ("primary losses (D-004)", len(eps)),
            ("tracked at loss (D0 defined)", len(tracked)),
            ("already organised at loss (D0 <= tau)", int((tracked["D0"] <= tracked["tau"]).sum())),
            ("disorganised at loss (D0 > tau)", len(needs)),
            ("  -> reorganised in window", int((needs["code"] == 1).sum())),
            ("  -> regained first (competing)", int((needs["code"] == 2).sum())),
            (
                "  -> censored (dead ball / shot / cap / period end)",
                int((needs["code"] == 0).sum()),
            ),
        ],
        columns=["step", "episodes"],
    )

    grid = np.round(np.arange(0, P.horizon_s + 1e-9, GRID_STEP_S), 1)
    cif_all = bootstrap_cif(eps, grid, pc["n_boot"], pc["seed"])
    cif_need = bootstrap_cif(needs, grid, pc["n_boot"], pc["seed"])
    plot_cif(
        cif_need,
        out / "cif_disorganised.png",
        title=f"League, losses disorganised at loss (n = {len(needs)}), 95% match bootstrap",
    )
    plot_cif(cif_all, out / "cif_all.png", title=f"League, all losses (n = {len(eps)})")

    def cif_at(df):
        c = cumulative_incidence(df["time_s"], df["code"], np.array(REPORT_HORIZONS_S))
        return c

    team_rows = []
    for tid, g in needs.groupby("losing_team_id"):
        c = cif_at(g)
        team_rows.append(
            {
                "team": names[tid],
                "matches": g["match_id"].nunique(),
                "episodes": len(g),
                "tau": round(taus[tid], 2),
                **{
                    f"reorg_by_{h:g}s": v
                    for h, v in zip(REPORT_HORIZONS_S, c["cif_1"], strict=True)
                },
                "regain_by_10s": float(c["cif_2"].iloc[2]),
            }
        )
    team_tab = pd.DataFrame(team_rows).sort_values("reorg_by_6s", ascending=False)
    team_tab.to_csv(out / "teams.csv", index=False)

    # Odd/even-match reliability of team-level reorganisation (exploratory preview of H1).
    needs = needs.copy()
    needs["half"] = (
        needs.groupby("losing_team_id")["match_id"].transform(lambda s: s.rank(method="dense")) % 2
    )
    halves = []
    for (tid, h), g in needs.groupby(["losing_team_id", "half"]):
        halves.append({"tid": tid, "half": h, "cif6": float(cif_at(g)["cif_1"].iloc[1])})
    hv = pd.DataFrame(halves).pivot(index="tid", columns="half", values="cif6").dropna()
    r_half = float(np.corrcoef(hv[0], hv[1])[0, 1]) if len(hv) > 2 else np.nan

    # Reliability regimes.
    epd_need = epd[np.isfinite(epd["D0"]) & (epd["D0"] > epd["tau"])]
    rel = reliability_report(regime_episodes(needs, epd_need, cfg))

    # D0 vs length of the losing possession (face validity / D-006).
    tracked = tracked.copy()
    tracked["a_poss_bin"] = pd.cut(
        tracked["a_possession_s"],
        [-0.01, 1, 2, 5, 10, 20, 1e9],
        labels=["<1s", "1-2s", "2-5s", "5-10s", "10-20s", ">20s"],
    )
    d0_tab = tracked.groupby("a_poss_bin", observed=True).agg(
        episodes=("D0", "size"),
        D0_median=("D0", "median"),
        share_disorganised=("D0", lambda s: float((s > tracked.loc[s.index, "tau"]).mean())),
    )
    d0_tab = d0_tab.reset_index()

    # Decomposition preview (descriptive): mean share of D^2 per component at each horizon.
    comps = ref.components
    dec = []
    for h in P.horizons_s:
        sub = needs[np.isfinite(needs[f"D_{h:g}s"]) & (needs[f"D_{h:g}s"] > needs["tau"])]
        dec.append(
            {
                "horizon": f"+{h:g}s",
                "late_episodes": len(sub),
                **{c: float(sub[f"share_{c}_{h:g}s"].mean()) for c in comps},
            }
        )
    dec_tab = pd.DataFrame(dec)

    # Prototype match: fastest / slowest / random examples.
    pm = needs[needs["match_id"] == pc["match_id"]]
    n_ex = int(pc["n_examples"])
    fast = pm[pm["code"] == 1].nsmallest(n_ex, "T_r")
    slow_r = pm[pm["code"] == 1].nlargest(n_ex // 2, "T_r")
    slow_c = pm[(pm["code"] != 1) & (pm["window_s"] >= 15)].nlargest(
        n_ex - len(slow_r), "debt_above_tau"
    )
    slow = pd.concat([slow_r, slow_c])
    rand = pm.iloc[rng.choice(len(pm), size=min(n_ex, len(pm)), replace=False)]
    md_p = next(md for md in mds if md.meta.match_id == pc["match_id"])
    ex_rows = []
    for kind, df in (("fast", fast), ("slow", slow), ("random", rand)):
        for j, r in enumerate(df.itertuples(index=False)):
            eid = f"{r.match_id}_{r.frame_loss}"
            tr = traces[eid]
            v = EpisodeView(md_p, ref, r.losing_team_id, r.frame_loss)
            status = (
                f"T_r = {r.T_r:.1f} s"
                if r.code == 1
                else f"not reorganised ({r.end_reason} at {r.window_s:.1f} s)"
            )
            team = names[r.losing_team_id]
            title = f"{team} lose the ball, P{r.period} frame {r.frame_loss}: {status}"
            stem = f"{kind}_{j}_{eid}"
            plot_snapshots(v, D=tr.D, tau=tr.tau, path=out / "anim" / f"{stem}.png", suptitle=title)
            if not a.no_anim:
                animate_episode(
                    v,
                    tr.D,
                    tr.tau,
                    tr.end,
                    str(out / "anim" / f"{stem}.mp4"),
                    title=title,
                    t_r=r.T_r if r.code == 1 else None,
                )
            ex_rows.append(
                {
                    "kind": kind,
                    "file": stem,
                    "team": names[r.losing_team_id],
                    "T_r": r.T_r,
                    "end_reason": r.end_reason,
                    "window_s": r.window_s,
                    "D0": r.D0,
                    "reliability_10s": r.reliability_10s,
                }
            )
    ex_tab = pd.DataFrame(ex_rows)
    if len(fast) and len(slow):
        f0, s0 = fast.iloc[0], slow.iloc[0]
        tf_ = traces[f"{f0.match_id}_{f0.frame_loss}"]
        ts_ = traces[f"{s0.match_id}_{s0.frame_loss}"]
        plot_d_curves(
            [tf_.D, ts_.D],
            [f"fast ({names[f0.losing_team_id]})", f"slow ({names[s0.losing_team_id]})"],
            [DEF, ATT],
            tf_.tau,
            P.fps,
            out / "d_curves_fast_slow.png",
            title="D(t) after two losses, match 2007721",
        )

    lines = ["<!-- generated by scripts/phase1_prototype.py; do not edit by hand -->", ""]
    lines += ["### Episode funnel (all 20 games)", "", md_table(funnel), ""]
    lines += [
        "### Reference: split-half (odd/even match) stability of team-specific deviations",
        "",
        md_table(sh, 2),
        "",
    ]
    lines += [
        f"tau per team (q = {P.tau_quantile}): "
        f"{dict(sorted((names[k], round(v, 2)) for k, v in taus.items()))}",
        "",
    ]
    c = cif_need.set_index("t").loc[list(REPORT_HORIZONS_S)]
    lines += [
        "### League cumulative incidence (disorganised-at-loss episodes, 95% match bootstrap)",
        "",
        md_table(
            c.reset_index()[["t", "cif_1", "cif_1_lo", "cif_1_hi", "cif_2", "cif_2_lo", "cif_2_hi"]]
        ),
        "",
    ]
    lines += [
        "### Per team (disorganised-at-loss episodes)",
        "",
        md_table(team_tab),
        "",
        f"Odd/even-match correlation of team P(reorganised by 6 s): r = {r_half:.2f} "
        f"({len(hv)} teams with >= 2 games; exploratory)",
        "",
    ]
    lines += ["### Reliability regimes", "", md_table(rel), ""]
    lines += ["### D at loss by length of the losing possession", "", md_table(d0_tab), ""]
    lines += [
        "### Decomposition preview: mean share of D² per component, late episodes (D > tau)",
        "",
        md_table(dec_tab),
        "",
    ]
    lines += ["### Face-validity examples (outputs/phase1/anim/)", "", md_table(ex_tab, 2), ""]
    (ROOT / "docs" / "gate1_generated.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
