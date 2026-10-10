"""Episode tables with outcomes, landmark samples and cross-validated predictions (H2/H4).

Shared by ``scripts/confirmatory.py`` (frozen-plan analyses) and ``scripts/sensitivity_effects.py``
(pre-registered sensitivity of the primary effects). Folds run in parallel worker processes;
results are concatenated in fold order, so they do not depend on the number of workers.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from reorg import stats
from reorg.outcomes import OutcomeParams, add_danger, landmark_sample
from reorg.reference import build_reference, organised_frames
from reorg.reorganisation import ReorgParams, episodes_for_match, team_taus

# H2 covariates (ANALYSIS_PLAN.md §5, H2): 5 adjustment terms + the exposure.
COVARIATES = ["ball_x", "wide_channel", "goal_side_at_loss", "ball_speed_3s", "log1p_poss"]
REDUCED_COVARIATES = ["ball_x", "goal_side_at_loss"]
ZONE = ["ball_x", "wide_channel"]
MODELS = {
    "M_zone": ZONE,
    "M_cov": COVARIATES,
    "M_D": ["still_disorganised", *COVARIATES],
    "M_compact": ["nn_mean", *COVARIATES],
    "M_phase": ["phase_org", *COVARIATES],
    "M_compact+D": ["nn_mean", "still_disorganised", *COVARIATES],
    "M_phase+D": ["phase_org", "still_disorganised", *COVARIATES],
}
REGIMES = ("all", "detected", "reliable")


@dataclass(frozen=True)
class Variant:
    """How the metric is computed (base settings unless a sensitivity variant says otherwise)."""

    cfg: dict
    team_specific: bool = True
    continuous: bool = True
    exclude_end_types: tuple[str, ...] = field(default_factory=tuple)

    @property
    def P(self) -> ReorgParams:
        return ReorgParams.from_config(self.cfg)

    @property
    def OP(self) -> OutcomeParams:
        return OutcomeParams.from_config(self.cfg)


def features(e: pd.DataFrame) -> pd.DataFrame:
    e = e.copy()
    e["wide_channel"] = ((e["cell"] % 3) != 1).astype(int)
    e["log1p_poss"] = np.log1p(e["a_possession_s"])
    return e


def fit_reference(mds_ref: list, v: Variant):
    ref = build_reference(organised_frames(mds_ref), v.cfg, team_specific=v.team_specific)
    return ref, team_taus(mds_ref, ref, v.P, continuous=v.continuous)


def episodes_with_outcome(mds_ref, mds_eval, v: Variant, min_frame_det=None, fitted=None):
    """Reference and tau from ``mds_ref`` (or ``fitted``); episodes + danger for ``mds_eval``."""
    ref, taus = fitted or fit_reference(mds_ref, v)
    P, OP = v.P, v.OP
    out = []
    for md in mds_eval:
        e = episodes_for_match(
            md, ref, taus, P, min_frame_det=min_frame_det, continuous=v.continuous
        )[0]
        out.append(add_danger(md, e, OP))
    e = features(pd.concat(out, ignore_index=True))
    if v.exclude_end_types:
        e = e[~e["a_end_type"].isin(v.exclude_end_types)].reset_index(drop=True)
    return e


def landmark(e: pd.DataFrame, lm: float) -> pd.DataFrame:
    s = landmark_sample(e, lm)
    s = s[np.isfinite(s[COVARIATES]).all(axis=1)].copy()
    s["nn_mean"] = s[f"nn_mean_0_{lm:g}s"]
    s["phase_org"] = s[f"phase_organised_at_{lm:g}s"].astype(int)
    return s


def h2_effect(sample, n_boot, seed, min_events, exposure="still_disorganised") -> dict:
    """H2 logistic model: OR of the exposure with match-cluster bootstrap CI and Wald p."""
    n_events = int(sample["y"].sum())
    cov = COVARIATES if n_events >= min_events else REDUCED_COVARIATES
    X = [exposure, *cov]
    r = stats.cluster_bootstrap_or(sample, "y", X, exposure, n_boot, seed)
    r.update(
        {
            "episodes": len(sample),
            "events": n_events,
            "model": "full" if cov is COVARIATES else "reduced",
            "p_one_sided": stats.wald_one_sided_p(sample, "y", X, exposure, "match_id"),
        }
    )
    return r


# ------------------------------------------------------------------------ cross-validation
_STATE: dict = {}  # set in the parent before forking; read by worker processes


def _fold(k: int) -> dict[str, pd.DataFrame]:
    s = _STATE
    mds, held, v = s["mds"], s["groups"][k], s["variant"]
    train = [md for md in mds if md.meta.match_id not in held]
    test = [md for md in mds if md.meta.match_id in held]
    fitted = fit_reference(train if s["rebuild"] else mds, v)
    cache: dict = {}
    out = {}
    for reg in s["regimes"]:
        mfd = s["det_thr"] if reg == "detected" else None
        key = "detected" if reg == "detected" else "all"
        if key not in cache:
            cache[key] = (
                landmark(episodes_with_outcome(train, train, v, mfd, fitted), s["landmark"]),
                landmark(episodes_with_outcome(train, test, v, mfd, fitted), s["landmark"]),
            )
        tr, te = cache[key]
        if reg == "reliable":
            tr = tr[tr["reliability_10s"] >= s["rel_thr"]]
            te = te[te["reliability_10s"] >= s["rel_thr"]]
        if te.empty:
            continue
        row = te[["match_id", "frame_loss", "y"]].copy()
        for name, X in s["models"].items():
            if reg != "all" and name not in ("M_cov", "M_D"):
                continue
            fit = stats.fit_logit(tr, "y", X)
            exog = stats.sm.add_constant(te[X].astype(float), has_constant="add")
            row[name] = fit.predict(exog).to_numpy()
        out[reg] = row
    return out


def cv_predictions(
    mds: list,
    v: Variant,
    landmark_s: float,
    det_thr: float,
    rel_thr: float,
    folds: int = 0,
    seed: int = 0,
    workers: int = 1,
    rebuild: bool = True,
    models: dict | None = None,
    regimes: tuple[str, ...] = REGIMES,
    progress=None,
) -> dict[str, pd.DataFrame]:
    """Out-of-fold predicted probabilities per regime.

    ``folds=0`` is leave-one-match-out (the frozen plan). ``rebuild=True`` rebuilds the reference
    and tau inside every fold from the training matches only.
    """
    ids = np.array([md.meta.match_id for md in mds])
    if folds:
        order = np.random.default_rng(seed).permutation(ids)
        fold_of = {m: i % folds for i, m in enumerate(order)}
        groups = [ids[[fold_of[m] == f for m in ids]] for f in range(folds)]
    else:
        groups = [np.array([m]) for m in ids]
    _STATE.update(
        mds=mds,
        groups=groups,
        variant=v,
        rebuild=rebuild,
        regimes=regimes,
        det_thr=det_thr,
        rel_thr=rel_thr,
        landmark=landmark_s,
        models=models or MODELS,
    )
    if workers > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(workers) as pool:
            results = []
            for k, r in enumerate(pool.imap(_fold, range(len(groups)))):
                results.append(r)
                if progress:
                    progress(k + 1, len(groups))
    else:
        results = []
        for k in range(len(groups)):
            results.append(_fold(k))
            if progress:
                progress(k + 1, len(groups))
    return {
        reg: pd.concat([r[reg] for r in results if reg in r], ignore_index=True) for reg in regimes
    }


def compare(pred: pd.DataFrame, a: str, b: str, n_boot: int, seed: int) -> dict:
    """Out-of-fold log-loss / Brier / AUC differences (a - b) with match-cluster bootstrap."""
    y, cl = pred["y"].to_numpy(), pred["match_id"].to_numpy()
    d = stats.paired_metric_diff(y, pred[a].to_numpy(), pred[b].to_numpy(), cl, n_boot, seed)
    d = d.set_index("metric")
    p = stats.bootstrap_one_sided_p(y, pred[a].to_numpy(), pred[b].to_numpy(), cl, n_boot, seed)
    return {
        "comparison": f"{a} vs {b}",
        "episodes": len(pred),
        "events": int(y.sum()),
        "dlogloss": d.loc["log_loss", "diff"],
        "dlogloss_lo": d.loc["log_loss", "lo"],
        "dlogloss_hi": d.loc["log_loss", "hi"],
        "p_one_sided": p,
        "dbrier": d.loc["brier", "diff"],
        "dauc": d.loc["auc", "diff"],
        "dauc_lo": d.loc["auc", "lo"],
        "dauc_hi": d.loc["auc", "hi"],
    }


def sensitivity_variants(cfg: dict) -> list[tuple[str, Variant]]:
    """The pre-registered sensitivity grid (ANALYSIS_PLAN.md §6)."""
    import copy

    out = [("base", Variant(cfg))]

    def mod(name: str, fn=None, **kw) -> None:
        c = copy.deepcopy(cfg)
        if fn:
            fn(c)
        out.append((name, Variant(c, **kw)))

    for q in (0.70, 0.80):
        mod(f"tau_q={q}", lambda c, q=q: c["reorganisation"].__setitem__("tau_quantile", q))
    for h in (1.5, 3.0):
        mod(f"hold={h}s", lambda c, h=h: c["reorganisation"].__setitem__("hold_s", h))
    mod("discrete_reference", continuous=False)
    mod("league_only_reference", team_specific=False)
    mod(
        "goal_side_floor=1.0",
        lambda c: c["reference"]["spread_floor"].__setitem__("goal_side", 1.0),
    )
    mod("no_goal_side", lambda c: c["shape"]["weights"].__setitem__("goal_side", 0.0))
    mod("exclude_after_shot_or_clearance", exclude_end_types=("shot", "clearance"))
    return out
