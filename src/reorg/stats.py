"""Confirmatory statistics (ANALYSIS_PLAN.md). Tested on synthetic data with planted effects.

* H1: match-preserving permutation test for between-team differences in reorganisation, and
  odd/even-match split-half reliability with the Spearman-Brown correction.
* H2: logistic landmark models with match-cluster bootstrap CIs for odds ratios.
* H4: leave-one-match-out predictions; log-loss / Brier / AUC differences with a match-cluster
  bootstrap.
* H3: component shares of D^2 with match-cluster bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from reorg.survival import cumulative_incidence

EPS = 1e-12


# ----------------------------------------------------------------------------------------- H1
def team_cif(df: pd.DataFrame, horizon_s: float, team_col: str = "losing_team_id") -> pd.Series:
    """Cumulative incidence of reorganisation by ``horizon_s`` per team."""
    g = np.array([horizon_s])
    return df.groupby(team_col).apply(
        lambda x: float(cumulative_incidence(x["time_s"], x["code"], g)["cif_1"].iloc[0]),
        include_groups=False,
    )


def permutation_team_effect(
    df: pd.DataFrame, horizon_s: float, n_perm: int, seed: int
) -> dict[str, float]:
    """Is the spread of team CIF(horizon) larger than chance?

    Statistic: variance across teams of P(reorganised by horizon). Null: within every match the
    two teams' labels are swapped with probability 1/2, which keeps match-level effects
    (weather, referee, opponent pairing, tracking quality) intact.
    """
    rng = np.random.default_rng(seed)
    obs = float(team_cif(df, horizon_s).var())
    pairs = df.groupby("match_id")[["losing_team_id", "gaining_team_id"]].first()
    # Array version of "relabel, then team_cif(...).var()": same draws, same per-team CIF
    # function and the same pandas variance, without copying the table each round.
    t = df["time_s"].to_numpy()
    c = df["code"].to_numpy()
    lose = df["losing_team_id"].to_numpy()
    gain = df["gaining_team_id"].to_numpy()
    match_pos = pd.Index(pairs.index).get_indexer(df["match_id"])
    g = np.array([horizon_s])
    perm_stats = np.empty(n_perm)
    for b in range(n_perm):
        swap = rng.random(len(pairs)) < 0.5
        lab = np.where(swap[match_pos], gain, lose)
        vals = [
            float(cumulative_incidence(t[lab == k], c[lab == k], g)["cif_1"].iloc[0])
            for k in np.unique(lab)
        ]
        perm_stats[b] = pd.Series(vals).var()
    p = (1 + np.sum(perm_stats >= obs)) / (1 + n_perm)
    return {"statistic": obs, "null_mean": float(perm_stats.mean()), "p_value": float(p)}


def split_half_reliability(df: pd.DataFrame, horizon_s: float) -> dict[str, float]:
    """Odd/even-match split per team; Pearson r of team CIF and Spearman-Brown 2r / (1 + r)."""
    d = df.copy()
    d["half"] = d.groupby("losing_team_id")["match_id"].rank(method="dense").astype(int) % 2
    a = team_cif(d[d["half"] == 0], horizon_s)
    b = team_cif(d[d["half"] == 1], horizon_s)
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    r = float(np.corrcoef(both["a"], both["b"])[0, 1]) if len(both) > 2 else np.nan
    return {"teams": len(both), "r": r, "spearman_brown": 2 * r / (1 + r) if r > -1 else np.nan}


# ----------------------------------------------------------------------------------------- H2
def fit_logit(df: pd.DataFrame, y: str, X: list[str]):
    """Logistic regression (statsmodels GLM, binomial) with an intercept."""
    exog = sm.add_constant(df[X].astype(float), has_constant="add")
    return sm.GLM(df[y].astype(float), exog, family=sm.families.Binomial()).fit()


def cluster_bootstrap_or(
    df: pd.DataFrame,
    y: str,
    X: list[str],
    term: str,
    n_boot: int,
    seed: int,
    cluster: str = "match_id",
) -> dict[str, float]:
    """Odds ratio of ``term`` with a match-cluster percentile bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    est = float(np.exp(fit_logit(df, y, X).params[term]))
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    boots = []
    for _ in range(n_boot):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        try:
            boots.append(np.exp(fit_logit(s, y, X).params[term]))
        except (np.linalg.LinAlgError, ValueError):
            continue
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"or": est, "lo": float(lo), "hi": float(hi), "n_boot_ok": len(boots)}


# ----------------------------------------------------------------------------------------- H4
def lomo_predict(df: pd.DataFrame, y: str, X: list[str], cluster: str = "match_id") -> np.ndarray:
    """Leave-one-match-out predicted probabilities."""
    pred = np.full(len(df), np.nan)
    idx = np.arange(len(df))
    for _, test in df.groupby(cluster).indices.items():
        train = np.setdiff1d(idx, test)
        model = fit_logit(df.iloc[train], y, X)
        exog = sm.add_constant(df.iloc[test][X].astype(float), has_constant="add")
        pred[test] = model.predict(exog)
    return pred


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC (ties count 1/2)."""
    y = np.asarray(y).astype(bool)
    pos, neg = p[y], p[~y]
    if not len(pos) or not len(neg):
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def paired_metric_diff(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int
) -> pd.DataFrame:
    """Metric(a) - metric(b) for log-loss, Brier and AUC, with match-cluster bootstrap CIs.

    Negative log-loss / Brier differences and positive AUC differences favour model a.
    """
    rng = np.random.default_rng(seed)
    fns = {"log_loss": log_loss, "brier": brier, "auc": auc}
    est = {k: f(y, p_a) - f(y, p_b) for k, f in fns.items()}
    uniq = np.unique(clusters)
    where = {c: np.flatnonzero(clusters == c) for c in uniq}
    boots = {k: [] for k in fns}
    for _ in range(n_boot):
        ii = np.concatenate([where[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        for k, f in fns.items():
            boots[k].append(f(y[ii], p_a[ii]) - f(y[ii], p_b[ii]))
    rows = []
    for k in fns:
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        rows.append({"metric": k, "diff": est[k], "lo": float(lo), "hi": float(hi)})
    out = pd.DataFrame(rows)
    # One-sided p for "a has lower log-loss than b", from the same resamples (+1 correction).
    ll = np.asarray(boots["log_loss"])
    out.attrs["p_log_loss_ge_0"] = float((1 + np.sum(ll >= 0)) / (1 + n_boot))
    return out


# ----------------------------------------------------------------------------------------- H3
def component_shares(
    df: pd.DataFrame, components: tuple[str, ...], horizon_s: float, n_boot: int, seed: int
) -> pd.DataFrame:
    """Mean share of D^2 per component among episodes still late (D > tau) at the horizon,
    with match-cluster bootstrap CIs, plus the CI of (largest - second largest)."""
    h = f"{horizon_s:g}s"
    late = df[np.isfinite(df[f"D_{h}"]) & (df[f"D_{h}"] > df["tau"])]
    cols = [f"share_{c}_{h}" for c in components]
    est = late[cols].mean().to_numpy()
    order = np.argsort(est)[::-1]
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in late.groupby("match_id")}
    keys = np.array(list(groups))
    boot, gap = [], []
    for _ in range(n_boot):
        s = pd.concat([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        m = s[cols].mean().to_numpy()
        boot.append(m)
        gap.append(m[order[0]] - m[order[1]])
    boot = np.vstack(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"component": components, "share": est, "lo": lo, "hi": hi})
    out = out.sort_values("share", ascending=False).reset_index(drop=True)
    g_lo, g_hi = np.percentile(gap, [2.5, 97.5])
    out.attrs.update({"p_gap_le_0": float((1 + np.sum(np.asarray(gap) <= 0)) / (1 + len(gap)))})
    out.attrs.update(
        {
            "late_episodes": len(late),
            "top_minus_second": float(est[order[0]] - est[order[1]]),
            "gap_lo": float(g_lo),
            "gap_hi": float(g_hi),
        }
    )
    return out


# ------------------------------------------------------------------------- p-values for Holm
def wald_one_sided_p(df: pd.DataFrame, y: str, X: list[str], term: str, cluster: str) -> float:
    """One-sided (coefficient > 0) Wald p with cluster-robust (match) covariance."""
    from scipy.stats import norm

    exog = sm.add_constant(df[X].astype(float), has_constant="add")
    res = sm.GLM(df[y].astype(float), exog, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df[cluster])[0]}
    )
    return float(1 - norm.cdf(res.params[term] / res.bse[term]))


def bootstrap_one_sided_p(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int
) -> float:
    """P(log-loss(a) - log-loss(b) >= 0) under the match-cluster bootstrap (+1 correction)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    where = {c: np.flatnonzero(clusters == c) for c in uniq}
    hits = 0
    for _ in range(n_boot):
        ii = np.concatenate([where[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        hits += log_loss(y[ii], p_a[ii]) - log_loss(y[ii], p_b[ii]) >= 0
    return float((1 + hits) / (1 + n_boot))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm step-down adjusted p-values and decisions."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = [], 0.0
    for i, (_, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    return pd.DataFrame(
        {
            "test": [k for k, _ in items],
            "p": [p for _, p in items],
            "p_holm": adj,
            "reject": [a <= alpha for a in adj],
        }
    )
