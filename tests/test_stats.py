import numpy as np
import pandas as pd
import pytest

from reorg import stats


def synthetic_episodes(team_effect: float, n_matches=24, n_per=60, seed=0):
    """Episodes from a round robin of 8 teams; team k reorganises faster by team_effect * k."""
    rng = np.random.default_rng(seed)
    teams = np.arange(8)
    rows = []
    for m in range(n_matches):
        a, b = rng.choice(teams, 2, replace=False)
        for lose, gain in ((a, b), (b, a)):
            rate = 0.15 * (1 + team_effect * lose)
            t = rng.exponential(1 / rate, n_per)
            c = rng.exponential(1 / 0.12, n_per)  # competing regain
            time = np.minimum(np.minimum(t, c), 20.0)
            code = np.where(time == t, 1, np.where(time == c, 2, 0))
            for ti, ci in zip(time, code, strict=True):
                rows.append(
                    {
                        "match_id": m,
                        "losing_team_id": lose,
                        "gaining_team_id": gain,
                        "time_s": ti,
                        "code": ci,
                    }
                )
    return pd.DataFrame(rows)


def test_permutation_detects_planted_team_effect():
    out = stats.permutation_team_effect(synthetic_episodes(0.4), 6.0, n_perm=99, seed=1)
    assert out["p_value"] <= 0.02


def test_permutation_null_not_significant():
    out = stats.permutation_team_effect(synthetic_episodes(0.0, seed=3), 6.0, n_perm=99, seed=1)
    assert out["p_value"] > 0.05


def test_split_half_reliability_high_with_effect():
    rel = stats.split_half_reliability(synthetic_episodes(0.4, n_matches=40), 6.0)
    assert rel["r"] > 0.6 and rel["spearman_brown"] > rel["r"]


def logit_data(beta=0.8, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    p = 1 / (1 + np.exp(-(-1.0 + beta * x + 0.3 * z)))
    return pd.DataFrame(
        {"x": x, "z": z, "y": rng.random(n) < p, "match_id": rng.integers(0, 20, n)}
    )


def test_cluster_bootstrap_or_covers_truth():
    out = stats.cluster_bootstrap_or(logit_data(), "y", ["x", "z"], "x", n_boot=100, seed=0)
    assert out["lo"] < np.exp(0.8) < out["hi"]
    assert out["n_boot_ok"] == 100


def test_lomo_and_metrics():
    df = logit_data()
    p_full = stats.lomo_predict(df, "y", ["x", "z"])
    p_null = stats.lomo_predict(df, "y", ["z"])
    y = df["y"].to_numpy().astype(int)
    assert stats.log_loss(y, p_full) < stats.log_loss(y, p_null)
    d = stats.paired_metric_diff(y, p_full, p_null, df["match_id"].to_numpy(), 100, 0)
    ll = d.set_index("metric").loc["log_loss"]
    assert ll["hi"] < 0  # informative model is better, CI excludes 0
    assert d.set_index("metric").loc["auc", "lo"] > 0


def test_auc_known_values():
    assert stats.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert stats.auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_component_shares_identifies_dominant_component():
    rng = np.random.default_rng(0)
    n = 600
    raw = rng.dirichlet([6, 2, 2], n)
    df = pd.DataFrame({"match_id": rng.integers(0, 20, n), "tau": 1.0, "D_6s": 2.0})
    for k, c in enumerate(("a", "b", "c")):
        df[f"share_{c}_6s"] = raw[:, k]
    out = stats.component_shares(df, ("a", "b", "c"), 6.0, n_boot=100, seed=0)
    assert out.iloc[0]["component"] == "a"
    assert out.attrs["gap_lo"] > 0


def test_holm_matches_hand_computation():
    out = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03}).set_index("test")
    assert out.loc["a", "p_holm"] == pytest.approx(0.03)
    assert out.loc["c", "p_holm"] == pytest.approx(0.06)
    assert out.loc["b", "p_holm"] == pytest.approx(0.06)  # monotone step-down
    assert out.loc["a", "reject"] and not out.loc["b", "reject"]


def test_wald_one_sided_p_small_for_true_positive_effect():
    df = logit_data()
    assert stats.wald_one_sided_p(df, "y", ["x", "z"], "x", "match_id") < 1e-6
