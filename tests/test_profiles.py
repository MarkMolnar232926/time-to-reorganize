import numpy as np
import pandas as pd

from reorg.profiles import late_shares, plot_team_card, team_profile
from reorg.shape import COMPONENTS


def fake_episodes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "team": rng.choice(["A", "B"], n),
            "match_id": rng.integers(0, 8, n),
            "disorganised_at_loss": rng.random(n) < 0.7,
            "time_s": rng.uniform(0, 20, n),
            "code": rng.choice([0, 1, 2], n),
            "tau": 1.0,
            "D_6s": rng.uniform(0.5, 3, n),
        }
    )
    shares = rng.dirichlet(np.ones(len(COMPONENTS)), n)
    for k, c in enumerate(COMPONENTS):
        df[f"share_{c}_6s"] = shares[:, k]
    return df


def test_late_shares_only_counts_late_episodes():
    df = fake_episodes()
    s = late_shares(df, 6.0)
    late = df[df["D_6s"] > 1.0]
    assert s.attrs["n_late"] == len(late)
    assert np.isclose(s.sum(), 1.0)


def test_team_profile_and_card(tmp_path):
    df = fake_episodes()
    grid = np.round(np.arange(0, 20.01, 0.5), 1)
    p = team_profile(df, "A", 6.0, grid, n_boot=20, seed=0)
    assert p["losses"] == int((df["team"] == "A").sum())
    lo, hi = p["shares_team_ci"]
    assert (lo <= hi).all()
    plot_team_card(p, tmp_path / "card.png")
    assert (tmp_path / "card.png").stat().st_size > 10_000
