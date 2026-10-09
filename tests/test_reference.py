import numpy as np
import pandas as pd
import pytest

from reorg.reference import LEAGUE, build_reference
from reorg.shape import COMPONENTS

CFG = {
    "reference": {
        "n_channels": 3,
        "n_depths": 3,
        "shrinkage_k_frames": 100,
        "spread_floor": {c: 0.1 for c in COMPONENTS},
    }
}
L, W = 105.0, 68.0


def synthetic_org(n_per_cell=200, seed=0):
    """Organised frames where every component's centre equals 10 * cell + team offset."""
    rng = np.random.default_rng(seed)
    rows = []
    for team, off, n in ((1, 0.0, n_per_cell), (2, 5.0, n_per_cell // 10)):
        for cell in range(9):
            vals = 10.0 * cell + off + rng.normal(0, 1, (n, len(COMPONENTS)))
            df = pd.DataFrame(vals, columns=COMPONENTS)
            df["team_id"], df["cell"], df["match_id"] = team, cell, 1
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_shrinkage_pulls_thin_team_toward_league():
    ref = build_reference(synthetic_org(), CFG)
    t = ref.table
    rich = t.loc[(1, 4), "c_depth"]
    thin = t.loc[(2, 4), "c_depth"]
    league = t.loc[(LEAGUE, 4), "c_depth"]
    assert abs(rich - 40) < 0.5
    assert league < thin < 45  # shrunk from 45 toward the pooled median
    assert t.loc[(2, 4), "w"] == pytest.approx(20 / 120)


def test_continuous_lookup_matches_cells_and_is_continuous():
    ref = build_reference(synthetic_org(), CFG)
    # Centre of cell (depth 1, channel 1) = cell 4.
    C, _ = ref.lookup_continuous(1, np.array([0.0]), np.array([0.0]), L, W)
    Cd, _ = ref.lookup(1, np.array([4]))
    assert np.allclose(C, Cd)
    # Crossing the channel boundary y = -W/6 changes the target smoothly.
    eps = 1e-3
    a, _ = ref.lookup_continuous(1, np.array([0.0]), np.array([-W / 6 - eps]), L, W)
    b, _ = ref.lookup_continuous(1, np.array([0.0]), np.array([-W / 6 + eps]), L, W)
    assert np.abs(a - b).max() < 0.01
    # Beyond the outermost centres the target is constant (clamped).
    e1, _ = ref.lookup_continuous(1, np.array([-L / 2]), np.array([-W / 2]), L, W)
    e2, _ = ref.lookup_continuous(1, np.array([-L / 2 + 5]), np.array([-W / 2 + 3]), L, W)
    assert np.allclose(e1, e2)
    nan, _ = ref.lookup_continuous(1, np.array([np.nan]), np.array([0.0]), L, W)
    assert np.isnan(nan).all()


def test_league_only_reference_ignores_team():
    org = synthetic_org()
    ref = build_reference(org, CFG, team_specific=False)
    assert set(ref.table.index.get_level_values(0)) == {LEAGUE}
    a, _ = ref.lookup_continuous(1, np.array([0.0]), np.array([0.0]), L, W)
    b, _ = ref.lookup_continuous(2, np.array([0.0]), np.array([0.0]), L, W)
    assert np.allclose(a, b)


def test_unvisited_cells_do_not_create_holes():
    org = synthetic_org()
    org = org[org["cell"].isin([3, 4, 5])]  # only the middle third was ever defended
    ref = build_reference(org, CFG)
    C, S = ref.lookup_continuous(1, np.linspace(-50, 50, 21), np.zeros(21), L, W)
    assert np.isfinite(C).all() and np.isfinite(S).all()
