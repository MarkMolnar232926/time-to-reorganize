import numpy as np
import pytest

from reorg.shape import COMPONENTS, hull_area, shape_components

L = 105.0


def grid_block():
    """A 4-4-2-like block: back 4 at x=-35, mids at x=-20, forwards at x=-5."""
    x = np.array([-35, -35, -35, -35, -20, -20, -20, -20, -5, -5], float)
    y = np.array([-15, -5, 5, 15, -15, -5, 5, 15, -5, 5], float)
    return x[None, :], y[None, :]


def test_components_on_known_block():
    X, Y = grid_block()
    s = shape_components(X, Y, np.array([0.0]), np.array([0.0]), L).iloc[0]
    assert set(COMPONENTS) == set(s.index)
    assert s.line_height == pytest.approx(-35 + L / 2)
    assert s.depth == pytest.approx(np.percentile(X, 90) - np.percentile(X, 10))
    assert s.cx_ball == pytest.approx(X.mean())
    assert s.cy_ball == pytest.approx(0.0)
    assert s.goal_side == 10
    assert s.nn_median == pytest.approx(10.0)


def test_permutation_invariance_and_nan():
    X, Y = grid_block()
    perm = np.random.default_rng(0).permutation(10)
    a = shape_components(X, Y, np.zeros(1), np.zeros(1), L)
    b = shape_components(X[:, perm], Y[:, perm], np.zeros(1), np.zeros(1), L)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    Xn = X.copy()
    Xn[0, 3] = np.nan
    assert shape_components(Xn, Y, np.zeros(1), np.zeros(1), L).isna().all(axis=None)


def test_goal_side_counts_relative_to_ball():
    X, Y = grid_block()
    s = shape_components(X, Y, np.array([-25.0]), np.array([0.0]), L).iloc[0]
    assert s.goal_side == 4


def test_hull_area():
    assert hull_area(np.array([0, 10, 10, 0.0]), np.array([0, 0, 5, 5.0])) == pytest.approx(50)
