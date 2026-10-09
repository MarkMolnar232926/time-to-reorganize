"""Shape descriptor S(t) of a team's 10 outfield players, in the team's own frame.

All components are permutation-invariant and vectorised over frames. Coordinates follow
:mod:`reorg.normalise`: own goal line at ``x = -L/2``, attacking towards +x.

Components (metres unless noted)
--------------------------------
line_height   mean x of the deepest 4 outfield players, measured from the own goal line
depth         robust length of the block: P90 - P10 of x
width         robust width of the block: P90 - P10 of y
cx_ball       centroid x minus ball x (negative = block behind the ball)
cy_ball       centroid y minus ball y (lateral shift relative to the ball)
nn_median     median nearest-neighbour distance (compactness)
goal_side     number of outfield players goal-side of the ball (x < ball x), count
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

COMPONENTS = ("line_height", "depth", "width", "cx_ball", "cy_ball", "nn_median", "goal_side")
N_DEEPEST = 4  # back line proxy: deepest four outfield players
P_LO, P_HI = 10, 90  # robust range percentiles


def shape_components(
    X: np.ndarray, Y: np.ndarray, ball_x: np.ndarray, ball_y: np.ndarray, pitch_length: float
) -> pd.DataFrame:
    """Compute S(t) for arrays of shape (n_frames, n_players). Rows with NaN give NaN."""
    xs = np.sort(X, axis=1)
    line_height = xs[:, :N_DEEPEST].mean(axis=1) + pitch_length / 2
    with np.errstate(all="ignore"):
        depth = np.percentile(X, P_HI, axis=1) - np.percentile(X, P_LO, axis=1)
        width = np.percentile(Y, P_HI, axis=1) - np.percentile(Y, P_LO, axis=1)
        cx = X.mean(axis=1) - ball_x
        cy = Y.mean(axis=1) - ball_y
        dx = X[:, :, None] - X[:, None, :]
        dy = Y[:, :, None] - Y[:, None, :]
        d = np.hypot(dx, dy)
        n = X.shape[1]
        d[:, np.arange(n), np.arange(n)] = np.inf
        nn = np.median(d.min(axis=2), axis=1)
        goal_side = (X < ball_x[:, None]).sum(axis=1).astype(float)
    goal_side[~np.isfinite(ball_x)] = np.nan
    out = pd.DataFrame(
        {
            "line_height": line_height,
            "depth": depth,
            "width": width,
            "cx_ball": cx,
            "cy_ball": cy,
            "nn_median": nn,
            "goal_side": goal_side,
        }
    )
    out.loc[np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1)] = np.nan
    return out


def hull_area(x: np.ndarray, y: np.ndarray) -> float:
    """Convex-hull area of one frame's players (m²); NaN if degenerate or missing."""
    pts = np.column_stack([x, y])
    if not np.isfinite(pts).all():
        return np.nan
    try:
        return float(ConvexHull(pts).volume)
    except QhullError:
        return np.nan


def hull_polygon(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convex-hull vertices (closed polygon) for plotting."""
    pts = np.column_stack([x, y])
    hull = ConvexHull(pts)
    v = pts[hull.vertices]
    return np.vstack([v, v[:1]])
