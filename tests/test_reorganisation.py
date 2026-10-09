import numpy as np
import pytest

from reorg.reorganisation import REORGANISED, first_hold, shape_distance, time_to_reorganise
from reorg.shape import shape_components

FPS = 10
HOLD = 20


def decay_trace(t_cross_s, tau=1.0, d0=5.0, n=201, noise=0.0, seed=0):
    """Exponential decay from d0 that crosses tau exactly at t_cross_s."""
    t = np.arange(n) / FPS
    k = np.log(d0 / tau) / t_cross_s
    d = d0 * np.exp(-k * t)
    rng = np.random.default_rng(seed)
    return d + noise * rng.standard_normal(n)


@pytest.mark.parametrize("t_cross", [1.0, 3.3, 6.0, 12.5])
def test_recovers_known_convergence_time(t_cross):
    D = decay_trace(t_cross)
    t, code = time_to_reorganise(D, 1.0, end=200, hold=HOLD)
    assert code == REORGANISED
    assert abs(t / FPS - t_cross) <= 0.11  # within one frame


def test_noisy_trace_within_tolerance():
    errs = []
    for seed in range(50):
        D = decay_trace(6.0, noise=0.02, seed=seed)
        t, code = time_to_reorganise(D, 1.0, end=200, hold=HOLD)
        assert code == REORGANISED
        errs.append(abs(t / FPS - 6.0))
    assert np.max(errs) <= 0.2 + 1e-9  # +-0.2 s target (2 frames)


def test_hysteresis_rejects_brief_dip():
    D = np.full(200, 3.0)
    D[30:40] = 0.5  # 1 s dip, shorter than the 2 s hold
    D[100:] = 0.5
    t, code = time_to_reorganise(D, 1.0, end=199, hold=HOLD)
    assert code == REORGANISED and t == 100


def test_censored_before_hold_completes():
    D = decay_trace(6.0)
    t, code = time_to_reorganise(D, 1.0, end=70, hold=HOLD)  # window ends 7 s, hold needs to 8 s
    assert code != REORGANISED and t == 50


def test_missing_frames_break_hold():
    D = np.full(200, 0.5)
    D[10] = np.nan
    assert first_hold(np.isfinite(D) & (D <= 1.0), HOLD, 200) == 11


def test_end_to_end_synthetic_block_convergence():
    """Players move linearly from a stretched shape to a target block, arriving at T* = 6 s.

    With the reference equal to the target's own shape, D falls monotonically to 0 at T*;
    T_r (tau = 0.25) must lie just before T* and never after it."""
    L = 105.0
    tx = np.array([-35, -35, -35, -35, -20, -20, -20, -20, -5, -5], float)
    ty = np.array([-15, -5, 5, 15, -15, -5, 5, 15, -5, 5], float)
    sx = tx * 0.4 + 20  # pushed high and compressed, as after an attack
    sy = ty * 1.8
    n, T = 201, 60
    a = np.clip(np.arange(n) / T, 0, 1)[:, None]
    X = (1 - a) * sx + a * tx
    Y = (1 - a) * sy + a * ty
    bx = np.full(n, 0.0)
    by = np.full(n, 0.0)
    S = shape_components(X, Y, bx, by, L).to_numpy()
    ref = S[-1]
    sigma = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0])
    z = (S - ref) / sigma
    D = shape_distance(z, np.ones(7))
    assert D[0] > 2 and D[-1] == pytest.approx(0)
    t, code = time_to_reorganise(D, 0.25, end=200, hold=HOLD)
    assert code == REORGANISED
    assert 4.5 <= t / FPS <= 6.0
