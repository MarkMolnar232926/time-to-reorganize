import numpy as np
import pandas as pd

from reorg.outcomes import OutcomeParams, first_danger, landmark_sample

P = OutcomeParams(fps=10, danger_window_s=15.0, min_box_frames=2)
L = 105.0


def path(x0, x1, n=200):
    return np.linspace(x0, x1, n), np.zeros(n)


def test_box_entry_detected_at_first_frame_inside():
    bx, by = path(0, -52.5, 200)  # ball travels toward the own goal
    t, kind = first_danger(bx, by, np.array([]), stop=200, L=L, params=P)
    first_inside = np.flatnonzero(bx <= -L / 2 + 16.5)[0]
    assert kind == "box_entry" and t == first_inside / 10


def test_shot_before_box_entry_wins():
    bx, by = path(0, -52.5, 200)
    t, kind = first_danger(bx, by, np.array([50]), stop=200, L=L, params=P)
    assert kind == "shot" and t == 5.0


def test_nothing_after_stop_or_window():
    bx, by = path(0, -52.5, 200)
    assert np.isnan(first_danger(bx, by, np.array([]), stop=50, L=L, params=P)[0])
    far = np.array([170])  # 17 s, beyond the 15 s window
    t, _ = first_danger(np.zeros(200), np.zeros(200), far, stop=200, L=L, params=P)
    assert np.isnan(t)


def test_ball_already_in_box_only_shot_counts():
    bx = np.full(100, -45.0)
    by = np.zeros(100)
    assert np.isnan(first_danger(bx, by, np.array([]), stop=100, L=L, params=P)[0])
    assert first_danger(bx, by, np.array([30]), stop=100, L=L, params=P)[1] == "shot"


def test_single_frame_blip_into_box_ignored():
    bx = np.zeros(100)
    bx[40] = -45.0
    assert np.isnan(first_danger(bx, np.zeros(100), np.array([]), 100, L, P)[0])


def test_landmark_sample_rules():
    e = pd.DataFrame(
        {
            "disorganised_at_loss": [True, True, True, True, False],
            "window_s": [10.0, 2.0, 10.0, 10.0, 10.0],
            "D_3s": [2.0, 2.0, 0.5, 2.0, 2.0],
            "tau": [1.0] * 5,
            "t_danger": [np.nan, 1.0, 8.0, 2.0, 8.0],
        }
    )
    s = landmark_sample(e, 3.0)
    # row 1: window ended before 3 s; row 3: danger before landmark; row 4: organised at loss.
    assert list(s.index) == [0, 2]
    assert list(s["y"]) == [0, 1]
    assert list(s["still_disorganised"]) == [1, 0]
