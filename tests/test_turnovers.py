import numpy as np
import pandas as pd

from reorg.turnovers import (
    LossParams,
    detect_losses_from_labels,
    find_losses_tracking,
    in_play_mask,
)

P = LossParams(fps=10, min_sustain_s=1.0, max_null_gap_s=1.0, require_ball_in_play=True)


def seq(*parts):
    """Build a label array from (label, n_frames) parts."""
    return np.array([lab for lab, n in parts for _ in range(n)], dtype=object)


def run(labels, in_play=None, **kw):
    in_play = np.ones(len(labels), bool) if in_play is None else in_play
    return detect_losses_from_labels(labels, in_play, P, **kw)


def test_clean_transfer_is_one_loss():
    out = run(seq(("A", 30), ("B", 30)))
    assert len(out) == 1
    assert out.loc[0, "losing"] == "A" and out.loc[0, "i_loss"] == 30


def test_short_blip_is_not_a_loss():
    assert run(seq(("A", 30), ("B", 5), ("A", 30))).empty


def test_short_null_gap_is_bridged():
    out = run(seq(("A", 30), (None, 8), ("B", 30)))
    assert len(out) == 1 and out.loc[0, "gap_frames"] == 8


def test_long_null_gap_is_rejected_but_override_counts_it():
    labels = seq(("A", 30), (None, 25), ("B", 30))
    assert run(labels).empty
    assert len(run(labels, max_null_gap_s=3.0)) == 1


def test_same_team_across_null_gap_is_not_a_loss():
    assert run(seq(("A", 30), (None, 5), ("A", 30))).empty


def test_b_must_sustain():
    assert run(seq(("A", 30), ("B", 9))).empty
    assert len(run(seq(("A", 30), ("B", 10)))) == 1


def test_dead_ball_transfer_excluded():
    labels = seq(("A", 30), (None, 5), ("B", 30))
    play = np.ones(len(labels), bool)
    play[32] = False
    assert run(labels, play).empty


def test_blip_then_real_loss_counts_once():
    out = run(seq(("A", 30), ("B", 4), ("A", 3), ("B", 40)))
    assert len(out) == 1


def test_in_play_mask_and_find_losses():
    n = 100
    frames = pd.DataFrame(
        {
            "frame": np.arange(n),
            "period": pd.array([1] * n, dtype="Int8"),
            "poss_group": list(seq(("home team", 50), ("away team", 50))),
        }
    )
    phases = pd.DataFrame({"frame_start": [0, 75], "frame_end": [70, 99]})
    assert in_play_mask(frames["frame"], phases).sum() == 71 + 25
    assert len(find_losses_tracking(frames, phases, P)) == 1
    # Dead ball inside the sustain window -> excluded.
    phases2 = pd.DataFrame({"frame_start": [0, 55], "frame_end": [52, 99]})
    assert find_losses_tracking(frames, phases2, P).empty


def _pp(rows):
    """Synthetic player_possession events: (frame_start, frame_end, team, interrupt_before,
    interrupt_after, end_type)."""
    return pd.DataFrame(
        [
            {
                "event_type": "player_possession",
                "period": 1,
                "frame_start": a,
                "frame_end": b,
                "team_id": t,
                "game_interruption_before": ib,
                "game_interruption_after": ia,
                "end_type": et,
            }
            for a, b, t, ib, ia, et in rows
        ]
    )


ALL_IN_PLAY = pd.DataFrame({"frame_start": [0], "frame_end": [10_000]})


def test_events_loss_basic_and_runs():
    from reorg.turnovers import find_losses, team_possession_runs

    ev = _pp(
        [
            (0, 10, 1, None, None, "pass"),
            (15, 30, 1, None, None, "possession_loss"),
            (32, 50, 2, None, None, "pass"),
            (55, 80, 2, None, None, "pass"),
        ]
    )
    runs = team_possession_runs(ev)
    assert list(runs["team_id"]) == [1, 2]
    out = find_losses(ev, ALL_IN_PLAY, P)
    assert len(out) == 1
    r = out.iloc[0]
    assert (r.losing_team_id, r.gaining_team_id, r.frame_last_a, r.frame_loss) == (1, 2, 30, 32)
    assert r.a_possession_s == 3.0


def test_events_loss_debounce_and_stoppage():
    from reorg.turnovers import find_losses

    # B holds only 0.5 s before A wins it back -> not a loss for A; A's regain is a loss for B
    # only if A then holds >= 1 s.
    ev = _pp(
        [
            (0, 30, 1, None, None, "pass"),
            (32, 34, 2, None, None, "possession_loss"),
            (37, 80, 1, None, None, "pass"),
        ]
    )
    out = find_losses(ev, ALL_IN_PLAY, P)
    assert list(out["losing_team_id"]) == [2]
    # A stoppage between runs means no open-play loss.
    ev2 = _pp(
        [(0, 30, 1, None, "throw_in_against", "pass"), (60, 90, 2, "throw_in_for", None, "pass")]
    )
    assert find_losses(ev2, ALL_IN_PLAY, P).empty
    # Ball out of play (no phase) during the transfer -> excluded.
    gap = pd.DataFrame({"frame_start": [0, 33], "frame_end": [30, 10_000]})
    ev3 = _pp([(0, 30, 1, None, None, "pass"), (35, 90, 2, None, None, "pass")])
    assert find_losses(ev3, gap, P).empty
    # Contiguous phases (shared boundary frame at the change of possession) are in play.
    contiguous = pd.DataFrame({"frame_start": [0, 31], "frame_end": [31, 10_000]})
    assert len(find_losses(ev3, contiguous, P)) == 1
