"""End-to-end run of the public API on a synthetic match with a known convergence time."""

import copy
from pathlib import Path

import yaml
from synthetic_match import write_match

import reorg

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())


def test_run_league_recovers_known_convergence(tmp_path):
    write_match(tmp_path, match_id=1, minutes=8, spell_s=30, converge_s=5, seed=1)
    cfg = copy.deepcopy(CFG)
    cfg["data"]["root"] = str(tmp_path)
    cfg["data"]["cache"] = None
    res = reorg.run_league(cfg)
    eps = res.episodes
    # 16 spells of 30 s -> 15 possession changes, minus the one at half-time (new period).
    assert len(eps) == 14
    dis = eps[eps["disorganised_at_loss"]]
    assert len(dis) == len(eps)  # every loss starts stretched and high
    t = dis.loc[dis["reorganised"], "T_r"]
    assert len(t) == len(dis)
    # True arrival is 5 s. tau is the 75th percentile of calm-defending D, so when the settled
    # team happens to be in one of its looser calm moments, T_r waits until it tightens:
    # most episodes sit at ~5 s and the rest are later, never earlier than the approach allows.
    assert 4.5 <= t.median() <= 5.5
    assert ((t >= 3.0) & (t <= 6.5)).mean() >= 0.6
    assert (t >= 3.0).all()
    teams = reorg.team_summary(eps)
    assert set(teams["losing_team_id"]) == {100, 200}
    assert (teams["share_organised_at_loss"] == 0).all()
