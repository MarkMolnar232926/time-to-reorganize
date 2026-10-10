# Using `reorg`

## Python
```python
import reorg

cfg = reorg.load_config("config.yaml")      # all thresholds, with a rationale for each
res = reorg.run_league(cfg)                 # every match under cfg["data"]["root"]

res.episodes                                # one row per possession loss: T_r, D at +3/+6/+10 s,
                                            # end reason, component z-scores, reliability, ...
reorg.team_summary(res.episodes)            # per team: share organised at loss, P(back in shape)
reorg.plot_reorganisation(res, 2007721, 18757, "episode.png")   # snapshots + D(t)
```

Lower-level building blocks:

| Function | What it does |
|---|---|
| `reorg.find_losses(events, phases)` | open-play possession losses from dynamic events (D-004) |
| `reorg.build_reference(org_frames, cfg)` | each team's organised block per ball context, shrunk toward the league |
| `reorg.time_to_reorganise(D, tau, end, hold)` | T_r from a D(t) trace with the 2 s hold rule |
| `reorg.reliability_report(regimes)` | results under the three tracking-reliability regimes |
| `reorg.profiles.team_profile(...)` / `plot_team_card(...)` | coach-facing "what goes wrong" card |

## Command line
```bash
reorg losses   --out outputs/losses.csv          # possession losses per match
reorg episodes                                   # outputs/episodes.csv + outputs/teams.csv
reorg team-cards [--team "Auckland FC"]          # outputs/teams/<team>.png
reorg episode --match 2007721 --frame 18757 --video   # snapshots + MP4 of one loss
```

## Reproducing every number in the README
```bash
scripts/download_data.sh && scripts/run_all.sh
```
