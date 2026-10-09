# Data audit: SkillCorner open data (A-League 2024/25)

*Phase 0, 9 Oct 2026. Upstream commit `4340d27` (14 Sep 2026), fetched by `scripts/download_data.sh`.*
All numbers below come from `scripts/audit_data.py`. Full per-match tables are in
[`audit_generated.md`](audit_generated.md) and CSVs in `outputs/audit/`. Rerun with
`python scripts/audit_data.py` (about 20 s once tracking is cached).

## 1. Summary for GATE 0

| Question | Answer |
|---|---|
| Games with XY tracking | **20** (all with dynamic events **and** phases of play; the upstream README text still says 10) |
| Body pose | 2 games (1925299, 1996435). Full files (~0.6 GB zipped each) are on Hugging Face, and the repo holds only a 12 s sample |
| Frame rate / clock | 10 fps, frames contiguous (no skipped frame numbers), `timestamp` is the period clock (P2 starts at 45:00) |
| Coordinates | metres, origin at the pitch centre, x = long axis. Pitch sizes 104–106 × 68 m and differ by venue |
| Attacking direction | `match.json/home_team_side` per period matches the GK positions in **20/20** games |
| Players per frame | always 22 (all players, extrapolated) or 0. **19–34%** of in-period frames per game have no players (replays/close-ups), but only **1.7%** of ball-in-play frames do |
| `is_detected` | 0.54–0.73 of in-play player-frames per game. **Central defenders are lowest (0.45–0.67)**, GKs 0.13–0.23 |
| Tracking `possession` field | **Not reliable enough to define losses** (see §4) |
| Proposed primary losses | **2,818** open-play losses from dynamic events across 20 games (~141/game). 83% are confirmed by phases of play within ±2 s |
| Usable for shape | 2,721 (97%) have ≥90% of the first 10 s tracked |
| Censoring | heavy: only 41% of losses have ≥10 s of open play before a regain or stoppage, and 29% have ≥15 s |

## 2. Inventory

* 20 games, Nov 2024 – May 2025, 13 teams. Matches per team: Auckland 7; CC Mariners, Melbourne
  Victory, Western United 4; Adelaide, Brisbane, Melbourne City, Wellington 3; Macarthur,
  Perth, Sydney FC, Western Sydney 2; **Newcastle 1**.
* Per game: 57.6k–72.1k frames. Games 2011166 onward include about 12k trailing frames with
  `period = null`, which are dropped.
* Events: 3.9k–5.7k rows per game, 322 columns. Phases: 383–490 rows per game.
* Season aggregates (physical / OBR / passing) exist but are not used by this project.
* Spec PDFs (Dynamic Events, Phases of Play) return HTTP 403 from this environment, so
  `docs/specs/` is empty (**blocker, see STATUS.md**). Everything below is checked empirically
  against the data.

## 3. Schemas (verified)

**`{id}_match.json`**: `players[].id` is the id used in tracking `player_id` (not
`trackable_object`). Role is in `players[].player_role.{acronym, position_group}`, and GKs have
acronym `GK` (position_group shows as "Other" in our tables). `playing_time.total.{start_frame,
end_frame}` gives on-pitch frames. `match_periods` gives start/end frames, and `pitch_length`
/ `pitch_width` are in metres.

**Tracking JSONL** (one object per frame): `frame, timestamp ("HH:MM:SS.ss" | null), period
(1|2|null), ball_data{x,y,z,is_detected}, possession{player_id, group ∈ {"home team","away
team", null}}, image_corners_projection{8 coords}, player_data[{x,y,player_id,is_detected}]`.

**Phases of play CSV**: one row per phase with `frame_start, frame_end, period,
team_in_possession_id, team_in/out_of_possession_phase_type` plus start/end location and team
width/length. Phases cover the ball-in-play time only. Consecutive phases **share their boundary
frame** at a change of possession, and a gap > 1 frame means the ball was dead.

**Dynamic events CSV**: `event_type ∈ {passing_option, player_possession, on_ball_engagement,
off_ball_run}`. `player_possession` rows give `frame_start/frame_end, team_id, start_type,
end_type, game_interruption_before/after` and much more (EPV, pressure, line-break features).

**Event coordinates.** The README says they are "not scaled to standard pitch size". We
verified that event `x/y` are **actual-pitch metres, mirrored into the attacking team's
direction**: multiply by −1 when `attacking_side == "right_to_left"`. They then match the
tracking position of the same player at `frame_start` with a median error of 0.00–0.17 m in
every game. Without mirroring, the median error is up to 9.5 m.

### Vocabularies (actual values)

* Out-of-possession phases: `medium_block` 2880, `chaotic` 1802, `low_block` 1623,
  `high_block` 1196, `defending_direct` 677, `defending_set_play` 360, `defending_transition`
  185, `defending_quick_break` 111. These pair one-to-one with in-possession `create, chaotic,
  finish, build_up, direct, set_play, transition, quick_break`. The event-level phase columns
  also contain `disruption` (21 rows in game 1874553), which never appears in the phases file.
* `player_possession.end_type`: pass, indirect_regain, possession_loss, indirect_disruption,
  direct_regain, shot, clearance, foul_suffered, foul_committed, direct_disruption, unknown.
* `start_type`: pass_reception, pass_interception, recovery, keep_possession,
  throw_in_reception, free_kick_reception, goal_kick_reception, corner_reception, the matching
  `*_interception` values, and unknown.
* `game_interruption_before/after`: {throw_in, free_kick, goal_kick, corner, goal,
  penalty} × {for, against}.
* `on_ball_engagement` subtypes: pressure, recovery_press, pressing, counter_press, other.
  `off_ball_run` subtypes include dropping_off, run_ahead_of_the_ball, support, etc.

## 4. The tracking `possession` field vs events and phases

The tracking field is set in 95–98% of ball-in-play frames, but:

* its team agrees with the phase-of-play team in only **77–88%** of in-play frames;
* hand inspection (game 1874553, frames 6000–7800) shows it holding one team for 41.6 s across
  a 4 s recovery spell by the other team that both events and phases report;
* debounced losses from this field (3,561) match an events-based loss within ±2 s only **34%**
  of the time. Conversely only 43% of events-based losses have a matching tracking-field loss.
  When it does switch, it lags the event-defined loss by a median of 0.0 s but a 90th
  percentile of **7.1 s**, and in 6.8% of losses it never switches within 30 s;
* `possession.player_id` is set in only 26–39% of in-play frames.

Dynamic events and phases of play (both from SkillCorner's Game Intelligence layer) agree with
each other much better (below), so **we propose events as the primary loss source and phases as
the cross-check.** Our null-gap bridging parameter turned out to be irrelevant: within play the
tracking field switches teams directly. Nulls occur only around dead balls.

## 5. Possession-loss candidates (counts across all 20 games)

| Definition | Losses | Agreement (±2 s, same losing team) |
|---|---|---|
| **A. Events (proposed primary):** consecutive team-possession runs of `player_possession` with no `game_interruption` between them, gaining team holds ≥ 1.0 s, ball in play (contiguous phases) from A's last touch to 1 s after B's first control | **2,818** | 83.1% found in phases. 77.7% of phase losses found in A |
| A with hold ≥ 0.5 s / ≥ 2.0 s | 2,883 / 2,563 | sensitivity |
| Raw event team changes (no debounce / in-play check) | 3,061 | |
| B. Phases of play: contiguous phases with a change of team | 3,017 | cross-check |
| C. Tracking `possession.group`, debounced 1 s, in play | 3,561 | 34% found in A (rejected, §4) |

* Loss time `t0` = frame of the gaining team's first control (`frame_start` of its first
  possession). We also store A's last touch (`frame_last_a`). The median gap between them is a
  few frames.
* The losing possession's last action: pass 1,942 (misplaced or intercepted), possession_loss
  658, clearance 109, shot 101, unknown 8.
* Losing team's possession before the loss: median 5.0 s, but **25% last < 1.3 s** (scrappy
  exchanges where the team never left its defensive shape). Gaining team's hold: median 7.3 s.
* Losses not confirmed by phases are mostly these short exchanges (median losing possession
  2.8 s vs 5.3 s for confirmed ones).
* Per team: 66 (Newcastle, 1 game) to 490 (Auckland, 7 games) losses.

## 6. Censoring and coverage of loss episodes

For the 2,818 primary losses, an open-play window runs until the first of: the losing team's
next control (regain), the ball going dead, period end, or a 20 s cap.

* Ends by regain 1,510 (54%), dead ball / period end 743 (26%), and reaches the 20 s cap 565 (20%).
* Fraction of episodes whose open-play window lasts at least 5 / 10 / 15 / 20 s:
  **66% / 41% / 29% / 20%**.
* Implication: T_r is observable only when reorganisation is fast. Censoring and competing
  risks are **essential** here. A naive "mean T_r of reorganised episodes" would be badly
  biased.

## 7. Detection (`is_detected`) and reliability

* In-play player-frames detected: 0.54–0.73 per game. By role: midfield 0.70–0.89, wide
  attackers 0.63–0.83, full backs 0.50–0.78, **central defenders 0.45–0.67**, GK 0.13–0.23.
  Detection barely differs between home/away or between periods.
* Over the first 10 s after a loss, the fraction of the losing team's outfield player-frames
  that are detected has quantiles (10/25/50/75/90%) of **0.36 / 0.51 / 0.65 / 0.77 / 0.87**.
* Implication: the deepest line, which drives line height and depth, is the most extrapolated
  part of the block. This is the strongest argument for the reliability layer (§5.6 of the
  brief). The 0.6 threshold placeholder in `config.yaml` sits near the median, so it would drop
  about 40% of episodes. It must be fixed before outcome analysis.

## 8. Organised-defence reference data

Frames where a team is out of possession in `low/medium/high_block`, tracked, and ≥15 s after
that team's last loss: **16 min (Newcastle) to 86 min (Auckland)** per team in total. Spread over
9 ball-context cells, many team × cell combinations will be thin. Hierarchical shrinkage toward
the league-pooled reference is required, not optional.

## 9. Anomalies and quirks

* No duplicate player-frames, no player ids missing from lineups, and always 11 per team
  including a GK. (Extrapolation fills in every player, so substitutions must be
  respected via lineup frames, and there were no red cards in this sample.)
* 0–266 player-frames per game lie more than 5 m outside the pitch, usually extrapolated
  players near the touchline. These are kept but flagged.
* Ball x/y is null in 0.3–4.2% of in-play frames, and the ball is detected in 92–96%.
* `period = null` trailing frames in 5 games, which are dropped.
* Phases of play cover only 45–63% of in-period frames (ball in play).
