# Decisions log

Format: **ID. Decision.** Alternatives considered. Reason. *(status)*

---

**D-001. Package name `reorg`, MIT licence, Python ≥ 3.11, uv for locking.**
Alternatives: `timetoreorg`, `defreorg`; Apache-2.0; pip-tools.
`reorg` is short and matches the brief. MIT is the simplest permissive licence for the
"open-source in perpetuity" rule. uv gives a fast, cross-platform lockfile. *(adopted)*

**D-002. Data is fetched from the upstream SkillCorner repo at a pinned commit and never committed.**
Alternatives: use the `opendata/` copy already committed in this repo; vendor the data.
Competition rules forbid redistributing data in the submission repo. Pinning commit
`4340d27` makes the run reproducible. `scripts/download_data.py` uses only the standard library
(no git, git-lfs or curl, so it also works on Windows and in a slim container). Small files come
from raw.githubusercontent.com. Tracking files (Git LFS) come from media.githubusercontent.com
and are verified against the SHA-256 in their LFS pointer. *(adopted; `opendata/` removed from the tree on 9 Oct. It still exists in
git history (commit `ca6fd71`). Purging it requires a history rewrite + force-push, which is Mark's call before the repo goes public)*

**D-003. Thin custom parser instead of kloppy.**
Alternatives: kloppy's SkillCorner loader.
We need `is_detected` per player-frame, the raw `possession` block and `image_corners_projection`
untouched, plus a cached parquet in long format. The parser is ~150 lines, parses all 20 games
in ~20 s with 4 processes, and adds no dependency. *(adopted)*

**D-004. Primary possession-loss source = dynamic events, not the tracking `possession` field.**
Alternatives: (a) tracking `possession.group` (the brief's default); (b) phases of play; (c) events.
The audit (docs/DATA_AUDIT.md §4–5) shows the tracking field lags and misses short spells. Only
34% of its debounced losses match an event-defined loss within ±2 s, and its switch lags by
p90 = 7 s. Events and phases agree at 78–83%. Events give frame-accurate first-control times and
the losing action type. Phases are coarser, so they serve as the cross-check. Rule: consecutive
team-possession runs with no game interruption between them, gaining team holds ≥ 1.0 s,
contiguous phases (ball in play) from A's last touch through B's first control + 1 s.
`t0` = B's first control. *(approved at GATE 0, 9 Oct)*

**D-005. Ball-in-play = inside a run of contiguous phases of play (gap ≤ 1 frame).**
Alternatives: ball detected / ball x not null; event interruptions only.
SkillCorner defines phases only while the ball is in play, and consecutive phases share their
boundary frame at a change of possession. *(adopted)*

**D-006. Keep short losing possessions (< 2 s) in the episode set but record `a_possession_s`.**
Alternatives: drop them (a team that never left its block has nothing to reorganise).
32% of losses follow a possession under 2 s. Dropping them would select on shape. Instead we
keep them and use `a_possession_s` as a stratifier / covariate, and decide at GATE 1 using
D(t0) itself, which measures how disorganised the team actually is. *(approved at GATE 0, 9 Oct)*

**D-007. Losses that follow a shot or clearance are kept but flagged via `a_end_type`.**
101 follow a shot and 109 a clearance. They are real transitions in a different context, so the
primary analysis keeps them and a sensitivity analysis excludes them. *(proposed)*

**D-008. Event coordinates are converted to tracking coordinates by mirroring when
`attacking_side == right_to_left`. No rescaling is applied.**
Verified empirically (median error ≤ 0.17 m after mirroring vs up to 9.5 m without). *(adopted)*

**D-009. Docker image needs no apt packages. A `BASE` build arg allows a different base image.**
The downloader is pure Python, so `python:3.11-slim` + `pip install uv` + `uv sync --frozen` is
enough. In this sandbox the build was verified on a locally derived base image that trusts the
sandbox proxy CA. Docker Hub and Debian mirrors are blocked or rate-limited here. *(adopted)*

**D-010. Prototype match for Phase 1 = 2007721 (Auckland FC v CC Mariners).**
Alternatives: 1953632 (highest detection), 1996435 (has pose).
Best in-play tracking coverage, above-median detection, and the two best-covered teams
(7 and 4 games). *(proposed)*

---
## Phase 1

**D-011. The reference is interpolated continuously in ball position (bilinear between the 9
context-cell centres, clamped at the outer centres).**
Alternatives: the discrete cell reference from the brief (§5.4).
Face-validity review of the first prototype runs showed D(t) jumping by 1–3 units when the ball
crossed a channel boundary, because the target's lateral offset jumped by up to 14 m in 0.4 s.
That is an artefact of the method, not of the defending. Cells are still used to *build* the
reference and as covariates. The discrete version is kept as a sensitivity analysis
(`z_scores(..., continuous=False)`). Disclosure: after the switch, the exploratory odd/even
correlation of team P(reorganised by 6 s) fell from 0.49 to 0.18 (12 teams, both within noise).
The change was made for face validity before that number was seen, and we keep it. *(adopted,
review at GATE 1)*

**D-012. D(t) is the weighted RMS of z (divided by Σw), not the root of the weighted sum.**
This keeps D in "typical deviations per component" whatever the number of components, so τ is
comparable across component sets (τ ≈ 1.0–1.15 for all teams). *(adopted)*

**D-013. T_r is the start of the first D ≤ τ stretch lasting `hold_s`, and the whole hold must
finish inside the open-play window.** For episodes that never reorganise, the analysis time is
`window − hold`, the last moment a confirmed hold could have started. A regain (competing event)
is timed at the regain itself. A first version used `window − hold` for regains too, which put
7% of regains at t = 0; this was caught in the GATE 1 figure review and fixed. Frames with missing D
break a hold. Tested on synthetic traces: recovery within ±0.2 s with noise, and exact
without it. *(adopted)*

**D-014. Window ends: regain by the losing team is a competing event. Opponent shot, dead ball,
period end and the 20 s cap are censoring.** The shot is kept as a separate `end_reason`
because it is outcome-related. Its treatment in the outcome models goes into ANALYSIS_PLAN.md.
*(adopted for Phase 1)*

**D-015. Proposed: survival analysis of T_r is restricted to losses where the team is
disorganised at the moment of loss (D0 > τ, 72%).** The other 28% (D0 ≤ τ) have nothing to
reorganise. They are reported as a separate team-level quantity: "share of losses suffered while
organised". D0 rises with the length of the losing possession, as expected (median 1.29 for
possessions < 1 s vs 1.78 for > 20 s), which supports D-006. *(proposed at GATE 1)*

**D-016. Phase 1 reliability regimes use placeholders (frame: ≥ 50% of defenders detected;
episode: mean ≥ 0.6 over the first 10 s).** They will be fixed in ANALYSIS_PLAN.md before any
outcome model. *(provisional)*
