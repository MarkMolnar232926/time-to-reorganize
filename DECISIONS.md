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
and are verified against the SHA-256 in their LFS pointer. *(adopted; the existing `opendata/` folder
in git history is an open question for Mark, see STATUS.md)*

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
`t0` = B's first control. *(proposed, awaiting GATE 0)*

**D-005. Ball-in-play = inside a run of contiguous phases of play (gap ≤ 1 frame).**
Alternatives: ball detected / ball x not null; event interruptions only.
SkillCorner defines phases only while the ball is in play, and consecutive phases share their
boundary frame at a change of possession. *(adopted)*

**D-006. Keep short losing possessions (< 2 s) in the episode set but record `a_possession_s`.**
Alternatives: drop them (a team that never left its block has nothing to reorganise).
32% of losses follow a possession under 2 s. Dropping them would select on shape. Instead we
keep them and use `a_possession_s` as a stratifier / covariate, and decide at GATE 1 using
D(t0) itself, which measures how disorganised the team actually is. *(proposed)*

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
