# Status

_Last updated: 9 Oct 2026. GATE 0 passed (D-004, D-006 approved). Phase 1 prototype built; **GATE 1 evidence ready (`docs/GATE1_REPORT.md`), waiting for review.**_

## Done
- Repo skeleton: `src/reorg/` package, `pyproject.toml` + `uv.lock`, `config.yaml`, tests, CI,
  Makefile, `scripts/run_all.sh`, MIT `LICENSE.md`, `.gitignore` (excludes `data/`, `outputs/`).
- `scripts/download_data.py` (stdlib only): pinned upstream commit, LFS tracking fetched over
  HTTPS with a SHA-256 check (1.8 GB, 20 games).
- Clean reproduction verified: a fresh Docker container (python:3.11-slim, run with the
  sandbox CA) downloads the data, passes the tests and reproduces the audit and the 2,818 losses
  in ~65 s. CI workflow mirrors this without data.
- `reorg.io`: match/tracking/events/phases parsers with a parquet cache (20 games in ~20 s).
- `reorg.turnovers`: events-based loss detector (proposed primary) plus phases and tracking-field
  cross-checks. 11 synthetic tests.
- `scripts/audit_data.py` → `docs/DATA_AUDIT.md` (narrative) + `docs/audit_generated.md`.

## Key findings
- 20 games, 13 teams (1–7 games each). Every game has tracking, events and phases. Pose for 2.
- Tracking `possession` field is too noisy for loss detection → propose events (D-004).
- 2,818 primary losses (≥500 target met). 97% are well tracked in the first 10 s.
- Censoring is heavy: only 41% of episodes have ≥10 s of open play. Survival/competing-risk
  framing is essential.
- Central defenders are the least-detected outfield role. The reliability layer matters.

## Phase 1 (done ahead of schedule)
- `normalise`, `shape`, `reference` (shrunk team × context, continuous in ball position),
  `reorganisation` (D(t), τ, T_r with hysteresis, censoring/competing regain, debt,
  decomposition), `reliability` (3 regimes), `survival` (KM, Aalen–Johansen, match bootstrap),
  `viz` (snapshots, D curves, CIF, MP4 animations). 32 tests.
- `scripts/phase1_prototype.py` → `docs/gate1_generated.md` + `outputs/phase1/` (18 face-validity
  animations from 2007721). Whole league in ~2 min without animations.
- Headline: 2,031 disorganised-at-loss episodes. Reorganised by 6 s: 30% [28, 32]. Regain first by
  6 s: 37%. Caveats: team-specific references are weakly reproducible, and team-level reliability
  of the metric is not yet shown (see report).

## Next
- GATE 1: you review the animations and the report, and decide D-015 (restrict T_r to D0 > τ).
- Then Phase 2: freeze ANALYSIS_PLAN.md (by 5 Nov), episode-level H1 model, sensitivity grid,
  package API.

## Blockers / questions for Mark
1. **Spec PDFs**: SkillCorner's HubSpot links return 403 from this environment. Please download
   the Dynamic Events and Phases of Play spec PDFs into `docs/specs/`. (Not blocking Phase 1.)
2. **`opendata/` in git history**: removed from the tree. It is still in commit `ca6fd71`. Purge it
   with a history rewrite + force-push before going public? (Your call.)
3. **Rules needing a human decision before submission**: AI-assistance disclosure, and any
   employer/university permission for a competition entry. To resolve by GATE 4.
