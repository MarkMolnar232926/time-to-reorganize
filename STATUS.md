# Status

_Last updated: 9 Oct 2026, Phase 0 (setup + data audit). **Waiting at GATE 0.**_

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

## Next (Phase 1, after GATE 0 approval)
See `docs/PHASE1_PLAN.md`.

## Blockers / questions for Mark
1. **Spec PDFs**: SkillCorner's HubSpot links return 403 from this environment. Please download
   the Dynamic Events and Phases of Play spec PDFs into `docs/specs/`. (Not blocking Phase 1.)
2. **`opendata/` folder committed in this repo** (commit `ca6fd71`): it includes SkillCorner
   CSV/JSON data. The submission repo must not contain data. We need to delete it, and ideally
   rewrite history before the repo is made public. Your call (rewriting history needs a force-push).
3. **Rules needing a human decision before submission**: AI-assistance disclosure, and any
   employer/university permission for a competition entry. To resolve by GATE 4.
