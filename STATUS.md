# Status

_Last updated: 9 Oct 2026. GATE 2 passed (plan frozen). **Confirmatory results are in (`docs/confirmatory_generated.md`), waiting for Mark's direction on framing.**_

## Confirmatory results (9 Oct, Holm-corrected)
- H3 confirmed: goal-side count carries the most remaining disorganisation (32% of D² at +6 s,
  robust to the floor).
- H2 not confirmed at the primary 3 s landmark (OR 1.34 [0.95, 2.02]). Secondary analyses (log D at
  3 s, and both exposures at 6 s) show clear positive associations.
- H1 not confirmed: team differences are not distinguishable from chance with 20 games
  (permutation p = 0.23, Spearman-Brown 0.32).
- H4 not confirmed: no out-of-fold predictive gain over the covariates (ΔAUC +0.001).
  SkillCorner's phase label predicts better than D.

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
- Headline: 2,031 disorganised-at-loss episodes. Reorganised by 6 s: 30% [28, 31]. Regain first by
  6 s: 33%. Caveats: team-specific references are weakly reproducible, and team-level reliability
  of the metric is not yet shown (see report).

## Phase 2 (in progress), GATE 2 = freeze ANALYSIS_PLAN.md
- GATE 1 passed 9 Oct (GO, D-011 and D-015 approved). PR #1 merged, and the branch restarted
  from main.
- Done: D-015 flag + `team_summary` (share of losses while organised); league-only reference
  option; pre-registered sensitivity grid (`scripts/sensitivity.py`, all T_r rank ρ ≥ 0.85);
  `reorg.outcomes` (danger definition, landmark sample) and `reorg.stats` (H1 permutation and
  split-half, H2 cluster-bootstrap OR, H4 leave-one-match-out comparison, H3 shares). All tested
  on synthetic data only, and **not run on real data**. Public API `reorg.run_league` etc. +
  `reorg episodes` CLI; end-to-end synthetic smoke test in CI. 48 tests.
- GATE 2 passed 9 Oct: plan frozen (baseline c / rβ dropped per the plan rule).
- After the freeze: the per-fold reference rebuild for H4, then running H1–H5.

## Blockers / questions for Mark
1. **Related-work papers + spec PDFs**: www.mdpi.com, arxiv.org and ncbi.nlm.nih.gov are blocked by
   this environment's network policy (needed for the rβ baseline, H4c). Also: SkillCorner's HubSpot links return 403 from this environment. Please download
   the Dynamic Events and Phases of Play spec PDFs into `docs/specs/`. (Not blocking Phase 1.)
2. **`opendata/` in git history**: removed from the tree. It is still in commit `ca6fd71`. Purge it
   with a history rewrite + force-push before going public? (Your call.)
3. **Rules needing a human decision before submission**: AI-assistance disclosure, and any
   employer/university permission for a competition entry. To resolve by GATE 4.
