# Analysis plan

**Status: DRAFT, not frozen.** To be frozen by 5 Nov 2026 (GATE 2), before any outcome model
is fitted. After freezing, every change is appended to the changelog at the bottom with a
date and reason, and analyses are labelled *confirmatory* or *exploratory*.

## Hypotheses (from the brief; operational details to be filled in after Phase 1)
- **H1 (confirmatory):** T_r differs reliably between teams/contexts (odd/even-match split, ICC).
- **H2 (confirmatory):** longer T_r → higher P(dangerous attack within 15 s), adjusted for loss
  location, goal-side count at loss, opponent ball speed in the first 3 s, and ball channel.
- **H3 (confirmatory):** one component dominates lateness (decomposition at +3/+6/+10 s).
- **H4 (confirmatory):** T_r adds value beyond baselines (loss zone; fixed-window compactness;
  rβ-style angular concentration; event-based transition duration).
- **H5 (confirmatory):** conclusions hold across the reliability regimes.

## Known constraints from the audit
- ~2.8k episodes, heavy censoring (41% have ≥10 s windows). Regain is a competing event.
- 13 teams with 1–7 games each. Newcastle has a single game, so it cannot enter split-half
  reliability by match.

## Notes from Phase 1 (to be turned into fixed rules at the freeze)
- Analysis population: losses with D0 > τ (D-015, pending approval). Report the "organised at
  loss" share separately.
- H1 must use an episode-level model (team random effect, ICC, match-cluster bootstrap): 12 teams
  are too few for a correlation of team means.
- H3: report component *ranking* with sensitivity to the spread floors and weights. Goal-side
  share moves from 32% to 24% with its floor, while T_r ranks are robust (Spearman 0.91 without
  goal-side).
- Rebuild the reference inside each CV training fold for H2/H4.
- Pre-registered sensitivity grid: τ quantile {0.70, 0.75, 0.80}, hold {1.5, 2, 3} s,
  discrete vs continuous reference, league-only reference, goal-side floor {0.5, 1.0}.

## Changelog
- 2026-10-09: draft created.
- 2026-10-09: Phase 1 notes added (no outcome data used).
