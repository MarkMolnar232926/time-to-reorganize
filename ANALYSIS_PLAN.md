# Analysis plan

**Status: FROZEN on 9 Oct 2026 (GATE 2, approved by Mark).** The freeze commit is the first
commit that contains this line, and it precedes every run of the outcome code (verifiable in git
history). Every change after this point is appended to the changelog with date and reason, and
every result is labelled **confirmatory** (pre-specified here) or **exploratory**.

## 0. What has and has not been seen before writing this plan
*Seen (no outcome association):* loss counts, the distributions of D(t), T_r, censoring and
competing regains, team cumulative incidences, the reliability-regime and sensitivity-grid
summaries (`docs/gate1_generated.md`, `docs/sensitivity_generated.md`), and the number of
episodes whose window ends with an opponent shot (191 of 2,818). This is a censoring reason we
needed for T_r.
*Not seen:* the box-entry outcome, the prevalence of "dangerous attack", and **any association
between an exposure (T_r, D, baselines) and any outcome**. The outcome code
(`reorg/outcomes.py`) is unit-tested on synthetic data only and has not been run on real matches.

## 1. Data and analysis population
* 20 A-League 2024/25 games (SkillCorner open data, pinned commit `4340d27`), 13 teams.
* Losses: D-004 (dynamic events, open play, gaining team holds ≥ 1 s): 2,818.
* **Analysis population: losses with D0 > τ (D-015): 2,031.** The share of losses suffered while
  organised is reported per team as a descriptive quantity.
* Episode counts are reported at every filtering step. If an analysis sample has fewer than 500
  episodes, it is flagged.

## 2. Metric, fixed as in `config.yaml` at the freeze
Continuous team reference (D-011), shrinkage k = 600 frames, 7 components with equal weights,
spread floors as in config. τ = 75th percentile of the team's organised-frame D. Hold 2 s.
Horizon 20 s. Regain is a competing event, and shot / dead ball / period end / cap are
censoring (D-013, D-014). Reliability regimes: frame ≥ 50% of defenders detected; episode mean
≥ 0.6 over the first 10 s (D-016). These thresholds are now fixed.

## 3. Outcome (§O)
**Dangerous attack conceded:** within 15 s after the loss, while the gaining team still has the
ball in open play (before the losing team's next control or a dead ball), either (a) a
gaining-team shot (`player_possession.end_type == "shot"`), or (b) a **box entry**: the ball
goes from outside to inside the losing team's penalty area (16.5 m × 40.32 m) and stays at least
0.2 s (2 frames). If the loss happens inside the box, only (a) counts. Implementation:
`reorg.outcomes.first_danger` / `add_danger`.

## 4. Why landmarks, not T_r directly
A dangerous attack ends or shortens the observation of reorganisation, so regressing danger on
T_r would be biased (immortal-time / reverse-causation). All outcome analyses therefore use a
**landmark design**. At landmark ℓ, the sample is episodes in the analysis population that are
still in open play at ℓ and danger-free up to ℓ. Exposure is measured at ℓ, and the outcome is
danger in (ℓ, 15 s]. **Primary ℓ = 3 s**; secondary ℓ = 6 s.

## 5. Hypotheses and confirmatory tests

**H1 (confirmatory): teams differ reliably in how fast they reorganise.**
* *Statistic:* variance across teams of P(reorganised by 6 s), the Aalen–Johansen cumulative
  incidence.
* *Test:* match-preserving permutation. Within every match, the two teams' labels are swapped
  with probability 1/2. 5,000 permutations, one-sided p.
* *Reliability:* odd/even-match split-half r of team P(reorganised by 6 s), with the
  Spearman–Brown correction (12 teams with ≥ 2 games). Pre-set reading: SB ≥ 0.5 is "moderately
  reliable", and below that we say team rankings are not yet reliable with 20 games.
* *Context (exploratory):* cumulative incidence by loss third, with match-bootstrap CIs.

**H2 (confirmatory): being still disorganised at 3 s predicts conceding a dangerous attack.**
* Logistic model on the ℓ = 3 s landmark sample:
  `danger ~ still_disorganised_3s + ball_x_at_loss + wide_channel_at_loss + goal_side_at_loss
  + ball_speed_first_3s + log1p(losing_possession_s)` (6 terms).
  `still_disorganised_3s` = D(3 s) > τ.
* *Effect:* odds ratio of `still_disorganised_3s` with a 95% match-cluster percentile bootstrap
  CI (2,000 resamples), and a cluster-robust Wald p (clusters = matches) for multiplicity.
* *Direction is pre-specified:* OR > 1.
* *Secondary:* log D(3 s) per SD instead of the binary exposure, and the same models at ℓ = 6 s.
* *Events-per-parameter rule:* if the landmark sample has fewer than 60 dangerous attacks, the
  model is reduced to exposure + ball_x + goal_side, and that is reported.

**H3 (confirmatory): one component carries most of the remaining disorganisation.**
* Among analysis-population episodes still late (D > τ) at +6 s: mean share of D² per component,
  with 95% match-bootstrap CIs (2,000).
* *Confirmed* if the CI of (largest share − second largest) excludes 0. It counts as *robust*
  only if the same component is largest with the goal-side spread floor at 1.0 (Phase 1 preview:
  goal-side ≈ 32%, or 24% with floor 1.0).
* Shares at +3 s and +10 s are reported descriptively.

**H4 (confirmatory): the metric adds predictive value beyond baselines.**
* *Validation:* leave-one-match-out (20 folds). In each fold, **the reference and τ are rebuilt
  from the 19 training matches**, and the held-out match's exposures are recomputed from them.
* *Models,* fitted on the ℓ = 3 s landmark sample:
  * **M_zone** (baseline a): ball_x + wide_channel at loss.
  * **M_cov**: the H2 covariates without the exposure.
  * **M_D**: M_cov + `still_disorganised_3s`.
  * **M_compact** (baseline b): M_cov + mean NN distance over 0–3 s.
  * **M_phase** (baseline d, event-based transition): M_cov + `phase_organised_at_3s`, which
    asks whether SkillCorner's phase label already says low/medium/high block.
  * **M_rβ** (baseline c): M_cov + an rβ-style angular-concentration indicator implemented from
    its published description. *Pending:* the paper is not reachable from this environment. If
    the description is not available at the freeze, baseline c is dropped and reported as not
    done. It will not be replaced post hoc.
    **At the freeze (9 Oct 2026) the description was not available (network policy blocks the
    publisher), so baseline c is dropped.**
* *Primary comparison:* out-of-fold log-loss of M_D − M_cov, with a 95% match-cluster bootstrap
  CI (2,000). Adds value if the upper bound is < 0.
* *Secondary:* M_D vs M_compact and M_D vs M_phase, head-to-head and as additions
  (M_compact + D, M_phase + D). Brier and AUC are also reported, as are M_zone and M_cov on their own.

**H5 (confirmatory robustness): conclusions hold across reliability regimes.**
The H2 primary OR and the H4 primary Δlog-loss are recomputed under the `all`, `detected` and
`reliable` regimes. Robust if all three agree in sign and CI exclusion. Otherwise we report
exactly where the conclusion changes.

**Multiplicity:** Holm correction across the four primary tests (H1 permutation, H2 OR, H3
gap, H4 Δlog-loss) at family-wise α = 0.05. Effect sizes with intervals are always reported,
not only p-values.

## 6. Pre-registered sensitivity grid (reported for the descriptive summaries and for the H2/H4 primary effects)
τ quantile {0.70, 0.80}; hold {1.5, 3} s; discrete vs continuous reference; league-only
reference; goal-side floor 1.0; no goal-side component; excluding losses after a shot or
clearance (D-007). The descriptive part has been run for every variant except the shot/clearance
exclusion (`docs/sensitivity_generated.md`):
episode-level T_r rank correlation with the base run is 0.85–0.99 for every variant, and team
rankings stay at ρ ≥ 0.92 except for τ = 0.70 (0.76), the discrete reference (0.80) and the
league-only reference (0.75).

## 7. Exploratory (labelled as such in all outputs)
Cause-specific Cox models (reorganisation vs regain) with the H2 covariates; a time-varying
model of the danger hazard on D(t); per-team component profiles; reorganisation debt as an
alternative exposure; the body-pose stretch (2 games); and whether reorganisation speed depends
on the length of the losing possession.

## 8. Rules
No threshold is tuned against outcomes. Seeds are fixed (`config.yaml: prototype.seed`). Every
number in the README is regenerated by `scripts/run_all.sh`. Deviations from this plan are
logged below and in `DECISIONS.md`, and results affected by a deviation are labelled.

## Changelog
- 2026-10-09: draft created.
- 2026-10-09: Phase 1 notes added (no outcome data used).
- 2026-10-09: full draft for GATE 2 (landmark design, outcome definition, H1–H5 tests, CV
  protocol, sensitivity grid). Still no outcome data used.
- 2026-10-09: **FROZEN** (GATE 2 approved). Baseline c (rβ) dropped under the rule above.
