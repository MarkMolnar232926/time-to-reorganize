# GATE 1 report: one-match prototype (go/no-go)

*Prepared 9 Oct 2026, ahead of the 29 Oct gate. Generated numbers are in
[`gate1_generated.md`](gate1_generated.md). Rerun with `python scripts/phase1_prototype.py`
(~2 min, or ~10 min with the 18 animations). No outcome (danger) variable has been touched.*

## Questions for the gate

**1. Do fast and slow reorganisations separate in a way a coach would recognise?**
Provisionally yes, but you need to eyeball the animations (`outputs/phase1/anim/`: 6 fast,
6 slow, 6 random from 2007721 Auckland v CC Mariners). Things I checked:

* **D at loss behaves sensibly.** It rises with the length of the possession just lost: median
  1.29 after possessions < 1 s vs 1.78 after > 20 s. The share of losses where the team is
  disorganised rises from 58–65% to 87%. Teams that have committed players forward are further
  from their block.
* **The worked example (frame 18757).** Auckland lose the ball high on the left. D climbs to a
  peak at +5 s when the ball is switched behind their high block, then falls steadily as they
  drop back, crossing τ at 10.7 s. This is the kind of sequence a coach would point at.
* **Fast episodes** are losses where D at loss was only just above τ (1.1–1.6) and the team
  settled within about 1 s, whatever the length of the lost possession (0–17 s). The **three
  slowest reorganised episodes** all followed long possessions (20–31 s). The three censored
  slow examples are mixed (possessions of 2–37 s), so don't read a pattern into them.

**2. Are there enough episodes across 20 games?** Yes.

| step | episodes |
|---|---|
| primary losses (D-004) | 2,818 |
| disorganised at loss (D0 > τ) | 2,031 |
| → reorganised within the window | 777 (38%) |
| → regained the ball first (competing event) | 830 (41%) |
| → censored (dead ball / shot / 20 s cap) | 424 (21%) |

League cumulative incidence of reorganisation (95% match-bootstrap CI): **18% [16, 20] by 3 s,
30% [28, 31] by 6 s, 39% [37, 41] by 10 s, 44% [42, 47] by 15 s**. Regain-first runs alongside
it at 16 / 33 / 43 / 48%. The
competing-risks framing is necessary: a naive mean T_r over reorganised episodes would ignore
the 41% that regain first.

**Recommendation: GO**, with the caveats below.

## What changed during Phase 1 (logged in DECISIONS.md)
* **D-013 fix.** Regains were first timed at `window − hold`, which put 7% of them at t = 0. They
  are now timed when the regain happens. This was caught while reviewing the CIF figure.
* **D-011, continuous reference.** With the brief's discrete 3×3 contexts, D(t) jumped whenever
  the ball crossed a cell boundary. The reference is now interpolated in ball position.
* **D-015 (needs your approval).** Analyse T_r only for losses where the team is disorganised
  at loss (72%), and report "share of losses suffered while organised" separately per team.

## Honest caveats

1. **Team-specific references are only weakly reproducible.** Odd/even-match correlation of
   each team's deviation from the league block, by component: line height 0.39, centroid-ball
   x 0.31, width 0.29, goal-side 0.22, and depth / NN distance / lateral offset ≤ 0.12. Most
   of the block's shape is explained by ball context. The team-specific layer is small and noisy
   with 1–7 games per team. This doesn't break the metric, since shrinkage handles thin cells,
   but it weakens the "team-referenced" claim. A sensitivity run with the league reference only
   is a must for Phase 3.
2. **Team-level differences in reorganisation speed are not yet shown to be reliable.**
   Exploratory odd/even-match correlation of team P(reorganised by 6 s): 0.19 (12 teams; 0.49
   before D-011). With 12 teams that is noise either way. H1 needs an episode-level model
   (team random effect with an ICC), not a correlation of 12 points. The raw team spread is
   wide, though (P(reorg by 6 s) from 0.18 to 0.41).
3. **Reliability matters.** In the "detected" regime, where D is undefined when < 50% of
   defenders are detected, reorganisation by 6 s falls from 0.30 to 0.24, because missing
   frames break holds. The "reliable episodes" regime leaves it unchanged (0.30). The
   back-line components are the most extrapolated.
4. **Decomposition preview (descriptive only).** Among late episodes, goal-side count carries
   the largest share of D² (32% at +6 s). That share is sensitive to its spread floor (24% with
   floor 1.0), but the metric is not: dropping goal-side entirely leaves T_r ranks almost
   unchanged (Spearman 0.91), and D at +6 s gives 0.96. Depth's share grows with time
   (11% → 20% from +3 s to +10 s): the block gets players behind the ball first and closes up
   vertically later. H3 should be framed as a component ranking with its sensitivity shown.
5. **Counter-pressing is not "disorganisation".** A team that counter-presses is deliberately far
   from its settled block, so its D stays high until it regains the ball. The competing-risks
   framing handles this (a successful counter-press is a regain), but the README must not imply
   that high D is always bad.
6. **Reference fitted on all games.** It uses shape only and no outcomes. For H2/H4
   cross-validation I propose rebuilding it inside each training fold anyway (cheap, and it
   removes any doubt).

## Fallback if you judge NO-GO
Reframe around D at fixed horizons (+3 / +6 s, observed in 66% / 41% of episodes) and
reorganisation debt, with T_r as a secondary summary.

## Phase 2 plan (30 Oct – 19 Nov), if GO
1. Freeze `ANALYSIS_PLAN.md` by 5 Nov. That covers the outcome definition (shot or box entry
   within 15 s), the H1 episode-level model (team random effect, ICC with match-cluster
   bootstrap), the H2 covariate set (≤ 6), leave-matches-out CV with the reference rebuilt per
   fold, the baselines for H4, and fixed reliability thresholds.
2. Survival: cause-specific Cox models (reorganisation vs regain) and the censored-version KM.
3. Package API: `find_losses`, `build_reference`, `time_to_reorganise`, `reliability_report`,
   `plot_reorganisation`, plus `reorg` CLI subcommands. Add the reference/episode stages to
   `run_all.sh`.
4. Sensitivity grid (pre-registered): τ quantile 0.70/0.75/0.80, hold 1.5/2/3 s, discrete vs
   continuous reference, league-only reference, goal-side floor.
