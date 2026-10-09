# Phase 1 plan: one-match prototype (16–29 Oct 2026), proposed at GATE 0

**Prototype match:** 2007721 Auckland FC v CC Mariners. It has the best tracking coverage
(99.2% of in-play frames), above-median detection (0.69), and the two teams with the most games
(7 and 4), so their reference blocks are the best supported. References are built from *all*
of a team's games. This uses shape data only and no outcomes, so it is not tuning on held-out data.

## Build order (each step with synthetic tests)
1. **`normalise`**: per period, flip coordinates so the *defending* team attacks +x (own goal at
   −x). Drop GKs (role `GK`) from outfield shape. Respect lineup on-pitch frames (substitutions).
   Velocities via Savitzky–Golay are used only where needed (opponent ball speed, first 3 s).
2. **`shape`**: vectorised S(t) per team-frame for the 10 outfield players: line height (mean x
   of the deepest 4, measured from the own goal line), block depth and width (P90–P10), centroid–ball
   offset (x, y), compactness (median nearest-neighbour distance and convex-hull area),
   goal-side count. Plus the per-frame detected fraction of the 10.
3. **`reference`**: organised frames = team out of possession in `low/medium/high_block`,
   ≥15 s after that team's last loss, ball in play, tracked. Context = ball channel (3 lateral
   bands) × ball depth third in the defending frame. Robust centre/spread (median / MAD) per
   team × context, with empirical-Bayes shrinkage to the league-pooled cell weighted by frame
   count. Split-half (odd/even games) stability check.
4. **`reorganisation`**: D(t) against the *time-varying* reference R(c(t)), since the target
   moves with the ball. τ = 75th percentile of the team's own organised-frame D. T_r = first time
   D ≤ τ holding for 2 s. Censoring reason ∈ {regain (competing), shot, dead ball, period end,
   20 s cap}. Also reorganisation debt (∫D over 15 s) and per-component contributions at
   +3/+6/+10 s.
5. **`reliability`**: per-frame detected fraction among the 10. Episode score = mean over the
   first 10 s. T_r recomputed under: all frames / detected-only components / reliable episodes.
6. **`viz`**: pitch snapshots (loss, +3 s, +6 s) with the actual hull vs the reference block, D(t)
   curves with τ, and MP4/GIF animations from tracking coordinates (ffmpeg is available).

## GATE 1 evidence I will bring (29 Oct)
- Episode funnel across all 20 games: losses → tracked → reference available → T_r observed vs
  censored by reason.
- Cumulative-incidence curves (reorganised vs regained vs censored) for the prototype teams and
  the league.
- 6 fastest / 6 slowest / 6 random animations for face validity (you eyeball them).
- Synthetic recovery tests: known convergence times recovered within ±0.2 s.
- Distribution of D(t0) by losing-possession length (relevant to D-006).
- No outcome (danger) analysis. That waits until ANALYSIS_PLAN.md is frozen (GATE 2).

## Risks and fallbacks
- **Censoring dominates** (only 41% of windows reach 10 s). If few T_r are observed, the primary
  metric becomes D at fixed horizons (+3/+6 s, observed for 66% at 5 s) plus reorganisation debt,
  with T_r reported via cumulative incidence (competing risks) rather than means.
- **Thin reference cells** (16–86 organised minutes per team): fall back to 3 depth contexts only,
  or the league reference plus a team offset.
- **Extrapolated back line**: if line-height conclusions flip between reliability regimes, we
  report that honestly and weight components by detection.
