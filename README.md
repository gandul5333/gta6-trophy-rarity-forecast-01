[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gandul5333/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gandul5333/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast — make.fun gta6-trophy-rarity-forecast-01

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, its earn rate
on Exophase at 2026-11-26T00:00:00Z, as five quantiles in percent. Entry
`main.ts` (Node 22, stdlib only, no network, deterministic).

**Approach:** counts, not rates. A trophy's earn rate is `r = E / N`, where `N`
is the players Exophase tracks and `E` the tracked players who earned the
trophy. The house baselines forecast the rate itself; that fails whenever
Exophase refreshes its tracked population, because a refresh grows `N` in a
jump while `E` barely moves, so every rate drops at once. This program fits
the two counts separately, both as power laws in `ln(1 + hours since
release)`, by least squares on the case's distinct snapshots (consecutive
reads with identical rates collapse to one point, so a frozen page does not
overweight the flat stretch), then evaluates both at the target hour and
divides. Slopes are shrunk toward their tier median, blended with a
recent-half fit, and capped; medians are capped at a per-tier plausible
week-one ceiling. The spread combines the fit residual, a term growing with
the horizon, and an allowance for a refresh between the last read and the
target.

When the history carries no signal (one distinct snapshot, as on launch
night when every read is the same frozen page), the program falls back to a
refresh prior: the page's first refresh multiplies the tracked population
several-fold while earners barely move, so rates fall to roughly a quarter
of their last read, with a wide spread. The ~100% trophy is forecast to
stay at ~99.99%.

**Validation (local):** on the practice cases the program scores a mean
pinball loss of ~0.29 on the logit scale against reconstructed practice
answers (the house baseline scores ~1.51–1.69 on the same answers — the
practice answers themselves are not in the kit, so they were reconstructed
from the known 04:30 refresh, 638 → 3,868 players tracked, and the live
Exophase page). On two synthetic final cases with rich histories (one where
the tracked population keeps exploding and rates fall, one where it
saturates and rates rise) it scores 0.12 and 0.72, against the baseline's
0.25 and 0.75. Two runs are byte-identical; runtime ~0.25 s per case.

**Run it:** `mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts`
