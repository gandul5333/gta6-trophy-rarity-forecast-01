// gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's earn rate on Exophase
// at the case's target, as five quantiles in percent.
//
// Model (counts, not rates). A trophy's earn rate is r = E / N, where N is the
// players Exophase tracks and E the tracked players who earned the trophy. The
// house baselines forecast the rate itself; that fails whenever Exophase
// refreshes its tracked population, because a refresh grows N in a jump while
// E barely moves, so every rate drops at once. This program instead fits the
// two counts separately, both as power laws in ln(1 + hours since release):
//
//   ln N(h)  = aN + bN * ln(1 + h)
//   ln E_i(h) = a_i + b_i * ln(1 + h)
//
// fitted by least squares on the case's distinct snapshots (consecutive reads
// with identical rates collapse to one point, so a frozen page does not
// overweight the flat stretch), then evaluated at the target hour and divided:
// r_i* = 100 * E_i* / N*. A trophy whose earners grow faster than the tracked
// population sees its rate rise; one whose earners stall sees it fall. Slopes
// are shrunk toward their tier median and blended with the recent-half fit,
// so one noisy trophy cannot run away, and a decelerating population does not
// extrapolate its launch-night explosion.
//
// When the history carries no signal (one distinct snapshot, as on launch
// night, when every read is the same frozen page), the program falls back to a
// refresh prior: the page's first refresh multiplies the tracked population
// several-fold while the earners barely move, so rates fall to roughly a
// quarter of their last read, with a wide spread because the size of the
// drop is uncertain. The one trophy at ~100% (everyone earns it on starting
// the game) is forecast to stay at ~99.99%.
//
// Spread: the quantiles are logistic(logit(median) + z_tau * sigma), where
// sigma combines the trophy's fit residual, a term growing with the horizon,
// and an allowance for a refresh between the last read and the target.
//
// Node 22, standard library only. Reads every *.in in the input directory
// (MAKEFUN_INPUT_DIR or /input), writes one *.out per case to the output
// directory (MAKEFUN_OUTPUT_DIR or /output). Deterministic: pure least
// squares, no randomness, so two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release instant: the cohort clock starts here. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

/** Refresh prior (flat history): median rate = last rate * DROP. */
const DROP_BASE = 0.22;
/** Extra drop relief for the highest-rate (earliest) trophies. */
const DROP_RATE_TILT = 0.1;
/** Spread of the refresh prior, in logit units. */
const PRIOR_SIGMA = 1.05;
/** Spread floor / cap for fitted forecasts, in logit units. */
const SIGMA_MIN = 0.7;
const SIGMA_MAX = 1.6;
/** Horizon term: 0.30 logit units per sqrt(day) of horizon. */
const HORIZON_SIGMA_PER_SQRT_DAY = 0.3;
/** Allowance for a refresh between the last read and the target. */
const REFRESH_SIGMA = 0.45;
/** Slope caps, per ln(1+hours): population growth and earner growth. */
const B_N_MAX = 1.5;
const B_E_MAX = 3.5;
/** Shrinkage: a trophy's slope is pulled toward its tier median with this
 *  pseudo-count (larger = more shrinkage). */
const SHRINK_PSEUDO_COUNT = 4;
/** Plausible day-one-week ceilings for a trophy's rate, by tier: a
 *  power-law earner fit must not extrapolate past them (earners cannot
 *  outgrow the tracked population, and a week after launch only a share of
 *  tracked players has reached any given trophy). */
const TIER_CEILING: Record<string, number> = { platinum: 5, gold: 30, silver: 60, bronze: 85 };

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const logistic = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const hoursSinceRelease = (iso: string): number =>
  Math.max(0, (Date.parse(iso) - RELEASE_MS) / 3600000);
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Least squares line y = a + b*x with recency weights (later points count
 *  up to 3x as much, so the fit tracks the current regime); returns [a, b]
 *  or null when degenerate. */
function leastSquares(xs: number[], ys: number[]): [number, number] | null {
  const n = xs.length;
  if (n < 2) return null;
  let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const w = 1 + 2 * (n === 1 ? 0 : i / (n - 1));
    sw += w; sx += w * xs[i]; sy += w * ys[i]; sxx += w * xs[i] * xs[i]; sxy += w * xs[i] * ys[i];
  }
  const den = sw * sxx - sx * sx;
  if (Math.abs(den) < 1e-12) return null;
  const b = (sw * sxy - sx * sy) / den;
  return [ (sy - b * sx) / sw, b ];
}

/** Fit on all points blended with a fit on the recent half (when enough
 *  points), so a decelerating regime is not dragged by its early explosion. */
function blendedFit(xs: number[], ys: number[]): [number, number] | null {
  const all = leastSquares(xs, ys);
  if (all === null) return null;
  if (xs.length < 6) return all;
  const half = Math.floor(xs.length / 2);
  const recent = leastSquares(xs.slice(half), ys.slice(half));
  if (recent === null) return all;
  // Average the two lines' predictions is not linear; average slopes and
  // re-anchor the intercept at the data's mean x so the line passes through
  // the cloud's centre.
  const b = (all[1] + recent[1]) / 2;
  const mx = xs.reduce((s, x) => s + x, 0) / xs.length;
  const my = ys.reduce((s, y) => s + y, 0) / ys.length;
  return [my - b * mx, b];
}

/** Quantile row for a median rate and a logit spread. */
function quantileRow(id: number, med: number, sigma: number): { id: number; q: number[] } {
  const l = logit(med);
  const q = Z.map((z) => round4(clampRate(logistic(l + z * sigma))));
  // Rounding can only break monotonicity by a hair; enforce it.
  for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
  return { id, q };
}

function forecastCase(c: Case): { id: number; q: number[] }[] {
  const nT = c.trophies.length;
  const horizonDays = Math.max(0, c.horizon_minutes) / 1440;
  const hTarget = hoursSinceRelease(c.target);

  // Last non-null rate per trophy, and a tier-median fallback rate.
  const lastRate: number[] = new Array(nT).fill(NaN);
  for (let j = c.history.length - 1; j >= 0; j--) {
    const r = c.history[j].rates;
    for (let i = 0; i < nT; i++) {
      if (Number.isNaN(lastRate[i]) && r[i] !== null && r[i] !== undefined) lastRate[i] = r[i];
    }
  }
  const tierLast: Record<string, number[]> = {};
  for (let i = 0; i < nT; i++) {
    if (!Number.isNaN(lastRate[i])) {
      (tierLast[c.trophies[i].tier] ??= []).push(lastRate[i]);
    }
  }
  const tierMedianRate: Record<string, number> = {};
  for (const [tier, rs] of Object.entries(tierLast)) tierMedianRate[tier] = median(rs);
  const fallbackRate = (i: number): number =>
    Number.isNaN(lastRate[i]) ? (tierMedianRate[c.trophies[i].tier] ?? 0.85) : lastRate[i];

  // Distinct snapshots: collapse consecutive reads with identical
  // (players_tracked, rates) so a frozen page counts once.
  const snaps: { h: number; x: number; n: number; rates: Array<number | null> }[] = [];
  for (const read of c.history) {
    const prev = snaps[snaps.length - 1];
    const same =
      prev !== undefined &&
      prev.n === read.players_tracked &&
      prev.rates.length === read.rates.length &&
      prev.rates.every((v, i) => v === read.rates[i]);
    if (same) continue;
    const h = hoursSinceRelease(read.at);
    snaps.push({ h, x: Math.log1p(h), n: read.players_tracked, rates: read.rates });
  }

  // No usable history at all: tier priors, very wide spread.
  if (snaps.length === 0) {
    return c.trophies.map((t, i) => quantileRow(t.id, fallbackRate(i), 1.5));
  }

  // Population fit: ln N vs ln(1+h).
  const nFit = blendedFit(snaps.map((s) => s.x), snaps.map((s) => Math.log(Math.max(1, s.n))));
  const hasTrend = snaps.length >= 3 && nFit !== null && snaps[snaps.length - 1].n > snaps[0].n;
  if (!hasTrend) {
    // Flat history: the refresh prior. The page's next refresh grows the
    // tracked population several-fold while earners barely move, so rates
    // fall to roughly a quarter of their last read. The ~100% trophy stays.
    return c.trophies.map((t, i) => {
      const r = fallbackRate(i);
      if (r >= 99.9) return quantileRow(t.id, 99.99, 0.35);
      const drop = DROP_BASE + DROP_RATE_TILT * Math.min(1, r / 2);
      return quantileRow(t.id, r * drop, PRIOR_SIGMA);
    });
  }

  const [, bNraw] = nFit as [number, number];
  const bN = Math.min(B_N_MAX, Math.max(0, bNraw));
  const aN = Math.log(Math.max(1, snaps[snaps.length - 1].n)) - bN * snaps[snaps.length - 1].x;
  const nTarget = Math.exp(aN + bN * Math.log1p(hTarget));

  // Per-trophy earner fits: ln E vs ln(1+h), E = rate * N / 100.
  interface TrophyFit { a: number; b: number; resid: number; points: number }
  const fits: (TrophyFit | null)[] = new Array(nT).fill(null);
  const tierSlopes: Record<string, number[]> = {};
  const tierResids: Record<string, number[]> = {};
  for (let i = 0; i < nT; i++) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const s of snaps) {
      const r = s.rates[i];
      if (r === null || r === undefined || r <= 0) continue;
      const e = (r * s.n) / 100;
      if (e <= 0) continue;
      xs.push(s.x);
      ys.push(Math.log(e));
    }
    if (xs.length < 3) continue;
    const f = blendedFit(xs, ys);
    if (f === null) continue;
    // Re-anchor the intercept at the last snapshot so the fit passes
    // through the most recent earner count.
    const eLast = ((fallbackRate(i) * snaps[snaps.length - 1].n) / 100);
    const a = Math.log(Math.max(0.05, eLast)) - f[1] * snaps[snaps.length - 1].x;
    let ss = 0;
    for (let k = 0; k < xs.length; k++) ss += (ys[k] - (a + f[1] * xs[k])) ** 2;
    const resid = Math.sqrt(ss / Math.max(1, xs.length - 2));
    fits[i] = { a, b: f[1], resid, points: xs.length };
    (tierSlopes[c.trophies[i].tier] ??= []).push(f[1]);
    (tierResids[c.trophies[i].tier] ??= []).push(resid);
  }
  const tierSlopeMed: Record<string, number> = {};
  const tierResidMed: Record<string, number> = {};
  for (const [tier, bs] of Object.entries(tierSlopes)) tierSlopeMed[tier] = median(bs);
  for (const [tier, rs] of Object.entries(tierResids)) tierResidMed[tier] = median(rs);
  const allSlopes = Object.values(tierSlopes).flat();
  const globalSlopeMed = allSlopes.length ? median(allSlopes) : 0;
  const allResids = Object.values(tierResids).flat();
  const globalResidMed = allResids.length ? median(allResids) : 0.3;

  const horizonSigma = HORIZON_SIGMA_PER_SQRT_DAY * Math.sqrt(horizonDays);
  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < nT; i++) {
    const rLast = fallbackRate(i);
    if (rLast >= 99.9) {
      rows.push(quantileRow(c.trophies[i].id, 99.99, 0.35));
      continue;
    }
    const tier = c.trophies[i].tier;
    const tierB = tierSlopeMed[tier] ?? globalSlopeMed;
    const f = fits[i];
    let b: number;
    let resid: number;
    if (f !== null) {
      // Shrink the trophy's own slope toward its tier median.
      b = (f.points * f.b + SHRINK_PSEUDO_COUNT * tierB) / (f.points + SHRINK_PSEUDO_COUNT);
      resid = f.resid;
    } else {
      // No fit of its own: the tier's median slope, anchored at its last
      // earner count.
      b = tierB;
      resid = tierResidMed[tier] ?? globalResidMed;
    }
    b = Math.min(B_E_MAX, Math.max(0, b));
    const eLast = Math.max(0.05, (rLast * snaps[snaps.length - 1].n) / 100);
    const a = Math.log(eLast) - b * snaps[snaps.length - 1].x;
    const eTarget = Math.exp(a + b * Math.log1p(hTarget));
    let med = (100 * eTarget) / Math.max(1, nTarget);
    // Loose sanity caps: never more than 200x the last read either way, and
    // never past the tier's plausible week-one ceiling.
    med = Math.min(rLast * 200, Math.max(rLast / 200, med));
    med = Math.min(med, TIER_CEILING[c.trophies[i].tier] ?? 85);
    med = clampRate(med);
    const sigma = Math.min(
      SIGMA_MAX,
      Math.max(SIGMA_MIN, Math.sqrt(resid * resid + horizonSigma * horizonSigma + REFRESH_SIGMA * REFRESH_SIGMA)),
    );
    rows.push(quantileRow(c.trophies[i].id, med, sigma));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
