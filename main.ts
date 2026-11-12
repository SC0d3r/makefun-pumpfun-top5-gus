// pumpfun-top5-forecast-01 entry: a self-calibrating Monte Carlo forecast of the
// top 5 pump.fun coins by market cap at the seven answer reads.
//
// The history is the house's 10-minute snapshots of every pump.fun coin launched
// since the opening. The answer reads are 7 to 13 hours after the history ends,
// so almost every coin visible in the history has decayed by then; the top 5 is
// dominated by coins born after the cut, which no entry may name. The model
// therefore forecasts the level and shape of the top 5 from the history's own
// long-lag behaviour: it fits how the 5th-largest cap drifts and spreads over
// time, how long top coins survive, and the distribution of the gaps between
// places 1 and 5, then simulates the union of the surviving named coins and a
// fresh pool of unseen coins at each read.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  slot: number;
  name: string;
  symbol: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
  quote_reserves_lamports: string | number;
}
interface Mark {
  schema: string;
  at: string;
  ok: boolean;
  error: string | null;
  kind: string;
  t0: string;
  counts: Record<string, number>;
  rows: Row[];
}
interface Task {
  places: number;
  output: string;
  reads: string[];
  chart_size: number;
  size: { edges: number[] };
  window: { deadline: string; answer_at: string };
  history_until: string;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const marks = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '')
  .map((l) => JSON.parse(l) as Mark)
  .filter((m) => m.ok === true && Array.isArray(m.rows) && m.rows.length > 0)
  .sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
if (marks.length === 0) throw new Error('the history holds no ok mark');

const places = task.places;
const reads = task.reads.map((s) => Date.parse(s));
const edges = task.size.edges;
const bins = edges.length + 1;
const MICRO = 1_000_000;

/** log10 of a cap in SOL, as the scorer takes it: log10(lamports) - 9. */
function sizeOf(lamports: string | number): number {
  return Math.log10(Number(BigInt(lamports))) - 9;
}

/** Deterministic PRNG: mulberry32. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(0x9e3779b9);
function gauss(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}
function sd(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / xs.length);
}
function quantile(xs: number[], q: number): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))));
  return s[i]!;
}
function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

// ---- per-mark level, per-coin series -------------------------------------
interface Pt {
  t: number;
  l: number;
  rank: number;
}
const series = new Map<string, Pt[]>();
const lastRows = new Map<string, Row>();
const level: number[] = [];
const level1: number[] = [];
for (const m of marks) {
  const t = Date.parse(m.at);
  const rows = [...m.rows].sort((a, b) => a.rank - b.rank);
  if (rows.length >= places) {
    level.push(sizeOf(rows[places - 1]!.market_cap_lamports));
    level1.push(sizeOf(rows[0]!.market_cap_lamports));
  } else {
    level.push(Number.NaN);
    level1.push(Number.NaN);
  }
  for (const r of rows) {
    const arr = series.get(r.mint) ?? [];
    arr.push({ t, l: sizeOf(r.market_cap_lamports), rank: r.rank });
    series.set(r.mint, arr);
    lastRows.set(r.mint, r);
  }
}
const validLevel = level.filter((x) => Number.isFinite(x));
if (validLevel.length === 0) throw new Error('no mark ranks the places');
const lastMark = marks[marks.length - 1]!;
const tNow = Date.parse(lastMark.at);
const lNow = validLevel[validLevel.length - 1]!;

/** Fit y = a + b*x over points (x, y), robust enough for short windows. */
function linfit(xs: number[], ys: number[]): { a: number; b: number } {
  const n = xs.length;
  if (n < 2) return { a: n === 1 ? ys[0]! : 0, b: 0 };
  const mx = mean(xs);
  const my = mean(ys);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i]! - mx) * (ys[i]! - my);
    den += (xs[i]! - mx) * (xs[i]! - mx);
  }
  const b = den === 0 ? 0 : num / den;
  return { a: my - b * mx, b };
}

/** Recent trend of the level, in log10 per hour, over the last `hours`. */
function recentTrend(hours: number): number {
  const idx: number[] = [];
  for (let i = 0; i < marks.length; i++) {
    const dt = (tNow - Date.parse(marks[i]!.at)) / 3_600_000;
    if (dt <= hours && Number.isFinite(level[i]!)) idx.push(i);
  }
  if (idx.length < 3) return 0;
  const xs = idx.map((i) => -((tNow - Date.parse(marks[i]!.at)) / 3_600_000));
  const ys = idx.map((i) => level[i]!);
  return linfit(xs, ys).b;
}
const gRecent = clamp(recentTrend(3), -0.6, 0.6);

// Long-lag calibration: dL(h) = level(t+h) - level(t), and the recent trend at t.
const lagSamples: Array<{ h: number; dL: number; g: number }> = [];
for (let i = 0; i < marks.length; i++) {
  if (!Number.isFinite(level[i]!)) continue;
  const gi = (() => {
    const idx: number[] = [];
    for (let k = Math.max(0, i - 18); k <= i; k++) if (Number.isFinite(level[k]!)) idx.push(k);
    if (idx.length < 3) return 0;
    const xs = idx.map((k) => (Date.parse(marks[k]!.at) - Date.parse(marks[i]!.at)) / 3_600_000);
    const ys = idx.map((k) => level[k]!);
    return linfit(xs, ys).b;
  })();
  for (let j = i + 1; j < marks.length; j++) {
    if (!Number.isFinite(level[j]!)) continue;
    const h = (Date.parse(marks[j]!.at) - Date.parse(marks[i]!.at)) / 3_600_000;
    if (h <= 0 || h > 11) continue;
    lagSamples.push({ h, dL: level[j]! - level[i]!, g: clamp(gi, -0.6, 0.6) });
  }
}
// damp: how much of the recent trend persists over the horizon.
let damp = 0.3;
{
  let num = 0;
  let den = 0;
  for (const s of lagSamples) {
    const x = s.g * s.h;
    num += x * s.dL;
    den += x * x;
  }
  if (den > 1e-9) damp = clamp(num / den, -0.4, 0.6);
}
// residual spread of the level change, by horizon (hours).
function levelSd(h: number): number {
  // nearest lag bucket with enough samples, spread scaled by sqrt(h / h_bucket)
  const hs = Array.from(new Set(lagSamples.map((s) => Math.round(s.h * 1000) / 1000))).sort((a, b) => a - b);
  let best: { hb: number; v: number } | null = null;
  for (const hb of hs) {
    const res = lagSamples
      .filter((s) => Math.abs(s.h - hb) < 1e-6)
      .map((s) => s.dL - s.g * s.h * damp);
    if (res.length >= 3) {
      const v = sd(res);
      if (best === null || Math.abs(hb - h) < Math.abs(best.hb - h)) best = { hb, v };
    }
  }
  if (best === null) return clamp(0.2 * Math.sqrt(h), 0.06, 0.7);
  return clamp(best.v * Math.sqrt(Math.max(0.2, h) / Math.max(0.2, best.hb)), 0.06, 0.7);
}

// Survival: fraction of a rank bucket at t still in the top 20 at t+h.
const bucketEdges = [0, 5, 10, 20, 50];
const survSamples: Array<{ h: number; alive: number }>[] = bucketEdges.slice(0, -1).map(() => []);
for (let i = 0; i < marks.length; i++) {
  for (let b = 0; b < bucketEdges.length - 1; b++) {
    const lo = bucketEdges[b]!;
    const hi = bucketEdges[b + 1]!;
    const top = new Set(marks[i]!.rows.slice(lo, hi).map((r) => r.mint));
    if (top.size === 0) continue;
    for (let j = i + 1; j < marks.length; j++) {
      const h = (Date.parse(marks[j]!.at) - Date.parse(marks[i]!.at)) / 3_600_000;
      if (h <= 0 || h > 11) continue;
      const later = new Set(marks[j]!.rows.slice(0, 20).map((r) => r.mint));
      let alive = 0;
      for (const mint of top) if (later.has(mint)) alive++;
      survSamples[b]!.push({ h, alive: alive / top.size });
    }
  }
}
function fitSurv(samples: Array<{ h: number; alive: number }>): (h: number) => number {
  let best = { p0: 0.7, pInf: 0.01, tau: 1.5, err: Infinity };
  for (let p0 = 0.35; p0 <= 0.95; p0 += 0.05) {
    for (let pInf = 0; pInf <= 0.06; pInf += 0.01) {
      for (let tau = 0.4; tau <= 4.0; tau += 0.1) {
        let err = 0;
        for (const s of samples) {
          const pred = pInf + (p0 - pInf) * Math.exp(-s.h / tau);
          err += (pred - s.alive) * (pred - s.alive);
        }
        if (err < best.err) best = { p0, pInf, tau, err };
      }
    }
  }
  return (h: number) => clamp(best.pInf + (best.p0 - best.pInf) * Math.exp(-h / best.tau), 0.0005, 0.95);
}
const survFits = survSamples.map((samples) => (samples.length >= 3 ? fitSurv(samples) : (h: number) => clamp(0.6 * Math.exp(-h / 1.5) + 0.01, 0.0005, 0.9)));
function survFor(rank: number, h: number): number {
  const b = rank <= 5 ? 0 : rank <= 10 ? 1 : rank <= 20 ? 2 : 3;
  const f = survFits[b]!(h);
  return rank > 50 ? f * 0.5 : f;
}

// Survivor path spread (idiosyncratic, market-adjusted), by horizon.
const pathSamples: Array<{ h: number; d: number }> = [];
for (let i = 0; i < marks.length; i++) {
  const top = marks[i]!.rows.slice(0, 20);
  for (let j = i + 1; j < marks.length; j++) {
    const h = (Date.parse(marks[j]!.at) - Date.parse(marks[i]!.at)) / 3_600_000;
    if (h <= 0 || h > 11) continue;
    const later = new Map(marks[j]!.rows.map((r) => [r.mint, sizeOf(r.market_cap_lamports)]));
    const dL = level[j]! - level[i]!;
    for (const r of top) {
      const lj = later.get(r.mint);
      if (lj === undefined) continue;
      pathSamples.push({ h, d: lj - sizeOf(r.market_cap_lamports) - dL });
    }
  }
}
function pathSd(h: number): number {
  // Nearest sampled horizon with enough observations, spread scaled by sqrt(h / h_bucket).
  const hs = Array.from(new Set(pathSamples.map((s) => Math.round(s.h * 100) / 100))).sort((a, b) => a - b);
  let best: { hb: number; v: number } | null = null;
  for (const hb of hs) {
    const res = pathSamples.filter((s) => Math.abs(s.h - hb) < 0.02).map((s) => s.d);
    if (res.length >= 10) {
      const v = sd(res);
      if (best === null || Math.abs(hb - h) < Math.abs(best.hb - h)) best = { hb, v };
    }
  }
  if (best === null) return clamp(0.25 * Math.sqrt(Math.max(0.2, h)), 0.08, 0.85);
  return clamp(best.v * Math.sqrt(Math.max(0.2, h) / Math.max(0.2, best.hb)), 0.08, 1.0);
}

// Gap vectors between the five places, relative to the place-5 level.
const gapVecs: number[][] = [];
{
  const start = Math.max(0, marks.length - 36);
  for (let i = start; i < marks.length; i++) {
    const rows = marks[i]!.rows;
    if (rows.length < places) continue;
    const l5 = sizeOf(rows[places - 1]!.market_cap_lamports);
    const v: number[] = [];
    for (let k = 0; k < places; k++) v.push(sizeOf(rows[k]!.market_cap_lamports) - l5);
    gapVecs.push(v);
  }
}
if (gapVecs.length === 0) gapVecs.push([0.3, 0.2, 0.12, 0.06, 0]);

// Dead level: where a coin that does not survive ends up.
const deadLevel = (() => {
  const rows = lastMark.rows;
  const tail = rows.slice(Math.max(0, rows.length - 60));
  const ls = tail.map((r) => sizeOf(r.market_cap_lamports));
  return clamp(mean(ls), 0.8, 2.2);
})();

// Candidate coins: the last mark's rows, plus any coin seen high in the last 6 marks.
const cand: string[] = [];
const seen = new Set<string>();
for (const r of lastMark.rows) {
  if (!seen.has(r.mint)) { seen.add(r.mint); cand.push(r.mint); }
}
for (let i = Math.max(0, marks.length - 6); i < marks.length; i++) {
  for (const r of marks[i]!.rows.slice(0, 40)) {
    if (!seen.has(r.mint)) { seen.add(r.mint); cand.push(r.mint); }
  }
}
const candInfo = cand.map((mint) => {
  const s = series.get(mint)!;
  const last = s[s.length - 1]!;
  const recent = s.filter((p) => tNow - p.t <= 3 * 3_600_000).slice(-18);
  let slope = 0;
  if (recent.length >= 3) {
    const xs = recent.map((p) => (p.t - tNow) / 3_600_000);
    const ys = recent.map((p) => p.l);
    slope = clamp(linfit(xs, ys).b, -0.8, 0.8);
  }
  return { mint, l0: last.l, rank: last.rank, slope, mayhem: lastRows.get(mint)?.mayhem ?? false };
});
// Keep the set bounded: highest last cap first.
candInfo.sort((a, b) => b.l0 - a.l0 || a.mint.localeCompare(b.mint));
const cands = candInfo.slice(0, 80);

// ---- simulation ----------------------------------------------------------
const SCEN = 2400;
const placeCaps: number[][] = Array.from({ length: places }, () => []);
const orderCount = new Map<string, number[]>(); // mint -> per-place counts
for (const c of cands) orderCount.set(c.mint, new Array(places).fill(0));

const tmpNamed: number[] = new Array(cands.length).fill(0);
const tmpIdx: number[] = new Array(cands.length).fill(0);
const tmpNew: number[] = new Array(places).fill(0);

for (let s = 0; s < SCEN; s++) {
  // latent survival quality per candidate, fixed over the seven reads
  const q: number[] = new Array(cands.length);
  for (let i = 0; i < cands.length; i++) q[i] = rnd();
  const gap = gapVecs[Math.floor(rnd() * gapVecs.length)]!;
  // A modest chance that the top places are further apart than the history's own gap vectors.
  const jump = rnd() < 0.15 ? 1.0 + rnd() * 0.45 : 1.0;
  for (let r = 0; r < reads.length; r++) {
    const h = Math.max(0.05, (reads[r]! - tNow) / 3_600_000);
    const dL = gRecent * h * damp + levelSd(h) * gauss() + (rnd() < 0.08 ? rnd() * 0.9 - 0.45 : 0);
    const L = lNow + dL;
    // newcomer top 5: a resampled gap vector around the future level
    for (let k = 0; k < places; k++) {
      const gk = clamp(gap[k]! * jump, -0.35, 1.9);
      tmpNew[k] = L + gk + 0.09 * gauss();
    }
    // named coins
    const psd = pathSd(h);
    for (let i = 0; i < cands.length; i++) {
      const c = cands[i]!;
      if (q[i]! < survFor(c.rank, h)) {
        tmpNamed[i] = c.l0 + dL + c.slope * h * 0.15 + psd * gauss();
      } else {
        // a coin that leaves the top 20 decays toward the dead level, not at once
        const deadPath = deadLevel + (c.l0 - deadLevel) * Math.exp(-h / 0.9);
        tmpNamed[i] = Math.max(deadLevel, deadPath) + 0.08 * gauss();
      }
      tmpIdx[i] = i;
    }
    // top 5 of the union: pick the 5 largest by log-cap
    const top: Array<{ l: number; i: number }> = [];
    for (let k = 0; k < places; k++) top.push({ l: tmpNew[k]!, i: -1 });
    for (let i = 0; i < cands.length; i++) {
      const l = tmpNamed[i]!;
      if (top.length < places || l > top[top.length - 1]!.l) {
        top.push({ l, i });
        top.sort((a, b) => b.l - a.l);
        if (top.length > places) top.pop();
      }
    }
    for (let k = 0; k < places; k++) {
      placeCaps[k]!.push(top[k]!.l);
      const i = top[k]!.i;
      if (i >= 0) {
        const mint = cands[i]!.mint;
        orderCount.get(mint)![k]!++;
      }
    }
  }
}

// ---- probabilities -------------------------------------------------------
const total = SCEN * reads.length;
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  const rows: Array<{ mint: string; p: number }> = [];
  for (const c of cands) {
    const n = orderCount.get(c.mint)![k]!;
    const p = n / total;
    if (p > 0) rows.push({ mint: c.mint, p });
  }
  rows.sort((a, b) => b.p - a.p || a.mint.localeCompare(b.mint));
  for (const row of rows.slice(0, 100)) {
    const u = Math.floor(row.p * MICRO);
    if (u > 0) probabilities.push({ mint: row.mint, place: k + 1, p: u / MICRO });
  }
}

// top5 tie-break: the named coins most often in the top 5 over the reads
const inTop = cands.map((c) => {
  const cnt = orderCount.get(c.mint)!;
  return { mint: c.mint, n: cnt.reduce((a, b) => a + b, 0), l0: c.l0 };
});
inTop.sort((a, b) => b.n - a.n || b.l0 - a.l0 || a.mint.localeCompare(b.mint));
const top5 = inTop.slice(0, places).map((x) => x.mint);
if (top5.length < places) {
  for (const c of cands) {
    if (top5.length >= places) break;
    if (!top5.includes(c.mint)) top5.push(c.mint);
  }
}

// sizes: histogram of the simulated place caps, with a small uniform prior
const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  // 5% uniform hedge so a bin outside the simulated range is not scored at the floor.
  const prior = (SCEN * reads.length * 0.05) / bins;
  const hist = new Array(bins).fill(prior);
  for (const l of placeCaps[k]!) {
    let bin = 0;
    if (l < edges[0]!) bin = 0;
    else if (l >= edges[edges.length - 1]!) bin = bins - 1;
    else {
      // binary search for the bin
      let lo = 0;
      let hi = edges.length - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (edges[mid]! <= l) lo = mid;
        else hi = mid - 1;
      }
      bin = lo + 1;
    }
    hist[bin] += 1;
  }
  const sum = hist.reduce((a, b) => a + b, 0);
  for (let bin = 0; bin < bins; bin++) {
    const u = Math.floor((hist[bin]! / sum) * MICRO);
    if (u > 0) sizes.push({ place: k + 1, bin, p: u / MICRO });
  }
}

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);

// Guard: every place must carry probability mass before the run is accepted.
for (const place of [1, 2, 3, 4, 5]) {
  if (!probabilities.some((row) => row.place === place)) {
    throw new Error(`no probability rows for place ${place}`);
  }
}

// Guard: every place must carry size mass before the run is accepted.
for (const place of [1, 2, 3, 4, 5]) {
  if (!sizes.some((row) => row.place === place)) {
    throw new Error(`no size rows for place ${place}`);
  }
}
