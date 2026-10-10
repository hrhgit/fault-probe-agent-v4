/**
 * Instrument-fault watch: decides when to report a fault from exposure scores and public bulletins alone.
 *
 * Published rules it rests on (participant guide, sections 5 and 6). For a hit target i of an exposure:
 *   completion quality  Q_i = eta * W * G_i     eta: instrument efficiency (a fault multiplies it),
 *   program quality     B_i =       W * G_i     W: site weather level, G_i = moon factor / airmass^beta (known geometry).
 * The score is weight * min(flux*T*Q_i/(f0*T0), 1) * m_i, where m_i is the declared program's bonus when B_i lies in
 * that program's band and 1 otherwise. So an unsaturated target shows u = eta*W (after dividing by G_i), and which
 * targets got the bonus bounds W without any instrument term. Two readings follow:
 *   - a bound  eta <= u / W_min  that no weather can explain when it is far below 1 (the "ruler");
 *   - the series of u inside one night: W is steady within a night, a fault is a sudden lasting drop (the "step").
 * Weather that is announced for the whole sky, a newly announced earthquake, directional notices over the pointing
 * and exposures with zero-score fibres are kept out of the step test. A report is itself a test: its result is exact,
 * and the first two wrong ones after a correct one are free, so a free report is spent on less evidence than a paid one.
 */
import { Site, unit_vector } from "../core/astro";
import { Vec } from "../core/compat";

export interface Hit {
  /** Index into the target arrays given at construction. */
  index: number;
  score: number;
}
export interface ExposureInput {
  start: number;
  duration: number;
  program: string;
  alt: number;
  az: number;
  hits: Hit[];
}
export interface Notice {
  kind: string;
  direction: string;
}
export interface Reading {
  start: number;
  end: number;
  night: number;
  alt: number;
  az: number;
  program: string;
  /**
   * efficiency x sky read as if no target got the program bonus, when every target is far from saturated and they
   * agree; null otherwise. With no saturated target the bonus cannot be seen in one exposure (it scales every score
   * alike), so this value is off by the bonus factor exactly when the declared program matched the sky.
   */
  raw: number | null;
  /** Smallest geometry factor among the scoring targets (the sky level a band needs is its edge over this). */
  gMin: number;
  /** Set (to u) when the standing worlds disagree on u by the bonus factor; u is then the largest of them. */
  open: number | null;
  /** The smallest u a standing world allows (equal to u unless the reading is open). */
  uLow: number | null;
  /** When a dimming directional notice covering this pointing first appeared (Infinity: none). */
  sectorSince: number;
  /** eta * W of this exposure (the largest value consistent with the scores), or null when nothing shows it. */
  u: number | null;
  etaHi: number;
  etaLo: number;
  /** The weather level the program bonus proves at least (0 when it proves none). */
  wLo: number;
  /** Unsaturated targets that agree on u; a single one may be a target partly behind an obstruction. */
  support: number;
  /** Some fibres scored zero: the field touches a closed region. */
  blocked: boolean;
  zero: boolean;
  /** Low toward a direction named for terrain: a target may be hidden for part of the exposure. */
  terrain: boolean;
  uneven: boolean;
  sector: boolean;
  weather: string[];
  quake: boolean;
  /** Nights since the earthquake notice now in force first appeared (0 on its first night); null without one. */
  quakeNights: number | null;
  usable: boolean;
}
export interface Advice {
  report: boolean;
  kind: "ruler" | "step" | "collapse" | null;
  reason: string;
}
export interface WatchConfig {
  site: Site;
  ra: number[];
  dec: number[];
  flux: number[];
  weight: number[];
  /** f0 * T0 of the score configuration. */
  unit: number;
  dark: number;
  bright: number;
  multipliers: Record<string, number>;
  nights: [number, number][];
  freeAllowance: number;
}

/**
 * What a drop of log(u) must show before it is reported, by size: [smallest drop, readings after it, minutes it has
 * held]. Weather inside a night moves a few per cent per slot and a fifth within an hour about once in a hundred
 * hours, so a moderate drop has to hold before it is taken for the instrument.
 * Every row asks for three readings. The site also closes without notice for a slot or a few (the official weather
 * generator's background closure), and an exposure that straddles the start or the end of such a closure scores a
 * part of its time: u falls by any factor. One closure can cut two exposures in a row this way, never three, since
 * an exposure between its two ends lies wholly inside it and scores nothing. (Two readings for a large drop were
 * enough on the self-made training cards; in the first platform runs they were the closure each time.)
 * Drops under a fifth are left to the ruler: on 30 self-made training cards a third row for them (0.15, held an hour)
 * added 16 wrong reports for one more fault found. With no free report left each row needs what the next one does
 * and the last row is not used.
 */
export const STEPS: [number, number, number][] = [
  [0.45, 3, 0],
  [0.25, 3, 20],
];
/** Smallest drop of log(u) from the lowest recent night to the start of this one that counts as a collapse (x0.30). */
export const COLLAPSE = 1.2;
/** Readings after a step must agree with each other this closely (log units): a fault is a constant factor. */
export const FLAT = 0.12;
/** An upper bound on eta below this is a fault whatever the weather (a healthy instrument stays well above). */
export const RULER = 0.8;
const COMPASS: Record<string, number> = { N: 0, NE: 45, E: 90, SE: 135, S: 180, SW: 225, W: 270, NW: 315 };
/**
 * Nights over which an earthquake's hit on the efficiency is assumed to fall to 1/e. The official simulator decays it
 * exponentially per night (its own cards use 4 nights); a slower 6 is assumed here so that the bound below errs on
 * the side of blaming the earthquake.
 */
export const QUAKE_DECAY_NIGHTS = 6;
/** Terrain is named once, in the first bulletins, and stays; it blocks below some altitude that is not published. */
const TERRAIN_BELOW = 65;
const NOT_WEATHER = new Set(["earthquake", "terrain_obstruction", "rocket_launch"]);

/**
 * Switches for offline comparisons; the defaults are the design. sectorSteps: whether a step may be reported from
 * readings under a directional notice (when the pointings lie close together and the notice is older). Off: the
 * sector's edge is not published, and a pointing that drifts across it steps down exactly like a fault.
 */
export const OPTIONS = { anyDirectional: true, rulerTwice: false, sectorSteps: false };

export class FaultWatch {
  readings: Reading[] = [];
  free: number;
  private vectors = new Map<number, Vec>();
  /** Bulletins of the last hours with, for each notice, the time it first appeared without interruption. */
  private history: { at: number; notices: Notice[]; since: Map<string, number> }[] = [];
  private noticeSince = new Map<string, number>();
  private terrain = new Set<string>();
  private quakeFirstNight: number | null = null;
  private quakeLastNight: number | null = null;
  private quakeRulerMuted = false;
  private bulletinAt = -Infinity;
  /** Readings before this index are not used as the level before a step (set at night change and after a report). */
  private seriesFrom = 0;
  /** First reading after the last report result (levels of earlier nights are not compared across a repair). */
  private sinceReport = 0;
  private stepsMutedUntil = -Infinity;
  private rulerMutedUntil = -Infinity;
  /** The largest settled u so far and the number of nights seen: the best sky this card has shown. */
  private skyBest = 0;
  private skyNights = 0;
  /** The rule behind the report this watch has just advised, until its result arrives; null for anyone else's. */
  private pending: "ruler" | "step" | null = null;
  /** Wrong reports in a row that cost points, per rule (since its last correct one), and until when it may not pay again. */
  private paidStreak = { ruler: 0, step: 0 };
  private paidMutedUntil = { ruler: -Infinity, step: -Infinity };

  constructor(
    readonly cfg: WatchConfig,
    private readonly log: (text: string) => void = () => {},
  ) {
    this.free = cfg.freeAllowance;
  }

  /**
   * Whether a rule may report with no free report left, when a wrong report costs 150 points. Each wrong report that
   * cost points doubles the time the rule must wait before it may pay again (the rest of that night, then two nights,
   * four, ...), until a correct report of its own clears the count. Where a rule keeps being wrong the paying thins
   * out to a handful of reports over a season instead of one every night, and it is never shut out for good: a rule
   * that could not pay at all stayed silent through a real fault of nineteen days on a self-made stress card.
   */
  private canPay(kind: "ruler" | "step", now: number): boolean {
    return now >= this.paidMutedUntil[kind];
  }

  nightOf(t: number): number {
    const nights = this.cfg.nights;
    for (let i = 0; i < nights.length; i++) if (t <= nights[i][1] + 1) return t >= nights[i][0] - 1 ? i : -1;
    return -1;
  }

  /** The night a time belongs to, or the next one when it falls in daytime. */
  private nightAround(t: number): number {
    const nights = this.cfg.nights;
    for (let i = 0; i < nights.length; i++) if (t <= nights[i][1] + 1) return i;
    return nights.length - 1;
  }

  /** Mean of moon factor / airmass^beta over the exposure, sampled like the judge (pieces of at most 120 s). */
  geometry(index: number, start: number, duration: number): number {
    let v = this.vectors.get(index);
    if (v === undefined) {
      v = unit_vector(this.cfg.ra[index], this.cfg.dec[index]);
      this.vectors.set(index, v);
    }
    const site = this.cfg.site,
      steps = Math.max(1, Math.ceil(duration / 120));
    let total = 0;
    for (let k = 0; k < steps; k++) {
      const t = start + ((k + 0.5) * duration) / steps,
        [, zen] = site.frame(t),
        sin_alt = v[0] * zen[0] + v[1] * zen[1] + v[2] * zen[2];
      total += site.lunar_factor(v[0], v[1], v[2], site.moon(t)) * site.transmission(sin_alt);
    }
    return total / steps;
  }

  bulletin(issued: number, notices: Notice[]): void {
    if (issued < this.bulletinAt) return;
    this.bulletinAt = issued;
    const present = new Set(notices.map((n) => `${n.kind}:${n.direction}`));
    for (const key of [...this.noticeSince.keys()]) if (!present.has(key)) this.noticeSince.delete(key);
    for (const key of present) if (!this.noticeSince.has(key)) this.noticeSince.set(key, issued);
    for (const n of notices) if (n.kind === "terrain_obstruction") this.terrain.add(n.direction);
    if (notices.some((n) => n.kind === "earthquake")) {
      // One episode of earthquake notices: nights in a row (a gap of one night allowed). A new one starts the count
      // again and lifts the mute a wrong report may have set.
      const night = this.nightAround(issued);
      if (this.quakeLastNight === null || night - this.quakeLastNight > 2) {
        this.quakeFirstNight = night;
        this.quakeRulerMuted = false;
      }
      this.quakeLastNight = night;
    }
    this.history.push({ at: issued, notices, since: new Map(this.noticeSince) });
    while (this.history.length > 64) this.history.shift();
  }

  /** Turn one finished exposure into a reading and keep it. */
  exposure(e: ExposureInput): Reading {
    const { cfg } = this,
      bonus = cfg.multipliers[e.program] ?? 1,
      tol = 5e-5;
    let zeros = 0;
    const live: { r: number; g: number; scale: number }[] = [];
    for (const hit of e.hits) {
      if (hit.score <= 1e-9) {
        zeros++; // closed for this target (terrain, a closed sector, the site): says nothing about the instrument
        continue;
      }
      const g = this.geometry(hit.index, e.start, e.duration);
      // scale: the u at which this target would just saturate.
      if (g > 0) live.push({ r: hit.score / cfg.weight[hit.index], g, scale: cfg.unit / (cfg.flux[hit.index] * e.duration * g) });
    }
    // Which targets got the bonus depends on one unknown, the weather level W: target i is in the declared band when
    // lo <= W * G_i < hi. Between the break points lo/G_i and hi/G_i the set of bonus targets is fixed, so each
    // interval of W is one candidate world. A world stands when a single u explains every score (unsaturated targets
    // agree on it, saturated ones need at least their own) without asking for an efficiency above 1.
    const [lo, hi] = e.program === "DARK" ? [cfg.dark, Infinity] : e.program === "BRIGHT" ? [cfg.bright, cfg.dark] : [0, cfg.bright],
      cuts = [0, Infinity];
    for (const t of live) {
      if (lo > 0) cuts.push(lo / t.g);
      if (hi < Infinity) cuts.push(hi / t.g);
    }
    cuts.sort((x, y) => x - y);
    const worlds: { u: number; n: number; wLo: number; wHi: number }[] = [];
    for (let k = 0; k + 1 < cuts.length && live.length; k++) {
      const a = cuts[k],
        b = cuts[k + 1];
      if (!(b > a * (1 + 1e-9))) continue;
      const w = b === Infinity ? a * 2 + 1 : (a + b) / 2,
        xs: number[] = [];
      let floor = 0,
        ok = true;
      for (const t of live) {
        const inBand = w * t.g >= lo && w * t.g < hi,
          g = t.r / (inBand ? bonus : 1);
        if (g > 1 + tol) {
          ok = false;
          break;
        }
        if (g >= 1 - tol) floor = Math.max(floor, t.scale);
        else xs.push(g * t.scale);
      }
      if (!ok) continue;
      if (!xs.length) {
        // Every target saturated in this world: it stands, but shows no u (only that u is at least `floor`).
        worlds.push({ u: NaN, n: 0, wLo: a, wHi: b });
        continue;
      }
      xs.sort((x, y) => x - y);
      const value = xs[xs.length >> 1];
      if (xs[xs.length - 1] / xs[0] > 1.015 || value < floor * (1 - 1e-3) || value / b > 1.02) continue;
      const last = worlds[worlds.length - 1];
      if (last !== undefined && last.wHi === a && last.n > 0 && Math.abs(Math.log(last.u / value)) < 1e-3) last.wHi = b;
      else worlds.push({ u: value, n: xs.length, wLo: a, wHi: b });
    }
    // The reading: the largest u any standing world allows (so a drop is never overstated), and bounds on the
    // efficiency that hold in every standing world.
    let u: number | null = null,
      uLow: number | null = null,
      open: number | null = null,
      support = 0,
      etaHi = Infinity,
      etaLo = 0,
      wLo = 0;
    const uneven = live.length > 0 && worlds.length === 0 && live.some((t) => t.r < 1 - tol);
    if (worlds.length && worlds.every((x) => x.n > 0)) {
      u = Math.max(...worlds.map((x) => x.u));
      support = Math.min(...worlds.map((x) => x.n));
      etaHi = Math.max(...worlds.map((x) => (x.wLo > 0 ? x.u / x.wLo : Infinity)));
      etaLo = Math.min(...worlds.map((x) => (x.wHi < Infinity ? x.u / x.wHi : 0)));
      wLo = Math.min(...worlds.map((x) => x.wLo));
      uLow = Math.min(...worlds.map((x) => x.u));
      if (uLow < u * (1 - 1e-3)) open = u;
    }
    const zero = e.hits.length > 0 && zeros === e.hits.length,
      end = e.start + e.duration,
      // Notices in force at any time during the exposure (a bulletin covers the slot that follows it).
      spanned = this.history.filter((h) => h.at > e.start - 900 && h.at <= end),
      during = spanned.length ? spanned : this.history.slice(-1),
      key = (n: Notice) => `${n.kind}:${n.direction}`,
      weather = [...new Set(during.flatMap((h) => h.notices.filter((n) => n.direction === "ALL" && !NOT_WEATHER.has(n.kind)).map((n) => n.kind)))],
      quake = during.some((h) => h.notices.some((n) => n.kind === "earthquake")),
      // A directional notice of any kind over the pointing: dimming there, or a closure that may cover only part of
      // the exposure or of a target's path. The named direction is the sector's centre to the nearest 45 degrees and
      // sectors are wide.
      toward = (direction: string) => direction in COMPASS && Math.abs(((e.az - COMPASS[direction] + 540) % 360) - 180) <= 90,
      covers = (n: Notice) => n.direction !== "ALL" && n.kind !== "terrain_obstruction" && (OPTIONS.anyDirectional || n.kind === "haze" || n.kind === "overcast") && toward(n.direction),
      sector = during.some((h) => h.notices.some(covers)),
      terrain = e.alt < TERRAIN_BELOW && [...this.terrain].some(toward),
      sectorSince = sector ? Math.max(...during.flatMap((h) => h.notices.filter(covers).map((n) => h.since.get(key(n)) ?? h.at))) : Infinity;
    const reading: Reading = {
      start: e.start,
      end,
      night: this.nightOf(e.start),
      alt: e.alt,
      az: e.az,
      program: e.program,
      open,
      uLow,
      sectorSince,
      u,
      etaHi,
      etaLo,
      wLo,
      support,
      blocked: zeros > 0,
      zero,
      terrain,
      uneven,
      sector,
      weather,
      quake,
      quakeNights: quake && this.quakeFirstNight !== null ? Math.max(0, this.nightOf(e.start) - this.quakeFirstNight) : null,
      usable: u !== null && !zero && !uneven && support >= 2,
      raw: null,
      gMin: live.length ? Math.min(...live.map((t) => t.g)) : 0,
    };
    if (live.length >= 2 && zeros === 0 && live.every((t) => t.r < 0.8)) {
      const xs = live.map((t) => t.r * t.scale).sort((x, y) => x - y);
      if (xs[xs.length - 1] / xs[0] <= 1.03) reading.raw = xs[xs.length >> 1];
    }
    const previous = this.readings[this.readings.length - 1];
    if (previous === undefined || previous.night !== reading.night) {
      this.seriesFrom = this.readings.length;
      if (previous !== undefined) this.skyNights++;
    }
    // u = efficiency x sky and the efficiency is at most 1, so every u is a sky level the card has reached.
    if (reading.usable && !reading.sector && reading.open === null && (reading.u as number) > this.skyBest) this.skyBest = reading.u as number;
    this.readings.push(reading);
    return reading;
  }

  /** The latest readings of this night since the last report that are deep below the best sky seen and show no bonus pattern. */
  private dim(): Reading[] {
    const n = this.readings.length,
      out: Reading[] = [];
    for (let i = n - 1; i >= Math.max(this.sinceReport, this.seriesFrom) && out.length < 4; i--) {
      const r = this.readings[i];
      if (r.night !== this.readings[n - 1].night || r.raw === null || r.sector || r.wLo > 0) break;
      if (!(this.skyNights >= 3 && r.raw < 0.3 * this.skyBest)) break;
      if (out.length && out[out.length - 1].start - r.end > 1800) break;
      out.push(r);
    }
    return out.reverse();
  }

  /**
   * The program to declare for the next exposure, or null to leave the choice to the agent. When the readings sit far
   * below the best sky this card has shown and no target is near saturation, nothing in one exposure tells a dead
   * instrument under a good sky from a healthy one under a terrible sky. Two exposures with different declared
   * programs do: the bonus goes to the one that names the true band. So the declaration alternates (DARK, BACKUP, ...)
   * while that lasts; a wrong declaration costs at most the bonus on an exposure that scores almost nothing.
   */
  wantProgram(now: number): string | null {
    const dim = this.dim();
    if (!dim.length || now - dim[dim.length - 1].end > 1800) return null;
    return dim[dim.length - 1].program === "DARK" ? "BACKUP" : "DARK";
  }

  /**
   * Fault shown by the contrast between declared programs: over the last three or four dim readings, find the bands
   * of the sky under which one level of efficiency x sky explains them all (within 10 %). When only a band with a
   * floor stands (DARK or BRIGHT), the sky is at least that floor and the efficiency at most reading / floor.
   */
  private contrast(now: number): string | null {
    const dim = this.dim();
    if (dim.length < 3 || new Set(dim.map((r) => r.program)).size < 2) return null;
    const mult = this.cfg.multipliers,
      stands: string[] = [],
      level: Record<string, number> = {};
    for (const band of ["DARK", "BRIGHT", "BACKUP"]) {
      const xs = dim.map((r) => (r.raw as number) / (r.program === band ? (mult[band] ?? 1) : 1));
      if (Math.max(...xs) / Math.min(...xs) <= 1.1) {
        stands.push(band);
        level[band] = Math.max(...xs);
      }
    }
    if (stands.length !== 1 || stands[0] === "BACKUP") return null;
    const floor = (stands[0] === "DARK" ? this.cfg.dark : this.cfg.bright) / Math.max(...dim.map((r) => r.gMin)),
      eta = level[stands[0]] / floor,
      quake = dim.some((r) => r.quake);
    // An earthquake by itself leaves at least a few per cent of the efficiency; only far less is a fault on top, and only a free report is spent on it.
    if (quake ? !(eta < 0.04 && this.free > 0) : !(eta < RULER)) return null;
    return `declared programs disagree by the bonus: sky in the ${stands[0]} band, efficiency at most ${eta.toFixed(3)}`;
  }

  /** Whether to report now, given everything read so far. */
  advise(now: number): Advice {
    const n = this.readings.length;
    this.pending = null;
    if (!n) return { report: false, kind: null, reason: "no readings" };
    // Only a reading taken after the last report can speak: the one before it shows the instrument as it was.
    if (n - 1 < this.sinceReport) return { report: false, kind: null, reason: "no reading since the last report" };
    const latest = this.readings[n - 1];
    // Ruler: the bonus says the sky is at least this good, the scores say far less arrived. A target partly hidden
    // by terrain reads the same way, so near terrain (a zero fibre, or low toward a named terrain direction) three
    // unsaturated targets must agree instead of two. (Asking for a second such reading in the night was tried on the
    // training cards: one wrong report fewer, a quarter of the faults found later.)
    // Under an earthquake notice the efficiency is low by itself, but only so low: k nights into the notice the
    // earthquake cannot hold it below 1 - exp(-k / decay). A bound under that share of the usual limit is a fault
    // on top of the earthquake. This rests on an assumed decay, so it only ever spends a free report, and a wrong
    // one silences it for the rest of that earthquake.
    const contrast = this.contrast(now);
    if (contrast !== null && now >= this.rulerMutedUntil && (this.free > 0 || this.canPay("ruler", now))) {
      this.pending = "ruler";
      return { report: true, kind: "ruler", reason: contrast };
    }
    const limit = (r: Reading) => (!r.quake ? RULER : r.quakeNights === null || this.free <= 0 || this.quakeRulerMuted ? 0 : RULER * (1 - Math.exp(-r.quakeNights / QUAKE_DECAY_NIGHTS))),
      ruled = (r: Reading) => r.u !== null && !r.zero && !r.uneven && !r.sector && r.support >= (r.blocked || r.terrain ? 3 : 2) && r.etaHi < limit(r);
    if (now >= this.rulerMutedUntil && ruled(latest) && (this.free > 0 || this.canPay("ruler", now))) {
      const earlier = this.readings.slice(this.seriesFrom, n - 1).filter((r) => r.night === latest.night && ruled(r));
      if (!OPTIONS.rulerTwice || earlier.length > 0) {
        this.pending = "ruler";
        return { report: true, kind: "ruler", reason: `efficiency at most ${Math.max(latest.etaHi, ...earlier.map((r) => r.etaHi)).toFixed(2)} by the program bonus` };
      }
    }
    // Step: a drop inside the night that holds over consecutive readings.
    if (now < this.stepsMutedUntil) return { report: false, kind: null, reason: "steps muted after a wrong report" };
    const usable: number[] = [];
    for (let i = this.seriesFrom; i < n; i++) if (this.readings[i].night === latest.night && this.readings[i].usable) usable.push(i);
    if (usable[usable.length - 1] !== n - 1) return { report: false, kind: null, reason: "latest reading not usable" };
    const value = (i: number) => Math.log(this.readings[i].u as number),
      tiers = this.free > 0 ? STEPS : STEPS.slice(0, -1).map(([size], k) => [size, STEPS[k + 1][1], STEPS[k + 1][2]] as [number, number, number]),
      need = tiers[0][1];
    let why = "no step";
    // Where did the drop happen? Try the last 2..6 readings as "after"; the reading just before them may be the
    // exposure during which the drop happened and is then left out of both sides.
    for (let count = need; count <= 8 && count < usable.length; count++)
      for (const skip of [0, 1]) {
        const after = usable.slice(-count),
          sameSky = (i: number) => this.readings[i].sector === this.readings[after[0]].sector,
          before = usable.slice(0, usable.length - count - skip).filter(sameSky).slice(-3);
        if (!before.length || !after.every(sameSky)) continue;
        // An open reading may be up to the bonus factor lower than its u: among the readings after the drop it takes
        // whichever of its two values lies nearer the settled ones (all at their largest when none is settled).
        const settled = after.filter((i) => this.readings[i].open === null).map(value).sort((x, y) => x - y),
          anchor = settled.length ? settled[settled.length >> 1] : null,
          lowOf = (i: number) => {
            const r = this.readings[i];
            if (r.open === null || anchor === null) return value(i);
            const a = value(i),
              b = Math.log(r.uLow as number);
            return Math.abs(a - anchor) <= Math.abs(b - anchor) ? a : b;
          };
        const high = before.map(value),
          low = after.map(lowOf),
          level = [...high].sort((x, y) => x - y)[high.length >> 1];
        const drop = level - Math.max(...low),
          held = (latest.end - this.readings[after[0]].start) / 60,
          tier = tiers.find(([size]) => drop >= size);
        if (tier === undefined || Math.max(...low) - Math.min(...low) > FLAT) continue;
        if (Math.max(...high) - Math.min(...high) > Math.max(FLAT, 0.4 * drop)) {
          why = "level before not steady";
          continue;
        }
        if (count < tier[1] || held < tier[2]) {
          why = "a drop that has not held long enough yet";
          continue;
        }
        const used = [...before, ...after].map((i) => this.readings[i]);
        if (used[0].sector) {
          if (!OPTIONS.sectorSteps) {
            why = "directional notice over these pointings";
            continue;
          }
          // Under a dimming sector: only pointings close together, and a notice older than the level before.
          const azSpan = Math.max(...used.map((r) => Math.abs(((r.az - used[0].az + 540) % 360) - 180))),
            altSpan = Math.max(...used.map((r) => r.alt)) - Math.min(...used.map((r) => r.alt));
          if (Math.max(...used.map((r) => r.sectorSince)) > used[0].start || azSpan > 20 || altSpan > 10) {
            why = "directional notice over these pointings";
            continue;
          }
        }
        let closed = false;
        // (An exposure with every fibre at zero low toward named terrain is the terrain, not the weather.)
        for (let i = before[0]; i < n; i++) closed ||= this.readings[i].zero && !this.readings[i].terrain;
        if (closed) {
          why = "a fully closed exposure in between";
          continue;
        }
        // Weather over the whole sky can explain a drop of u only if the sky itself fell by that factor. The bonus
        // proves a sky level after the drop; if even the best sky this card has shown (with a margin), reduced by the
        // drop, would lie below it, the drop is in the instrument and the notice explains nothing.
        const proven = Math.min(...after.map((i) => this.readings[i].wLo)),
          notWeather = this.skyBest > 0 && this.skyNights >= 5 && Math.exp(-drop) * 1.3 * this.skyBest < proven;
        if (!notWeather && after.some((i) => this.readings[i].weather.length > 0)) {
          why = "all-sky weather notice";
          continue;
        }
        const first = this.readings[after[0]];
        if (this.bulletinAt < first.start) {
          why = "no bulletin since the drop";
          continue;
        }
        const quakeSince = this.noticeSince.get("earthquake:ALL");
        if (quakeSince !== undefined && quakeSince >= this.readings[before[before.length - 1]].start - 900) {
          why = "earthquake newly announced";
          continue;
        }
        if (this.free <= 0 && !this.canPay("step", now)) {
          why = "no free report left and waiting after a paid wrong report";
          continue;
        }
        this.pending = "step";
        return { report: true, kind: "step", reason: `efficiency x${Math.exp(Math.min(...low) - level).toFixed(2)} to x${Math.exp(Math.max(...low) - level).toFixed(2)} over ${count} readings` };
      }
    // Collapse: the night starts far below every recent night, with no notice of weather. Weather moves the nightly
    // level by a third or so between nights; a factor of more than three with a clear forecast is the instrument.
    if (this.free > 0) {
      const tonight = usable,
        nights = new Map<number, number[]>();
      for (let i = this.sinceReport; i < n; i++) {
        const r = this.readings[i];
        if (r.usable && r.night !== latest.night) nights.set(r.night, [...(nights.get(r.night) ?? []), Math.log(r.u as number)]);
      }
      const levels = [...nights.entries()].filter(([, v]) => v.length >= 3).sort((a, b) => b[0] - a[0]).slice(0, 5)
        .map(([, v]) => [...v].sort((x, y) => x - y)[v.length >> 1]);
      if (tonight.length >= 3 && tonight.length <= 8 && levels.length >= 1) {
        const low = tonight.map(value),
          held = (latest.end - this.readings[tonight[0]].start) / 60;
        if (Math.min(...levels) - Math.max(...low) >= COLLAPSE && Math.max(...low) - Math.min(...low) <= 2 * FLAT && held >= 20) {
          const quakeSince = this.noticeSince.get("earthquake:ALL");
          let clear = !tonight.some((i) => this.readings[i].weather.length > 0 || this.readings[i].sector);
          for (let i = tonight[0]; i < n; i++) clear &&= !this.readings[i].zero;
          if (clear && !(quakeSince !== undefined && quakeSince >= this.cfg.nights[latest.night][0] - 86400)) {
            this.pending = "step";
            return { report: true, kind: "collapse", reason: `the night starts at x${Math.exp(Math.max(...low) - Math.min(...levels)).toFixed(2)} of the lowest recent night` };
          }
          why = "a collapse under notices";
        }
      }
    }
    return { report: false, kind: null, reason: why };
  }

  /**
   * The judge's answer to a report, whoever sent it. A wrong one means the instrument was healthy at that moment, so
   * the readings start afresh either way; only this watch's own wrong report silences the rule that made it.
   */
  reportResult(at: number, correct: boolean): void {
    const own = this.pending,
      paid = this.free <= 0;
    this.pending = null;
    this.seriesFrom = this.readings.length;
    this.sinceReport = this.readings.length;
    if (correct) {
      this.free = this.cfg.freeAllowance;
      if (own !== null) {
        this.paidStreak[own] = 0;
        this.paidMutedUntil[own] = -Infinity;
      }
      return;
    }
    this.free = Math.max(0, this.free - 1);
    if (own === null) return;
    const night = this.nightOf(at),
      end = night >= 0 ? this.cfg.nights[night][1] : at + 6 * 3600;
    if (paid) {
      // Wait out this night, then 2, 4, ... nights (counting this one) before paying again.
      const last = night >= 0 ? night + 2 ** this.paidStreak[own] - 1 : -1;
      this.paidStreak[own]++;
      this.paidMutedUntil[own] = last < 0 ? end : last < this.cfg.nights.length ? this.cfg.nights[last][1] : Infinity;
    }
    if (own === "ruler") {
      this.rulerMutedUntil = end;
      const last = this.readings[this.readings.length - 1];
      if (last !== undefined && last.quake) this.quakeRulerMuted = true;
    } else this.stepsMutedUntil = end;
    this.log(`fault watch: wrong report (${own})${paid ? ", paid" : ""}, muted until the end of the night; free reports left ${this.free}`);
  }
}
