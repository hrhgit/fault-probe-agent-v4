import { Site, parse_utc } from "./astro";
import { Catalog } from "./catalog";
import { Sky } from "./conditions";
import { OffsetEstimator } from "./pointing";
import { Payload, median, sum, tupleCompare } from "./compat";
export type Knobs = Record<string, number>;
type LedgerRow = [number, number, number];
export class Survey {
  T_REF = 900;
  MU_REF: number;
  site: Site;
  scoring: Payload;
  n_fibers: number;
  n_side: number;
  pitch: number;
  fov: number;
  t_min: number;
  t_max: number;
  f0t0: number;
  thr_req: number;
  p_req: number;
  thr_unif: number;
  u_weight: number;
  slot: number;
  nights: [number, number][];
  survey_end: number;
  total_night_seconds: number;
  cat: Catalog;
  sky: Sky;
  offset: OffsetEstimator;
  best: number[];
  gmax: number[];
  need_req: number[];
  ledger = new Map<number, LedgerRow[]>();
  band_done = new Map<number, number>();
  band_bonus = new Map<number, number>();
  any_band_bonus = false;
  zpot: number[];
  zdeep: number[];
  zreq: number[][];
  zflag: boolean[];
  urgency: number[];
  sci_urgency: number[];
  urgency_night = -1;
  extra = new Map<number, [number, number, number]>();
  zextra = new Map<number, number[]>();
  rate_gain: number;
  rate_time = 3600;
  observes = 0;
  required_left: number;
  /** Start-of-card plan (src/plan/cardplan.ts): multiplier of every coarse cell for a night; null = no plan. */
  plan_cells: ((night: number) => number[]) | null = null;
  plan_mult: number[] | null = null;
  /** Further plan levers for a night, all 1 without a plan: exposure depth (divides the time price), weight of
   * required targets, value of deepening fields already observed, value of timed requests. */
  plan_night: ((night: number) => { depth: number; required: number; revisit: number; requests: number }) | null = null;
  plan_levers = { depth: 1, required: 1, revisit: 1, requests: 1 };
  /** Foresight test (ORACLE_FUTURE, never a candidate): the score every target ended with in a reference run
   * of the same card and when it got it. Until that moment the target counts as already holding that score
   * (times future_weight), so an exposure is valued only by what it adds to the season's outcome. */
  future: { score: number[]; at: number[]; weight: number } | null = null;
  floor: number[] | null = null;
  set_now(t: number): void {
    if (this.future === null) return;
    const { score, at, weight } = this.future;
    this.floor = score.map((f, j) => (t < at[j] - 1 ? weight * f : 0));
  }
  held(j: number): number {
    const floor = this.floor;
    return floor !== null && floor[j] > this.best[j] ? floor[j] : this.best[j];
  }
  constructor(
    payload: Payload,
    public knobs: Knobs,
  ) {
    const site_cfg = payload.site,
      scoring = payload.scoring,
      inst = payload.instrument;
    this.scoring = scoring;
    this.site = new Site(
      site_cfg.latitude_deg,
      site_cfg.longitude_deg,
      site_cfg.minimum_altitude_deg ?? 30,
      scoring.airmass_exponent ?? 0.6,
      scoring.lunar_model ?? {},
    );
    this.n_fibers = Math.trunc(Number(inst.n_fibers));
    this.n_side = Math.trunc(
      inst.grid_side || Math.floor(Math.sqrt(this.n_fibers)),
    );
    this.pitch = Number(
      inst.pitch_deg ||
        Math.sqrt(Number(inst.fiber_area_deg2)) + Number(inst.gap_deg ?? 0),
    );
    this.fov = this.n_side * this.pitch;
    const exposure = inst.exposure ?? {};
    this.t_min = Math.trunc(exposure.min_duration_seconds ?? 60);
    this.t_max = Math.trunc(exposure.max_duration_seconds ?? 3600);
    this.f0t0 =
      Number(scoring.flux_zero_point) *
      Number(scoring.exposure_zero_point_seconds);
    this.MU_REF = Math.max(
      ...Object.values(scoring.program.multipliers).map(Number),
    );
    const required = scoring.required ?? {},
      uniformity = scoring.uniformity ?? {};
    this.thr_req = Number(required.observed_factor_threshold ?? 0.5);
    this.p_req = Number(required.penalty_per_missing ?? 50);
    this.thr_unif = Number(uniformity.observed_factor_threshold ?? 0.5);
    this.u_weight = Number(uniformity.weight ?? 0);
    this.slot = Number(payload.survey.slot_seconds ?? 900);
    this.nights = payload.survey.nights.map(
      (n: Payload) =>
        [parse_utc(n.observing_start_utc), parse_utc(n.observing_end_utc)] as [
          number,
          number,
        ],
    );
    this.nights.sort(tupleCompare);
    this.survey_end = this.nights[this.nights.length - 1][1];
    this.total_night_seconds = sum(
      this.nights.map(([start, end]) => end - start),
    );
    this.cat = new Catalog(
      payload,
      this.site,
      this.fov,
      Number(uniformity.ra_band_width_deg ?? 10),
    );
    this.cat.build_season(this.nights);
    this.sky = new Sky(scoring);
    this.offset = new OffsetEstimator(this.n_side, this.pitch);
    const cat = this.cat,
      n = cat.n;
    this.best = Array(n).fill(0);
    this.gmax = Array(n).fill(0);
    this.need_req = cat.flux.map(
      (f) => (this.thr_req * this.f0t0) / Math.max(f, 1e-9),
    );
    for (const b of cat.band_total.keys()) {
      this.band_done.set(b, 0);
      this.band_bonus.set(b, 0);
    }
    this.zpot = Array(cat.nz).fill(0);
    this.zdeep = Array(cat.nz).fill(0);
    this.zreq = Array.from({ length: cat.nz }, () => []);
    this.zflag = Array(cat.nz).fill(false);
    for (let z = 0; z < cat.nz; z++) this.refresh_zone(z);
    this.urgency = Array(cat.nc).fill(1);
    this.sci_urgency = Array(cat.nc).fill(1);
    const potentials = [...this.zpot].sort((a, b) => a - b),
      typical = potentials.length
        ? potentials[Math.trunc(0.6 * (potentials.length - 1))]
        : 0;
    this.rate_gain = ((0.8 * typical) / this.T_REF) * 3600;
    this.required_left = cat.required.filter(Boolean).length;
  }
  refresh_zone(z: number): void {
    const cat = this.cat,
      scale = this.T_REF / this.f0t0,
      deep_scale = (1.2 * this.t_max) / this.f0t0,
      values: number[] = [],
      deep: number[] = [],
      req: number[] = [];
    for (const j of cat.zmembers[z]) {
      let g = cat.flux[j] * scale,
        v = cat.weight[j] * this.MU_REF * Math.min(g, 1) - this.held(j);
      if (v > 0) values.push(v);
      else {
        g = cat.flux[j] * deep_scale;
        v = cat.weight[j] * this.MU_REF * Math.min(g, 1) - this.held(j);
        if (v > 1e-3) deep.push(v);
      }
      if (cat.required[j] && this.gmax[j] < this.thr_req) req.push(j);
    }
    if (values.length > this.n_fibers) {
      values.sort((a, b) => b - a);
      values.length = this.n_fibers;
    }
    if (deep.length > this.n_fibers) {
      deep.sort((a, b) => b - a);
      deep.length = this.n_fibers;
    }
    this.zpot[z] = sum(values);
    this.zdeep[z] = sum(deep);
    this.zreq[z] = req;
    this.zflag[z] = req.length > 0 || this.zextra.has(z);
  }
  get price(): number {
    // price_fixed > 0 (experiment): a constant price of telescope time in points per second for the whole
    // season, in place of the trailing one (kappa x the recent gain rate).
    if ((this.knobs.price_fixed ?? 0) > 0) return this.knobs.price_fixed / this.plan_levers.depth;
    return (this.knobs.kappa * this.rate_gain) / this.rate_time / this.plan_levers.depth;
  }
  night_index(t: number): number {
    let lo = 0,
      hi = this.nights.length;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (this.nights[mid][1] <= t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  apply_hits(
    observe_index: number,
    rows: [number, number, number, number][],
    teff: number,
    closed: boolean,
  ): number {
    const touched = new Set<number>();
    let gain = 0;
    for (const [j, score, , g_low] of rows) {
      if (score > this.best[j]) {
        gain += score - this.best[j];
        this.best[j] = score;
        touched.add(this.cat.zone_of[j]);
      }
      if (g_low > this.gmax[j]) {
        const before = this.gmax[j];
        this.gmax[j] = g_low;
        if (before < this.thr_unif && this.thr_unif <= g_low) {
          const b = this.cat.band[j];
          this.band_done.set(b, this.band_done.get(b)! + 1);
        }
        if (
          this.cat.required[j] &&
          before < this.thr_req &&
          this.thr_req <= g_low
        )
          this.required_left--;
        touched.add(this.cat.zone_of[j]);
      }
    }
    this.ledger.set(
      observe_index,
      rows.map(([j, score, , g_low]) => [j, score, g_low]),
    );
    for (const z of touched) this.refresh_zone(z);
    if (!closed && teff > 0) {
      const decay = Math.exp(-teff / (this.knobs.rate_hours * 3600));
      this.rate_gain = this.rate_gain * decay + gain;
      this.rate_time = this.rate_time * decay + teff;
    }
    this.observes++;
    return gain;
  }
  resync(message: Payload): number {
    const window = message.invalidated_window ?? {},
      first = Math.trunc(window.action_index_start ?? 0),
      last = Math.trunc(window.action_index_end_exclusive ?? 0);
    let dropped = 0;
    for (let i = first; i < last; i++) if (this.ledger.delete(i)) dropped++;
    const n = this.cat.n,
      best = Array(n).fill(0),
      gmax = Array(n).fill(0);
    for (const rows of this.ledger.values())
      for (const [j, score, low] of rows) {
        if (score > best[j]) best[j] = score;
        if (low > gmax[j]) gmax[j] = low;
      }
    if (Array.isArray(message.best_scores)) {
      const official = Array(n).fill(0);
      for (const row of message.best_scores) {
        const j = this.cat.index.get(row.target_id);
        if (j !== undefined) official[j] = Number(row.best_score ?? 0);
      }
      for (let j = 0; j < n; j++) {
        if (official[j] < best[j] - 1e-6) {
          best[j] = official[j];
          gmax[j] = Math.min(
            gmax[j],
            official[j] / Math.max(this.cat.weight[j] * this.MU_REF, 1e-9),
          );
        } else if (official[j] > best[j]) best[j] = official[j];
      }
    }
    this.best = best;
    this.gmax = gmax;
    this.band_done = new Map(
      [...this.cat.band_total.keys()].map((b) => [b, 0]),
    );
    let left = 0;
    for (let j = 0; j < n; j++) {
      const b = this.cat.band[j];
      if (gmax[j] >= this.thr_unif)
        this.band_done.set(b, this.band_done.get(b)! + 1);
      if (this.cat.required[j] && gmax[j] < this.thr_req) left++;
    }
    this.required_left = left;
    for (let z = 0; z < this.cat.nz; z++) this.refresh_zone(z);
    return dropped;
  }
  set_requests(active: Payload[] | null | undefined, now: number): void {
    const old_zones = new Set(this.zextra.keys()),
      extra = new Map<number, [number, number, number]>();
    for (const request of active ?? []) {
      const need =
        Math.trunc(request.minimum_completed ?? 0) -
        Math.trunc(request.completed_count ?? 0);
      if (need <= 0) continue;
      let deadline: number;
      try {
        deadline = parse_utc(request.deadline_utc);
      } catch {
        continue;
      }
      const done = new Set(request.completed_target_ids ?? []),
        pending: number[] = [];
      for (const id of request.target_ids ?? []) {
        const j = this.cat.index.get(id);
        if (j !== undefined && !done.has(id)) pending.push(j);
      }
      if (pending.length < need || deadline - now < 60) continue;
      const reward = Number(request.completion_reward ?? 0),
        threshold = Number(request.completion_factor_threshold ?? 0.5);
      if (!Number.isFinite(reward) || !Number.isFinite(threshold)) continue;
      if (reward <= 0 && this.knobs.zero_reward_skip > 0) continue;
      let share = Math.max(
        (this.knobs.request_gain * reward) / need,
        this.knobs.request_floor * reward,
      ) * this.plan_levers.requests;
      try {
        const issued = parse_utc(request.issued_at_utc),
          used = (now - issued) / Math.max(1, deadline - issued);
        share *= 1 + 2 * Math.max(0, Math.min(1, used)) ** 2;
      } catch {
        /* missing issue time: baseline share */
      }
      for (const j of pending) {
        const p = extra.get(j);
        extra.set(
          j,
          p === undefined
            ? [share, deadline, threshold]
            : [
                p[0] + share,
                Math.min(p[1], deadline),
                Math.max(p[2], threshold),
              ],
        );
      }
    }
    this.extra = extra;
    this.zextra = new Map();
    for (const j of extra.keys()) {
      const z = this.cat.zone_of[j],
        group = this.zextra.get(z);
      if (group) group.push(j);
      else this.zextra.set(z, [j]);
    }
    for (const z of new Set([...old_zones, ...this.zextra.keys()]))
      this.zflag[z] = this.zreq[z].length > 0 || this.zextra.has(z);
  }
  update_band_bonus(): void {
    const totals = this.cat.band_total;
    if (this.u_weight <= 0 || totals.size < 2) return;
    const ratios = new Map(
      [...totals].map(([b, total]) => [b, this.band_done.get(b)! / total]),
    );
    const s1 = sum(ratios.values()),
      s2 = sum([...ratios.values()].map((r) => r * r)),
      n = ratios.size;
    let any_bonus = false;
    for (const [b, r] of ratios) {
      let value =
        s2 <= 0
          ? 0
          : (this.u_weight * (((2 * s1) / (n * s2)) * (1 - (r * s1) / s2))) /
            totals.get(b)!;
      value = Math.max(0, Math.min(value, 10));
      if (value < 0.05) value = 0;
      this.band_bonus.set(b, value);
      any_bonus ||= value > 0;
    }
    this.any_band_bonus = any_bonus;
  }
  update_urgency(night: number): void {
    if (night === this.urgency_night || night >= this.nights.length) return;
    this.urgency_night = night;
    const cat = this.cat,
      power_s = this.knobs.sci_urgency_power;
    if (power_s > 0) {
      const weight = new Map<number, number>();
      for (let z = 0; z < cat.nz; z++)
        if (this.zpot[z] > 0) {
          const c = cat.zcoarse[z];
          weight.set(c, (weight.get(c) ?? 0) + this.zpot[z]);
        }
      const left = new Map(
        [...weight.keys()]
          .filter((c) => cat.season_vis[c][night] > 0)
          .map((c) => [c, cat.season_vis[c][night]]),
      );
      const sci = Array(cat.nc).fill(1);
      if (left.size) {
        const ordered = [...left].sort((a, b) => a[1] - b[1]),
          half = 0.5 * sum(ordered.map(([c]) => weight.get(c)!));
        let run = 0,
          med = ordered[ordered.length - 1][1];
        for (const [c, value] of ordered) {
          run += weight.get(c)!;
          if (run >= half) {
            med = value;
            break;
          }
        }
        for (const [c, value] of left)
          sci[c] = Math.max(
            1,
            Math.min(
              this.knobs.sci_urgency_cap,
              (med / Math.max(value, 1800)) ** power_s,
            ),
          );
      }
      this.sci_urgency = sci;
    }
    if (this.plan_night !== null) this.plan_levers = this.plan_night(night);
    if (this.plan_cells !== null) {
      const mult = this.plan_cells(night);
      this.plan_mult = mult;
      if (power_s > 0) this.sci_urgency = this.sci_urgency.map((u, c) => u * mult[c]);
    }
    const cells = new Set<number>();
    for (let z = 0; z < cat.nz; z++)
      if (this.zreq[z].length) cells.add(cat.zcoarse[z]);
    if (!cells.size) return;
    const remaining = new Map(
        [...cells].map((c) => [c, cat.season_vis[c][night]]),
      ),
      positive = [...remaining.values()].filter((v) => v > 0);
    if (!positive.length) return;
    const med = median(positive),
      urgency = Array(cat.nc).fill(1);
    for (const [c, value] of remaining)
      urgency[c] = Math.max(
        1,
        Math.min(
          this.knobs.urgency_cap,
          (med / Math.max(value, 1800)) ** this.knobs.urgency_power,
        ),
      );
    this.urgency = urgency;
  }
}
