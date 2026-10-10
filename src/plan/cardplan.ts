/**
 * Start-of-card plan (experiment). Once per card the public input is condensed into a per-region digest
 * (how much there is to observe, how many dark hours each region is above the altitude limit in each third
 * of the survey); a language model answers with a priority multiplier per region and third. The multipliers
 * scale the planner's own value of the region's fields (Survey.plan_mult); pointing, fibres and exposure
 * length stay with the planner.
 *
 * Nothing here is fitted to a card: the regions are RA sectors of the card's own footprint, the thirds are
 * thirds of its own night list.
 */
import { readFileSync } from "node:fs";
import { mod } from "../core/compat";
import { Survey } from "../core/state";

export const PHASES = 3;
export const MULT_MIN = 0.5;
export const MULT_MAX = 2;
const MAX_REGIONS = 12;

export interface Region {
  id: string;
  ra_from: number;
  ra_to: number;
  dec_min: number;
  dec_max: number;
  targets: number;
  required: number;
  fields: number;
  /** Points available with every target exposed for the reference time, and for the longest allowed exposure. */
  value: number;
  value_longest: number;
  required_penalty: number;
  hours_needed: number;
  hours_needed_longest: number;
  visible_hours: number[];
  last_night_visible: number;
}
export interface Digest {
  nights: number;
  phase_nights: [number, number][];
  dark_hours: number[];
  reference_exposure_seconds: number;
  longest_exposure_seconds: number;
  hours_needed_total: number;
  hours_needed_longest_total: number;
  value_total: number;
  value_longest_total: number;
  /** Card-wide figures for the wider plan (prompt v2 only). */
  overview: {
    fibres: number;
    targets_per_field: number;
    visits_to_reach_every_target: number;
    exposure_curve: { seconds: number; points_if_every_target_gets_it: number; hours_for_every_visit: number }[];
    required: { count: number; penalty_each: number; penalty_total: number; seconds_needed_in_good_sky_median: number; seconds_needed_in_good_sky_p90: number; share_needing_more_than_reference: number; share_out_of_reach: number };
  };
  regions: Region[];
  /** Region index of every coarse cell (not sent to the model). */
  cell_region: number[];
}
export interface CardPlan {
  /** Multiplier per region id and phase. */
  regions: Record<string, number[]>;
  /** Wider plan (prompt v2), each per third of the survey unless noted; absent = 1. */
  depth?: number[];
  required?: number[];
  revisit?: number[];
  requests?: number;
  reason?: string;
  notes?: Record<string, string>;
}
export const LEVERS = ["depth", "required", "revisit"] as const;

function phaseOf(night: number, nights: number): number {
  return Math.min(PHASES - 1, Math.trunc((night * PHASES) / Math.max(nights, 1)));
}

/** RA sectors that start at the widest gap of the footprint, so a region is never split across 0/360. */
export function buildDigest(sv: Survey): Digest {
  const cat = sv.cat,
    nn = sv.nights.length;
  const sums = Array.from({ length: cat.nc }, () => [0, 0, 0]);
  for (let z = 0; z < cat.nz; z++) {
    const s = sums[cat.zcoarse[z]];
    s[0] += cat.zcx[z];
    s[1] += cat.zcy[z];
    s[2] += cat.zcz[z];
  }
  const cellRa = sums.map(([x, y]) => mod((Math.atan2(y, x) * 180) / Math.PI, 360));
  const sorted = [...new Set(cellRa.map((ra) => Math.round(ra * 100) / 100))].sort((a, b) => a - b);
  let origin = 0,
    gap = -1;
  for (let i = 0; i < sorted.length; i++) {
    const next = i + 1 < sorted.length ? sorted[i + 1] : sorted[0] + 360,
      width = next - sorted[i];
    if (width > gap) {
      gap = width;
      origin = mod(next, 360);
    }
  }
  const span = Math.max(1, 360 - gap + 1e-6),
    width = Math.max(15, Math.ceil(span / MAX_REGIONS / 5) * 5),
    count = Math.max(1, Math.ceil(span / width));
  const cell_region = cellRa.map((ra) => Math.min(count - 1, Math.trunc(mod(ra - origin, 360) / width)));
  const phase_nights: [number, number][] = [],
    dark_hours = Array(PHASES).fill(0);
  for (let p = 0; p < PHASES; p++) phase_nights.push([Math.ceil((p * nn) / PHASES), Math.ceil(((p + 1) * nn) / PHASES) - 1]);
  sv.nights.forEach(([start, end], i) => {
    dark_hours[phaseOf(i, nn)] += (end - start) / 3600;
  });
  const regions: Region[] = Array.from({ length: count }, (_, r) => ({
    id: `R${r + 1}`,
    ra_from: Math.round(mod(origin + r * width, 360)),
    ra_to: Math.round(mod(origin + Math.min(span, (r + 1) * width), 360)),
    dec_min: 90,
    dec_max: -90,
    targets: 0,
    required: 0,
    fields: 0,
    value: 0,
    value_longest: 0,
    required_penalty: 0,
    hours_needed: 0,
    hours_needed_longest: 0,
    visible_hours: Array(PHASES).fill(0),
    last_night_visible: 0,
  }));
  const weight = Array(count).fill(0);
  for (let z = 0; z < cat.nz; z++) {
    const c = cat.zcoarse[z],
      region = regions[cell_region[c]],
      members = cat.zmembers[z],
      required = members.filter((j) => cat.required[j]).length;
    region.targets += members.length;
    region.required += required;
    region.fields += 1;
    for (const j of members) {
      region.value += cat.weight[j] * sv.MU_REF * Math.min(1, (cat.flux[j] * sv.T_REF) / sv.f0t0);
      region.value_longest += cat.weight[j] * sv.MU_REF * Math.min(1, (cat.flux[j] * sv.t_max) / sv.f0t0);
    }
    region.dec_min = Math.min(region.dec_min, cat.zdec[z]);
    region.dec_max = Math.max(region.dec_max, cat.zdec[z]);
    // Visible hours of the region: mean over its fields (each field takes its coarse cell's value).
    weight[cell_region[c]] += 1;
    const vis = cat.season_night[c];
    for (let i = 0; i < nn; i++) {
      region.visible_hours[phaseOf(i, nn)] += vis[i] / 3600;
      if (vis[i] >= 3600) region.last_night_visible = Math.max(region.last_night_visible, i + 1);
    }
  }
  regions.forEach((region, r) => {
    const n = Math.max(1, weight[r]);
    region.visible_hours = region.visible_hours.map((h) => Math.round(h / n));
    region.hours_needed = Math.round((region.fields * sv.T_REF) / 3600);
    region.required_penalty = region.required * sv.p_req;
    region.hours_needed_longest = Math.round((region.fields * sv.t_max) / 3600);
    region.value = Math.round(region.value);
    region.value_longest = Math.round(region.value_longest);
    region.dec_min = Math.round(region.dec_min);
    region.dec_max = Math.round(region.dec_max);
  });
  const kept = regions.filter((region) => region.targets > 0),
    renumber = regions.map((region) => kept.indexOf(region));
  kept.forEach((region, r) => (region.id = `R${r + 1}`));
  let visits = 0;
  for (const members of cat.zmembers) visits += Math.ceil(members.length / sv.n_fibers);
  const exposure_curve = [sv.T_REF / 2, sv.T_REF, sv.T_REF * 2, sv.t_max]
    .filter((t, i, all) => t >= sv.t_min && t <= sv.t_max && all.indexOf(t) === i)
    .map((t) => {
      let points = 0;
      for (let j = 0; j < cat.n; j++) points += cat.weight[j] * sv.MU_REF * Math.min(1, (cat.flux[j] * t) / sv.f0t0);
      return { seconds: t, points_if_every_target_gets_it: Math.round(points), hours_for_every_visit: Math.round((visits * t) / 3600) };
    });
  const needs = sv.need_req.filter((_, j) => cat.required[j]).sort((a, b) => a - b),
    at = (q: number) => (needs.length ? Math.round(needs[Math.min(needs.length - 1, Math.trunc(q * needs.length))]) : 0),
    share = (limit: number) => (needs.length ? Number((needs.filter((n) => n > limit).length / needs.length).toFixed(2)) : 0);
  return {
    nights: nn,
    phase_nights,
    overview: {
      fibres: sv.n_fibers,
      targets_per_field: Number((cat.n / Math.max(1, cat.nz)).toFixed(1)),
      visits_to_reach_every_target: visits,
      exposure_curve,
      required: { count: needs.length, penalty_each: sv.p_req, penalty_total: needs.length * sv.p_req, seconds_needed_in_good_sky_median: at(0.5), seconds_needed_in_good_sky_p90: at(0.9),
        share_needing_more_than_reference: share(sv.T_REF), share_out_of_reach: share(sv.t_max) },
    },
    dark_hours: dark_hours.map((h) => Math.round(h)),
    reference_exposure_seconds: sv.T_REF,
    longest_exposure_seconds: sv.t_max,
    hours_needed_total: kept.reduce((a, r) => a + r.hours_needed, 0),
    hours_needed_longest_total: kept.reduce((a, r) => a + r.hours_needed_longest, 0),
    value_total: kept.reduce((a, r) => a + r.value, 0),
    value_longest_total: kept.reduce((a, r) => a + r.value_longest, 0),
    regions: kept,
    cell_region: cell_region.map((r) => renumber[r]),
  };
}

export const PLAN_SYSTEM = `You plan one season of a multi-fibre sky survey before it starts. A separate planner chooses every pointing, fibre assignment and exposure length; you only tell it which sky regions to favour in which part of the season.

Facts about the survey:
- One telescope, one field at a time. A target scores only its best single exposure, so a field is normally observed once, well; more time on a field gives more points up to a cap per target.
- A region can only be observed while it is above the altitude limit during dark time. Regions rise and set with the season: hours lost in a region that will not be visible later cannot be made up.
- Each required target that is not observed well enough costs the stated penalty. Sky left unobserved also costs a small uniformity penalty.
- Weather is unknown in advance and removes an unpredictable share of the dark hours.
- The planner already multiplies a region's value by up to 2.5 when the region has less visible time left than the median region. Your numbers are applied on top of that, so 1.0 means "leave it to the planner".

The input lists, per region: its RA range and declination range, number of targets and required targets, number of telescope fields, the points available if every target got the reference exposure and if every target got the longest allowed exposure, the penalty at stake for its required targets, the telescope hours needed to cover every field once at the reference exposure and at the longest exposure, and the hours the region is observable in the early, middle and late third of the survey (and the last night it is observable for at least an hour). It also gives the dark hours in each third and the totals.

Work out where telescope time is scarce: compare what each third's dark hours can cover with what the regions observable in that third need, taking into account that regions compete for the same hours. Decide which regions must be done early because they will set, which can wait, and which are not worth their time if not everything fits.

Answer with one JSON object and nothing else:
{"regions": {"R1": [early, middle, late], ...}, "reason": "two or three sentences"}
Every region must appear. Each number is a multiplier between ${MULT_MIN} and ${MULT_MAX}: above 1 favours the region in that third, below 1 holds it back. Use 1.0 wherever you see no reason to depart from the planner.`;

export const PLAN_SYSTEM_V2 = `You plan one season of a multi-fibre sky survey before it starts. A separate planner chooses every pointing, fibre assignment and exposure length, one exposure at a time, greedily. It cannot look ahead over the season. You set the season-level policy it works under.

Facts about the survey:
- One telescope, one field at a time; each exposure serves at most as many targets as there are fibres, so a field with more targets than fibres needs several visits.
- A target scores only its best single exposure; exposures do not add up. Points grow with exposure length up to a cap per target, so bright targets saturate early and faint ones need long exposures. Poor sky lowers what an exposure of a given length achieves.
- Each required target that never reaches the stated threshold costs the stated penalty; that penalty can exceed everything else on the card.
- A region can only be observed while it is above the altitude limit in dark time. Regions rise and set with the season; hours lost in a region that will not come back cannot be made up.
- Weather is unknown in advance and removes an unpredictable share of the dark hours; some nights are lost entirely. Timed requests with rewards arrive during the season without warning.

The input gives, card-wide: fibres, mean targets per field, the number of visits needed to reach every target once, an exposure curve (for several exposure lengths: the points if every target got that exposure in good sky, and the telescope hours all those visits would take), and for the required targets their number, penalty, the exposure seconds they need in good sky (median and 90th percentile), the share needing more than the reference exposure and the share that cannot reach the threshold even at the longest exposure. Per region it gives the RA and declination range, targets, required targets and their penalty, fields, points at the reference and at the longest exposure, hours for one pass over its fields at both lengths, the hours it is observable in the early, middle and late third of the survey, and the last night it is observable for an hour. It also gives the dark hours of each third.

Your levers. Every number is a multiplier on the planner's own setting, between ${MULT_MIN} and ${MULT_MAX}; 1.0 means "leave it to the planner". Give a value for the early, middle and late third unless stated otherwise.
- "regions": per region, how strongly to favour it. The planner already raises a region's value by up to 2.5 when it has less observable time left than the median region.
- "depth": above 1 makes the planner choose longer exposures on fewer fields (it lowers the price it puts on telescope time); below 1 makes it choose shorter exposures and move on sooner.
- "required": weight of securing required targets relative to ordinary points.
- "revisit": value of going back to fields already observed to give their faint targets a longer exposure; the planner's default treats this as worth a quarter of fresh points.
- "requests": one number, how much to pay in telescope time for a timed request relative to its reward.

Do the arithmetic before deciding: how many dark hours survive plausible weather; what one pass over everything costs at each exposure length and what it earns per hour; whether the required targets can all be secured and by when; which regions set early and whether their work fits before they go; what is left for depth and revisits and when that time is best spent. Then set the levers so that the greedy planner ends up doing what your arithmetic says is best. Where the arithmetic gives no reason to depart from the planner, use 1.0; a confident wrong push costs points.

Answer with one JSON object and nothing else:
{"regions": {"R1": [early, middle, late], ...}, "depth": [early, middle, late], "required": [early, middle, late], "revisit": [early, middle, late], "requests": number, "notes": {"regions": "one sentence", "depth": "one sentence", "required": "one sentence", "revisit": "one sentence", "requests": "one sentence"}}
Every region must appear.`;

export function planUser(digest: Digest, version = 1): string {
  const { cell_region: _hidden, overview, ...shown } = digest;
  return JSON.stringify(version >= 2 ? { overview, ...shown } : shown);
}

/** Checked plan, or null with the reason. Missing regions and out-of-range numbers make the plan invalid. */
export function parsePlan(raw: unknown, digest: Digest): { plan: CardPlan | null; error: string | null } {
  const regions = (raw as { regions?: unknown } | null)?.regions;
  if (regions === null || typeof regions !== "object") return { plan: null, error: "no regions object" };
  const out: Record<string, number[]> = {};
  for (const region of digest.regions) {
    const row = (regions as Record<string, unknown>)[region.id];
    if (!Array.isArray(row) || row.length !== PHASES) return { plan: null, error: `${region.id}: expected ${PHASES} numbers` };
    const values = row.map(Number);
    if (values.some((v) => !Number.isFinite(v) || v < MULT_MIN - 1e-9 || v > MULT_MAX + 1e-9)) return { plan: null, error: `${region.id}: number outside ${MULT_MIN}-${MULT_MAX}` };
    out[region.id] = values;
  }
  const reason = (raw as { reason?: unknown }).reason,
    plan: CardPlan = { regions: out, reason: typeof reason === "string" ? reason : undefined },
    inRange = (v: number) => Number.isFinite(v) && v >= MULT_MIN - 1e-9 && v <= MULT_MAX + 1e-9;
  for (const lever of LEVERS) {
    const row = (raw as Record<string, unknown>)[lever];
    if (row === undefined) continue;
    if (!Array.isArray(row) || row.length !== PHASES || !row.map(Number).every(inRange)) return { plan: null, error: `${lever}: expected ${PHASES} numbers in ${MULT_MIN}-${MULT_MAX}` };
    plan[lever] = row.map(Number);
  }
  const requests = (raw as { requests?: unknown }).requests;
  if (requests !== undefined) {
    if (!inRange(Number(requests))) return { plan: null, error: "requests: number out of range" };
    plan.requests = Number(requests);
  }
  const notes = (raw as { notes?: unknown }).notes;
  if (notes !== null && typeof notes === "object") plan.notes = notes as Record<string, string>;
  return { plan, error: null };
}

export function loadPlan(file: string, digest: Digest): CardPlan {
  const { plan, error } = parsePlan(JSON.parse(readFileSync(file, "utf8")), digest);
  if (plan === null) throw new Error(`card plan ${file}: ${error}`);
  return plan;
}

/** Multiplier of every coarse cell for one night. */
export function cellMultipliers(plan: CardPlan, digest: Digest, night: number): number[] {
  const phase = phaseOf(night, digest.nights);
  return digest.cell_region.map((r) => plan.regions[digest.regions[r].id][phase]);
}

/** The other levers for one night. */
export function nightLevers(plan: CardPlan, digest: Digest, night: number): { depth: number; required: number; revisit: number; requests: number } {
  const phase = phaseOf(night, digest.nights);
  return { depth: plan.depth?.[phase] ?? 1, required: plan.required?.[phase] ?? 1, revisit: plan.revisit?.[phase] ?? 1, requests: plan.requests ?? 1 };
}
