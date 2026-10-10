/** public-env-features-v1: same field order, history padding and geometry as Python. */
import { performance } from "node:perf_hooks";
import { airmass, DEG, moon_radec, RAD, sun_radec } from "../../core/astro";
import { mod } from "../../core/compat";

export const FEATURE_VERSION = "public-env-features-v1";
export const KINDS = ["rain", "overcast", "haze", "cold_snap", "storm", "rocket_launch", "earthquake", "terrain_obstruction"];
export const DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "ALL"];
export interface PublicNotice { event_kind: string; direction: string; nights?: string[] }
export interface Bulletin { issued_at_utc: string; notices: PublicNotice[] }
export interface Forecast extends Bulletin { coverage_start_utc: string; coverage_end_utc: string }
export interface SiteConfig { latitude_deg: number; longitude_deg: number; minimum_altitude_deg: number }
export interface ScoreConfig {
  q0: number; airmass_exponent: number;
  program: { bands: { DARK: number; BRIGHT: number } };
  lunar_model: { maximum_penalty: number; altitude_exponent: number; angular_decay_scale_deg: number };
}
export interface PublicInitialize {
  site: SiteConfig; scoring: ScoreConfig;
  survey: { nights: { observing_start_utc: string; observing_end_utc: string; night_date: string }[] };
  instrument?: { exposure?: { min_duration_seconds: number; max_duration_seconds: number } };
}
export interface Candidate {
  candidate_id?: string; ra_deg: number; dec_deg: number; start_utc: string; duration_seconds: number;
}
export interface ConditionsRequest {
  initialize: PublicInitialize; now_utc: string; bulletins?: Bulletin[]; forecasts?: Forecast[]; candidates: Candidate[];
}
export interface PredictionRow {
  candidate_id: string; eligible: boolean; reason?: string;
  known_geometry_factor?: number; actual_duration_seconds?: number;
  mean_quality?: number; q05?: number; q50?: number; q95?: number;
  usable_fraction?: number; u05?: number; u50?: number; u95?: number;
  severe_block_probability?: number; p_dark?: number; p_bright?: number; p_backup?: number;
}
export interface FeatureGroup {
  context: Float32Array; history: Float32Array[]; candidates: Float32Array[]; indices: number[];
}
export interface PreparedConditions {
  rows: PredictionRow[]; groups: FeatureGroup[]; preprocessing_seconds: number;
}
const clipped = (v: number): number => Math.max(-1, Math.min(1, v));
const age = (seconds: number): number => Math.log1p(Math.max(0, seconds)) / Math.log1p(7 * 86400);
export function utc(value: string): number {
  if (typeof value !== "string" || !/(Z|[+-]\d\d:\d\d)$/.test(value)) throw new Error("timestamp must include a timezone");
  const timestamp = Date.parse(value) / 1000;
  if (!Number.isFinite(timestamp)) throw new Error(`invalid timestamp: ${value}`);
  const fraction = /\.(\d+)(?:Z|[+-]\d\d:\d\d)$/.exec(value)?.[1];
  return timestamp + (fraction ? Number("0." + fraction) - Number("0." + fraction.slice(0, 3)) : 0);
}
function noticeMatrix(notices: PublicNotice[]): Float64Array {
  if (!Array.isArray(notices)) throw new Error("notices must be an array");
  const result = new Float64Array(72);
  for (const notice of notices) {
    const kind = KINDS.indexOf(notice.event_kind), direction = DIRECTIONS.indexOf(notice.direction);
    if (kind < 0 || direction < 0) throw new Error(`unsupported public notice: ${notice.event_kind}/${notice.direction}`);
    result[kind * 9 + direction] = 1;
  }
  return result;
}
function rightIndex(times: number[], value: number): number {
  let low = 0, high = times.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (times[middle] <= value) low = middle + 1; else high = middle;
  }
  return low - 1;
}
/** Append new messages once. Snapshot uses published times, including persistent terrain. */
export class PublicTimeline {
  private bulletins: Bulletin[] = [];
  private forecasts: Forecast[] = [];
  private times: number[] = [];
  private forecastTimes: number[] = [];
  private matrices: Float64Array[] = [];
  private terrain: Float64Array[] = [];
  private starts: Float64Array[] = [];
  private indexedBulletins: Bulletin[] = [];
  private seen = new Set<string>();
  constructor(bulletins: Bulletin[] = [], forecasts: Forecast[] = []) { this.add(bulletins, forecasts); }
  add(bulletins: Bulletin[], forecasts: Forecast[]): void {
    const pending: { kind: "bulletin" | "forecast"; row: Bulletin | Forecast; key: string }[] = [];
    for (const [kind, records] of [["bulletin", bulletins], ["forecast", forecasts]] as const) {
      for (const record of records) {
        // Copy only public model fields; other protocol fields are not features.
        const row: Bulletin | Forecast = kind === "forecast" ? {
          issued_at_utc: record.issued_at_utc,
          coverage_start_utc: (record as Forecast).coverage_start_utc,
          coverage_end_utc: (record as Forecast).coverage_end_utc,
          notices: record.notices.map(n => ({ event_kind: n.event_kind, direction: n.direction, nights: n.nights?.slice() })),
        } : { issued_at_utc: record.issued_at_utc,
          notices: record.notices.map(n => ({ event_kind: n.event_kind, direction: n.direction })) };
        utc(row.issued_at_utc); noticeMatrix(row.notices);
        if (kind === "forecast") {
          const f = row as Forecast;
          if (utc(f.coverage_end_utc) <= utc(f.coverage_start_utc)) throw new Error("empty forecast coverage");
        }
        const key = kind + JSON.stringify(row);
        if (this.seen.has(key)) continue;
        pending.push({ kind, row, key });
      }
    }
    if (!pending.length) return;
    // Validate the whole update before changing persistent state.
    let bulletinChanged = false, forecastChanged = false, first = this.times.length;
    const lastTime = this.times[this.times.length - 1];
    for (const { kind, row, key } of pending) {
      if (this.seen.has(key)) continue;
      this.seen.add(key);
      if (kind === "forecast") { this.forecasts.push(row as Forecast); forecastChanged = true; }
      else {
        this.bulletins.push(row); bulletinChanged = true;
        if (lastTime !== undefined && utc(row.issued_at_utc) < lastTime) first = 0;
      }
    }
    const sort = (a: Bulletin, b: Bulletin): number => utc(a.issued_at_utc) - utc(b.issued_at_utc);
    if (forecastChanged) {
      this.forecasts.sort(sort); this.forecastTimes = this.forecasts.map(f => utc(f.issued_at_utc));
    }
    if (!bulletinChanged) return;
    if (first === 0) {
      this.bulletins.sort(sort);
      while (first < this.times.length && this.indexedBulletins[first] === this.bulletins[first]) first++;
    } else {
      const appended = this.bulletins.splice(first); appended.sort(sort);
      for (const bulletin of appended) this.bulletins.push(bulletin);
    }
    this.times.length = this.matrices.length = this.terrain.length = this.starts.length = first;
    this.indexedBulletins.length = first;
    for (let i = first; i < this.bulletins.length; i++) {
      const time = utc(this.bulletins[i].issued_at_utc), matrix = noticeMatrix(this.bulletins[i].notices);
      const terrain = new Float64Array(9), starts = new Float64Array(72);
      for (let j = 0; j < 9; j++) terrain[j] = Math.max(i ? this.terrain[i - 1][j] : 0, matrix[63 + j]);
      for (let j = 0; j < 72; j++) starts[j] = matrix[j] ?
        (i && this.matrices[i - 1][j] ? this.starts[i - 1][j] : time) : NaN;
      this.times.push(time); this.matrices.push(matrix); this.terrain.push(terrain); this.starts.push(starts);
      this.indexedBulletins.push(this.bulletins[i]);
    }
  }
  snapshot(now: number, start: number, requested: number, actual: number, nightDate: string,
    nightEnd: number, site: SiteConfig, score: ScoreConfig): { context: Float32Array; history: Float32Array[] } {
    const index = rightIndex(this.times, now), first = Math.max(0, index - 31);
    const history = Array.from({ length: 32 }, () => new Float32Array(75));
    const count = Math.max(0, index - first + 1);
    for (let i = first; i <= index; i++) {
      const row = history[32 - count + i - first];
      row.set(this.matrices[i]); row[72] = age(now - this.times[i]);
      row[73] = age(this.times[i] - (i ? this.times[i - 1] : this.times[i])); row[74] = 1;
    }
    const fi = rightIndex(this.forecastTimes, now), forecast = this.forecasts[fi];
    const covers = !!forecast && utc(forecast.coverage_start_utc) <= start && start < utc(forecast.coverage_end_utc);
    const context = new Float32Array(242);
    if (index >= 0) {
      context.set(this.matrices[index], 0); context.set(this.terrain[index], 144);
      for (let j = 0; j < 72; j++) context[153 + j] = Number.isNaN(this.starts[index][j]) ? 0 : age(now - this.starts[index][j]);
    }
    if (covers) context.set(noticeMatrix(forecast.notices.filter(n => n.nights?.includes(nightDate))), 72);
    const date = new Date(start * 1000), day = Math.floor((Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()) -
      Date.UTC(date.getUTCFullYear(), 0, 1)) / 86400000);
    const season = 2 * Math.PI * day / 365.2425, lon = site.longitude_deg * RAD;
    context.set([index >= 0 ? age(now - this.times[index]) : -1, fi >= 0 ? age(now - this.forecastTimes[fi]) : -1,
      Number(covers), site.latitude_deg / 90, Math.sin(lon), Math.cos(lon), site.minimum_altitude_deg / 90,
      score.q0, score.airmass_exponent, score.program.bands.DARK, score.program.bands.BRIGHT,
      Math.sin(season), Math.cos(season), (start - now) / 3600, requested / 3600, actual / 3600, (nightEnd - start) / 43200], 225);
    if (!context.every(Number.isFinite)) throw new Error("nonfinite public context");
    return { context, history };
  }
}
function altaz(ra: number, decDegrees: number, time: number, site: SiteConfig, official = false): [number, number] {
  const d = official ? time / 86400 + 2440587.5 - 2451545 : time / 86400 - 10957.5;
  const lst = mod(280.46061837 + 360.98564736629 * d + site.longitude_deg, 360);
  const hour = (official ? mod(lst - ra + 180, 360) - 180 : lst - ra) * RAD;
  const dec = decDegrees * RAD, lat = site.latitude_deg * RAD;
  const altitude = Math.asin(clipped(Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(hour)));
  const cosine = Math.max(1e-12, Math.cos(altitude));
  const az = Math.atan2(-Math.sin(hour) * Math.cos(dec) / cosine,
    (Math.sin(dec) - Math.sin(altitude) * Math.sin(lat)) / (cosine * Math.max(1e-12, Math.cos(lat))));
  return [altitude * DEG, mod(az * DEG, 360)];
}
function moonGeometry(time: number, site: SiteConfig): number[] {
  const [ra, dec] = moon_radec(time), [sr, sd] = sun_radec(time);
  const illumination = (1 - clipped(Math.sin(sd * RAD) * Math.sin(dec * RAD) +
    Math.cos(sd * RAD) * Math.cos(dec * RAD) * Math.cos((sr - ra) * RAD))) / 2;
  return [ra, dec, illumination, altaz(ra, dec, time, site, true)[0]];
}
function geometry(ra: number, dec: number, time: number, site: SiteConfig, score: ScoreConfig, moon: number[]): number[] {
  const [alt, az] = altaz(ra, dec, time, site), [mr, md, illumination, moonAlt] = moon;
  const separation = Math.acos(clipped(Math.sin(dec * RAD) * Math.sin(md * RAD) +
    Math.cos(dec * RAD) * Math.cos(md * RAD) * Math.cos((ra - mr) * RAD))) * DEG;
  const lunar = score.lunar_model;
  const penalty = lunar.maximum_penalty * illumination * Math.sin(Math.max(0, moonAlt) * RAD) ** lunar.altitude_exponent;
  const factor = 1 - penalty * Math.exp(-separation / lunar.angular_decay_scale_deg), air = airmass(alt);
  return [alt / 90, Math.sin(az * RAD), Math.cos(az * RAD), air, factor, moonAlt / 90,
    illumination, separation / 180, factor / (score.q0 * air ** score.airmass_exponent)];
}
export function candidateFeatures(candidates: Candidate[], start: number, end: number,
  site: SiteConfig, score: ScoreConfig): { features: Float32Array; eligible: boolean }[] {
  const moments = [start, (start + end) / 2, end], moons = moments.map(t => moonGeometry(t, site));
  const shared = new Map<string, { features: Float32Array; eligible: boolean }>();
  return candidates.map(candidate => {
    const { ra_deg: ra, dec_deg: dec } = candidate, key = `${ra},${dec}`;
    const previous = shared.get(key); if (previous) return previous;
    const parts = moments.map((t, i) => geometry(ra, dec, t, site, score, moons[i]));
    let minimum = Math.min(parts[0][0], parts[2][0]) * 90;
    const hour = mod(mod(280.46061837 + 360.98564736629 * (start / 86400 - 10957.5) + site.longitude_deg, 360) - ra, 360);
    if (mod(180 - hour, 360) <= 360.98564736629 * (end - start) / 86400) {
      const lat = site.latitude_deg * RAD, d = dec * RAD;
      minimum = Math.min(minimum, Math.asin(clipped(Math.sin(lat) * Math.sin(d) - Math.cos(lat) * Math.cos(d))) * DEG);
    }
    const features = new Float32Array([...parts.flat(), minimum / 90, parts.reduce((sum, p) => sum + p[8], 0) / 3]);
    const value = { features, eligible: minimum >= site.minimum_altitude_deg };
    shared.set(key, value); return value;
  });
}
export function prepareConditions(request: ConditionsRequest, timeline?: PublicTimeline): PreparedConditions {
  const began = performance.now();
  if (!request || Object.keys(request).some(k => !["initialize", "now_utc", "bulletins", "forecasts", "candidates"].includes(k)))
    throw new Error("unsupported conditions request fields");
  if (!Array.isArray(request.candidates)) throw new Error("candidates must be an array");
  const { site, scoring: score, survey } = request.initialize, now = utc(request.now_utc);
  const nights = survey.nights.map(n => ({ ...n, start: utc(n.observing_start_utc), end: utc(n.observing_end_utc) }));
  const publicHistory = timeline ?? new PublicTimeline(request.bulletins, request.forecasts);
  const rows: PredictionRow[] = [], grouped = new Map<string, FeatureGroup & { raw: Candidate[]; start: number; end: number }>();
  request.candidates.forEach((candidate, index) => {
    if (Object.keys(candidate).some(k => !["candidate_id", "ra_deg", "dec_deg", "start_utc", "duration_seconds"].includes(k)))
      throw new Error("candidate contains unsupported fields");
    const start = utc(candidate.start_utc), duration = candidate.duration_seconds;
    const limits = request.initialize.instrument?.exposure ?? { min_duration_seconds: 60, max_duration_seconds: 3600 };
    if (!Number.isInteger(duration) || duration < limits.min_duration_seconds || duration > limits.max_duration_seconds)
      throw new Error("requested exposure outside public limits");
    if (![0, 900, 1800, 3600].includes(start - now)) throw new Error("model supports starts at now, +15, +30, +60 minutes");
    if (!Number.isFinite(candidate.ra_deg) || !Number.isFinite(candidate.dec_deg) ||
      candidate.ra_deg < 0 || candidate.ra_deg >= 360 || candidate.dec_deg < -90 || candidate.dec_deg > 90)
      throw new Error("invalid sky coordinates");
    const row: PredictionRow = { candidate_id: candidate.candidate_id ?? String(index), eligible: false };
    rows.push(row);
    const night = nights.find(n => n.start <= start && start < n.end);
    if (!night) { row.reason = "outside observing night"; return; }
    const end = Math.min(start + duration, night.end), key = `${start},${duration}`;
    if (!grouped.has(key)) grouped.set(key, { ...publicHistory.snapshot(now, start, duration, end - start,
      night.night_date, night.end, site, score), raw: [], candidates: [], indices: [], start, end });
    const group = grouped.get(key)!; group.raw.push(candidate); group.indices.push(index);
  });
  const groups: FeatureGroup[] = [];
  for (const group of grouped.values()) {
    const computed = candidateFeatures(group.raw, group.start, group.end, site, score), indices: number[] = [];
    computed.forEach(({ features, eligible }, i) => {
      const index = group.indices[i];
      if (!eligible) { rows[index].reason = "below minimum altitude during exposure"; return; }
      if (!features.every(Number.isFinite)) throw new Error("nonfinite eligible candidate features");
      Object.assign(rows[index], { eligible: true, known_geometry_factor: features[28], actual_duration_seconds: group.end - group.start });
      group.candidates.push(features); indices.push(index);
    });
    groups.push({ context: group.context, history: group.history, candidates: group.candidates, indices });
  }
  return { rows, groups, preprocessing_seconds: (performance.now() - began) / 1000 };
}
