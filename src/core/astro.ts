/** Public sky geometry; same formulas and operation order as the Python baseline. */
import { mod, Vec, Payload } from "./compat";
export const RAD = Math.PI / 180,
  DEG = 180 / Math.PI;
export const SIDEREAL_DEG_PER_DAY = 360.98564736629,
  OMEGA = SIDEREAL_DEG_PER_DAY / 86400;
const ZENITH_RAW = 1 / (1 + 0.50572 * 96.07995 ** -1.6364);
export function parse_utc(text: string): number {
  const value = Date.parse(text) / 1000;
  if (!Number.isFinite(value)) throw new Error(`invalid UTC time: ${text}`);
  return value;
}
export function format_utc(ts: number): string {
  return new Date(Math.floor(ts) * 1000)
    .toISOString()
    .replace(/\.\d{3}Z$/, "Z");
}
export function days_since_j2000(ts: number): number {
  return ts / 86400 + 2440587.5 - 2451545;
}
export function lst_deg(ts: number, longitude_deg: number): number {
  return mod(
    280.46061837 + 360.98564736629 * days_since_j2000(ts) + longitude_deg,
    360,
  );
}
export function unit_vector(ra_deg: number, dec_deg: number): Vec {
  const ra = ra_deg * RAD,
    dec = dec_deg * RAD,
    c = Math.cos(dec);
  return [c * Math.cos(ra), c * Math.sin(ra), Math.sin(dec)];
}
export function sun_radec(ts: number): [number, number] {
  const d = days_since_j2000(ts),
    longitude = mod(280.46 + 0.9856474 * d, 360);
  const g = mod(357.528 + 0.9856003 * d, 360) * RAD;
  const lam =
    mod(longitude + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g), 360) * RAD;
  const eps = (23.439 - 0.0000004 * d) * RAD;
  return [
    mod(Math.atan2(Math.cos(eps) * Math.sin(lam), Math.cos(lam)) * DEG, 360),
    Math.asin(Math.sin(eps) * Math.sin(lam)) * DEG,
  ];
}
export function moon_radec(ts: number): [number, number] {
  const d = days_since_j2000(ts),
    lm = mod(218.316 + 13.176396 * d, 360) * RAD;
  const mm = mod(134.963 + 13.064993 * d, 360) * RAD,
    fm = mod(93.272 + 13.22935 * d, 360) * RAD;
  const lam = lm + 6.289 * RAD * Math.sin(mm),
    beta = 5.128 * RAD * Math.sin(fm),
    eps = (23.439 - 0.0000004 * d) * RAD;
  const x = Math.cos(lam) * Math.cos(beta),
    y =
      Math.sin(lam) * Math.cos(beta) * Math.cos(eps) -
      Math.sin(beta) * Math.sin(eps);
  const z =
    Math.sin(lam) * Math.cos(beta) * Math.sin(eps) +
    Math.sin(beta) * Math.cos(eps);
  return [mod(Math.atan2(y, x) * DEG, 360), Math.asin(z) * DEG];
}
export function airmass(alt_deg: number): number {
  if (alt_deg <= 0) return Infinity;
  const z = 90 - alt_deg;
  return (
    1 / (Math.cos(z * RAD) + 0.50572 * (96.07995 - z) ** -1.6364) / ZENITH_RAW
  );
}
export type Frame = [number, Vec, Vec, Vec];
export type Moon = [number, number, number, number];
export class Site {
  readonly TABLE = 2000;
  lat: number;
  lon: number;
  sin_lat: number;
  cos_lat: number;
  min_alt: number;
  sin_min_alt: number;
  beta: number;
  moon_penalty: number;
  moon_alt_exp: number;
  moon_scale: number;
  atab: number[];
  constructor(
    latitude: number,
    longitude: number,
    minimum: number,
    exponent: number,
    lunar: Payload,
  ) {
    this.lat = Number(latitude);
    this.lon = Number(longitude);
    this.sin_lat = Math.sin(this.lat * RAD);
    this.cos_lat = Math.cos(this.lat * RAD);
    this.min_alt = Number(minimum);
    this.sin_min_alt = Math.sin(this.min_alt * RAD);
    this.beta = Number(exponent);
    this.moon_penalty = Number(lunar.maximum_penalty ?? 0);
    this.moon_alt_exp = Number(lunar.altitude_exponent ?? 1);
    this.moon_scale = Number(lunar.angular_decay_scale_deg ?? 35);
    this.atab = Array(this.TABLE + 2).fill(0);
    for (let i = 1; i <= this.TABLE; i++)
      this.atab[i] =
        airmass(Math.asin(Math.min(1, i / this.TABLE)) * DEG) ** -this.beta;
    this.atab[this.TABLE + 1] = this.atab[this.TABLE];
  }
  transmission(sin_alt: number): number {
    return sin_alt <= 0
      ? 0
      : airmass(Math.asin(Math.min(1, sin_alt)) * DEG) ** -this.beta;
  }
  frame(ts: number): Frame {
    const lst = lst_deg(ts, this.lon),
      cl = Math.cos(lst * RAD),
      sl = Math.sin(lst * RAD);
    return [
      lst,
      [this.cos_lat * cl, this.cos_lat * sl, this.sin_lat],
      [-this.sin_lat * cl, -this.sin_lat * sl, this.cos_lat],
      [-sl, cl, 0],
    ];
  }
  moon(ts: number): Moon {
    const [mra, mdec] = moon_radec(ts),
      [sra, sdec] = sun_radec(ts),
      m = unit_vector(mra, mdec),
      s = unit_vector(sra, sdec);
    const cos_psi = Math.max(
      -1,
      Math.min(1, m[0] * s[0] + m[1] * s[1] + m[2] * s[2]),
    );
    const illumination = (1 - cos_psi) / 2,
      [, zen] = this.frame(ts),
      sin_alt = m[0] * zen[0] + m[1] * zen[1] + m[2] * zen[2];
    return [
      m[0],
      m[1],
      m[2],
      sin_alt <= 0 || this.moon_penalty <= 0
        ? 0
        : this.moon_penalty * illumination * sin_alt ** this.moon_alt_exp,
    ];
  }
  lunar_factor(ux: number, uy: number, uz: number, moon: Moon): number {
    if (moon[3] <= 0) return 1;
    const cos_sep = Math.max(
      -1,
      Math.min(1, ux * moon[0] + uy * moon[1] + uz * moon[2]),
    );
    return Math.max(
      0,
      Math.min(
        1,
        1 - moon[3] * Math.exp((-Math.acos(cos_sep) * DEG) / this.moon_scale),
      ),
    );
  }
  hour_angle_limit(dec_deg: number): number {
    const dec = dec_deg * RAD,
      denom = this.cos_lat * Math.cos(dec);
    if (Math.abs(denom) < 1e-12)
      return this.sin_lat * Math.sin(dec) >= this.sin_min_alt ? 180 : -1;
    const cos_h = (this.sin_min_alt - this.sin_lat * Math.sin(dec)) / denom;
    return cos_h > 1 ? -1 : cos_h < -1 ? 180 : Math.acos(cos_h) * DEG;
  }
  altaz(vec: Vec, ts: number): [number, number] {
    const [, zen, north, east] = this.frame(ts),
      sin_alt = vec[0] * zen[0] + vec[1] * zen[1] + vec[2] * zen[2];
    return [
      Math.asin(Math.max(-1, Math.min(1, sin_alt))) * DEG,
      mod(
        Math.atan2(
          vec[0] * east[0] + vec[1] * east[1] + vec[2] * east[2],
          vec[0] * north[0] + vec[1] * north[1] + vec[2] * north[2],
        ) * DEG,
        360,
      ),
    ];
  }
}
