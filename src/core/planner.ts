/** The baseline search, including its placement, duration and assignment order. */
import { DEG, Moon, OMEGA, RAD } from "./astro";
import { mod, tupleCompare, Vec } from "./compat";
import { Survey } from "./state";
import { appendFileSync } from "node:fs";

const PICK_LOG = process.env.PICK_LOG;

export const BASE_DURATIONS = [
  60, 90, 120, 150, 180, 240, 300, 360, 420, 480, 540, 600, 720, 840, 900, 1080,
  1260, 1440, 1620, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
];
/** Extra exposure lengths each candidate field is also placed for, as fractions of the card's longest
 * allowed exposure (1800, 450, 3000, 1350 s when that is one hour); see knob place_refs. */
export const PLACE_FRACTIONS = [1 / 2, 1 / 8, 5 / 6, 3 / 8];
type Info = [number, number, number, number, number, number, number, number];
type Member = [Info, number, number];
type Cells = Map<number, [number, number][]>;
type Context = [
  number,
  number,
  Vec,
  Vec,
  Vec,
  Moon,
  number,
  number,
  number,
  number,
  number,
  number,
  boolean,
  number,
];
export type Pick = [number, number, number];
export interface Plan {
  curve: [number, number, number, number][];
  centre: Vec;
  zone: number;
  q: number;
  anchor: number;
  cell_keys: number[];
  cell_lists: Member[][];
  weights: [number, number];
  u_s: number;
  duration: number;
  nv: number;
  picks: Pick[];
}
export class Planner {
  tab: number;
  cos_reach: number;
  sin_scan: number;
  shifts: number[];
  stat_plans = 0;
  stat_negative = 0;
  stat_negative_seconds = 0;
  constructor(public sv: Survey) {
    this.tab = sv.site.TABLE;
    this.cos_reach = Math.cos((sv.fov * Math.sqrt(2) + 0.02) * RAD);
    this.sin_scan = Math.sin((sv.site.min_alt + 1) * RAD);
    this.shifts = sv.n_side >= 5 ? [-1 / 3, 0, 1 / 3] : [-0.25, 0, 0.25];
    if ((sv.knobs.fine_shifts ?? 0) > 0)
      this.shifts = [-0.4, -0.2, 0, 0.2, 0.4];
  }
  _zone_bonus_weight(
    z: number,
    a_clear: number,
    east: Vec,
    night: number,
  ): number {
    const sv = this.sv,
      cat = sv.cat,
      c = cat.zcoarse[z];
    const rising =
      cat.zcx[z] * east[0] + cat.zcy[z] * east[1] + cat.zcz[z] * east[2] > 0;
    let best = cat.season_future[c][night];
    if (rising && cat.season_peak[c][night] > best)
      best = cat.season_peak[c][night];
    if (a_clear > best) best = a_clear;
    const timing = best > 0 ? a_clear / best : 1,
      urgency = sv.urgency[c];
    return (sv.plan_mult?.[c] ?? 1) * sv.plan_levers.required * sv.p_req * urgency * timing ** (sv.knobs.timing_power / urgency);
  }
  /** Experiment (knob sci_timing_power, 0 = off): science value scaled by how close the zone is to the best
   * transmission it will still reach (later tonight if it is rising, or on a later night), to that power. */
  _sci_timing(z: number, a_clear: number, east: Vec, night: number): number {
    const power = this.sv.knobs.sci_timing_power ?? 0;
    if (power <= 0) return 1;
    const cat = this.sv.cat,
      c = cat.zcoarse[z],
      rising = cat.zcx[z] * east[0] + cat.zcy[z] * east[1] + cat.zcz[z] * east[2] > 0;
    let best = cat.season_future[c][night];
    if (rising && cat.season_peak[c][night] > best) best = cat.season_peak[c][night];
    return best > a_clear ? (a_clear / best) ** power : 1;
  }
  plan(
    t: number,
    night: number,
    k_zones: number,
    t_floor: number,
    t_ceiling: number,
    calibrate: boolean,
  ): Plan | null {
    const sv = this.sv,
      cat = sv.cat,
      site = sv.site,
      sky = sv.sky,
      [lst, zen, north, east] = site.frame(t),
      [zx, zy, zz] = zen;
    const rem = sv.nights[night][1] - t;
    let t_cap = Math.trunc(Math.min(sv.t_max, rem, t_ceiling));
    if (t_cap < sv.t_min) t_cap = sv.t_min;
    if (t_floor > t_cap) t_floor = t_cap;
    const moon = site.moon(t + Math.min(t_cap, 1200) / 2),
      [rho, w_level] = sky.levels();
    let price = sv.price;
    const directional = sky.directional,
      moon_on = moon[3] > 0.01,
      [ex, ey, ez] = east,
      [nx, ny, nz] = north,
      scored: [number, number][] = [];
    for (let z = 0; z < cat.nz; z++) {
      const sa = cat.zcx[z] * zx + cat.zcy[z] * zy + cat.zcz[z] * zz;
      if (sa < this.sin_scan) continue;
      let pot = sv.zpot[z];
      if (pot <= 1e-9) pot = sv.knobs.deep_weight * sv.plan_levers.revisit * sv.zdeep[z];
      const flag = sv.zflag[z];
      if (pot <= 1e-9 && !flag && !calibrate) continue;
      const a_clear = site.atab[Math.trunc(sa * this.tab)];
      let a = a_clear;
      if (moon_on)
        a *= site.lunar_factor(cat.zcx[z], cat.zcy[z], cat.zcz[z], moon);
      if (directional) {
        const alt = Math.asin(sa) * DEG;
        if (alt < 82) {
          const az = mod(
            Math.atan2(
              cat.zcx[z] * ex + cat.zcy[z] * ey + cat.zcz[z] * ez,
              cat.zcx[z] * nx + cat.zcy[z] * ny + cat.zcz[z] * nz,
            ) * DEG,
            360,
          );
          a *= sky.dir_factor(alt, az);
        }
      }
      const q = rho * a;
      if (q <= 1e-4) continue;
      let s = pot * (q < 1.25 ? q : 1.25) * sv.sci_urgency[cat.zcoarse[z]] * this._sci_timing(z, a_clear, east, night);
      if (flag) {
        const req = sv.zreq[z];
        if (req.length) {
          const weight = this._zone_bonus_weight(z, a_clear, east, night);
          for (const j of req) {
            const need = (sv.need_req[j] / q) * 1.25;
            if (need <= t_cap)
              s += weight - price * (need > sv.T_REF ? need - sv.T_REF : 0);
            else if ((sv.need_req[j] / q) * 0.95 <= t_cap) s += 0.25 * weight;
          }
        }
        for (const j of sv.zextra.get(z) ?? []) {
          const [bonus, deadline, threshold] = sv.extra.get(j)!,
            need = ((threshold * sv.f0t0) / cat.flux[j] / q) * 1.25;
          if (need <= t_cap && t + need <= deadline)
            s += bonus - price * (need > sv.T_REF ? need - sv.T_REF : 0);
        }
      }
      if (calibrate) s += 1e-3 * cat.zmembers[z].length;
      if (s > 0) scored.push([s, z]);
    }
    if (!scored.length) return null;
    // Python nlargest compares the full (score, zone) tuple. Short lists retain input order.
    const zones =
      scored.length > k_zones
        ? scored.sort((a, b) => tupleCompare(b, a)).slice(0, k_zones)
        : scored;
    const ctx: Context = [
        t,
        lst,
        zen,
        north,
        east,
        moon,
        rho,
        w_level,
        t_cap,
        t_floor,
        price,
        night,
        calibrate,
        rem,
      ],
      plans: Plan[] = [];
    // A field is placed for one reference exposure length. With place_refs > 0 it is also placed
    // for shorter and longer ones, and all placements compete under the same rule below.
    const refs = calibrate
      ? []
      : PLACE_FRACTIONS.slice(0, Math.trunc(sv.knobs.place_refs ?? 0)).map((f) =>
          Math.round(f * sv.t_max),
        );
    for (const [, z] of zones) {
      const plan = this._evaluate(z, ctx);
      if (plan !== null) plans.push(plan);
      for (const ref of refs) {
        if (ref > t_cap || ref < t_floor) continue;
        const other = this._evaluate(z, ctx, ref);
        if (other !== null) plans.push(other);
      }
    }
    if (!plans.length) return null;
    if (!calibrate) {
      const lo = Math.min(300, t_cap);
      let rate_now = 0;
      for (const plan of plans)
        for (const [duration, eff, total] of plan.curve)
          if (duration >= lo && eff > 0 && total / eff > rate_now)
            rate_now = total / eff;
      const capped = sv.knobs.price_cap * rate_now;
      this.stat_plans++;
      if (capped < price) {
        price = capped;
        this.stat_negative++;
      }
    }
    let best_plan: Plan | null = null,
      best_score: number | null = null;
    for (const plan of plans) {
      let top: [number, number, number] | null = null;
      for (const [duration, eff, total, science] of plan.curve) {
        const nv = total - price * eff;
        if (top === null || nv > top[0] + 1e-9) top = [nv, duration, science];
      }
      if (top === null) continue;
      const score = top[0] + (plan.u_s - 1) * top[2];
      if (best_score === null || score > best_score) {
        best_score = score;
        best_plan = plan;
        plan.duration = top[1];
        plan.nv = top[0];
      }
    }
    if (best_plan === null) return null;
    const picks = this._picks(best_plan, t, rem, calibrate);
    if (!picks.length) return null;
    best_plan.picks = picks;
    return best_plan;
  }
  _evaluate(z: number, ctx: Context, t_ref = this.sv.T_REF): Plan | null {
    const [
      t,
      lst,
      zen,
      north,
      east,
      moon,
      rho,
      w_level,
      t_cap,
      t_floor,
      price,
      night,
      calibrate,
      rem,
    ] = ctx;
    const sv = this.sv,
      cat = sv.cat,
      site = sv.site,
      sky = sv.sky,
      [zx, zy, zz] = zen,
      [cx, cy, cz] = [cat.zcx[z], cat.zcy[z], cat.zcz[z]];
    const sa = cx * zx + cy * zy + cz * zz;
    if (sa <= 0) return null;
    const a_clear = site.atab[Math.trunc(sa * this.tab)],
      a = a_clear * site.lunar_factor(cx, cy, cz, moon);
    let d_factor = 1;
    if (sky.directional) {
      const alt = Math.asin(sa) * DEG,
        az = mod(
          Math.atan2(
            cx * east[0] + cy * east[1] + cz * east[2],
            cx * north[0] + cy * north[1] + cz * north[2],
          ) * DEG,
          360,
        );
      d_factor = sky.dir_factor(alt, az);
    }
    const q = rho * a * d_factor;
    if (q <= 1e-4) return null;
    const qf = q / sv.f0t0,
      mhat = sky.expected_multiplier(w_level * a, t),
      thr = sv.thr_req,
      thr_u = sv.thr_unif;
    const breq =
        sv.required_left > 0
          ? this._zone_bonus_weight(z, a_clear, east, night)
          : 0,
      band_bonus = sv.any_band_bonus ? sv.band_bonus : null;
    const t0 = Math.min(t_cap, Math.max(t_floor, t_ref)),
      floor_value = calibrate ? 1e-4 : 0,
      u_s = sv.sci_urgency[cat.zcoarse[z]] * this._sci_timing(z, a_clear, east, night),
      delta = sv.knobs.partial_discount;
    const bonus_of = (j: number): [number, number, number] => {
      let w_b = 0,
        th = thr,
        dl = 1e18;
      if (cat.required[j] && sv.gmax[j] < thr) w_b = breq;
      if (band_bonus !== null && sv.gmax[j] < thr_u) {
        const bb = band_bonus.get(cat.band[j])!;
        if (bb > 0) {
          w_b += bb;
          if (thr_u > th) th = thr_u;
        }
      }
      const e = sv.extra.get(j);
      if (e !== undefined) {
        w_b += e[0];
        dl = e[1];
        if (e[2] > th) th = e[2];
      }
      return [w_b, th, dl];
    };
    let anchor = -1,
      anchor_value = 0,
      t_plan = t0;
    for (const j of cat.zmembers[z]) {
      const hm = cat.hmax[j];
      let tmax: number;
      if (hm < 180) {
        const h = mod(lst - cat.ra[j] + 180, 360) - 180;
        if (h < -hm || h > hm) continue;
        tmax = (hm - h) / OMEGA - 8;
        if (tmax < sv.t_min) continue;
      } else tmax = 1e9;
      let tj = t0 <= tmax ? t0 : tmax;
      const rate = cat.flux[j] * qf;
      let g = rate * tj,
        v = cat.weight[j] * mhat * Math.min(g, 1) - sv.held(j);
      if (v <= 0) {
        v = 0;
        const t_long = Math.min(t_cap, tmax);
        if (t_long > tj) {
          const g_long = rate * t_long,
            v_long = cat.weight[j] * mhat * Math.min(g_long, 1) - sv.held(j);
          if (v_long > 0) {
            v = 0.25 * v_long;
            tj = t_long;
            g = g_long;
          }
        }
      }
      if (g < 1) v *= delta;
      const [w_b, th, dl] = bonus_of(j);
      if (w_b > 0 && rate > 0) {
        let tb = (th * 1.25) / rate;
        const limit = Math.min(tmax, t_cap, dl - t);
        if (tb > limit) tb = limit;
        if (tb >= sv.t_min) {
          const ratio = (rate * tb) / th,
            conf = ratio >= 1.25 ? 1 : ratio > 0.85 ? (ratio - 0.85) * 2.5 : 0;
          if (conf > 0) {
            const gb = rate * tb;
            let vb = Math.max(
              0,
              cat.weight[j] * mhat * Math.min(gb, 1) - sv.held(j),
            );
            if (gb < 1) vb *= delta;
            vb += w_b * conf - price * (tb > t0 ? tb - t0 : 0);
            if (vb > v) {
              v = vb;
              tj = Math.max(tb, t_floor);
            }
          }
        }
      }
      if (v + floor_value > anchor_value) {
        anchor_value = v + floor_value;
        anchor = j;
        t_plan = tj;
      }
    }
    if (anchor < 0) return null;
    if (calibrate) t_plan = sv.t_min;
    const [ax, ay, az_] = [cat.ux[anchor], cat.uy[anchor], cat.uz[anchor]],
      s_a = ax * zx + ay * zy + az_ * zz,
      c_a = Math.sqrt(Math.max(1e-12, 1 - s_a * s_a));
    const [upx, upy, upz] = [
      (zx - s_a * ax) / c_a,
      (zy - s_a * ay) / c_a,
      (zz - s_a * az_) / c_a,
    ];
    const [rtx, rty, rtz] = [
      (ay * zz - az_ * zy) / c_a,
      (az_ * zx - ax * zz) / c_a,
      (ax * zy - ay * zx) / c_a,
    ];
    const items: [number, number, number, number][] = [],
      info = new Map<number, Info>(),
      beacons: [number, number, number][] = [];
    for (const j of cat.reach(z)) {
      const d = cat.ux[j] * ax + cat.uy[j] * ay + cat.uz[j] * az_;
      if (d < this.cos_reach) continue;
      const hm = cat.hmax[j];
      let tmax: number;
      if (hm < 180) {
        const h = mod(lst - cat.ra[j] + 180, 360) - 180;
        if (h < -hm || h > hm) continue;
        tmax = (hm - h) / OMEGA - 8;
        if (tmax < sv.t_min) continue;
      } else tmax = 1e9;
      const rate = cat.flux[j] * qf,
        tj = Math.min(t_plan, tmax),
        g = rate * tj;
      let v = Math.max(0, cat.weight[j] * mhat * Math.min(g, 1) - sv.held(j));
      if (g < 1) v *= delta;
      let [w_b, th, dl] = bonus_of(j);
      if (w_b > 0) {
        if (t + tj <= dl) {
          const ratio = g / th;
          if (ratio >= 1.25) v += w_b;
          else if (ratio > 0.85) v += w_b * (ratio - 0.85) * 2.5;
        } else w_b = 0;
      }
      v += floor_value;
      if (v <= 1e-9) {
        if (rate * t_plan >= 2.5 && tmax >= t_plan)
          beacons.push([rate, j, tmax]);
        continue;
      }
      const x =
          ((cat.ux[j] * rtx + cat.uy[j] * rty + cat.uz[j] * rtz) / d) * DEG,
        y = ((cat.ux[j] * upx + cat.uy[j] * upy + cat.uz[j] * upz) / d) * DEG;
      items.push([x, y, v, j]);
      info.set(j, [
        rate,
        cat.weight[j] * mhat,
        sv.held(j),
        w_b,
        th,
        dl,
        tmax,
        v,
      ]);
    }
    const weights: [number, number] = [1, delta];
    if (!items.length) return null;
    const n = sv.n_side,
      p = sv.pitch,
      side = 2 * n - 1,
      stride = side + 1,
      mid = (n - 1) / 2,
      grid = Array(side * side).fill(0);
    // The fibre lattice is laid with the anchor at a cell centre and the best n x n window of cells taken.
    // lattice_search > 0 (experiment) also tries the lattice displaced by every fine shift before the window
    // is chosen, so the window and the displacement are found together rather than one after the other.
    const lattice: [number, number][] =
      (sv.knobs.lattice_search ?? 0) > 0 && !calibrate
        ? [[0, 0] as [number, number], ...this.shifts.flatMap((sx) => this.shifts.filter((sy) => sx !== 0 || sy !== 0).map((sy) => [sx, sy] as [number, number]))]
        : [[0, 0]];
    const table = Array(stride * stride).fill(0);
    let best_sum = -1,
      ar_best = 0,
      ac_best = 0,
      lx_best = 0,
      ly_best = 0;
    for (const [lx, ly] of lattice) {
      grid.fill(0);
      for (const [x, y, v] of items) {
        const dc = Math.floor(x / p + 0.5 - lx),
          dr = Math.floor(y / p + 0.5 - ly);
        if (-n < dc && dc < n && -n < dr && dr < n) {
          const k = (dr + n - 1) * side + dc + n - 1;
          if (v > grid[k]) grid[k] = v;
        }
      }
      for (let r = 0; r < side; r++) {
        let run = 0;
        const base = (r + 1) * stride,
          prev = r * stride,
          row = r * side;
        for (let c = 0; c < side; c++) {
          run += grid[row + c];
          table[base + c + 1] = table[prev + c + 1] + run;
        }
      }
      for (let ar = 0; ar < n; ar++) {
        const r0 = n - 1 - ar,
          top_row = (r0 + n) * stride,
          low_row = r0 * stride;
        for (let ac = 0; ac < n; ac++) {
          const c0 = n - 1 - ac;
          let total =
            table[top_row + c0 + n] -
            table[low_row + c0 + n] -
            table[top_row + c0] +
            table[low_row + c0];
          total -= 1e-7 * (Math.abs(ar - mid) + Math.abs(ac - mid));
          if (total > best_sum) {
            best_sum = total;
            ar_best = ar;
            ac_best = ac;
            lx_best = lx;
            ly_best = ly;
          }
        }
      }
    }
    const xa = (ac_best + 0.5 - n / 2 - lx_best) * p * RAD,
      ya = (ar_best + 0.5 - n / 2 - ly_best) * p * RAD;
    let vx = ax - xa * rtx - ya * upx,
      vy = ay - xa * rty - ya * upy,
      vz = az_ - xa * rtz - ya * upz,
      norm = Math.sqrt(vx * vx + vy * vy + vz * vz);
    let centre: Vec = [vx / norm, vy / norm, vz / norm];
    const candidates = items.map((item) => item[3]);
    if (beacons.length && !calibrate) {
      beacons.sort((a, b) => tupleCompare(b, a));
      for (const [rate_b, j_b, tmax_b] of beacons.slice(0, 8))
        info.set(j_b, [rate_b, 0, 0, 0, thr, 1e18, tmax_b, 0]);
    }
    const frame = this._field_frame(centre, zen);
    if (frame === null) return null;
    const [, up_c, rt_c] = frame,
      scale = DEG / p,
      half = n / 2,
      projected: [number, number, number][] = [];
    for (const j of candidates) {
      const d =
        cat.ux[j] * centre[0] + cat.uy[j] * centre[1] + cat.uz[j] * centre[2];
      if (d <= 0.5) continue;
      const gx =
          ((cat.ux[j] * rt_c[0] + cat.uy[j] * rt_c[1] + cat.uz[j] * rt_c[2]) /
            d) *
            scale +
          half,
        gy =
          ((cat.ux[j] * up_c[0] + cat.uy[j] * up_c[1] + cat.uz[j] * up_c[2]) /
            d) *
            scale +
          half;
      if (-0.6 < gx && gx < n + 0.6 && -0.6 < gy && gy < n + 0.6)
        projected.push([gx, gy, info.get(j)![7]]);
    }
    let best_shift: [number, number] = [0, 0],
      best_value = -1;
    for (const sx of this.shifts)
      for (const sy of this.shifts) {
        const seen = new Map<number, number>();
        let value = 0;
        for (const [gx, gy, v] of projected) {
          const hx = gx - sx,
            hy = gy - sy;
          if (0 <= hx && hx < n && 0 <= hy && hy < n) {
            const k = Math.trunc(hy) * n + Math.trunc(hx),
              old = seen.get(k) ?? 0;
            if (v > old) {
              value += v - old;
              seen.set(k, v);
            }
          }
        }
        if (sx === 0 && sy === 0) value += 1e-7;
        if (value > best_value) {
          best_value = value;
          best_shift = [sx, sy];
        }
      }
    if (best_shift[0] !== 0 || best_shift[1] !== 0) {
      const sx = best_shift[0] * p * RAD,
        sy = best_shift[1] * p * RAD;
      vx = centre[0] + sx * rt_c[0] + sy * up_c[0];
      vy = centre[1] + sx * rt_c[1] + sy * up_c[1];
      vz = centre[2] + sx * rt_c[2] + sy * up_c[2];
      norm = Math.sqrt(vx * vx + vy * vy + vz * vz);
      centre = [vx / norm, vy / norm, vz / norm];
    }
    const margin_alt = (sv.offset.margin_alt + 1e-5) / p;
    if (beacons.length && !calibrate)
      candidates.push(...beacons.slice(0, 8).map((b) => b[1]));
    const best_cells = this._exact_cells(
      centre,
      zen,
      candidates,
      margin_alt,
      sv.offset.margin_az,
      !sv.offset.active,
    );
    if (best_cells === null) return null;
    const durations = this._durations(
        best_cells,
        info,
        t_floor,
        t_cap,
        calibrate,
      ),
      cell_keys = [...best_cells.keys()],
      cell_lists = [...best_cells.values()].map((members) =>
        members.map(([j, phit]) => [info.get(j)!, phit, j] as Member),
      ),
      curve: Plan["curve"] = [];
    for (const duration of durations) {
      const eff = Math.min(duration, rem);
      let total = 0,
        science = 0;
      for (const members of cell_lists) {
        let top_v = 0,
          top_sci = 0;
        for (const [[rate, cap, base, w_b, th, dl, tmax], phit] of members) {
          if (tmax < eff) continue;
          const g = rate * eff;
          let v = Math.max(0, cap * Math.min(g, 1) - base);
          v *= g >= 1 ? weights[0] : weights[1];
          const sci = v;
          if (w_b > 0 && t + eff <= dl) {
            const ratio = g / th;
            if (ratio >= 1.25) v += w_b;
            else if (ratio > 0.85) v += w_b * (ratio - 0.85) * 2.5;
          }
          v *= phit;
          if (v > top_v) {
            top_v = v;
            top_sci = sci * phit;
          }
        }
        total += top_v;
        science += top_sci;
      }
      curve.push([duration, eff, total, science]);
    }
    if (!curve.length) return null;
    return {
      curve,
      centre,
      zone: z,
      q,
      anchor,
      cell_keys,
      cell_lists,
      weights,
      u_s,
      duration: 0,
      nv: 0,
      picks: [],
    };
  }
  _picks(plan: Plan, t: number, rem: number, calibrate: boolean): Pick[] {
    const eff = Math.min(plan.duration, rem),
      floor_value = calibrate ? 1e-4 : 0,
      picks: Pick[] = [],
      spare: [number, number, number][] = [];
    for (let i = 0; i < plan.cell_keys.length; i++) {
      const cell = plan.cell_keys[i],
        members = plan.cell_lists[i];
      let top_v = floor_value * 0.5,
        top_j = -1,
        top_phit = 0;
      for (const [[rate, cap, base, w_b, th, dl, tmax], phit, j] of members) {
        if (tmax < eff) continue;
        const g = rate * eff;
        let v = Math.max(0, cap * Math.min(g, 1) - base);
        v *= g >= 1 ? plan.weights[0] : plan.weights[1];
        if (w_b > 0 && t + eff <= dl) {
          const ratio = g / th;
          if (ratio >= 1.25) v += w_b;
          else if (ratio > 0.85) v += w_b * (ratio - 0.85) * 2.5;
        }
        v = (v + floor_value) * phit;
        if (v > top_v) {
          top_v = v;
          top_j = j;
          top_phit = phit;
        }
      }
      if (top_j >= 0) picks.push([cell, top_j, top_phit]);
      else if (!calibrate)
        for (const [[rate, cap, , , , , tmax], phit, j] of members)
          if (cap === 0 && phit >= 0.999 && tmax >= eff && rate * eff >= 2)
            spare.push([rate, cell, j]);
    }
    // Diagnostic record (PICK_LOG=<file>): per exposure, for every fibre cell the chosen target and the other
    // candidates that could have been observed for the same duration.
    if (PICK_LOG && !calibrate) {
      const ids = this.sv.cat.ids,
        chosen = new Map(picks.map(([cell, j]) => [cell, j])),
        cells = plan.cell_keys.map((cell, i) => [
          chosen.has(cell) ? ids[chosen.get(cell)!] : null,
          plan.cell_lists[i].filter(([info, , j]) => info[6] >= eff && info[1] > 0 && j !== chosen.get(cell)).map(([, , j]) => ids[j]),
        ]);
      appendFileSync(PICK_LOG, JSON.stringify({ t, duration: eff, cells }) + "\n");
    }
    if (picks.length && spare.length) {
      spare.sort((a, b) => tupleCompare(b, a));
      const used = new Set<number>();
      for (const [, cell, j] of spare)
        if (!used.has(cell)) {
          used.add(cell);
          picks.push([cell, j, 1]);
          if (used.size === 2) break;
        }
    }
    return picks;
  }
  _field_frame(centre: Vec, zen: Vec): [number, Vec, Vec] | null {
    const [cx, cy, cz] = centre,
      [zx, zy, zz] = zen,
      sa = cx * zx + cy * zy + cz * zz;
    if (sa <= 0.02 || sa >= 0.9986) return null;
    const ca = Math.sqrt(1 - sa * sa);
    return [
      sa,
      [(zx - sa * cx) / ca, (zy - sa * cy) / ca, (zz - sa * cz) / ca],
      [
        (cy * zz - cz * zy) / ca,
        (cz * zx - cx * zz) / ca,
        (cx * zy - cy * zx) / ca,
      ],
    ];
  }
  _exact_cells(
    centre: Vec,
    zen: Vec,
    candidates: number[],
    margin_alt: number,
    margin_az_deg: number,
    hard: boolean,
  ): Cells | null {
    const sv = this.sv,
      cat = sv.cat,
      [cx, cy, cz] = centre,
      frame = this._field_frame(centre, zen);
    if (frame === null) return null;
    const [sa, [upx, upy, upz], [rtx, rty, rtz]] = frame,
      ca = Math.sqrt(1 - sa * sa),
      n = sv.n_side,
      scale = DEG / sv.pitch,
      half = n / 2,
      margin_az = (margin_az_deg * ca + 1e-5) / sv.pitch,
      cells: Cells = new Map();
    for (const j of candidates) {
      const d = cat.ux[j] * cx + cat.uy[j] * cy + cat.uz[j] * cz;
      if (d <= 0.5) continue;
      const fx =
        ((cat.ux[j] * rtx + cat.uy[j] * rty + cat.uz[j] * rtz) / d) * scale +
        half;
      if (fx < 0 || fx >= n) continue;
      const fy =
        ((cat.ux[j] * upx + cat.uy[j] * upy + cat.uz[j] * upz) / d) * scale +
        half;
      if (fy < 0 || fy >= n) continue;
      const col = Math.trunc(fx),
        row = Math.trunc(fy),
        ex = fx - col,
        ey = fy - row,
        mx = Math.min(ex, 1 - ex),
        my = Math.min(ey, 1 - ey);
      let phit = 1;
      if (mx < margin_az) {
        if (hard) continue;
        phit *= 0.5 + (0.5 * mx) / margin_az;
      }
      if (my < margin_alt) {
        if (hard) continue;
        phit *= 0.5 + (0.5 * my) / margin_alt;
      }
      const key = row * n + col,
        members = cells.get(key);
      if (members === undefined) cells.set(key, [[j, phit]]);
      else members.push([j, phit]);
    }
    return cells.size ? cells : null;
  }
  _durations(
    cells: Cells,
    info: Map<number, Info>,
    t_floor: number,
    t_cap: number,
    calibrate: boolean,
  ): number[] {
    const sv = this.sv;
    if (calibrate) return [sv.t_min];
    const lo = Math.max(sv.t_min, t_floor),
      values = new Set(BASE_DURATIONS.filter((d) => lo <= d && d <= t_cap));
    values.add(t_cap);
    values.add(Math.min(lo, t_cap));
    for (const members of cells.values())
      for (const [j] of members) {
        const [rate, , , w_b, th, , tmax] = info.get(j)!;
        if (rate <= 0) continue;
        if (w_b > 0) {
          const tb = Math.ceil((th * 1.25) / rate);
          if (lo <= tb && tb <= t_cap) values.add(tb);
        }
        if (tmax < t_cap) {
          const tm = Math.trunc(tmax);
          if (lo <= tm) values.add(tm);
        }
        const sat = Math.ceil(1.02 / rate / 30) * 30;
        if (lo <= sat && sat <= t_cap) values.add(sat);
      }
    return [...values].sort((a, b) => a - b);
  }
}
