import { DEG, OMEGA, RAD, lst_deg, Site, unit_vector } from "./astro";
import { floorDiv, mod, Payload, sum } from "./compat";
const overlap = (a0: number, a1: number, b0: number, b1: number): number =>
  Math.max(0, Math.min(a1, b1) - Math.max(a0, b0));
const key = (b: number, k: number): string => `${b},${k}`;
export class Grid {
  nb: number;
  ncell: number[] = [];
  constructor(public size: number) {
    this.nb = Math.ceil(180 / size);
    for (let b = 0; b < this.nb; b++)
      this.ncell.push(
        Math.max(
          1,
          Math.trunc((360 * Math.cos((-90 + (b + 0.5) * size) * RAD)) / size),
        ),
      );
  }
  locate(ra: number, dec: number): [number, number] {
    const b = Math.min(
        this.nb - 1,
        Math.max(0, Math.trunc((dec + 90) / this.size)),
      ),
      n = this.ncell[b];
    return [b, Math.min(n - 1, Math.trunc((ra / 360) * n))];
  }
  cells_near(b: number, k: number, radius: number): [number, number][] {
    const reach_bands = Math.ceil(radius / this.size),
      width = 360 / this.ncell[b],
      ra_lo = k * width,
      ra_hi = ra_lo + width;
    const extreme = Math.max(
      Math.abs(-90 + b * this.size - radius),
      Math.abs(-90 + (b + 1) * this.size + radius),
    );
    const delta = extreme >= 88 ? 360 : radius / Math.cos(extreme * RAD),
      out: [number, number][] = [];
    for (
      let bb = Math.max(0, b - reach_bands);
      bb <= Math.min(this.nb - 1, b + reach_bands);
      bb++
    ) {
      const n = this.ncell[bb],
        w = 360 / n;
      if (ra_hi - ra_lo + 2 * delta >= 360 - w) {
        for (let kk = 0; kk < n; kk++) out.push([bb, kk]);
        continue;
      }
      const seen = new Set<number>();
      for (
        let kk = Math.floor((ra_lo - delta) / w);
        kk <= Math.floor((ra_hi + delta) / w);
        kk++
      ) {
        const wrapped = mod(kk, n);
        if (!seen.has(wrapped)) {
          seen.add(wrapped);
          out.push([bb, wrapped]);
        }
      }
    }
    return out;
  }
}
export class Catalog {
  n: number;
  ids: string[];
  ra: number[];
  dec: number[];
  flux: number[];
  weight: number[];
  required: boolean[];
  index: Map<string, number>;
  ux: number[] = [];
  uy: number[] = [];
  uz: number[] = [];
  hmax: number[] = [];
  band_width: number;
  band: number[];
  band_total = new Map<number, number>();
  fov: number;
  grid!: Grid;
  zone_of: number[] = [];
  zmembers: number[][] = [];
  zkeys: [number, number][] = [];
  key_to_zone = new Map<string, number>();
  nz = 0;
  zcx: number[] = [];
  zcy: number[] = [];
  zcz: number[] = [];
  zra: number[] = [];
  zdec: number[] = [];
  zhmax: number[] = [];
  private _reach = new Map<number, number[]>();
  reach_radius = 0;
  zcoarse: number[] = [];
  nc = 0;
  season_vis: number[][] = [];
  /** Visible seconds of each coarse cell in each night (season_vis is its running total to the end). */
  season_night: number[][] = [];
  season_peak: number[][] = [];
  season_future: number[][] = [];
  constructor(
    payload: Payload,
    public site: Site,
    fov: number,
    band_width: number,
  ) {
    const table = payload.targets,
      col: Record<string, number> = {};
    table.columns.forEach((name: string, i: number) => {
      col[name] = i;
    });
    const rows: any[][] = table.rows;
    this.n = rows.length;
    this.ids = rows.map((r) => r[col.target_id]);
    this.ra = rows.map((r) => Number(r[col.ra_deg]));
    this.dec = rows.map((r) => Number(r[col.dec_deg]));
    this.flux = rows.map((r) => Number(r[col.feature_flux]));
    this.weight = rows.map((r) => Number(r[col.science_weight]));
    this.required = rows.map(
      (r) =>
        r[col.required] === true ||
        String(r[col.required]).trim().toLowerCase() === "true",
    );
    this.index = new Map(this.ids.map((id, i) => [id, i]));
    for (let i = 0; i < this.n; i++) {
      const [x, y, z] = unit_vector(this.ra[i], this.dec[i]);
      this.ux.push(x);
      this.uy.push(y);
      this.uz.push(z);
      this.hmax.push(site.hour_angle_limit(this.dec[i]));
    }
    this.band_width = band_width;
    this.band = this.ra.map((r) => floorDiv(r, band_width));
    for (const b of this.band)
      this.band_total.set(b, (this.band_total.get(b) ?? 0) + 1);
    this.fov = fov;
    this._build_zones();
  }
  private _build_zones(): void {
    this.grid = new Grid(this.fov);
    for (let i = 0; i < this.n; i++) {
      const cell = this.grid.locate(this.ra[i], this.dec[i]),
        k = key(...cell);
      let z = this.key_to_zone.get(k);
      if (z === undefined) {
        z = this.zmembers.length;
        this.key_to_zone.set(k, z);
        this.zmembers.push([]);
        this.zkeys.push(cell);
      }
      this.zmembers[z].push(i);
      this.zone_of.push(z);
    }
    this.nz = this.zmembers.length;
    for (const idx of this.zmembers) {
      const sx = sum(idx.map((i) => this.ux[i])),
        sy = sum(idx.map((i) => this.uy[i])),
        sz = sum(idx.map((i) => this.uz[i]));
      const norm = Math.sqrt(sx * sx + sy * sy + sz * sz) || 1;
      this.zcx.push(sx / norm);
      this.zcy.push(sy / norm);
      this.zcz.push(sz / norm);
      this.zdec.push(Math.asin(Math.max(-1, Math.min(1, sz / norm))) * DEG);
      this.zra.push(mod(Math.atan2(sy, sx) * DEG, 360));
    }
    this.zhmax = this.zdec.map((d) => this.site.hour_angle_limit(d));
    this.reach_radius = this.fov * Math.sqrt(2) + 0.05;
  }
  reach(z: number): number[] {
    let cached = this._reach.get(z);
    if (cached === undefined) {
      cached = [];
      const [b, k] = this.zkeys[z];
      for (const cell of this.grid.cells_near(b, k, this.reach_radius)) {
        const other = this.key_to_zone.get(key(...cell));
        if (other !== undefined) cached.push(...this.zmembers[other]);
      }
      this._reach.set(z, cached);
    }
    return cached;
  }
  build_season(nights: [number, number][], cell_deg = 5): void {
    const site = this.site,
      coarse = new Grid(cell_deg),
      key_to_cell = new Map<string, number>(),
      sums: number[][] = [];
    for (let z = 0; z < this.nz; z++) {
      const k = key(...coarse.locate(this.zra[z], this.zdec[z]));
      let c = key_to_cell.get(k);
      if (c === undefined) {
        c = sums.length;
        key_to_cell.set(k, c);
        sums.push([0, 0, 0]);
      }
      this.zcoarse.push(c);
      sums[c][0] += this.zcx[z];
      sums[c][1] += this.zcy[z];
      sums[c][2] += this.zcz[z];
    }
    this.nc = sums.length;
    const nn = nights.length,
      night_lst = nights.map(([start]) => lst_deg(start, site.lon));
    const night_span = nights.map(([start, end]) => OMEGA * (end - start));
    for (const [sx, sy, sz] of sums) {
      const norm = Math.sqrt(sx * sx + sy * sy + sz * sz) || 1,
        dec = Math.asin(Math.max(-1, Math.min(1, sz / norm))) * DEG;
      const ra = mod(Math.atan2(sy, sx) * DEG, 360),
        hm = site.hour_angle_limit(dec),
        sd = Math.sin(dec * RAD),
        cd = Math.cos(dec * RAD);
      const a_culm = site.transmission(
        Math.sin(Math.min(90, 90 - Math.abs(dec - site.lat)) * RAD),
      );
      const vis = Array(nn + 1).fill(0),
        peak = Array(nn).fill(0);
      for (let i = 0; i < nn; i++) {
        if (hm < 0) continue;
        const h0 = mod(night_lst[i] - ra + 180, 360) - 180,
          h1 = h0 + night_span[i];
        vis[i] =
          hm >= 180
            ? night_span[i] / OMEGA
            : (overlap(h0, h1, -hm, hm) + overlap(h0, h1, 360 - hm, 360 + hm)) /
              OMEGA;
        if (vis[i] <= 0) continue;
        if ((h0 <= 0 && 0 <= h1) || (h0 <= 360 && 360 <= h1)) peak[i] = a_culm;
        else {
          let best = 0;
          for (const h of [h0, h1])
            if (Math.abs(mod(h + 180, 360) - 180) <= hm)
              best = Math.max(
                best,
                site.transmission(
                  site.sin_lat * sd + site.cos_lat * cd * Math.cos(h * RAD),
                ),
              );
          peak[i] = best;
        }
      }
      const remaining = Array(nn + 1).fill(0),
        future = Array(nn + 1).fill(0);
      for (let i = nn - 1; i >= 0; i--) {
        remaining[i] = remaining[i + 1] + vis[i];
        future[i] = Math.max(future[i + 1], i + 1 < nn ? peak[i + 1] : 0);
      }
      this.season_vis.push(remaining);
      this.season_peak.push(peak);
      this.season_future.push(future);
      this.season_night.push(vis.slice(0, nn));
    }
  }
  /**
   * Recompute what is left of the season when part of some nights is known to be lost
   * (open[i] = usable share of night i, 1 when nothing is known).
   */
  set_open_nights(open: number[]): void {
    const nn = open.length;
    for (let c = 0; c < this.nc; c++) {
      const vis = this.season_night[c],
        peak = this.season_peak[c],
        remaining = this.season_vis[c],
        future = this.season_future[c];
      for (let i = nn - 1; i >= 0; i--) {
        remaining[i] = remaining[i + 1] + vis[i] * open[i];
        future[i] = Math.max(future[i + 1], i + 1 < nn && open[i + 1] > 0.25 ? peak[i + 1] : 0);
      }
    }
  }
}
