/** Hidden pointing offset: retained search schedule, evidence and uncertainty box. */
import { DEG, RAD } from "./astro";
import { appendBounded } from "./compat";
export type OffsetItem = [number, number, number, number, number, boolean];
type RecordRow = [number, number, OffsetItem[]];
type Box = [number, number, number, number, number];
export class OffsetEstimator {
  n: number;
  half: number;
  alt = 0;
  az = 0;
  margin_alt = 0;
  margin_az = 0;
  active = false;
  records: RecordRow[] = [];
  solves = 0;
  since_solve = 0;
  best_mismatch = 0;
  evidence = 0;
  constructor(
    n_side: number,
    public pitch: number,
  ) {
    this.n = Math.trunc(n_side);
    this.half = (this.n * pitch) / 2;
  }
  add(cmd_alt: number, cmd_az: number, items: OffsetItem[]): void {
    if (items.length) appendBounded(this.records, [cmd_alt, cmd_az, items], 60);
  }
  private _mismatch(da: number, dz: number, records: RecordRow[]): number {
    const n = this.n,
      pitch = this.pitch,
      half = this.half,
      mid = n / 2;
    let total = 0;
    for (const [cmd_alt, cmd_az, items] of records) {
      const alt_deg = cmd_alt + da;
      if (!(0 <= alt_deg && alt_deg <= 90)) {
        for (const item of items) if (item[5]) total++;
        continue;
      }
      const alt = alt_deg * RAD,
        az = (cmd_az + dz) * RAD,
        ca = Math.cos(alt),
        sa = Math.sin(alt),
        cz = Math.cos(az),
        sz = Math.sin(az);
      const cx = ca * cz,
        cy = ca * sz,
        cu = sa,
        ux = -sa * cz,
        uy = -sa * sz,
        uu = ca,
        rx = -sz,
        ry = cz;
      for (const [hn, he, hu, row, col, hit] of items) {
        const depth = hn * cx + he * cy + hu * cu;
        let predicted = false;
        if (depth > 0) {
          const d_alt = ((hn * ux + he * uy + hu * uu) / depth) * DEG,
            d_az = ((hn * rx + he * ry) / depth) * DEG;
          if (
            -half <= d_alt &&
            d_alt <= half &&
            -half <= d_az &&
            d_az <= half
          ) {
            const r = Math.min(n - 1, Math.floor(d_alt / pitch + mid)),
              c = Math.min(n - 1, Math.floor(d_az / pitch + mid));
            predicted = r === row && c === col;
          }
        }
        if (predicted !== hit) total++;
      }
    }
    return total;
  }
  solve(): void {
    if (!this.records.length) return;
    this.solves++;
    this.since_solve = 0;
    const recent: RecordRow[] = [];
    let count = 0;
    for (let i = this.records.length - 1; i >= 0; i--) {
      recent.push(this.records[i]);
      count += this.records[i][2].length;
      if (count >= 300) break;
    }
    let step = Math.min(0.02, this.pitch / 10),
      span = Math.max(0.26, 13 * step);
    let centre: [number, number] =
        this.active && this.solves > 1 ? [this.alt, this.az] : [0, 0],
      best!: Box;
    for (let i = 0; i < 3; i++) {
      best = this._grid(centre, span, span, step, recent);
      if (
        best[1] - centre[0] > -span + 1.5 * step &&
        best[2] - centre[0] < span - 1.5 * step &&
        best[3] - centre[1] > -span + 1.5 * step &&
        best[4] - centre[1] < span - 1.5 * step
      )
        break;
      centre = [0.5 * (best[1] + best[2]), 0.5 * (best[3] + best[4])];
      span *= 1.6;
    }
    for (let i = 0; i < 3; i++) {
      if (step <= 0.0012) break;
      const [, lo_a, hi_a, lo_z, hi_z] = best,
        span_a = 0.5 * (hi_a - lo_a) + step,
        span_z = 0.5 * (hi_z - lo_z) + step;
      const fine = Math.max(Math.max(span_a, span_z) / 7, step / 6, 0.0008);
      if (fine >= step) break;
      best = this._grid(
        [0.5 * (lo_a + hi_a), 0.5 * (lo_z + hi_z)],
        span_a,
        span_z,
        fine,
        recent,
      );
      step = fine;
    }
    const [mismatch, lo_a, hi_a, lo_z, hi_z] = best;
    this.alt = 0.5 * (lo_a + hi_a);
    this.az = 0.5 * (lo_z + hi_z);
    this.margin_alt = 0.5 * (hi_a - lo_a) + step;
    this.margin_az = 0.5 * (hi_z - lo_z) + step;
    this.best_mismatch = mismatch;
    this.active = true;
  }
  private _grid(
    centre: [number, number],
    span_a: number,
    span_z: number,
    step: number,
    records: RecordRow[],
  ): Box {
    const na = Math.ceil(span_a / step),
      nz = Math.ceil(span_z / step);
    let best_m: number | null = null,
      lo_a = 0,
      hi_a = 0,
      lo_z = 0,
      hi_z = 0;
    for (let i = -na; i <= na; i++) {
      const da = centre[0] + i * step;
      for (let k = -nz; k <= nz; k++) {
        const dz = centre[1] + k * step,
          m = this._mismatch(da, dz, records);
        if (best_m === null || m < best_m) {
          best_m = m;
          lo_a = hi_a = da;
          lo_z = hi_z = dz;
        } else if (m === best_m) {
          lo_a = Math.min(lo_a, da);
          hi_a = Math.max(hi_a, da);
          lo_z = Math.min(lo_z, dz);
          hi_z = Math.max(hi_z, dz);
        }
      }
    }
    return [best_m!, lo_a, hi_a, lo_z, hi_z];
  }
  observe(
    cmd_alt: number,
    cmd_az: number,
    items: OffsetItem[],
    unexpected: number,
  ): void {
    this.add(cmd_alt, cmd_az, items);
    this.since_solve++;
    if (!this.active) {
      if (unexpected > 0) this.solve();
      return;
    }
    this.evidence += items.length;
    if (this.solves >= 40) return;
    const widest = Math.max(this.margin_alt, this.margin_az);
    if (
      (widest > this.pitch / 25 && this.since_solve >= 1) ||
      (widest > this.pitch / 120 && this.since_solve >= 8) ||
      (unexpected > 0 && this.since_solve >= 5) ||
      this.since_solve >= 250
    )
      this.solve();
  }
  get uncertain(): boolean {
    return (
      this.active && Math.max(this.margin_alt, this.margin_az) > this.pitch / 12
    );
  }
}
