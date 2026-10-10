/**
 * Native Node.js 22 agent for participant-agent-protocol-v4.
 *
 * A language model reads the handover notes that arrive in `observation_request.reason` (handover/reader.ts)
 * and its reading becomes facts (facts/store.ts) that change what this loop does: instrument-fault reports,
 * closed sectors and stops, ridge heights, thin versus thick cloud. Pointing, fibre assignment and exposure
 * length stay with the deterministic planner. Optional trained sidecars run only when their local
 * Python runtime is already present; the submitted project must still work without them.
 */
import { FaultWatch } from "./faults/watch";
import { buildDigest, cellMultipliers, loadPlan, nightLevers } from "./plan/cardplan";
import { once } from "node:events";
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import * as path from "node:path";
import { StringDecoder } from "node:string_decoder";
import { DEG, format_utc, parse_utc } from "./core/astro";
import { asciiJson, mod, Payload } from "./core/compat";
import { COMPASS, Exposure, ExposureItem, NoteView } from "./core/conditions";
import { FactStore, FaultFact } from "./facts/store";
import { HandoverReader } from "./handover/reader";
import { SimClock } from "./sim/clock";
import { OffsetItem } from "./core/pointing";
import { Plan, Planner } from "./core/planner";
import { Knobs, Survey } from "./core/state";
import { BaselineModels, SelectedModelsSession, SELECTED_MODEL_IDS } from "./models";

export const PROTOCOL = "participant-agent-protocol-v4";
export const KNOBS: Knobs = {
  kappa: 0.7,
  price_cap: 0.8,
  rate_hours: 6,
  request_gain: 3,
  request_floor: 2.5,
  urgency_power: 0.5,
  urgency_cap: 3,
  sci_urgency_power: 0.5,
  sci_urgency_cap: 2.5,
  partial_discount: 0.85,
  deep_weight: 0.25,
  dark_probe_every: 8,
  timing_power: 3,
  k_max: 14,
  fine_shifts: 1,
  cpu_share: 0.85,
  wall_share: 0.8,
  calib_shots: 1,
  final_probe: 1,
  log_every: 250,
  // Number of extra reference exposure lengths (planner.ts PLACE_FRACTIONS) each field is also placed for.
  place_refs: 4,
  // Handover notes: longest wait for one model reading (seconds), margin above a stated ridge (degrees).
  note_wait_max: 30,
  note_first_wait: 75,
  ridge_margin: 1.5,
  // With nothing worth observing (gain rate below idle_rate points per second): seconds to wait at a time.
  idle_wait: 3600,
  idle_rate: 2e-4,
  // Requests that carry no reward (the handover notes arrive as such) add nothing to a field's value.
  zero_reward_skip: 1,
  // Recovery probe. When the planner finds nothing for probe_after seconds although no bulletin or note closes
  // the sky, the throughput estimate (rho) is put back to its prior once and the planner asked again for the
  // shortest exposure; at most one probe per probe_every seconds. A correct report makes the next such
  // decision probe at once. probe_every = 0 switches this off. Without it, rho measured near zero (an
  // unreported fault, a closed sky) is never measured again: no exposure is planned, so nothing updates it.
  probe_after: 1800,
  probe_every: 3600,
  // Experiment (state.ts price): constant price of time in points per second; 0 = the trailing price.
  price_fixed: 0,
  // Experiment (planner.ts _sci_timing): exponent on (transmission now / best transmission still to come).
  sci_timing_power: 0,
  // Experiment (planner.ts): choose the fibre lattice's displacement and the field window together.
  lattice_search: 0,
};
export function cpu_seconds(): number {
  const c = process.cpuUsage();
  return (c.user + c.system) / 1e6;
}
export function monotonic(): number {
  return Number(process.hrtime.bigint()) / 1e9;
}
/**
 * What goes into agent.log. The platform keeps the last 2 MB of it, so the default is a compact record that fits
 * a whole card: a start line, one line per exposure ("x"), per changed bulletin ("b"), per request ("q") and its
 * result ("qr"), the full text of every handover note once ("note"), every report and its result, a status
 * line every few hundred decisions, and an end line with the totals. Two switches add the bulky records:
 * BASE_INPUT_LOG=on, the verbatim input (it alone fills the log within a few hundred requests), and
 * BASE_MODELS_LOG=on, the trained models' record for every decision.
 */
const INPUT_LOG = process.env.BASE_INPUT_LOG === "on";
const MODEL_LOG = Boolean(process.env.BASE_MODELS_LOG) && process.env.BASE_MODELS_LOG !== "off";
/** Which build is running: the first characters of the hash of this script. */
function build_id(): string {
  try {
    return createHash("sha256").update(readFileSync(__filename)).digest("hex").slice(0, 10);
  } catch {
    return "unknown";
  }
}
/** AGENT_TRACE_FILE: also append the log, without the per-decision model records, to this file (local debugging). */
const TRACE_FILE = process.env.AGENT_TRACE_FILE || "";
export function log(...parts: unknown[]): void {
  const line =
    parts
      .map((p) => (typeof p === "string" ? p : JSON.stringify(p)))
      .join(" ") + "\n";
  process.stderr.write(line);
  if (TRACE_FILE && !line.startsWith("SURVEY26_FAULT_MODEL_V1") && !line.startsWith("SURVEY26_MODELS_V1"))
    try {
      appendFileSync(TRACE_FILE, line);
    } catch {
      /* tracing is best effort */
    }
}
export function load_knobs(): Knobs {
  const knobs = { ...KNOBS };
  for (const name of Object.keys(knobs)) {
    const raw = process.env["BASE_" + name.toUpperCase()];
    if (raw?.trim()) {
      const value = Number(raw);
      if (!Number.isNaN(value)) knobs[name] = value;
    }
  }
  return knobs;
}
/**
 * The competition image starts with no Python model environment.  Keeping this check here prevents
 * an optional local model from turning a valid JSON-Lines agent into a failed initialization.
 */
export function selectedModelRuntimeAvailable(): boolean {
  const python = process.env.BASE_MODEL_PYTHON || path.join(__dirname, "..", ".runtime", "venv", "bin", "python");
  return existsSync(python);
}
export class Agent {
  knobs: Knobs;
  watch: FaultWatch | null = null;
  sv: Survey;
  planner: Planner;
  readonly models: BaselineModels;
  readonly modelSession: SelectedModelsSession;
  /** false: the fault model alone decides reports; true: the rule check must also allow them. */
  readonly faultGuard: boolean;
  modelCompleted: Payload | undefined;
  model_cpu = 0;
  pending: Exposure | null = null;
  night_seen = -1;
  calib_left: number;
  calib_extra = 0;
  final_probe_done = false;
  decisions = 0;
  suffix: number[];
  avg_t = 900;
  k = 8;
  own = 0.02;
  engine = 0.02;
  engine0: [number, number] | null = null;
  entries = 0;
  requests_seen = 0;
  now = 0;
  reply_at: number | null = null;
  t_floor: number;
  misses = 0;
  assigned = 0;
  predicted_gain = 0;
  realised_gain = 0;
  readonly reader: HandoverReader;
  /** The note that the report now awaiting its result was sent for (null: a rule or the fault model sent it). */
  report_note: FaultFact | null = null;
  /** Clean throughput readings [exposure start, level], newest last. */
  readings: [number, number][] = [];
  wall_spare: number | null = null;
  /** Mean exposure length chosen while no floor was imposed, and whether the exposure now pending was such a one. */
  avg_free = 600;
  plan_free = false;
  /** Seconds of spare wall time earned by the notes received and not yet spent waiting for a reading. */
  note_credit = 0;
  note_reports = 0;
  /** Who sent the report that is waiting for its result. */
  report_source: "note" | "watch" | "detector" | "other" | null = null;
  /** The detector's wrong reports in a row that cost points, and until when it may not pay again (fault watch on). */
  detector_paid_streak = 0;
  detector_paid_muted_until = -Infinity;
  /** Reports sent for a handover note that came back correct. */
  note_reports_right = 0;
  /** Totals for the end-of-run line. */
  tally = { exposures: 0, zero_exposures: 0, exposure_hours: 0, zero_hours: 0, gain: 0, reports_note: [0, 0], reports_detector: [0, 0], detector_reports_withheld: 0 };
  last_notices = "";
  /** The last reason a report of the agent's own detector was withheld (logged once per reason). */
  held_reason = "";
  facts_revision = -1;
  facts_slot = -1;
  lost_key = "";
  stop_probed = "";
  /** Set when the note-handling code threw: from then on the notes are ignored and the rules run as before. */
  notes_broken = false;
  /** HANDOVER=off ignores the notes altogether (the baseline behaviour, for comparisons). */
  readonly read_notes = process.env.HANDOVER !== "off";
  oracle_reports: number[] = (process.env.ORACLE_REPORT_AT ?? "").split(",").filter(Boolean).map((text) => parse_utc(text.trim())).sort((x, y) => x - y);
  /** Local evaluation only (SIM_WALL=1): real time is charged from a model of the platform instead of measured. */
  readonly sim = SimClock.fromEnv();
  constructor(payload: Payload) {
    this.knobs = load_knobs();
    const started = cpu_seconds();
    this.sv = new Survey(payload, this.knobs);
    this.planner = new Planner(this.sv);
    // Fault watch (src/faults/watch.ts): FAULT_WATCH=on asks it first whether to report; the detector, the notes'
    // own reports and the free report at the end of the season stay as they are.
    if (process.env.FAULT_WATCH === "on") {
      const scoring = payload.scoring ?? {},
        cat = this.sv.cat;
      this.watch = new FaultWatch({
        site: this.sv.site, ra: cat.ra, dec: cat.dec, flux: cat.flux, weight: cat.weight, unit: this.sv.f0t0,
        dark: Number(scoring.program?.bands?.DARK ?? 0.65), bright: Number(scoring.program?.bands?.BRIGHT ?? 0.4),
        multipliers: scoring.program?.multipliers ?? {}, nights: this.sv.nights,
        freeAllowance: Number(scoring.reporting?.false_report_free_allowance ?? 2),
      }, (text) => log(text));
    }
    this.models = new BaselineModels();
    if (process.env.BASE_CONDITIONS_MODELS && !["off", "reliability", "conditions-reliability"].includes(process.env.BASE_CONDITIONS_MODELS))
      throw new Error("BASE_CONDITIONS_MODELS must be reliability or off; GRU/MLP are legacy comparison models.");
    if (process.env.BASE_FAULT_GUARD && !["off", "rule"].includes(process.env.BASE_FAULT_GUARD))
      throw new Error("BASE_FAULT_GUARD must be rule or off.");
    const modelRuntime = process.env.BASE_MODEL_RUNTIME ?? "auto";
    if (!["auto", "on", "off"].includes(modelRuntime))
      throw new Error("BASE_MODEL_RUNTIME must be auto, on, or off.");
    const runtimeAvailable = selectedModelRuntimeAvailable();
    if (modelRuntime === "on" && !runtimeAvailable)
      throw new Error("BASE_MODEL_RUNTIME=on requires BASE_MODEL_PYTHON or .runtime/venv/bin/python.");
    const useSelectedModels = modelRuntime !== "off" && runtimeAvailable;
    const enabled = useSelectedModels ? [...SELECTED_MODEL_IDS].filter(id => id === "conditions-reliability"
      ? process.env.BASE_CONDITIONS_MODELS !== "off" : process.env.BASE_FAULT_MODEL !== "off") : [];
    if (!useSelectedModels)
      log(`selected models: off (BASE_MODEL_RUNTIME=${modelRuntime}; no local model runtime)`);
    this.faultGuard = process.env.BASE_FAULT_GUARD === "rule";
    this.modelSession = new SelectedModelsSession(payload, this.models, enabled);
    const sv = this.sv;
    sv.sky.debug = Boolean(process.env.BASE_DEBUG_SKY);
    sv.sky.dark_probe_every = Math.trunc(this.knobs.dark_probe_every);
    this.calib_left = Math.trunc(this.knobs.calib_shots);
    this.suffix = Array(sv.nights.length + 1).fill(0);
    for (let i = sv.nights.length - 1; i >= 0; i--)
      this.suffix[i] = this.suffix[i + 1] + sv.nights[i][1] - sv.nights[i][0];
    this.t_floor = sv.t_min;
    this.reader = new HandoverReader({ utcOffsetHours: Number(payload.site?.utc_offset_hours ?? 0), nights: sv.nights }, (text) => log(text));
    this.reader.sim = this.sim;
    // Without a model nothing is read from the notes, and then nothing else departs from the baseline either:
    // the zero-reward requests that carry the notes go through the planner as before, and idle stretches are
    // slept slot by slot. (Checked on the local cards: the score is then the baseline's to the last digit.)
    if (!this.read_notes || !this.reader.enabled) {
      this.knobs.zero_reward_skip = 0;
      this.knobs.idle_wait = 0;
    }
    log("SURVEY26_RUN_V1", { build: build_id(), node: process.version, trained_models: enabled.length ? enabled : "off",
      notes: !this.read_notes ? "off (HANDOVER=off)" : this.reader.enabled ? "read by the model" : "not read (no model available)",
      llm: this.reader.llm.providers.map((p) => `${p.name}:${p.model}`), thinking: this.reader.llm.thinking });
    if (this.sim !== null) log("SIM_WALL: virtual real-time clock in use", this.sim.summary());
    log(`handover reader: model ${this.reader.enabled ? "on" : "off"} (${this.reader.llm.providers.map((p) => `${p.name}:${p.model}`).join(" > ") || "no provider"}; thinking ${this.reader.llm.thinking})`);
    // Start-of-card plan experiment: CARD_PLAN_DIGEST_OUT writes the digest the model is asked about,
    // CARD_PLAN_FILE applies a plan made from it (tools/card-plan), CARD_PLAN_ONLY ends the run at the first decision.
    if (process.env.CARD_PLAN_DIGEST_OUT || process.env.CARD_PLAN_FILE) {
      const digest = buildDigest(sv);
      if (process.env.CARD_PLAN_DIGEST_OUT) writeFileSync(process.env.CARD_PLAN_DIGEST_OUT, JSON.stringify(digest));
      if (process.env.CARD_PLAN_FILE) {
        const plan = loadPlan(process.env.CARD_PLAN_FILE, digest);
        sv.plan_cells = (night) => cellMultipliers(plan, digest, night);
        sv.plan_night = (night) => nightLevers(plan, digest, night);
        log("card plan:", process.env.CARD_PLAN_FILE, JSON.stringify(plan));
      }
    }
    if (process.env.ORACLE_FUTURE) {
      const table = JSON.parse(readFileSync(process.env.ORACLE_FUTURE, "utf8")) as Record<string, [number, number]>;
      sv.future = { score: sv.cat.ids.map((id) => table[id]?.[0] ?? 0), at: sv.cat.ids.map((id) => table[id]?.[1] ?? 0), weight: Number(process.env.ORACLE_FUTURE_WEIGHT ?? 1) };
      log(`oracle future: ${Object.keys(table).length} targets from ${process.env.ORACLE_FUTURE}, weight ${sv.future.weight}`);
    }
    const cat = sv.cat;
    log(
      `init: ${cat.n} targets, ${cat.required.filter(Boolean).length} required, ${cat.nz} zones, ${cat.nc} season cells, ${sv.nights.length} nights, ${sv.n_fibers} fibres pitch ${sv.pitch.toFixed(4)} deg, price0 ${sv.price.toFixed(5)}/s, cpu ${(cpu_seconds() - started).toFixed(2)}s`,
    );
  }
  _engine_seconds(
    n_observes: number,
    nights_left: number,
    speed: number,
  ): number {
    const sv = this.sv,
      entries = this.entries,
      per_exposure =
        sv.observes >= 10 ? entries / sv.observes : 0.8 * sv.n_fibers;
    const i =
      this.night_seen < 0
        ? sv.nights.length - 1
        : Math.min(this.night_seen, sv.nights.length - 1);
    const done =
      1 -
      (Math.max(0, sv.nights[i][1] - this.now) +
        this.suffix[Math.min(this.night_seen + 1, sv.nights.length)]) /
        sv.total_night_seconds;
    const requests_end =
        done > 0.15
          ? this.requests_seen / done
          : Math.max(this.requests_seen, Math.min(6, sv.nights.length / 14)),
      load_now = entries * (1 + this.requests_seen);
    if (this.engine0 === null && sv.observes >= 40)
      this.engine0 = [this.engine, load_now];
    let slope = 0.25e-6 * speed,
      base = this.engine;
    if (this.engine0 !== null) {
      base = this.engine0[0];
      if (load_now - this.engine0[1] > 30000)
        slope = Math.max(
          0.05e-6 * speed,
          (this.engine - this.engine0[0]) / (load_now - this.engine0[1]),
        );
      base = Math.max(0.002, base - slope * this.engine0[1]);
    }
    const entries_mid = entries + (per_exposure * n_observes) / 2,
      requests_mid = 0.5 * (this.requests_seen + requests_end),
      per_observe = base + slope * entries_mid * (1 + requests_mid);
    return (
      n_observes * per_observe +
      nights_left * 12 * slope * entries_mid * requests_mid
    );
  }
  _pace(payload: Payload, t: number, night: number): void {
    const sv = this.sv,
      clock = payload.wallclock || {},
      cpu_left = Number(
        clock.remaining_real_cpu_seconds ?? clock.remaining_seconds ?? 900,
      ),
      wall_left = clock.wall_remaining_seconds,
      speed = Number(clock.speed_factor || 1);
    this.now = t;
    const remaining =
        Math.max(0, sv.nights[night][1] - t) + this.suffix[night + 1],
      nights_left = sv.nights.length - night,
      n_observes = (remaining / Math.max(this.avg_t, 60)) * 1.05,
      n_other = 2 * nights_left + 5;
    let per_decision =
        (cpu_left * this.knobs.cpu_share) / (n_observes + n_other),
      floor = sv.t_min;
    if (wall_left !== undefined && wall_left !== null) {
      const budget = Number(wall_left) * this.knobs.wall_share,
        minimum = 0.004 * speed,
        idle = budget - this._engine_seconds(n_observes, nights_left, speed);
      // Wall time left over at the pace the planner picks when nothing holds it back. Only this is spent waiting
      // for note readings: a second waited beyond it is paid for with fewer, longer exposures. It is measured
      // against the unthrottled pace because a throttled pace always looks as if it had time to spare.
      const n_free = (remaining / Math.max(Math.min(this.avg_t, this.avg_free), 60)) * 1.05;
      this.wall_spare = budget - this._engine_seconds(n_free, nights_left, speed) - (n_free + n_other) * minimum;
      const per_wall = idle / (n_observes + n_other);
      if (per_wall < per_decision) per_decision = per_wall;
      if (per_wall < minimum) {
        let stretch = 1;
        for (let i = 0; i < 12; i++) {
          stretch *= 1.2;
          const fewer = n_observes / stretch;
          if (
            this._engine_seconds(fewer, nights_left, speed) +
              (fewer + n_other) * minimum <=
            budget
          )
            break;
        }
        floor = Math.trunc(Math.min(sv.t_max, this.avg_t * stretch));
        per_decision = minimum;
      }
      if (Number(wall_left) < 15 || cpu_left < 1.5) {
        per_decision = 0;
        floor = sv.t_max;
      }
    }
    per_decision = Math.max(0, per_decision - this.model_cpu);
    const cheapest = Math.max(0.002, (this.own * 2) / Math.max(this.k, 2));
    if (0 < per_decision && per_decision < cheapest)
      floor = Math.max(
        floor,
        Math.trunc(Math.min(sv.t_max, (this.avg_t * cheapest) / per_decision)),
      );
    const ratio = per_decision / Math.max(this.own, 1e-4);
    this.k = Math.max(
      2,
      Math.min(
        this.knobs.k_max,
        this.k * Math.max(0.6, Math.min(1.4, Math.sqrt(Math.max(ratio, 0)))),
      ),
    );
    this.t_floor = Math.max(sv.t_min, floor);
  }
  _digest(last: Payload): void {
    const rec = this.pending!,
      sv = this.sv,
      cat = sv.cat,
      hits = new Map<number, number>();
    for (const hit of last.hits || []) {
      const j = cat.index.get(hit.target_id);
      if (j !== undefined) hits.set(j, Number(hit.score ?? 0));
    }
    rec.notices = sv.sky.notices_between(rec.t0, rec.t0 + rec.teff, sv.slot);
    this.watch?.exposure({ start: rec.t0, duration: rec.teff, program: rec.program, alt: rec.cmd_alt, az: rec.cmd_az, hits: [...hits].map(([index, score]) => ({ index, score })) });
    const outcomes: OffsetItem[] = [];
    let unexpected = 0;
    for (const item of rec.items) {
      const hit = hits.has(item[0]);
      outcomes.push([item[10], item[11], item[12], item[1], item[2], hit]);
      if (!hit) {
        this.misses++;
        if (item[8]) unexpected++;
      }
    }
    this.assigned += rec.items.length;
    this.entries += hits.size;
    const was_active = sv.offset.active;
    sv.offset.observe(rec.cmd_alt, rec.cmd_az, outcomes, unexpected);
    if (sv.offset.active && (!was_active || unexpected))
      log(
        `pointing: offset estimate alt ${sv.offset.alt.toFixed(4)} az ${sv.offset.az.toFixed(4)} deg (+-${sv.offset.margin_alt.toFixed(4)}/${sv.offset.margin_az.toFixed(4)}), ${unexpected} unexpected misses, mismatch ${sv.offset.best_mismatch}`,
      );
    const factors = sv.sky.update(rec, hits, rec.night),
      rows: [number, number, number, number][] = [];
    for (const [j, score] of hits) {
      const f = factors.get(j);
      if (f !== undefined) rows.push([j, score, f[0], f[1]]);
    }
    const closed =
        hits.size > 0 && [...hits.values()].every((score) => score <= 0),
      index = last.observe_index;
    const gain = sv.apply_hits(
      Number.isInteger(index) ? index : sv.observes,
      rows,
      rec.teff,
      closed,
    );
    if (!closed) {
      this.predicted_gain += rec.predicted;
      this.realised_gain += gain;
      // Only an exposure that measured the throughput counts: a long one whose targets all saturate tells nothing
      // about a drop, and taking it for a reading would hold back the report of an announced fault.
      if (sv.sky.measured_at === rec.t0 && !sv.sky.closed_suspect && [...hits.values()].some((score) => score > 0)) {
        this.readings.push([rec.t0, sv.sky.rho]);
        if (this.readings.length > 40) this.readings.shift();
      }
    }
    this.modelCompleted = {
      observe_index: Number.isInteger(last.observe_index) ? last.observe_index : sv.observes,
      start_utc: format_utc(rec.t0), end_utc: format_utc(rec.t0 + rec.teff),
      program: rec.program, night: rec.night, pointing: { alt_deg: rec.cmd_alt, az_deg: rec.cmd_az },
      assigned_count: rec.items.length, hits: last.hits || [],
      target_ids: rec.items.map(item => cat.ids[item[0]]), items: rec.items.map(item => item.slice(0, 10)),
      estimated_efficiency: sv.sky.e, suspected_fault: sv.sky.suspect, quake_notice: sv.sky.quake_now,
    };
    const scored = [...hits.values()].filter((score) => score > 0).length;
    this.tally.exposures++;
    this.tally.exposure_hours += rec.teff / 3600;
    this.tally.gain += gain;
    if (hits.size > 0 && scored === 0) {
      this.tally.zero_exposures++;
      this.tally.zero_hours += rec.teff / 3600;
    }
    // x <start> <seconds> <alt> <az> <program> fibres/returned/scored gain rho W e [flags]
    log(`x ${format_utc(rec.t0)} ${Math.round(rec.teff)} ${rec.cmd_alt.toFixed(1)} ${rec.cmd_az.toFixed(1)} ${rec.program} ${rec.items.length}/${hits.size}/${scored} g=${gain.toFixed(2)} rho=${sv.sky.rho.toFixed(3)} W=${sv.sky.W.toFixed(3)} e=${sv.sky.e.toFixed(2)}${sv.sky.suspect ? " suspect" : ""}${sv.sky.closed_suspect ? " closed?" : ""}`);
    this.avg_t += 0.05 * (rec.teff - this.avg_t);
    if (this.plan_free) this.avg_free += 0.05 * (rec.teff - this.avg_free);
  }
  async decide(payload: Payload): Promise<Payload> {
    if (process.env.CARD_PLAN_ONLY) return { action: "finish" };
    const began = this.models.worker.cpuSeconds;
    this.why = "";
    try {
      const action = await this.decideAction(payload);
      this.logWait(payload, action);
      return action;
    }
    finally {
      const elapsed = this.models.worker.cpuSeconds - began;
      this.model_cpu += .1 * (elapsed - this.model_cpu);
    }
  }
  /** Recovery probe state (knobs probe_after / probe_every). */
  null_since: number | null = null;
  last_probe = -1e18;
  probe_now = false;
  probes = 0;
  /** Why the last decision was a wait (set at each wait below); logged when it changes, and once per night. */
  why = "";
  last_why = "";
  last_night_logged = -1;
  private logWait(payload: Payload, action: Payload): void {
    const sv = this.sv, t = parse_utc(payload.now_utc), night = sv.night_index(t);
    if (night !== this.last_night_logged && night < sv.nights.length) {
      this.last_night_logged = night;
      const store = this.reader.store;
      log(`n ${payload.now_utc} night=${night} required_left=${sv.required_left} facts: windows=${store.windows.length} active=${store.active(t).length} terrain=${store.terrain.length} faults=${store.faults.length} open=${store.openFaults().length} closed_suspect=${sv.sky.closed_suspect} e=${sv.sky.e.toFixed(2)}`);
    }
    const now = action.action === "wait" ? this.why || "other" : "";
    if (now !== this.last_why) {
      this.last_why = now;
      log(now === "" ? `w ${payload.now_utc} resumed (${action.action})` : `w ${payload.now_utc} wait: ${now}`);
    }
  }
  private async decideAction(payload: Payload): Promise<Payload> {
    const sv = this.sv,
      sky = sv.sky,
      t = parse_utc(payload.now_utc),
      resyncs: Payload[] = [],
      notes: Payload[] = [];
    this.modelCompleted = undefined;
    this.now = t;
    for (const message of payload.new_messages || []) {
      if (
        message === null ||
        typeof message !== "object" ||
        Array.isArray(message)
      )
        continue;
      if (message.record_type === "bulletin") {
        try {
          sky.bulletin(message, parse_utc(message.issued_at_utc));
          this.watch?.bulletin(parse_utc(message.issued_at_utc), (Array.isArray(message.notices) ? message.notices : []).map((n: Payload) => ({ kind: String(n?.event_kind), direction: String(n?.direction) })));
        } catch {
          /* invalid bulletin: original behavior */
        }
        const notices = (Array.isArray(message.notices) ? message.notices : []).map((n: Payload) => `${n?.event_kind}:${n?.direction}`).join(",");
        if (notices !== this.last_notices) {
          this.last_notices = notices;
          log(`b ${message.issued_at_utc} ${notices || "-"}`);
        }
      } else if (message.record_type === "state_resync") resyncs.push(message);
      else if (message.record_type === "forecast")
        log(`f ${message.issued_at_utc} ${JSON.stringify(message.notices ?? [])}`);
      else if (message.record_type === "observation_request_result")
        log(`qr ${message.issued_at_utc} ${message.request_id} ${message.status} ${message.completed_count}/${message.minimum_completed}`);
      else if (message.record_type === "observation_request") {
        log(`q ${message.issued_at_utc} ${message.request_id} reward=${message.completion_reward} need=${message.minimum_completed}/${(message.target_ids ?? []).length} at>=${message.completion_factor_threshold} until=${message.deadline_utc}`);
        if (HandoverReader.isNote(message.reason)) log(`note ${message.request_id} ${JSON.stringify(message.reason)}`);
        this.requests_seen++;
        if (this.read_notes && HandoverReader.isNote(message.reason)) notes.push(message);
      }
    }
    // Notes are submitted after the bulletins of this request, so the reader knows the announced mountains.
    if (!this.notes_broken)
      try {
        for (const message of notes) {
          let issued = t;
          try {
            issued = parse_utc(message.issued_at_utc);
          } catch {
            /* a note without a readable issue time is dated now */
          }
          this.reader.terrainDirections = sky.terrain.map((terrain) => Object.keys(COMPASS).find((name) => COMPASS[name] === terrain.centre) ?? "");
          this.reader.submit(String(message.request_id ?? `note-${this.requests_seen}`), issued, message.reason);
        }
        if (notes.length && this.reader.enabled) this._note_earn(payload, t, notes.length);
        const waited = this.reader.waited;
        await this.reader.settle(this._note_wait(payload));
        this.note_credit = Math.max(0, this.note_credit - (this.reader.waited - waited));
        this._apply_notes(t);
      } catch (error) {
        this._notes_failed(error);
      }
    const last = payload.last_result;
    if (last && typeof last === "object" && !Array.isArray(last)) {
      if (last.action === "observe" && this.pending !== null)
        this._digest(last);
      else if (last.action === "report") {
        const correct = Boolean(last.correct);
        sky.report_result(correct, t);
        this.watch?.reportResult(t, correct);
        if (correct) this.probe_now = true;
        this.reader.store.reportResult(t, correct, this.report_note);
        if (correct && this.report_note !== null) this.note_reports_right++;
        if (this.report_source === "detector") {
          if (correct) {
            this.detector_paid_streak = 0;
            this.detector_paid_muted_until = -Infinity;
          } else if (Number(last.score_delta) < 0) {
            // Wait out this night, then 2, 4, ... nights (counting this one) before paying again.
            const night = sv.night_index(t),
              until = night + 2 ** this.detector_paid_streak - 1;
            this.detector_paid_streak++;
            this.detector_paid_muted_until = until < sv.nights.length ? sv.nights[until][1] : Infinity;
          }
        }
        this.report_source = null;
        (this.report_note !== null ? this.tally.reports_note : this.tally.reports_detector)[correct ? 0 : 1]++;
        log(
          `report at ${payload.now_utc}: correct=${correct} delta=${last.score_delta} free_left=${sky.free_left} source=${this.report_note === null ? "rules/model" : `note ${this.report_note.src} (${this.report_note.kind})`}`,
        );
        this.report_note = null;
      }
    }
    this.pending = null;
    for (const message of resyncs) {
      const dropped = sv.resync(message);
      if (this.modelSession.enabled.includes("instrument-fault-report")) {
        sky.suspect = false; sky.low_count = 0; sky.low_seconds = 0; sky.tonight = [];
      }
      log(
        `state_resync: window ${JSON.stringify(message.invalidated_window)} dropped ${dropped} of our exposures`,
      );
    }
    const messages = (payload.new_messages || []).filter((m: Payload) => m && typeof m === "object");
    if (last?.action === "report" && !messages.some((m: Payload) => m.record_type === "report_result"))
      messages.push({ record_type: "report_result", issued_at_utc: payload.now_utc,
        correct: Boolean(last.correct), repaired: Boolean(last.repaired), score_delta: last.score_delta });
    const nextNight = sv.night_index(t), nightRemaining = nextNight < sv.nights.length ? Math.max(0, sv.nights[nextNight][1] - t) : 0;
    const remainingScience = sv.rate_time > 0 ? .5 * sv.rate_gain / sv.rate_time * (nightRemaining + (this.suffix[nextNight + 1] || 0)) : 0;
    await this.modelSession.update({ now_utc: payload.now_utc, messages, completed_exposure: this.modelCompleted,
      baseline_state: this.modelState(), sky: this.modelSky(), remaining_gain: remainingScience });
    if (this.modelSession.enabled.includes("instrument-fault-report")) {
      const prediction = await this.modelSession.faults(payload.now_utc, !this.faultGuard || sky.want_report(t, remainingScience));
      if (MODEL_LOG) log("SURVEY26_FAULT_MODEL_V1", prediction);
    }
    sv.set_requests(payload.active_requests, t);
    const night = sv.night_index(t);
    if (night >= sv.nights.length) {
      this.why = "season_over";
      return { action: "wait", duration_seconds: sv.t_max };
    }
    const [start, end] = sv.nights[night];
    if (t < start) { this.why = "before_night"; return { action: "wait", until_utc: format_utc(start) }; }
    const rem = end - t,
      last_night = night === sv.nights.length - 1;
    if (
      last_night &&
      rem <= 1500 &&
      !this.final_probe_done &&
      this.knobs.final_probe > 0 &&
      sky.free_left >= 1
    ) {
      this.final_probe_done = true;
      this.report_source = "other";
      return { action: "report" };
    }
    if (rem < 45) {
      this.why = "night_ending";
      return {
        action: "wait",
        until_utc: format_utc(last_night ? end : sv.nights[night + 1][0]),
      };
    }
    const remaining_gain =
      ((0.5 * sv.rate_gain) / sv.rate_time) * (rem + this.suffix[night + 1]);
    // Upper-bound arm with the answers (development cards only, never a candidate): ORACLE_REPORT_AT lists the
    // true fault onsets; the first decision at or after each one reports. ORACLE_REPORT_ONLY=1 also silences the detector.
    if (this.oracle_reports.length && t >= this.oracle_reports[0]) {
      while (this.oracle_reports.length && t >= this.oracle_reports[0]) this.oracle_reports.shift();
      log(`report: oracle (true onset) at ${payload.now_utc}`);
      this.report_source = "other";
      return { action: "report" };
    }
    const note_fault = this._guard(() => this._note_fault(t), null);
    if (note_fault !== null) {
      this.report_note = note_fault;
      this.report_source = "note";
      note_fault.state = "reported";
      this.note_reports++;
      log(`report: handover note ${note_fault.src} (${note_fault.kind}, onset ${note_fault.onset === null ? "not given" : format_utc(note_fault.onset)}, level ${note_fault.level ?? "?"}) at ${payload.now_utc}`);
      return { action: "report", reason: `handover note ${note_fault.src}: ${note_fault.kind} fault` };
    }
    if (this.watch !== null) {
      const advice = this.watch.advise(t);
      if (advice.report) {
        log(`report: fault watch (${advice.kind}): ${advice.reason} at ${payload.now_utc}`);
        this.report_source = "watch";
        return { action: "report" };
      }
    }
    // FAULT_WATCH_ONLY=1 silences the detector (for comparisons); otherwise it still speaks when the watch is silent:
    // it finds some faults the watch cannot see (a fault under an earthquake notice with no step, for one).
    const wanted = (this.watch === null || !process.env.FAULT_WATCH_ONLY) && Boolean(this.modelSession.enabled.includes("instrument-fault-report")
        ? this.modelSession.latestFault?.recommend_report : sky.want_report(t, remaining_gain)),
      held = wanted ? this._guard(() => this._quake_explains(t) ?? this._paid_alarm(t), null) : null;
    if (held !== null && held !== this.held_reason) {
      this.held_reason = held;
      this.tally.detector_reports_withheld++;
      log(`report withheld at ${payload.now_utc}: ${held} (e=${sky.e.toFixed(3)} free_left=${sky.free_left})`);
    }
    if (wanted && held === null && !process.env.ORACLE_REPORT_ONLY) {
      log(
        `report: e=${sky.e.toFixed(3)} W=${sky.W.toFixed(3)} rho=${sky.rho.toFixed(3)} low_count=${sky.low_count} at ${payload.now_utc}`,
      );
      this.report_source = "detector";
      return { action: "report" };
    }
    const boundary = (Math.floor(t / sv.slot) + 1) * sv.slot,
      stop = this._guard(() => this.reader.store.siteClosure(t), null);
    // A stop the notes announce (work on the telescope, a power cut, "we are stopped tonight") appears in no
    // bulletin. It is checked before it is believed: one shortest exposure (once per stop and night), and only
    // when that comes back empty is the rest of the stop, or of the night, sat out in a single wait. A misread
    // note then costs a minute.
    let probe_stop = false;
    if (stop !== null && stop.until - t >= 45) {
      const key = `${stop.until}|${night}`;
      if (this.stop_probed === key && sky.closed_suspect) {
        this.why = `note_stop:${stop.kind} until ${format_utc(stop.until)}`;
        return { action: "wait", until_utc: format_utc(Math.min(stop.until, end)), reason: `handover note: ${stop.kind} stop` };
      }
      if (this.stop_probed !== key) {
        this.stop_probed = key;
        probe_stop = true;
      }
    }
    if (sky.site_closed_now) { this.why = "bulletin_site_closed"; return { action: "wait", until_utc: format_utc(Math.min(boundary, end)) }; }
    if (sv.future !== null) {
      sv.set_now(t);
      if (night !== this.night_seen) for (let z = 0; z < sv.cat.nz; z++) sv.refresh_zone(z);
    }
    if (night !== this.night_seen) {
      this.night_seen = night;
      sv.update_urgency(night);
      sv.update_band_bonus();
    }
    this._pace(payload, t, night);
    let ceiling = sv.t_max;
    if (sky.closed_suspect) {
      ceiling = Math.trunc(boundary - t);
      if (ceiling < 240) ceiling += Math.trunc(sv.slot);
    }
    if (probe_stop) ceiling = sv.t_min;
    // Do not let an exposure run into a stop the notes announce.
    const ahead = this._guard(() => this.reader.store.nextSiteClosure(t, sv.t_max), null);
    if (ahead !== null) {
      const room = Math.trunc(ahead.start - t);
      if (room < sv.t_min + 30) {
        this.why = `stop_ahead:${ahead.kind} at ${format_utc(ahead.start)}`;
        return { action: "wait", until_utc: format_utc(Math.min(ahead.start, end)), reason: "handover note: stop about to begin" };
      }
      if (room < ceiling) ceiling = room;
    }
    const calibrate =
        this.calib_left > 0 || (sv.offset.uncertain && this.calib_extra < 8),
      floor = calibrate || sky.closed_suspect ? sv.t_min : this.t_floor,
      began = cpu_seconds();
    this.plan_free = !calibrate && !sky.closed_suspect && !probe_stop && this.t_floor <= sv.t_min && ceiling >= sv.t_max;
    let plan = this.planner.plan(
      t,
      night,
      Math.trunc(this.k),
      floor,
      ceiling,
      calibrate,
    );
    let probed = false;
    if (plan === null && this.knobs.probe_every > 0) {
      if (this.null_since === null) this.null_since = t;
      if ((this.probe_now || t - this.null_since >= this.knobs.probe_after) && t - this.last_probe >= this.knobs.probe_every) {
        this.last_probe = t;
        const saved = sky.snapshot();
        sky.reset_for_probe();
        plan = this.planner.plan(t, night, Math.trunc(this.k), sv.t_min, sv.t_min, false);
        if (plan === null) sky.restore(saved);
        else {
          probed = true;
          this.probes++;
          log(`p ${payload.now_utc} probe: rho ${saved.rho.toFixed(5)} -> ${sky.rho.toFixed(2)} after ${Math.round(t - this.null_since)} s of no plan${this.probe_now ? " (after a repair)" : ""}`);
        }
      }
    }
    if (plan !== null) { this.null_since = null; this.probe_now = false; }
    this.own += 0.1 * (cpu_seconds() - began - this.own);
    if (plan === null) {
      // Nothing worth a pointing. When that is the state of the survey (the recent gain rate is next to
      // nothing and no request is open) rather than of this slot, look again in an hour, not every slot:
      // every turn costs engine time against the real-time cap. Known events ahead still wake the agent.
      let until = Math.min(boundary, end);
      if (this.knobs.idle_wait > 0 && sv.extra.size === 0 && sv.rate_gain / sv.rate_time < this.knobs.idle_rate)
        until = Math.min(end, Math.max(until, Math.min(t + this.knobs.idle_wait, this._guard(() => this._next_note_event(t), Number.POSITIVE_INFINITY))));
      this.why = `plan_null: gain_rate=${(sv.rate_gain / Math.max(sv.rate_time, 1e-9)).toExponential(2)} idle_rate=${this.knobs.idle_rate} extra=${sv.extra.size} floor=${floor} ceiling=${ceiling} closed_suspect=${sky.closed_suspect} probe_stop=${probe_stop}`;
      return { action: "wait", until_utc: format_utc(until) };
    }
    if (this.calib_left > 0) this.calib_left--;
    else if (calibrate) this.calib_extra++;
    return this._action(plan, t, night, rem);
  }
  async _action(plan: Plan, t: number, night: number, rem: number): Promise<Payload> {
    const sv = this.sv,
      site = sv.site,
      cat = sv.cat,
      sky = sv.sky;
    let duration = Math.trunc(plan.duration);
    // While the fault watch is comparing declared programs the exposures score next to nothing; short ones get the answer sooner.
    if (this.watch?.wantProgram(t) != null) duration = Math.min(duration, Math.max(sv.t_min, 300));
    if (sky.suspect && duration > 2 * sv.t_min)
      duration = Math.max(sv.t_min, Math.trunc(0.8 * duration));
    const teff = Math.min(duration, rem),
      centre = plan.centre,
      [, zen, north, east] = site.frame(t),
      sin_alt = centre[0] * zen[0] + centre[1] * zen[1] + centre[2] * zen[2];
    const alt = Math.asin(Math.max(-1, Math.min(1, sin_alt))) * DEG,
      az = mod(
        Math.atan2(
          centre[0] * east[0] + centre[1] * east[1] + centre[2] * east[2],
          centre[0] * north[0] + centre[1] * north[1] + centre[2] * north[2],
        ) * DEG,
        360,
      );
    const cmd_alt = Math.min(90, Math.max(0, alt - sv.offset.alt));
    let cmd_az = mod(az - sv.offset.az, 360);
    if (cmd_az >= 360) cmd_az = 0;
    const samples = [1 / 6, 0.5, 5 / 6].map((x) => t + teff * x),
      frames = samples.map((s) => site.frame(s)),
      moons = samples.map((s) => site.moon(s)),
      [rho, w_level] = sky.levels(),
      directional = sky.directional,
      n = sv.n_side;
    const assignments: Record<string, string> = {},
      items: ExposureItem[] = [],
      pairs: [number, number][] = [];
    let clear = true,
      predicted = 0;
    for (const [cell, j, phit] of plan.picks) {
      if (Object.hasOwn(assignments, String(cell))) continue;
      const [x, y, z] = [cat.ux[j], cat.uy[j], cat.uz[j]];
      let a_sum = 0;
      for (let i = 0; i < frames.length; i++) {
        const zen_k = frames[i][1];
        a_sum +=
          site.transmission(x * zen_k[0] + y * zen_k[1] + z * zen_k[2]) *
          site.lunar_factor(x, y, z, moons[i]);
      }
      const a = a_sum / 3,
        unit = (cat.flux[j] * teff * a) / sv.f0t0,
        [, zen_m, north_m, east_m] = frames[1],
        alt_j =
          Math.asin(
            Math.max(
              -1,
              Math.min(1, x * zen_m[0] + y * zen_m[1] + z * zen_m[2]),
            ),
          ) * DEG;
      const az_j = mod(
          Math.atan2(
            x * east_m[0] + y * east_m[1] + z * east_m[2],
            x * north_m[0] + y * north_m[1] + z * north_m[2],
          ) * DEG,
          360,
        ),
        d = directional ? sky.dir_factor(alt_j, az_j) : 1;
      if (d < 0.999) clear = false;
      const row = Math.floor(cell / n),
        col = cell % n;
      items.push([
        j,
        row,
        col,
        unit,
        a,
        d,
        alt_j,
        az_j,
        phit >= 0.999,
        cat.weight[j],
        x * north[0] + y * north[1] + z * north[2],
        x * east[0] + y * east[1] + z * east[2],
        x * zen[0] + y * zen[1] + z * zen[2],
      ]);
      assignments[String(cell)] = cat.ids[j];
      const g = unit * rho * d;
      pairs.push([cat.weight[j] * Math.min(g, 1) + 1e-6, w_level * a]);
      const expected =
        cat.weight[j] *
          Math.min(g, 1) *
          sky.expected_multiplier(w_level * a, t) -
        sv.best[j];
      if (expected > 0) predicted += expected;
    }
    const program = (this.watch?.wantProgram(t) as ReturnType<typeof sky.choose_program> | null | undefined) ?? sky.choose_program(pairs, t);
    if (this.modelSession.enabled.includes("conditions-reliability")) {
      const began = this.models.worker.cpuSeconds;
      const candidates = [900, 1800, 3600].flatMap(lead => items.map(item => ({
        candidate_id: `${cat.ids[item[0]]}:${lead}`, ra_deg: cat.ra[item[0]], dec_deg: cat.dec[item[0]],
        start_utc: format_utc(t + lead), duration_seconds: duration,
      })));
      const predictions = await this.modelSession.conditions(format_utc(t), candidates);
      const elapsed = this.models.worker.cpuSeconds - began;
      if (MODEL_LOG) log("SURVEY26_MODELS_V1", { now_utc: format_utc(t), candidates: candidates.length, cpu_seconds: elapsed,
        models: [{ model_id: predictions.model_id, model_version: predictions.model_version,
          eligible: predictions.predictions.filter(r => r.eligible).length, final_test_passed: predictions.final_test_passed }] });
      if (process.env.BASE_MODELS_LOG === "full") log("SURVEY26_MODEL_PREDICTIONS_V1", predictions);
    }
    this.pending = {
      t0: t,
      teff,
      program,
      cmd_alt,
      cmd_az,
      items,
      night,
      clear_dirs: clear,
      notices: [],
      predicted,
    };
    return {
      action: "observe",
      pointing: { alt_deg: cmd_alt, az_deg: cmd_az },
      assignments,
      duration_seconds: duration,
      program,
    };
  }
  /** Run a piece of note logic; if it throws, stop using the notes rather than lose decisions. */
  _guard<T>(run: () => T, otherwise: T): T {
    if (this.notes_broken) return otherwise;
    try {
      return run();
    } catch (error) {
      this._notes_failed(error);
      return otherwise;
    }
  }
  _notes_failed(error: unknown): void {
    this.notes_broken = true;
    this.sv.sky.set_notes(null);
    for (const terrain of this.sv.sky.terrain) terrain.claim = null;
    log("handover: note handling failed and is switched off for the rest of the run:\n" + (error instanceof Error ? error.stack : String(error)));
  }
  /** Notes still to come this survey, from the rate so far (used to keep wall time for reading them). */
  _notes_ahead(night: number): number {
    const sv = this.sv,
      done = 1 - (Math.max(0, sv.nights[Math.min(night, sv.nights.length - 1)][1] - this.now) + this.suffix[Math.min(night + 1, sv.nights.length)]) / sv.total_night_seconds;
    if (!this.reader.enabled) return 0;
    if (done < 0.05) return Math.max(this.reader.notes, sv.nights.length / 3);
    return Math.min(400, (this.reader.notes / done) * (1 - done));
  }
  /**
   * Credit the notes just received with their share of the spare wall time. With plenty to spare every note
   * earns a full wait; with little, the credit builds up over several notes and pays for a wait now and then;
   * with none (the real-time cap already limits the number of exposures) nothing is earned and no note is waited for.
   */
  _note_earn(payload: Payload, t: number, count: number): void {
    const wall_left = Number((payload.wallclock || {}).wall_remaining_seconds),
      cap = this.knobs.note_wait_max;
    if (!Number.isFinite(wall_left)) {
      this.note_credit = cap;
      return;
    }
    const night = Math.min(this.sv.night_index(t), this.sv.nights.length - 1),
      spare = this.wall_spare ?? 0.25 * wall_left,
      each = Math.max(0, spare) / (this._notes_ahead(night) + count);
    this.note_credit = Math.min(2 * cap, this.note_credit + count * Math.min(cap, each));
  }
  /** How long this decision may wait for model readings in flight. Waiting costs wall time only, never survey time. */
  _note_wait(payload: Payload): number {
    if (!this.reader.pending) return 0;
    const first = this.reader.modelReadings === 0 && this.reader.failedReadings === 0,
      wall_left = Number((payload.wallclock || {}).wall_remaining_seconds);
    if (Number.isFinite(wall_left) && wall_left < 60) return 0;
    // A wait shorter than a reading takes buys nothing, so the credit has to cover one. The first reading of a
    // run may take longer (a new connection, a provider that turns out to be exhausted).
    if (this.note_credit < 1.3 * this.reader.latency) return 0;
    return first ? Math.min(this.knobs.note_first_wait, 3 * this.note_credit) : Math.min(this.knobs.note_wait_max, this.note_credit);
  }
  /** Hand the facts in force at t to the sky model and the terrain profiles. */
  _apply_notes(t: number): void {
    const store = this.reader.store,
      sky = this.sv.sky,
      slot = Math.floor(t / this.sv.slot);
    if (store.revision === this.facts_revision && slot === this.facts_slot) return;
    this.facts_revision = store.revision;
    this.facts_slot = slot;
    for (const terrain of sky.terrain) {
      const claim = store.ridgeFor(terrain.centre, this.knobs.ridge_margin);
      if (claim !== null && terrain.claim_conflicts < 3 && terrain.claim !== claim) {
        log(`handover: mountain at azimuth ${terrain.centre} clear above ${claim.toFixed(1)} deg`);
        terrain.claim = claim;
      }
    }
    const view: NoteView = { sectors: [], all: [] };
    for (const w of store.active(t)) {
      const firm = w.conf === "firm";
      if (w.all) view.all.push({ effect: w.effect, kind: w.kind, firm });
      else view.sectors.push({ azLo: w.azLo, azSpan: w.azSpan, effect: w.effect, kind: w.kind, firm, key: `${w.t0}|${w.azLo}|${w.azSpan}|${w.effect}` });
    }
    sky.set_notes(store.windows.length || store.terrain.length ? view : null);
    this._apply_lost_nights(t);
  }
  /**
   * Nights the notes say will be lost (announced stops, forecast rain) no longer count as time in which
   * a field can still be observed, so fields that would lose their window move up before the stop.
   */
  _apply_lost_nights(t: number): void {
    const sv = this.sv,
      closed = this.reader.store.closedIntervals(t),
      key = closed.map(([a, b]) => `${a}-${b}`).join(",");
    if (key === this.lost_key) return;
    this.lost_key = key;
    const open = sv.nights.map(([start, end]) => {
      let lost = 0;
      for (const [a, b] of closed) lost += Math.max(0, Math.min(b, end) - Math.max(a, start, t));
      return Math.max(0, 1 - lost / Math.max(1, end - start));
    });
    sv.cat.set_open_nights(open);
    const night = sv.night_index(t);
    if (night < sv.nights.length) {
      sv.urgency_night = -1;
      sv.update_urgency(night);
    }
    const gone = open.reduce((sum, share, i) => sum + (1 - share) * (sv.nights[i][1] - sv.nights[i][0]), 0);
    log(`handover: ${closed.length} announced whole-sky stops ahead, ${(gone / 3600).toFixed(1)} night hours taken out of the visibility table`);
  }
  /** The next moment the notes say something will need attention (an announced fault, an incoming request). */
  _next_note_event(t: number): number {
    let next = Number.POSITIVE_INFINITY;
    for (const fault of this.reader.store.openFaults()) {
      const due = FactStore.faultOnset(fault) + 900;
      if (due > t && due < next) next = due;
    }
    for (const request of this.reader.store.requests) if (request.arrive > t && request.arrive < next) next = request.arrive;
    return next;
  }
  /** Median throughput reading before and after a moment; null where there is none. */
  _levels_around(moment: number): [number | null, number | null] {
    const before = this.readings.filter(([t0]) => t0 < moment - 60 && t0 > moment - 30 * 3600).slice(-3).map(([, level]) => level),
      after = this.readings.filter(([t0]) => t0 >= moment - 60).map(([, level]) => level),
      mid = (values: number[]) => (values.length ? [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] : null);
    return [mid(before), mid(after)];
  }
  /**
   * The handover note, if any, that warrants a fault report now.
   * - A note that states a drop that is not weather and asks for a ticket is reported at once.
   * - An announced job ("efficiency will fall from about 3:15") or an undecided note ("counts are low since
   *   3:05, can't tell why") is reported once this agent's own readings show the drop, or, with no usable
   *   reading 45 minutes later and both free reports in hand, on a free report.
   */
  _note_fault(t: number): FaultFact | null {
    const store = this.reader.store,
      sky = this.sv.sky;
    for (const fault of store.openFaults()) {
      const onset = FactStore.faultOnset(fault);
      if (onset > t) continue;
      const [before, after] = this._levels_around(onset),
        dropped = (sky.suspect && sky.e < 0.85 * sky.e_ref) || (before !== null && after !== null && after < 0.8 * before);
      if (fault.kind === "instrument" && fault.action === "report") {
        // With the free reports used up a wrong report costs points. A ticket the model read out of the note
        // (who says so, since when, that it is not the sky) is still acted on: such notes have matched a real
        // fault every time they could be checked, and waiting for the agent's own readings has cost days when
        // an earthquake hid them.
        return fault;
      }
      // An announced job is checked against the readings whichever word the model put in "action".
      if (fault.action !== "verify" && !(fault.kind === "scheduled" && fault.action === "report")) continue;
      if (t > onset + 60 * 3600) {
        fault.state = "done";
        continue;
      }
      if (t < onset + 600) continue;
      // A note that names no time says nothing about when to report: what is left is this agent's own reading of a
      // drop, which the fault watch and the detector judge with their own guards. Such a note may use a free report
      // and never a paid one (in the first platform run with the watch they cost five paid reports, none correct).
      if (fault.onset === null && sky.free_left < 1) continue;
      if (dropped) return fault;
      // No usable reading since: a free report is spent only on a job that was announced for this time.
      // "Counts look low, can't tell why" with no announcement has been thin cloud as often as a fault.
      const announced = fault.kind === "scheduled" || store.faults.some((other) => other.kind === "scheduled" && other.onset !== null && Math.abs(other.onset - onset) <= 2 * 3600);
      if (announced && after === null && t >= onset + 2700 && sky.free_left >= 2) return fault;
    }
    return null;
  }
  /**
   * The detector's alarms are worth less once the notes drive the reports. While nobody repairs a fault for
   * days, an alarm usually lands on a real one; when each fault is reported within the hour because a note
   * announced it, what is left for the detector is weather, earthquakes and stops, and its alarms are wrong
   * most of the time (in the local runs 1 right in 10). A wrong report is free twice after every correct one
   * and costs 150 points after that, so the detector keeps its free reports and loses the paid ones.
   */
  _paid_alarm(t: number): string | null {
    if (this.sv.sky.free_left >= 1) return null;
    if (this.note_reports_right >= 1) return "the free reports are used up and the handover notes have been naming the faults";
    // With the fault watch on, the free reports are shared with it and run out sooner. Each wrong report of the
    // detector that cost points doubles the time before it may pay again (the rest of that night, then two nights,
    // four, ...) until one of its reports is correct: the rule the watch applies to itself.
    if (this.watch !== null && t < this.detector_paid_muted_until)
      return "the free reports are used up and the detector is waiting after a paid wrong report";
    return null;
  }
  /**
   * Why a report from the agent's own detector should be withheld now, or null.
   * After an earthquake the readings stay low for about a week and recover night by night with nothing broken.
   * The detector cannot tell that from a fault; a note that names the shaking and says to leave it alone can.
   * A fault a note names is reported through _note_fault whatever this says, and a fresh step down in the
   * agent's own readings (an earthquake only gets better) lifts the hold.
   */
  _quake_explains(t: number): string | null {
    const store = this.reader.store,
      sky = this.sv.sky,
      recovery = 7 * 86400;
    let quake: FaultFact | null = null,
      began = -1e18;
    for (const fault of store.faults) {
      if (fault.kind !== "earthquake") continue;
      // The note is written after the event; the bulletin that announced the shaking dates it.
      let at = Math.min(fault.onset ?? fault.issued, fault.issued);
      if (sky.quake_since <= fault.issued && fault.issued - sky.quake_since < 48 * 3600) at = Math.min(at, sky.quake_since);
      if (at <= t && at > began) {
        quake = fault;
        began = at;
      }
    }
    if (quake === null || t - began > recovery) return null;
    const repaired = store.lastCorrectReport >= began - 3600;
    // A note that points at a fault of its own since the shaking is a reason to report, not to hold.
    if (store.openFaults().some((fault) => FactStore.faultOnset(fault) <= t && FactStore.faultOnset(fault) >= began - 3600)) return null;
    const since = Math.max(began, store.lastCorrectReport),
      earlier = this.readings.filter(([t0]) => t0 > since && t0 < t - 5 * 3600).map(([, level]) => level),
      latest = this.readings.filter(([t0]) => t0 >= t - 3 * 3600).slice(-3).map(([, level]) => level),
      mid = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
    if (earlier.length >= 2 && latest.length >= 2 && mid(latest) < 0.75 * mid(earlier)) return null;
    return `note ${quake.src} puts the low readings down to the earthquake of ${format_utc(began)}${repaired ? ", and the fault since then is repaired" : ""}`;
  }
  modelState(): Payload {
    const sky = this.sv.sky, [rho, weather] = sky.levels();
    return { weather, throughput: rho, efficiency: sky.e, has_measurement: sky.measured,
      last_measurement_utc: sky.tonight.length ? format_utc(sky.tonight[sky.tonight.length - 1][0]) : null,
      suspected_fault: sky.suspect, suspected_closure: sky.closed_suspect || sky.site_closed_now, quake_notice: sky.quake_now,
      terrain: sky.terrain.map(p => ({ centre: p.centre, sure: p.sure, top: p.top, floor: p.floor, box: p.box })),
      directional_notices: sky.dir_now };
  }
  modelSky(): Payload {
    const sky = this.sv.sky;
    return { b_dark: sky.b_dark, b_bright: sky.b_bright, W: sky.W, e: sky.e, e_ref: sky.e_ref,
      suspect: sky.suspect, low_count: sky.low_count, low_seconds: sky.low_seconds, closed_suspect: sky.closed_suspect,
      probe: sky.probe, free_left: sky.free_left, reports_false: sky.reports_false, cooldown_until: sky.cooldown_until };
  }
  async close(): Promise<void> { await this.modelSession.close(); }
  /** With the virtual clock used up the platform would have ended the run: no further exposure is taken. */
  expired(payload: Payload): Payload {
    const sv = this.sv, sim = this.sim!;
    if (sim.expiredAt === null) {
      sim.expiredAt = String(payload.now_utc);
      log(`SIM_WALL: the ${sim.cap} s cap is used up at ${payload.now_utc} after ${sim.decisions} decisions; the rest of the survey is not observed`);
    }
    this.pending = null;
    return { action: "wait", until_utc: format_utc(sv.nights[sv.nights.length - 1][1]) };
  }
  status(payload: Payload): string {
    const sv = this.sv,
      sky = sv.sky,
      clock = payload.wallclock || {};
    return `${payload.now_utc} obs=${sv.observes} total=${payload.running_total} req_left=${sv.required_left} rho=${sky.rho.toFixed(3)} W=${sky.W.toFixed(3)} e=${sky.e.toFixed(2)} price=${sv.price.toFixed(5)} avgT=${this.avg_t.toFixed(0)} k=${this.k.toFixed(1)} own=${(this.own * 1000).toFixed(1)}ms eng=${(this.engine * 1000).toFixed(1)}ms floor=${this.t_floor} miss=${this.misses}/${this.assigned} gain/pred=${(this.realised_gain / Math.max(this.predicted_gain, 1e-9)).toFixed(3)} capped=${this.planner.stat_negative}/${this.planner.stat_plans} cpu_left=${clock.remaining_real_cpu_seconds} wall_left=${clock.wall_remaining_seconds} wall_spare=${this.wall_spare === null ? "?" : this.wall_spare.toFixed(0)} avgFree=${this.avg_free.toFixed(0)} notes=${this.reader.notes}/${this.reader.modelReadings} waited=${this.reader.waited.toFixed(0)}s credit=${this.note_credit.toFixed(1)}`;
  }
}
export function safe_wait(_payload: Payload): Payload {
  return { action: "wait", duration_seconds: 900 };
}
/** Retains full incoming lines, including the last line without a newline. */
export async function* raw_lines(
  input: NodeJS.ReadableStream,
): AsyncGenerator<string> {
  const decoder = new StringDecoder("utf8");
  let pending = "";
  for await (const chunk of input) {
    pending += typeof chunk === "string" ? chunk : decoder.write(chunk);
    let end: number;
    while ((end = pending.indexOf("\n")) >= 0) {
      yield pending.slice(0, end + 1);
      pending = pending.slice(end + 1);
    }
  }
  pending += decoder.end();
  if (pending) yield pending;
}
export async function main(): Promise<number> {
  let agent: Agent | null = null,
    received_sequence = 0,
    payload_last: Payload | null = null;
  try {
  for await (const line of raw_lines(process.stdin)) {
    received_sequence++;
    // The verbatim input is written only on request (BASE_INPUT_LOG=on): it fills the run log within a few hundred requests.
    if (
      INPUT_LOG &&
      !process.stderr.write(
        "SURVEY26_INPUT_V1 " +
          asciiJson({ received_sequence, raw: line }) +
          "\n",
      )
    )
      await once(process.stderr, "drain");
    if (!line.trim()) continue;
    let message: Payload;
    try {
      message = JSON.parse(line);
    } catch {
      continue;
    }
    if (!message || typeof message !== "object" || Array.isArray(message))
      continue;
    const kind = message.message_type;
    if (kind === "initialize") {
      try {
        await agent?.close();
        agent = new Agent(message.payload);
      } catch (error) {
        log(
          "initialize failed:\n" +
            (error instanceof Error ? error.stack : String(error)),
        );
        agent = null;
      }
      continue;
    }
    if (kind === "finish") {
      if (agent !== null) {
        log("finish:", message.payload ?? {});
        log("SURVEY26_HANDOVER_V1", { ...agent.reader.summary(), note_reports: agent.note_reports });
        const tally = agent.tally, round = (value: number) => Number(value.toFixed(1));
        log("SURVEY26_SUMMARY_V1", { build: build_id(), decisions: agent.decisions, exposures: tally.exposures, exposure_hours: round(tally.exposure_hours),
          all_zero_exposures: tally.zero_exposures, all_zero_hours: round(tally.zero_hours), gain_seen: round(tally.gain),
          reports_for_notes_right_wrong: tally.reports_note, reports_by_detector_right_wrong: tally.reports_detector,
          detector_reports_withheld: tally.detector_reports_withheld, requests_seen: agent.requests_seen,
          last_wall_left: (payload_last?.wallclock ?? {}).wall_remaining_seconds ?? null });
        if (agent.sim !== null) log("SURVEY26_SIM_WALL_V1", agent.sim.summary());
      }
      break;
    }
    if (kind !== "decision_request") continue;
    const arrived = monotonic(),
      payload = message.payload || {};
    payload_last = payload;
    let action: Payload | null = null;
    if (agent !== null) {
      const sim = agent.sim;
      if (sim !== null) {
        // The engine's time for this request, by the platform's measured law rather than this machine's clock.
        const cost = sim.engine(agent.assigned, agent.requests_seen);
        sim.used += cost;
        sim.decisions++;
        agent.engine += 0.05 * (cost - agent.engine);
        payload.wallclock = { ...(payload.wallclock || {}), wall_remaining_seconds: Number(sim.left.toFixed(3)) };
      } else if (agent.reply_at !== null)
        agent.engine += 0.05 * (arrived - agent.reply_at - agent.engine);
      const cpu_before = cpu_seconds() + agent.models.worker.cpuSeconds;
      try {
        action = sim !== null && sim.left <= 0 ? agent.expired(payload) : await agent.decide(payload);
      } catch (error) {
        log(
          "decide failed:\n" +
            (error instanceof Error ? error.stack : String(error)),
        );
        agent.pending = null;
      }
      if (sim !== null) sim.used += sim.ownScale * Math.max(0, cpu_seconds() + agent.models.worker.cpuSeconds - cpu_before);
      agent.decisions++;
      if (agent.decisions % Math.trunc(agent.knobs.log_every) === 0)
        log(agent.status(payload));
    }
    action ??= safe_wait(payload);
    const response = {
      protocol_version: message.protocol_version ?? PROTOCOL,
      message_type: "decision_response",
      decision_sequence: message.decision_sequence ?? null,
      ...action,
    };
    await new Promise<void>((resolve, reject) =>
      process.stdout.write(JSON.stringify(response) + "\n", (error) =>
        error ? reject(error) : resolve(),
      ),
    );
    if (agent !== null) agent.reply_at = monotonic();
  }
  } finally { await agent?.close(); }
  return 0;
}
if (require.main === module)
  main().catch((error) => {
    log(error instanceof Error ? error.stack : String(error));
    process.exitCode = 1;
  });
