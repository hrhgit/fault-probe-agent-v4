/**
 * Reads handover notes during a run: every note is sent to the model in the background, the answer is
 * validated and added to the fact store. A note is read by the model or not at all: without a model
 * (no key, OBSERVER_MODEL_DISABLED=1) or when a call fails, nothing is taken from it and the agent
 * goes on by its ordinary rules.
 */
import { FactStore, NoteFacts } from "../facts/store";
import { Json, LLM } from "../llm/client";
import { SimClock } from "../sim/clock";
import { codebookLines, factsFromAnswer, noteContext, SiteCalendar } from "./extract";
import { buildUserPrompt, EXTRACT_SYSTEM } from "./prompt";
import { annotate } from "./text";

const GENERIC_REASONS = new Set(["", "time-critical follow-up"]);

interface Job {
  src: string;
  issued: number;
  started: number;
  finished: boolean;
  promise: Promise<void>;
  /** The validated reading, once the answer is in (null: the call failed). */
  facts: NoteFacts | null;
  /** With a virtual clock: the virtual time at which the answer counts as arrived. */
  readyAt: number;
}

const utc = (t: number | null) => (t === null ? null : new Date(t * 1000).toISOString().slice(5, 16).replace("T", " "));
/** What a reading put into the store, in UTC and compact: one line per note in agent.log. */
export function describeFacts(f: NoteFacts): Json {
  return {
    w: f.windows.map((w) => `${utc(w.t0)}..${utc(w.t1)} ${w.all ? "ALL" : `${w.azLo}+${w.azSpan}`} ${w.kind}/${w.effect}/${w.conf}`),
    t: f.terrain.map((t) => `${t.azLo}+${t.azSpan}${t.single ? "s" : ""} ridge=${t.ridge} ok>=${t.okAbove}`),
    f: f.faults.map((x) => `${x.kind}/${x.action} onset=${utc(x.onset)} level=${x.level}`),
    n: f.nights.map(([n, q]) => `${n}:${q}`),
    r: f.requests.map((r) => `${utc(r.arrive)}..${utc(r.deadline)}`),
  } as unknown as Json;
}

export class HandoverReader {
  readonly store = new FactStore();
  readonly enabled: boolean;
  terrainDirections: string[] = [];
  private jobs: Job[] = [];
  private readonly seen = new Set<string>();
  /** Local evaluation with a virtual clock (sim/clock.ts): readings arrive on that clock, not in real time. */
  sim: SimClock | null = null;
  notes = 0;
  modelReadings = 0;
  failedReadings = 0;
  waited = 0;
  /** Typical seconds one model reading takes (running mean), for the wait budget. */
  latency = 8;

  constructor(
    readonly cal: SiteCalendar,
    private readonly log: (text: string) => void,
    readonly llm: LLM = new LLM(log),
    readonly callTimeout = Number(process.env.LLM_CALL_TIMEOUT || 75),
  ) {
    this.enabled = this.llm.available && process.env.OBSERVER_MODEL_DISABLED !== "1";
  }

  /** A request's reason text is a note when somebody wrote more than the stock phrase. */
  static isNote(reason: unknown): reason is string {
    return typeof reason === "string" && !GENERIC_REASONS.has(reason.trim()) && (reason.includes("【") || reason.trim().length >= 40);
  }

  /** Readings asked for and not yet in the fact store. */
  get pending(): number {
    return this.sim !== null ? this.jobs.length : this.jobs.filter((job) => !job.finished).length;
  }

  submit(src: string, issued: number, reason: string): void {
    if (this.seen.has(src)) return;
    this.seen.add(src);
    this.notes++;
    if (!this.enabled) return;
    this.store.learnCodebook(codebookLines(reason));
    const context = noteContext(issued, this.cal, this.terrainDirections, this.store.codebook.slice(0, 40)),
      { text, decoded } = annotate(reason),
      user = buildUserPrompt(context, text),
      job: Job = { src, issued, started: performance.now(), finished: false, promise: Promise.resolve(), facts: null,
        readyAt: this.sim !== null ? this.sim.used + this.sim.readingSeconds(src) : 0 };
    job.promise = this.llm
      .chat(`note ${src}`, EXTRACT_SYSTEM, user, this.callTimeout, 2200)
      .then((result) => {
        const seconds = this.sim !== null ? this.sim.readingSeconds(src) : (performance.now() - job.started) / 1000;
        if (result.json === null) {
          this.failedReadings++;
          return;
        }
        if (!result.cached || this.sim !== null) this.latency += 0.25 * (seconds - this.latency);
        const facts = factsFromAnswer(result.json as Json, src, issued, this.cal);
        job.facts = facts;
        this.modelReadings++;
        this.log(
          `handover ${src}: model reading in ${seconds.toFixed(1)}s (${result.provider}${result.cached ? ", cached" : ""}; ${decoded} decoded runs) ` +
            `windows=${facts.windows.length} terrain=${facts.terrain.length} faults=${facts.faults.length} nights=${facts.nights.length}`,
        );
        this.log(`facts ${src} ${JSON.stringify(describeFacts(facts))}`);
      })
      .catch((error) => {
        this.failedReadings++;
        this.log(`handover ${src}: reading failed (${error instanceof Error ? error.message : String(error)})`);
      })
      .finally(() => {
        job.finished = true;
      });
    this.jobs.push(job);
  }

  /** Move finished readings into the store. Returns how many were applied. */
  drain(): number {
    let count = 0;
    const arrived = (job: Job) => job.finished && (this.sim === null || job.readyAt <= this.sim.used);
    for (const job of this.jobs)
      if (arrived(job) && job.facts !== null) {
        this.store.apply(job.facts);
        count++;
      }
    this.jobs = this.jobs.filter((job) => !arrived(job));
    return count;
  }

  /** Wait up to `seconds` for the readings in flight, then apply what has arrived. */
  async settle(seconds: number): Promise<number> {
    if (this.sim !== null) {
      if (this.jobs.length && seconds > 0) {
        // On the virtual clock: until every reading is in, or the limit, whichever comes first.
        const limit = this.sim.used + seconds,
          last = Math.max(...this.jobs.map((job) => job.readyAt)),
          until = last <= limit ? last : limit,
          spent = Math.max(0, until - this.sim.used);
        this.sim.used += spent;
        this.sim.waited += spent;
        this.waited += spent;
      }
      // The answers themselves come from the local cache (or a real call, whose real duration is not charged).
      await Promise.all(this.jobs.filter((job) => job.readyAt <= this.sim!.used).map((job) => job.promise));
      return this.drain();
    }
    const open = this.jobs.filter((job) => !job.finished);
    if (open.length && seconds > 0) {
      const began = performance.now();
      let timer: NodeJS.Timeout | undefined;
      await Promise.race([
        Promise.all(open.map((job) => job.promise)),
        new Promise<void>((resolve) => (timer = setTimeout(resolve, seconds * 1000))),
      ]);
      clearTimeout(timer);
      this.waited += (performance.now() - began) / 1000;
    }
    return this.drain();
  }

  summary(): Json {
    return {
      notes: this.notes, model_readings: this.modelReadings, failed: this.failedReadings, waited_seconds: Number(this.waited.toFixed(1)),
      windows: this.store.windows.length, terrain: this.store.terrain.length, faults: this.store.faults.length, llm: this.llm.summary(),
    };
  }
}
