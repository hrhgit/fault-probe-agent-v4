/**
 * Local evaluation only (SIM_WALL=1): a virtual real-time clock.
 *
 * On the platform the 1800 s real-time cap, not the CPU budget, is what limits this agent, and nearly all
 * of that time is the engine's: in the downloaded formal runs one decision costs about
 *
 *     base + slope * fibres_assigned_so_far * (1 + requests_issued_so_far)
 *
 * (20 ms at the start of a card, 400-1100 ms at the end). A development machine has another speed and
 * another load every hour, so real time measured locally says little about the platform, and two runs
 * compared under different load are not comparable at all. With this clock the agent charges itself the
 * platform's engine time for every decision and a fixed time for every model reading instead of measuring
 * them, and stops observing when the cap is used up. Runs become repeatable, and waiting for the model
 * costs what it would cost on the platform.
 *
 * The numbers come from tools/run_eval.py (--sim-wall), which holds the fit for each card. Nothing here
 * is active unless SIM_WALL=1; a platform run never sets it.
 */
export class SimClock {
  /** Virtual seconds used since the first decision request. */
  used = 0;
  /** Survey time at which the cap was reached, or null. */
  expiredAt: string | null = null;
  decisions = 0;
  waited = 0;

  constructor(
    readonly cap: number,
    readonly base: number,
    readonly slope: number,
    readonly ownScale: number,
    readonly latency: number,
  ) {}

  static fromEnv(env: NodeJS.ProcessEnv = process.env): SimClock | null {
    if (env.SIM_WALL !== "1") return null;
    const number = (name: string, fallback: number) => {
      const value = Number(env[name]);
      return env[name]?.trim() && Number.isFinite(value) ? value : fallback;
    };
    return new SimClock(
      number("SIM_WALL_CAP", 1800),
      number("SIM_ENGINE_BASE_MS", 15) / 1000,
      number("SIM_ENGINE_SLOPE", 220) / 1e9,
      number("SIM_OWN_SCALE", 2),
      number("SIM_LLM_LATENCY", 8),
    );
  }

  get left(): number {
    return this.cap - this.used;
  }

  /** Engine seconds for one decision at this point of the run. */
  engine(assigned: number, requests: number): number {
    return Math.max(0.002, this.base + this.slope * assigned * (1 + requests));
  }

  /** Seconds one model reading takes: the configured mean spread over 0.8-1.3x, fixed per note. */
  readingSeconds(key: string): number {
    let hash = 2166136261;
    for (let i = 0; i < key.length; i++) hash = Math.imul(hash ^ key.charCodeAt(i), 16777619);
    return this.latency * (0.8 + 0.5 * ((hash >>> 0) % 1000) / 1000);
  }

  summary(): Record<string, unknown> {
    return {
      cap: this.cap, used: Number(this.used.toFixed(1)), expired_at: this.expiredAt, decisions: this.decisions,
      waited_for_model: Number(this.waited.toFixed(1)), engine_base_ms: this.base * 1000, engine_slope: this.slope * 1e9,
      own_scale: this.ownScale, reading_seconds: this.latency,
    };
  }
}
