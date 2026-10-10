/**
 * OpenAI-compatible chat client with a provider chain (Node built-ins only: fetch, no SDK).
 *
 * Providers are tried in this order; one whose key is missing is skipped:
 *   1. OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL          the team's model service on the platform
 *   2. KIMI_RELAY_API_KEY (+ KIMI_RELAY_BASE_URL / _MODEL)     organizer quota
 *   3. KIMI_API_KEY       (+ KIMI_BASE_URL / KIMI_MODEL)       Kimi Code subscription
 *   4. DEEPSEEK_API_KEY   (+ DEEPSEEK_BASE_URL)               DeepSeek Flash
 * A provider that answers "no credit / not authorised" is dropped for the rest of the run and the next one
 * takes over; rate limits, 5xx and timeouts are retried once and then also fall through for that call.
 *
 * LLM_THINKING = off | low | default selects the reasoning mode (off: `thinking: {type: "disabled"}`,
 * low: `reasoning_effort: "low"`). LLM_CACHE_DIR (local development only) stores every answer on disk keyed
 * by model and prompt, so a repeated local run spends no quota and repeats the same answers;
 * LLM_CACHE_LATENCY=<seconds> then delays each cached answer by that long.
 * Key values are never logged.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import * as path from "node:path";

export type Json = Record<string, unknown>;
export type Thinking = "off" | "low" | "default" | "high" | "max";

export interface Provider {
  name: string;
  baseUrl: string;
  model: string;
  key: string;
  downUntil: number;
  failures: number;
  /** Requests allowed at once (the organizer relay allows two per team). */
  limit: number;
  inFlight: number;
  calls: number;
  ok: number;
  seconds: number;
  tokens: number;
}

export interface ChatResult {
  json: Json | null;
  error: string | null;
  provider: string;
  seconds: number;
  cached: boolean;
}

const KEY_NAMES = [
  "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
  "KIMI_RELAY_API_KEY", "KIMI_RELAY_BASE_URL", "KIMI_RELAY_MODEL",
  "KIMI_API_KEY", "KIMI_BASE_URL", "KIMI_MODEL",
  "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
];
const DEFAULT_MODEL = "kimi-for-coding";
/** A 429 that means "out of credit" rather than "slow down"; concurrency and rate limits must not match. */
const QUOTA_WORDS = /quota|balance|insufficient|credit|billing|额度|余额|欠费|用完/i;
const BUSY_WORDS = /concurrent|rate.?limit|too many|同时|频率|并发/i;

/** Local development: fill missing model variables from another .env (LLM_ENV_FILE). Only the names above are read. */
export function loadEnvFile(file: string | undefined): string[] {
  if (!file) return [];
  let text: string;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    return [];
  }
  const loaded: string[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const at = line.indexOf("="),
      name = line.slice(0, at).trim(),
      value = line.slice(at + 1).trim().replace(/^["']+|["']+$/g, "");
    if (KEY_NAMES.includes(name) && value && !process.env[name]) {
      process.env[name] = value;
      loaded.push(name);
    }
  }
  return loaded;
}

const env = (name: string): string => (process.env[name] ?? "").trim();
const sleep = (seconds: number) => new Promise<void>((resolve) => setTimeout(resolve, Math.max(0, seconds) * 1000));

export function providersFromEnv(): Provider[] {
  const make = (name: string, key: string, baseUrl: string, model: string): Provider | null =>
    key && baseUrl
      ? {
          name, key, baseUrl: baseUrl.replace(/\/+$/, ""), model, downUntil: 0, failures: 0, calls: 0, ok: 0, seconds: 0, tokens: 0,
          limit: Math.max(1, Number(env("LLM_MAX_CONCURRENCY")) || (name === "kimi_relay" ? 2 : 4)), inFlight: 0,
        }
      : null;
  const candidates: Record<string, Provider | null> = {
    openai: make("openai", env("OPENAI_API_KEY"), env("OPENAI_BASE_URL"), env("OPENAI_MODEL") || DEFAULT_MODEL),
    kimi_relay: make(
      "kimi_relay",
      env("KIMI_RELAY_API_KEY"),
      env("KIMI_RELAY_BASE_URL") || "https://vdiemcofukuxglqsmlyz.supabase.co/functions/v1/kimi-relay/v1",
      env("KIMI_RELAY_MODEL") || DEFAULT_MODEL,
    ),
    kimi: make("kimi", env("KIMI_API_KEY"), env("KIMI_BASE_URL") || "https://api.kimi.com/coding/v1", env("KIMI_MODEL") || DEFAULT_MODEL),
    deepseek: make("deepseek", env("DEEPSEEK_API_KEY"), env("DEEPSEEK_BASE_URL") || "https://api.deepseek.com", "deepseek-flash"),
  };
  const order = (env("LLM_PROVIDER_ORDER") || "openai,kimi_relay,kimi,deepseek").split(",").map((s) => s.trim());
  return order.map((name) => candidates[name]).filter((p): p is Provider => p != null);
}

class Retryable extends Error {
  constructor(message: string, readonly retryAfter: number | null = null) {
    super(message);
  }
}
/** The provider cannot serve this run any more (no credit, bad key): move on and stop asking it. */
class Exhausted extends Error {}

export function extractJson(text: string): Json | null {
  const start = text.indexOf("{"),
    end = text.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    const parsed = JSON.parse(text.slice(start, end + 1));
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Json) : null;
  } catch {
    return null;
  }
}

export class LLM {
  readonly providers: Provider[];
  readonly thinking: Thinking;
  readonly cacheDir: string;
  cacheHits = 0;
  /** Failed attempts by provider and kind (HTTP status, timeout, network, bad reply), for the end-of-run line. */
  readonly errors = new Map<string, number>();
  slowest = 0;
  private dropThinking = new Set<string>();

  constructor(private readonly log: (text: string) => void = () => {}) {
    const loaded = loadEnvFile(env("LLM_ENV_FILE"));
    if (loaded.length) this.log(`llm: read ${loaded.length} model variables from LLM_ENV_FILE`);
    this.providers = providersFromEnv();
    const mode = env("LLM_THINKING").toLowerCase();
    this.thinking = mode === "low" || mode === "default" || mode === "high" || mode === "max" ? mode : "off";
    this.cacheDir = env("LLM_CACHE_DIR");
    if (this.cacheDir) mkdirSync(this.cacheDir, { recursive: true });
  }

  get available(): boolean {
    return this.providers.length > 0;
  }

  private thinkingBody(provider: Provider): Json {
    if (this.dropThinking.has(provider.name) || this.thinking === "default") return {};
    // high / max (card-plan experiment): thinking switched on explicitly with that effort, for providers whose default is not to think.
    if (this.thinking === "high" || this.thinking === "max") return { thinking: { type: "enabled" }, reasoning_effort: this.thinking };
    return this.thinking === "off" ? { thinking: { type: "disabled" } } : { reasoning_effort: "low" };
  }

  private cachePath(provider: Provider, system: string, user: string, maxTokens: number): string {
    const digest = createHash("sha256")
      .update(JSON.stringify([provider.model, this.thinking, maxTokens, system, user]))
      .digest("hex");
    return path.join(this.cacheDir, `${digest}.json`);
  }

  /** One HTTP attempt against one provider. */
  private async request(provider: Provider, system: string, user: string, maxTokens: number, timeout: number): Promise<{ json: Json; tokens: number }> {
    if (timeout <= 0.5) throw new Error("deadline");
    let response: Response;
    try {
      response = await fetch(provider.baseUrl + "/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer " + provider.key },
        body: JSON.stringify({
          model: provider.model,
          messages: [
            { role: "system", content: system },
            { role: "user", content: user },
          ],
          max_tokens: maxTokens,
          ...this.thinkingBody(provider),
        }),
        signal: AbortSignal.timeout(Math.max(1, Math.round(timeout * 1000))),
      });
    } catch (error) {
      // Keep what the network layer said: a timeout, or "fetch failed" with its code (ECONNRESET, ENOTFOUND, ...).
      const cause = error instanceof Error ? (error as Error & { cause?: { code?: string; message?: string } }).cause : undefined,
        timeout_hit = error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError");
      throw new Retryable(timeout_hit ? "timeout" : `network: ${cause?.code ?? cause?.message ?? (error instanceof Error ? error.message : "error")}`);
    }
    if (!response.ok) {
      let body = "";
      try {
        body = (await response.text()).slice(0, 2000);
      } catch {
        /* the status alone decides */
      }
      const status = response.status;
      if (status === 400 && /thinking|reasoning_effort/i.test(body) && !this.dropThinking.has(provider.name)) {
        this.dropThinking.add(provider.name);
        throw new Retryable("thinking parameter refused; retrying without it", 0);
      }
      if (status === 401 || status === 402 || status === 403 || (status === 429 && QUOTA_WORDS.test(body) && !BUSY_WORDS.test(body)))
        throw new Exhausted(`HTTP ${status}`);
      if (status === 429 || status >= 500) {
        const header = Number(response.headers.get("retry-after") ?? NaN);
        throw new Retryable(`HTTP ${status}`, Number.isFinite(header) && header >= 0 ? header : null);
      }
      throw new Error(`HTTP ${status}`);
    }
    let data: { choices?: { message?: { content?: string | null }; finish_reason?: string }[]; usage?: { total_tokens?: number } };
    try {
      data = (await response.json()) as typeof data;
    } catch {
      throw new Retryable("reply is not JSON");
    }
    const json = extractJson(data.choices?.[0]?.message?.content ?? "");
    if (json === null) throw new Retryable(`no JSON object in the reply (finish ${data.choices?.[0]?.finish_reason ?? "?"})`);
    return { json, tokens: Number(data.usage?.total_tokens ?? 0) };
  }

  /** Ask for one JSON object; walks the provider chain until one answers or the deadline passes. */
  async chat(tag: string, system: string, user: string, timeoutSeconds: number, maxTokens = 1500): Promise<ChatResult> {
    const started = performance.now(),
      deadline = started + timeoutSeconds * 1000;
    let error = "no model provider is configured",
      used = "";
    for (const provider of this.providers) {
      if (provider.downUntil > Date.now()) continue;
      used = provider.name;
      if (this.cacheDir) {
        const file = this.cachePath(provider, system, user, maxTokens);
        if (existsSync(file)) {
          try {
            const json = JSON.parse(readFileSync(file, "utf8")) as Json;
            this.cacheHits++;
            // Local evaluation: make a cached answer take as long as a real call, so wall-time behaviour is realistic.
            const replay = Number(env("LLM_CACHE_LATENCY"));
            if (replay > 0) await sleep(Math.min(replay, Math.max(0, (deadline - performance.now()) / 1000)));
            return { json, error: null, provider: provider.name, seconds: (performance.now() - started) / 1000, cached: true };
          } catch {
            /* unreadable cache entry: ask again */
          }
        }
      }
      for (let attempt = 0; attempt < 4; attempt++) {
        while (provider.inFlight >= provider.limit && performance.now() < deadline - 2000) await sleep(0.1);
        const began = performance.now();
        provider.calls++;
        provider.inFlight++;
        try {
          const { json, tokens } = await this.request(provider, system, user, maxTokens, (deadline - began) / 1000).finally(() => {
            provider.inFlight--;
          });
          provider.ok++;
          provider.failures = 0;
          provider.tokens += tokens;
          provider.seconds += (performance.now() - began) / 1000;
          this.slowest = Math.max(this.slowest, (performance.now() - began) / 1000);
          if (attempt > 0) this.log(`llm: ${tag} answered by ${provider.name} on attempt ${attempt + 1}`);
          if (this.cacheDir) writeFileSync(this.cachePath(provider, system, user, maxTokens), JSON.stringify(json));
          return { json, error: null, provider: provider.name, seconds: (performance.now() - started) / 1000, cached: false };
        } catch (exc) {
          const took = (performance.now() - began) / 1000;
          provider.seconds += took;
          const what = exc instanceof Error ? exc.message || exc.name : String(exc),
            kind = /^HTTP \d+/.test(what) ? what.slice(0, 8) : what === "timeout" ? "timeout" : what.startsWith("network") ? "network" : "bad reply";
          this.errors.set(`${provider.name} ${kind}`, (this.errors.get(`${provider.name} ${kind}`) ?? 0) + 1);
          error = `${provider.name}: ${what}`;
          this.log(`llm: ${tag} attempt ${attempt + 1} on ${provider.name} failed after ${took.toFixed(1)}s: ${what}`);
          if (exc instanceof Exhausted) {
            provider.downUntil = Number.POSITIVE_INFINITY;
            this.log(`llm: ${provider.name} refused (${exc.message}); switching to the next provider for the rest of the run`);
            break;
          }
          if (!(exc instanceof Retryable) || (attempt >= 1 && !/HTTP 429/.test(exc.message))) break;
          const pause = exc.retryAfter ?? 0.8 + attempt;
          if (performance.now() + pause * 1000 > deadline - 1500) break;
          await sleep(pause);
        }
      }
      if (provider.downUntil === 0 && ++provider.failures >= 3) {
        provider.downUntil = Date.now() + 120_000;
        provider.failures = 0;
        this.log(`llm: ${provider.name} failed three calls in a row; pausing it for two minutes`);
      }
      if (performance.now() > deadline - 1500) break;
    }
    this.log(`llm: ${tag} failed (${error})`);
    return { json: null, error, provider: used, seconds: (performance.now() - started) / 1000, cached: false };
  }

  summary(): Json {
    return {
      thinking: this.thinking,
      cache_hits: this.cacheHits,
      errors: Object.fromEntries(this.errors),
      slowest_call_seconds: Number(this.slowest.toFixed(1)),
      providers: this.providers.map((p) => ({
        name: p.name, model: p.model, calls: p.calls, ok: p.ok, seconds: Number(p.seconds.toFixed(1)), tokens: p.tokens,
        dropped: p.downUntil === Number.POSITIVE_INFINITY,
      })),
    };
  }
}
