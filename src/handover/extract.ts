/**
 * One handover note -> facts. The model reads the note (prompt.ts); this module supplies its context,
 * checks every field of the answer and does the arithmetic the model should not be trusted with:
 * calendar dates, local time to UTC, "that night" semantics, compass ranges.
 * Nothing here interprets a note without the model: there is no keyword reading.
 */
import { COMPASS } from "../core/conditions";
import { FactStore, FaultFact, FaultKind, NoteFacts, TerrainFact, WindowFact } from "../facts/store";
import { Json } from "../llm/client";
import { NoteContext } from "./prompt";
import { cleanText } from "./text";

export interface SiteCalendar {
  utcOffsetHours: number;
  /** [start, end] of every night in UTC seconds, ascending. */
  nights: [number, number][];
}

const WEEK = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
const DAY = 86400;
const pad = (n: number) => String(n).padStart(2, "0");

/** Local calendar fields of a UTC instant. */
function localParts(t: number, offsetHours: number): { y: number; m: number; d: number; hh: number; mm: number; dow: number } {
  const date = new Date((t + offsetHours * 3600) * 1000);
  return { y: date.getUTCFullYear(), m: date.getUTCMonth() + 1, d: date.getUTCDate(), hh: date.getUTCHours(), mm: date.getUTCMinutes(), dow: date.getUTCDay() };
}

/** UTC seconds of local midnight starting the given local calendar day. */
function localMidnight(y: number, m: number, d: number, offsetHours: number): number {
  return Date.UTC(y, m - 1, d) / 1000 - offsetHours * 3600;
}

/** The night a note belongs to: the one in progress, else the next one. Returns the local evening date as day number. */
export function nightOf(t: number, cal: SiteCalendar): { index: number; evening: number } {
  let index = cal.nights.findIndex(([, end]) => end > t);
  if (index < 0) index = cal.nights.length - 1;
  const start = cal.nights[index][0],
    p = localParts(start, cal.utcOffsetHours);
  // A night that starts after local midnight still belongs to the evening before.
  const evening = localMidnight(p.y, p.m, p.d, cal.utcOffsetHours) - (p.hh < 12 ? DAY : 0);
  return { index, evening };
}

function mmdd(midnight: number, offsetHours: number): string {
  const p = localParts(midnight + 12 * 3600, offsetHours);
  return `${pad(p.m)}/${pad(p.d)}`;
}

export function noteContext(issued: number, cal: SiteCalendar, terrainDirections: string[], codebook: string[]): NoteContext {
  const p = localParts(issued, cal.utcOffsetHours),
    tonight = nightOf(issued, cal).evening,
    today = localMidnight(p.y, p.m, p.d, cal.utcOffsetHours),
    days: string[] = [];
  for (let k = -3; k <= 12; k++) {
    const q = localParts(today + k * DAY + 12 * 3600, cal.utcOffsetHours);
    days.push(`${pad(q.m)}/${pad(q.d)} ${WEEK[q.dow]}`);
  }
  return {
    issuedLocal: `${p.y}-${pad(p.m)}-${pad(p.d)} ${WEEK[p.dow]} ${pad(p.hh)}:${pad(p.mm)}`,
    utcOffsetHours: cal.utcOffsetHours,
    tonight: mmdd(tonight, cal.utcOffsetHours),
    lastNight: mmdd(tonight - DAY, cal.utcOffsetHours),
    tomorrowNight: mmdd(tonight + DAY, cal.utcOffsetHours),
    calendar: days.join("，"),
    terrainDirections,
    codebook,
  };
}

/** "MM/DD" near the issue date -> UTC seconds of that local midnight (year chosen to be closest to the note). */
function resolveDate(text: unknown, issued: number, cal: SiteCalendar, utc: boolean): number | null {
  if (typeof text !== "string") return null;
  const match = /^\s*(\d{1,2})\s*[\/月.-]\s*(\d{1,2})/.exec(text);
  if (!match) return null;
  const month = Number(match[1]),
    day = Number(match[2]);
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;
  const offset = utc ? 0 : cal.utcOffsetHours,
    year = localParts(issued, offset).y;
  let best: number | null = null;
  for (const y of [year - 1, year, year + 1]) {
    const probe = new Date(Date.UTC(y, month - 1, day));
    if (probe.getUTCMonth() !== month - 1) continue;
    const midnight = localMidnight(y, month, day, offset);
    if (best === null || Math.abs(midnight - issued) < Math.abs(best - issued)) best = midnight;
  }
  return best;
}

type Clock = { seconds: number } | "dusk" | "dawn";
function parseClock(text: unknown): Clock | null {
  if (typeof text !== "string") return null;
  const lower = text.trim().toLowerCase();
  if (lower === "dusk" || lower === "dawn") return lower;
  const match = /^(\d{1,2})\s*[:：点时]\s*(\d{1,2})?/.exec(lower);
  if (!match) return null;
  const hh = Number(match[1]),
    mm = Number(match[2] ?? 0);
  if (hh > 24 || mm > 59) return null;
  return { seconds: hh * 3600 + mm * 60 };
}

/** The observing night whose evening is the given local midnight (the night may start after midnight). */
function nightBounds(evening: number, cal: SiteCalendar): [number, number] {
  for (const [start, end] of cal.nights) if (start >= evening + 10 * 3600 && start < evening + 34 * 3600) return [start, end];
  return [evening + 19 * 3600, evening + 29 * 3600];
}

/** One instant from a date, a clock token and the "that night" flag. */
function instant(date: number, clock: Clock, night: boolean, cal: SiteCalendar, utc: boolean): number {
  if (clock === "dusk") return nightBounds(date, cal)[0];
  if (clock === "dawn") return nightBounds(date, cal)[1];
  let t = date + clock.seconds;
  if (night && !utc && clock.seconds < 12 * 3600) t += DAY;
  return t;
}

/** "E-SE" -> clockwise arc; a single point gets half a compass step on either side. */
export function resolveArea(text: unknown): { all: boolean; azLo: number; azSpan: number; single?: boolean } | null {
  if (typeof text !== "string") return null;
  const upper = text.trim().toUpperCase();
  if (upper === "ALL" || upper === "") return upper === "ALL" ? { all: true, azLo: 0, azSpan: 360 } : null;
  const parts = upper.split(/\s*[-–~到至]\s*/).filter(Boolean);
  const first = COMPASS[parts[0]],
    last = COMPASS[parts[parts.length - 1]];
  if (first === undefined || last === undefined) return null;
  if (parts.length === 1 || first === last) return { all: false, azLo: (first - 22.5 + 360) % 360, azSpan: 45, single: true };
  const clockwise = (last - first + 360) % 360;
  // Sectors are at most half the horizon; a longer arc means the two ends were given the other way round.
  return clockwise <= 180 ? { all: false, azLo: first, azSpan: clockwise } : { all: false, azLo: last, azSpan: 360 - clockwise };
}

const KINDS = new Set(["thin_cloud", "thin_haze", "thick_cloud", "rain", "wind", "maintenance", "power", "other"]);
const FAULT_KINDS = new Set(["instrument", "scheduled", "unclear", "earthquake", "watch"]);
const list = (value: unknown): Json[] => (Array.isArray(value) ? value.filter((v): v is Json => v !== null && typeof v === "object" && !Array.isArray(v)) : []);

/** Validate a model answer and convert it; anything that does not check out is dropped, never guessed. */
export function factsFromAnswer(answer: Json, src: string, issued: number, cal: SiteCalendar): NoteFacts {
  const note: NoteFacts = { src, issued, windows: [], terrain: [], faults: [], nights: [], requests: [], codebook: [], };
  for (const item of list(answer.codebook)) {
    const phrase = String(item.phrase ?? "").trim(),
      means = String(item.means ?? "").trim();
    if (phrase && means && phrase.length <= 40 && means.length <= 80) note.codebook.push(`「${phrase}」= ${means}`);
  }
  for (const item of list(answer.terrain)) {
    const area = resolveArea(item.dir);
    if (area === null || area.all) continue;
    const ridge = Number(item.ridge_deg),
      ok = Number(item.ok_above_deg);
    const fact: TerrainFact = {
      azLo: area.azLo,
      azSpan: area.azSpan,
      single: area.single === true,
      ridge: item.ridge_deg != null && ridge >= 20 && ridge <= 70 ? ridge : null,
      okAbove: item.ok_above_deg != null && ok >= 20 && ok <= 75 ? ok : null,
      issued,
      src,
    };
    // The safe altitude lies above the ridge; the other way round one of the two numbers is a zenith distance.
    if (fact.ridge !== null && fact.okAbove !== null && fact.ridge > fact.okAbove) fact.ridge = null;
    if (fact.ridge !== null || fact.okAbove !== null) note.terrain.push(fact);
  }
  const seen = new Set<string>();
  for (const item of list(answer.windows)) {
    const conf = String(item.conf ?? "firm");
    if (conf === "rumor") continue;
    const utc = item.utc === true,
      night = item.night === true,
      area = resolveArea(item.area),
      kind = String(item.kind ?? "other"),
      effect = String(item.effect ?? "");
    const d1 = resolveDate(item.d1, issued, cal, utc),
      t1 = parseClock(item.t1),
      t2 = parseClock(item.t2);
    if (area === null || d1 === null || t1 === null || t2 === null) continue;
    if (effect !== "ok" && effect !== "degraded" && effect !== "closed") continue;
    const d2 = item.d2 == null ? null : resolveDate(item.d2, issued, cal, utc);
    const start = instant(d1, t1, night, cal, utc);
    let end: number;
    if (t2 === "dawn" && d2 !== null) {
      // "Dawn" on a calendar date ends the night before it; on an evening date it ends that night. Take the first after the start.
      const early = nightBounds(d2 - DAY, cal)[1];
      end = early > start ? early : nightBounds(d2, cal)[1];
    } else end = instant(d2 ?? d1, t2, night && d2 === null, cal, utc);
    if (d2 === null && end <= start) end += DAY;
    if (!(end > start) || end - start > 20 * DAY || start < issued - 4 * DAY || start > issued + 40 * DAY) continue;
    const key = `${start}|${end}|${area.all}|${area.azLo}|${area.azSpan}|${effect}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const fact: WindowFact = {
      t0: start, t1: end, all: area.all, azLo: area.azLo, azSpan: area.azSpan, kind: KINDS.has(kind) ? kind : "other", effect,
      conf: conf === "low" ? "low" : "firm", issued, src,
    };
    note.windows.push(fact);
  }
  for (const item of list(answer.nights)) {
    const date = resolveDate(item.d, issued, cal, false),
      quality = String(item.quality ?? "");
    if (date === null || (quality !== "good" && quality !== "poor")) continue;
    const [start] = nightBounds(date, cal),
      index = cal.nights.findIndex(([s]) => s === start);
    if (index >= 0) note.nights.push([index, quality]);
  }
  for (const item of list(answer.faults)) {
    const kind = String(item.kind ?? "");
    if (!FAULT_KINDS.has(kind)) continue;
    let action = String(item.action ?? "none");
    if (action !== "report" && action !== "verify") action = "none";
    const date = resolveDate(item.d, issued, cal, false),
      clock = parseClock(item.t);
    let onset: number | null = null;
    if (date !== null && clock !== null && clock !== "dusk" && clock !== "dawn") {
      // An announced job is dated by the calendar; "last night at 3" belongs to the night it names.
      // A time given on another clock ("UTC", "+9" for Tokyo) is converted here, not by the model.
      const zone = typeof item.tz === "string" ? item.tz.trim().toUpperCase() : "",
        shift = zone === "UTC" ? 0 : /^(UTC)?[+-]\d{1,2}(\.\d+)?$/.test(zone) ? Number(zone.replace("UTC", "")) : null;
      onset = shift === null ? instant(date, clock, item.night === true && kind !== "scheduled", cal, false) : (resolveDate(item.d, issued, cal, true) ?? date) + clock.seconds - shift * 3600;
      // A drop somebody already saw cannot lie in the future: a wrong "that night" flag is off by a day.
      if (kind !== "scheduled") while (onset > issued + 600) onset -= DAY;
      if (onset < issued - 6 * DAY || onset > issued + 12 * DAY) onset = null;
    }
    const level = Number(item.level);
    const fault: FaultFact = {
      onset, kind: kind as FaultKind, level: item.level != null && level > 0 && level < 1 ? level : null,
      action: action as FaultFact["action"], issued, src, state: "open",
    };
    // A report is only ever asked for an instrument fault; a quake heals by itself.
    if (fault.kind === "earthquake" || fault.kind === "watch") fault.action = "none";
    // An announced stop of the telescope is a closed window, not a loss of efficiency.
    if (fault.kind === "scheduled" && onset !== null && note.windows.some((w) => w.all && w.effect === "closed" && onset >= w.t0 - 1800 && onset <= w.t1)) continue;
    note.faults.push(fault);
  }
  for (const item of list(answer.requests)) {
    const d1 = resolveDate(item.d1, issued, cal, false),
      t1 = parseClock(item.t1),
      d2 = resolveDate(item.d2, issued, cal, false),
      t2 = parseClock(item.t2);
    if (d1 === null || d2 === null || t1 === null || t2 === null || t1 === "dusk" || t1 === "dawn" || t2 === "dusk" || t2 === "dawn") continue;
    const arrive = d1 + t1.seconds,
      deadline = d2 + t2.seconds;
    if (deadline > arrive && arrive > issued - DAY && arrive < issued + 14 * DAY) note.requests.push({ arrive, deadline });
  }
  return note;
}

/**
 * Code-word definitions written as 「phrase」＝meaning, copied verbatim so the model is shown them again with
 * every later note. This copies text; what a code word implies is still the model's reading.
 */
export function codebookLines(raw: string): string[] {
  const out: string[] = [];
  for (const match of cleanText(raw).matchAll(/「([^」]{2,30})」\s*=\s*([^;。「」]{2,60})/g)) out.push(`「${match[1]}」= ${match[2].trim()}`);
  return out;
}

export { FactStore };
