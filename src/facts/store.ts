/**
 * Facts taken from handover notes, in the units the planner works in (UTC seconds, azimuth degrees).
 * Each fact keeps where it came from. Bulletins and exposure feedback stay the reference for what is
 * happening now; these facts say what a public notice means (thin cloud or a closed sector), what is
 * not announced at all (maintenance stops, ridge heights) and what is coming.
 */

export type Effect = "ok" | "degraded" | "closed";
export type Confidence = "firm" | "low";

export interface WindowFact {
  t0: number;
  t1: number;
  /** Whole sky; otherwise the arc [azLo, azLo + azSpan] clockwise. */
  all: boolean;
  azLo: number;
  azSpan: number;
  kind: string;
  effect: Effect;
  conf: Confidence;
  issued: number;
  src: string;
}

export interface TerrainFact {
  /** The stretch of horizon the statement is about, clockwise from azLo. */
  azLo: number;
  azSpan: number;
  /** True when the note named one compass direction rather than a stretch between two. */
  single: boolean;
  /** Height the ridge blocks up to, and the altitude stated as safe; either may be missing. */
  ridge: number | null;
  okAbove: number | null;
  issued: number;
  src: string;
}

export type FaultKind = "instrument" | "scheduled" | "unclear" | "earthquake" | "watch";
export interface FaultFact {
  onset: number | null;
  kind: FaultKind;
  level: number | null;
  action: "report" | "verify" | "none";
  issued: number;
  src: string;
  state: "open" | "reported" | "rejected" | "done";
}

export interface NoteFacts {
  src: string;
  issued: number;
  windows: WindowFact[];
  terrain: TerrainFact[];
  faults: FaultFact[];
  /** Night index -> outlook. */
  nights: [number, "good" | "poor"][];
  requests: { arrive: number; deadline: number }[];
  codebook: string[];
}

/** Angular distance from an azimuth to an arc; 0 inside the arc. */
export function arcDistance(az: number, lo: number, span: number): number {
  const rel = (((az - lo) % 360) + 360) % 360;
  if (rel <= span) return 0;
  return Math.min(rel - span, 360 - rel);
}

export class FactStore {
  windows: WindowFact[] = [];
  terrain: TerrainFact[] = [];
  faults: FaultFact[] = [];
  nights = new Map<number, "good" | "poor">();
  requests: { arrive: number; deadline: number }[] = [];
  codebook: string[] = [];
  notes = new Map<string, NoteFacts>();
  revision = 0;

  /** Add what the model read in one note. */
  apply(note: NoteFacts): void {
    this.notes.set(note.src, note);
    this.windows.push(...note.windows);
    this.windows.sort((a, b) => a.t0 - b.t0);
    this.terrain.push(...note.terrain);
    this.faults.push(...note.faults);
    for (const [night, quality] of note.nights) this.nights.set(night, quality);
    this.requests.push(...note.requests);
    // Code words are kept as the notes wrote them (learnCodebook); the model's paraphrase is used only when
    // a note defined code words in some form that was not copied.
    if (this.codebook.length === 0) this.learnCodebook(note.codebook);
    this.revision++;
  }

  /** Remember code-word definitions so that later notes are read with them. */
  learnCodebook(lines: string[]): void {
    for (const line of lines) if (!this.codebook.includes(line)) this.codebook.push(line);
  }

  /**
   * What the notes say about the mountain announced in compass direction `centre`: the altitude above
   * which it does not block. A height given for that very direction wins over a statement about a wider
   * stretch; the newest note wins among equals. Null when no note covers the direction.
   */
  ridgeFor(centre: number, margin: number): number | null {
    let ridge: TerrainFact | null = null,
      safe: number | null = null;
    for (const item of this.terrain) {
      if (arcDistance(centre, item.azLo, item.azSpan) > 12) continue;
      if (item.ridge !== null && item.single && arcDistance(centre, item.azLo + item.azSpan / 2, 0) < 1 && (ridge === null || item.issued >= ridge.issued)) ridge = item;
      if (item.okAbove !== null && (safe === null || item.okAbove < safe)) safe = item.okAbove;
    }
    if (ridge !== null) return ridge.ridge! + margin;
    if (safe !== null) return safe;
    let wide: TerrainFact | null = null;
    for (const item of this.terrain)
      if (item.ridge !== null && arcDistance(centre, item.azLo, item.azSpan) <= 12 && (wide === null || item.issued >= wide.issued)) wide = item;
    return wide === null ? null : wide.ridge! + margin;
  }

  /** Windows in force at t. */
  active(t: number): WindowFact[] {
    const out: WindowFact[] = [];
    for (const w of this.windows) {
      if (w.t0 > t) break;
      if (t < w.t1) out.push(w);
    }
    return out;
  }

  /**
   * Whole-sky stops that no bulletin shows: work on the telescope, a power cut, or a bare "we are stopped
   * tonight" (kind "other"). Checked against the exposure results of the formal runs, 95-100 % of the fibres
   * inside such windows scored zero, whatever confidence the reader gave them; the few "other" windows that
   * were misreadings (a neighbour's dome, a forecast) are caught by the one short exposure the agent takes
   * before it believes a stop. A note's "rain, dome closed" or "cloud too thick" is not a stop: there the
   * bulletin decides (see Sky.set_notes).
   */
  static isStop(w: WindowFact): boolean {
    return w.all && w.effect === "closed" && (w.kind === "maintenance" || w.kind === "power" || w.kind === "other");
  }

  /** A stop sure enough to plan the season around: firm, and with a stated cause. */
  static isPlanned(w: WindowFact): boolean {
    return w.all && w.effect === "closed" && w.conf === "firm" && (w.kind === "maintenance" || w.kind === "power");
  }

  /** A whole-sky stop covering t: when it ends and what it is. */
  siteClosure(t: number): { until: number; kind: string } | null {
    let best: { until: number; kind: string } | null = null;
    for (const w of this.active(t))
      if (FactStore.isStop(w) && (best === null || w.t1 > best.until)) best = { until: w.t1, kind: w.kind };
    return best;
  }

  /** Planned whole-sky stops from t on, merged into disjoint intervals. */
  closedIntervals(from: number): [number, number][] {
    const out: [number, number][] = [];
    for (const w of this.windows) {
      if (!FactStore.isPlanned(w) || w.t1 <= from) continue;
      const last = out[out.length - 1];
      if (last !== undefined && w.t0 <= last[1]) last[1] = Math.max(last[1], w.t1);
      else out.push([w.t0, w.t1]);
    }
    return out;
  }

  /** Start of the next whole-sky stop in (t, t + horizon], or null. */
  nextSiteClosure(t: number, horizon: number): { start: number; until: number; kind: string } | null {
    for (const w of this.windows) {
      if (w.t0 <= t) continue;
      if (w.t0 > t + horizon) break;
      if (FactStore.isStop(w)) return { start: w.t0, until: w.t1, kind: w.kind };
    }
    return null;
  }

  lastCorrectReport = -1e18;

  /** When the fault a note describes began; a note without a time is taken to describe the last day. */
  static faultOnset(fault: FaultFact): number {
    return fault.onset ?? fault.issued - 24 * 3600;
  }

  /**
   * Fault notes that still ask for an action. Faults follow one another without overlap and a correct
   * report repairs the one active at that moment, so a note about an onset before the last correct
   * report is about a fault that is already repaired (or can no longer be).
   */
  openFaults(): FaultFact[] {
    return this.faults.filter(
      (f) => f.state === "open" && f.action !== "none" && FactStore.faultOnset(f) > this.lastCorrectReport,
    );
  }

  reportResult(t: number, correct: boolean, reported: FaultFact | null): void {
    if (reported !== null) reported.state = correct ? "done" : "rejected";
    // A report that comes back wrong says nothing was broken at that moment, whoever sent it. Every note
    // claiming "a ticket is due" for a fault that had begun by then is contradicted and not tried again.
    if (!correct)
      for (const other of this.faults)
        if (other.state === "open" && other.action === "report" && FactStore.faultOnset(other) <= t) other.state = "rejected";
    if (correct) this.lastCorrectReport = t;
  }
}
