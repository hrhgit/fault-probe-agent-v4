import { ModelAdapter } from "./registry";
import { PythonTransport } from "./python";
import { Candidate } from "./conditions/features";
export const SELECTED_MODEL_IDS = ["conditions-reliability", "instrument-fault-report"] as const;
export interface ReliabilityRequest {
  initialize: Record<string, any>;
  now_utc: string;
  baseline_state: Record<string, unknown>;
  feedback_history?: Record<string, unknown>[];
  bulletins?: Record<string, unknown>[];
  forecasts?: Record<string, unknown>[];
  candidates: Candidate[];
}
export interface SessionReliabilityRequest { session: Omit<ReliabilityRequest, "feedback_history" | "bulletins" | "forecasts"> }
export type ReliabilityInput = ReliabilityRequest | SessionReliabilityRequest;
export interface ReliabilityResponse {
  model_id: "conditions-reliability";
  model_version: string;
  method: string;
  final_test_passed: boolean;
  automatic_actions: false;
  predictions: Array<Record<string, any>>;
  [key: string]: unknown;
}
export interface FaultRequest { now_utc: string; guard_satisfied: boolean }
export interface FaultResponse {
  model_id: "instrument-fault-report";
  model_version: string;
  report_score: number | null;
  score_is_calibrated_fault_probability: false;
  recommend_report: boolean;
  decision_due: boolean;
  status: string;
  [key: string]: unknown;
}
export class ReliabilityAdapter implements ModelAdapter<ReliabilityInput, Promise<ReliabilityResponse>> {
  readonly id = "conditions-reliability";
  constructor(readonly worker: PythonTransport) {}
  predict(request: ReliabilityInput): Promise<ReliabilityResponse> { return "session" in request ? this.worker.request("session_conditions", request.session) : this.worker.request("conditions", request); }
}
export class FaultAdapter implements ModelAdapter<FaultRequest, Promise<FaultResponse>> {
  readonly id = "instrument-fault-report";
  constructor(readonly worker: PythonTransport) {}
  predict(request: FaultRequest): Promise<FaultResponse> { return this.worker.request("fault", request); }
}
export interface SessionUpdate {
  now_utc: string;
  messages: Record<string, any>[];
  completed_exposure?: Record<string, any>;
  baseline_state: Record<string, any>;
  sky: Record<string, any>;
  remaining_gain: number;
}
export class SelectedModelsSession {
  latestConditions: ReliabilityResponse | null = null;
  latestFault: FaultResponse | null = null;
  private initialized = false;
  private state?: SessionUpdate;
  private readonly context: Record<string, any>;
  constructor(readonly initialize: Record<string, any>, readonly models: {
    worker: PythonTransport;
    predictReliability: (input: ReliabilityInput) => Promise<ReliabilityResponse>;
    predictFault: (input: FaultRequest) => Promise<FaultResponse>;
  }, readonly enabled: readonly string[] = SELECTED_MODEL_IDS) {
    for (const id of enabled) if (!(SELECTED_MODEL_IDS as readonly string[]).includes(id)) throw new Error(`Unknown selected model ${id}`);
    const { site, scoring, survey, instrument } = initialize;
    this.context = JSON.parse(JSON.stringify({ site, scoring, survey, instrument }));
  }
  async update(input: SessionUpdate): Promise<void> {
    this.state = input;
    if (!this.enabled.length) return;
    if (!this.initialized) {
      await this.models.worker.request("initialize", { initialize: this.initialize, enabled: this.enabled });
      this.initialized = true;
    }
    await this.models.worker.request("sync", input);
  }
  async conditions(now_utc: string, candidates: Candidate[]): Promise<ReliabilityResponse> {
    if (!this.state) throw new Error("Update model session first");
    // Let the worker add its completed feedback and published-message history.
    const result = await this.models.predictReliability({ session: {
      initialize: this.context, now_utc, candidates, baseline_state: this.state.baseline_state,
    } });
    this.latestConditions = result;
    return result;
  }
  async faults(now_utc: string, guard_satisfied: boolean): Promise<FaultResponse> {
    this.latestFault = await this.models.predictFault({ now_utc, guard_satisfied });
    return this.latestFault;
  }
  async close(): Promise<void> { await this.models.worker.close(); }
}
