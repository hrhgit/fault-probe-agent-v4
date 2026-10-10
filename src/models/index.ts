import { PythonTransport } from "./python";
import { ReliabilityAdapter, ReliabilityInput, ReliabilityResponse, FaultAdapter, FaultRequest, FaultResponse } from "./selected";
import { existsSync } from "node:fs";
import * as path from "node:path";
import { ModelRegistry } from "./registry";
import { ConditionsModel, ConditionsResponse } from "./conditions/model";
import { Bulletin, Candidate, ConditionsRequest, Forecast, PreparedConditions, prepareConditions, PublicInitialize, PublicTimeline } from "./conditions/features";

/** Extend this task map (or use a separate ModelRegistry<MyTasks>) for new model tasks. */
export interface BaselineModelTasks {
  reliability: { input: ReliabilityInput; output: Promise<ReliabilityResponse> };
  instrumentFault: { input: FaultRequest; output: Promise<FaultResponse> };
  conditions: { input: PreparedConditions; output: ConditionsResponse };
}
export const CONDITION_MODEL_IDS = ["conditions-gru", "conditions-mlp"] as const;
export class BaselineModels {
  readonly worker = new PythonTransport();
  readonly registry = new ModelRegistry<BaselineModelTasks>();
  constructor(directory = path.resolve(__dirname, "../../models")) {
    this.registry.register("reliability", "conditions-reliability", () => new ReliabilityAdapter(this.worker));
    this.registry.register("instrumentFault", "instrument-fault-report", () => new FaultAdapter(this.worker));
    for (const id of CONDITION_MODEL_IDS) if (existsSync(path.join(directory, id, "manifest.json"))) this.registry.register("conditions", id, () => new ConditionsModel(path.join(directory, id)));
  }
  predictReliability(request: ReliabilityInput): Promise<ReliabilityResponse> { return this.registry.predict("reliability", "conditions-reliability", request); }
  predictFault(request: FaultRequest): Promise<FaultResponse> { return this.registry.predict("instrumentFault", "instrument-fault-report", request); }
  async close(): Promise<void> { await this.worker.close(); }
  predictConditions(ids: readonly string[], request: ConditionsRequest, timeline?: PublicTimeline): ConditionsResponse[] {
    const prepared = prepareConditions(request, timeline);
    return ids.map(id => this.registry.predict("conditions", id, prepared));
  }
}
/** One session per initialized scenario; no feedback scores or hidden data enter weather features. */
export class ConditionsSession {
  readonly timeline = new PublicTimeline();
  readonly initialize: PublicInitialize;
  latest: ConditionsResponse[] = [];
  constructor(initialize: PublicInitialize, readonly models: BaselineModels, readonly modelIds: readonly string[]) {
    const { site, scoring, survey, instrument } = initialize;
    // Clone the public context so later caller changes cannot alter this session.
    this.initialize = JSON.parse(JSON.stringify({ site, scoring, survey, instrument }));
    for (const id of modelIds) models.registry.resolve("conditions", id);
  }
  ingest(messages: unknown[]): void {
    const bulletins: Bulletin[] = [], forecasts: Forecast[] = [];
    for (const value of messages) {
      if (!value || typeof value !== "object" || Array.isArray(value)) continue;
      const message = value as Bulletin & Forecast & { record_type: string };
      if (message.record_type === "bulletin") bulletins.push(message);
      if (message.record_type === "forecast") forecasts.push(message);
    }
    this.timeline.add(bulletins, forecasts);
  }
  predict(now_utc: string, candidates: Candidate[], ids = this.modelIds): ConditionsResponse[] {
    this.latest = this.models.predictConditions(ids, { initialize: this.initialize, now_utc, candidates }, this.timeline);
    return this.latest;
  }
}
export * from "./registry";
export * from "./conditions/features";
export * from "./conditions/model";

export * from "./selected";
export * from "./python";
