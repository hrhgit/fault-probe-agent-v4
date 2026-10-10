/** Frozen conditions-information-v3 adapter. Returns every candidate in input order. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import * as path from "node:path";
import { performance } from "node:perf_hooks";
import { Dense, GRU, relu, sigmoid, softmax, softplus, TensorStore, WeightManifest } from "../neural";
import { ModelAdapter } from "../registry";
import { FEATURE_VERSION, FeatureGroup, PreparedConditions, PredictionRow } from "./features";

export const OUTPUT_NAMES = ["mean_quality", "q05", "q50", "q95", "usable_fraction", "u05", "u50", "u95",
  "severe_block_probability", "p_dark", "p_bright", "p_backup"] as const;
interface Calibration {
  blocked_temperature: number; band_temperature: number; interval_correction: number;
  usable_interval_correction: number;
}
export interface ConditionsManifest extends WeightManifest {
  normalization: Record<string, number[] | number>;
  metadata: {
    version: string; architecture: "gru" | "mlp"; context_dim: number; history_dim: number; candidate_dim: number;
    history_length: number; feature_version: string; quality_scale: number;
    calibration: Calibration;
    region_calibration: { corrections: { quality: number; usable: number }; limitation: string; independent_scenes: number };
    source: string; quality_definition: string;
    reliability_evidence: Record<string, unknown> & { validation_reliable: boolean; final_test_passed: boolean | null };
  };
}
export interface ConditionsResponse {
  model_id: string; model_version: string; architecture: string; quality_definition: string;
  training_source: string; interval_coverage_target: number; validation_reliable: boolean;
  final_test_passed: boolean | null; reliability_evidence: Record<string, unknown>;
  role: string; automatic_actions: false; joint_success_probability: null;
  predictions: PredictionRow[]; timing: { preprocessing_seconds: number; prediction_seconds: number };
}
export class ConditionsModel implements ModelAdapter<PreparedConditions, ConditionsResponse> {
  readonly id: string;
  readonly manifest: ConditionsManifest;
  readonly version: string;
  private shared: Dense;
  private candidate: Dense;
  private fusion: Dense[];
  private head: Dense;
  private encoder?: GRU;
  constructor(directory: string) {
    const text = readFileSync(path.join(directory, "manifest.json"), "utf8");
    this.manifest = JSON.parse(text);
    const m = this.manifest, metadata = m.metadata;
    this.id = m.model_id;
    if (m.task !== "observing-conditions" || m.backend !== "information-net-v3" ||
      metadata.version !== "conditions-information-v3" || metadata.feature_version !== FEATURE_VERSION ||
      !["gru", "mlp"].includes(metadata.architecture) || m.model_id !== `conditions-${metadata.architecture}` ||
      metadata.context_dim !== 242 || metadata.history_dim !== 75 || metadata.candidate_dim !== 29 || metadata.history_length !== 32)
      throw new Error(`unsupported conditions model contract: ${this.id}`);
    for (const [key, size] of [["context", 242], ["history", 75], ["candidate", 29]] as const) {
      for (const part of ["mean", "std"]) {
        const values = m.normalization[key + "_" + part];
        if (!Array.isArray(values) || values.length !== size || values.some(v => !Number.isFinite(v) || (part === "std" && v <= 0)))
          throw new Error(`invalid normalization: ${key}/${part}`);
      }
    }
    if (!(metadata.quality_scale > 0) || !Number.isFinite(metadata.quality_scale)) throw new Error("invalid quality scale");
    const calibration = metadata.calibration;
    for (const [key, value] of Object.entries(calibration)) {
      if (["blocked_temperature", "band_temperature", "interval_correction", "usable_interval_correction"].includes(key) &&
        (typeof value !== "number" || !Number.isFinite(value) || value < 0 || (key.endsWith("temperature") && value === 0)))
        throw new Error(`invalid calibration: ${key}`);
    }
    for (const key of ["blocked_temperature", "band_temperature", "interval_correction", "usable_interval_correction"] as const)
      if (!Number.isFinite(calibration[key])) throw new Error(`missing calibration: ${key}`);
    this.version = metadata.version + "-" + createHash("sha256").update(text).digest("hex").slice(0, 16);
    const tensors = new TensorStore(directory, m), mlp = metadata.architecture === "mlp";
    if (!mlp) this.encoder = new GRU(tensors, "encoder", 75, 64, 2);
    this.shared = new Dense(tensors, "shared", 242 + (mlp ? 32 * 75 : 64), 128);
    this.candidate = new Dense(tensors, "candidate", 29, 128, false);
    this.fusion = mlp ? [new Dense(tensors, "fusion.2", 128, 128), new Dense(tensors, "fusion.5", 128, 64)] :
      [new Dense(tensors, "fusion.2", 128, 64)];
    this.head = new Dense(tensors, "head", 64, 13);
  }
  private normalize(key: string, input: Float32Array): Float32Array {
    const mean = this.manifest.normalization[key + "_mean"] as number[], std = this.manifest.normalization[key + "_std"] as number[];
    if (input.length !== mean.length) throw new Error(`input shape mismatch: ${key}`);
    const output = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) output[i] = Math.fround(input[i] - mean[i]) / std[i];
    if (key === "history") for (let i = 0; i < output.length; i++) output[i] *= input[input.length - 1];
    if (!output.every(Number.isFinite)) throw new Error(`nonfinite normalized input: ${key}`);
    return output;
  }
  /** Numeric boundary also supports independent Python/Node parity checks. */
  numeric(group: FeatureGroup): Float32Array[] {
    const context = this.normalize("context", group.context), history = group.history.map(h => this.normalize("history", h));
    if (history.length !== 32) throw new Error("history must have 32 rows");
    const encoded = this.encoder ? this.encoder.forward(history) : new Float32Array(history.flatMap(h => Array.from(h)));
    const input = new Float32Array(context.length + encoded.length); input.set(context); input.set(encoded, context.length);
    const shared = this.shared.forward(input), f = Math.fround, cache = new Map<Float32Array, Float32Array>();
    const candidateBuffer = new Float32Array(128), fusionBuffers = this.fusion.map(layer => new Float32Array(layer.outputSize));
    const headBuffer = new Float32Array(13);
    return group.candidates.map(candidate => {
      const cached = cache.get(candidate); if (cached) return cached;
      let hidden = this.candidate.forward(this.normalize("candidate", candidate), candidateBuffer);
      for (let i = 0; i < hidden.length; i++) hidden[i] += shared[i];
      hidden = relu(hidden);
      for (let i = 0; i < this.fusion.length; i++) hidden = relu(this.fusion[i].forward(hidden, fusionBuffers[i]));
      const raw = this.head.forward(hidden, headBuffer), p = new Float32Array(12), scale = this.manifest.metadata.quality_scale;
      p[0] = f(softplus(raw[0])) * scale;
      let quality = 0;
      for (let i = 1; i <= 3; i++) { quality = f(quality + f(softplus(raw[i]))); p[i] = quality * scale; }
      p[4] = sigmoid(raw[4]);
      const increments = [5, 6, 7, 8].map(i => f(f(softplus(raw[i])) + 1e-6));
      const total = f(increments.reduce((a, b) => a + b, 0));
      let usable = 0;
      for (let i = 0; i < 3; i++) { usable = f(usable + increments[i]); p[5 + i] = usable / total; }
      p[8] = sigmoid(raw[9]); p.set(softmax(raw.subarray(10, 13)), 9);
      const c = this.manifest.metadata.calibration;
      const probability = Math.max(1e-7, Math.min(1 - 1e-7, p[8]));
      p[8] = sigmoid(Math.log(probability / (1 - probability)) / c.blocked_temperature);
      p.set(softmax(Array.from(p.subarray(9, 12), v => Math.log(Math.max(1e-7, Math.min(1, v))) / c.band_temperature)), 9);
      p[1] = Math.max(0, p[1] - c.interval_correction); p[3] += c.interval_correction;
      p[5] = Math.max(0, p[5] - c.usable_interval_correction); p[7] = Math.min(1, p[7] + c.usable_interval_correction);
      if (!p.every(Number.isFinite)) throw new Error(`nonfinite prediction: ${this.id}`);
      cache.set(candidate, p); return p;
    });
  }
  predict(prepared: PreparedConditions): ConditionsResponse {
    const begin = performance.now(), predictions = prepared.rows.map(row => ({ ...row }));
    for (const group of prepared.groups) {
      if (!group.indices.length) continue;
      const outputs = this.numeric(group);
      group.indices.forEach((index, row) => OUTPUT_NAMES.forEach((name, col) => { predictions[index][name] = outputs[row][col]; }));
    }
    const metadata = this.manifest.metadata, evidence = metadata.reliability_evidence;
    return { model_id: this.id, model_version: this.version, architecture: metadata.architecture,
      quality_definition: metadata.quality_definition, training_source: metadata.source,
      interval_coverage_target: .9, validation_reliable: evidence.validation_reliable,
      final_test_passed: evidence.final_test_passed, reliability_evidence: evidence,
      role: "objective_information_for_agent_reasoning", automatic_actions: false, joint_success_probability: null,
      predictions, timing: { preprocessing_seconds: prepared.preprocessing_seconds, prediction_seconds: (performance.now() - begin) / 1000 } };
  }
}
