/** Small CPU primitives. PyTorch GRU uses reset/update/new gates in that order. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import * as path from "node:path";

export interface TensorSpec { shape: number[]; byte_offset: number; elements: number }
export interface WeightManifest {
  schema: string;
  model_id: string;
  task: string;
  backend: string;
  weights_file: string;
  weights_sha256: string;
  tensors: Record<string, TensorSpec>;
}
export class TensorStore {
  private tensors = new Map<string, { shape: number[]; values: Float32Array }>();
  constructor(directory: string, manifest: WeightManifest) {
    if (manifest.schema !== "observer-neural-weights-v1" || manifest.weights_file !== "weights.f32")
      throw new Error("unsupported neural weight schema/file");
    const buffer = readFileSync(path.join(directory, manifest.weights_file));
    if (createHash("sha256").update(buffer).digest("hex") !== manifest.weights_sha256)
      throw new Error(`weight checksum mismatch: ${manifest.model_id}`);
    let expectedOffset = 0;
    for (const [name, spec] of Object.entries(manifest.tensors)) {
      if (!spec.shape.length || spec.shape.some(n => !Number.isInteger(n) || n < 1) ||
          spec.shape.reduce((a, b) => a * b, 1) !== spec.elements ||
          spec.byte_offset !== expectedOffset || spec.byte_offset + spec.elements * 4 > buffer.length)
        throw new Error(`invalid tensor layout: ${name}`);
      const values = new Float32Array(spec.elements);
      for (let i = 0; i < values.length; i++) {
        values[i] = buffer.readFloatLE(spec.byte_offset + i * 4);
        if (!Number.isFinite(values[i])) throw new Error(`nonfinite tensor: ${name}`);
      }
      this.tensors.set(name, { shape: spec.shape, values });
      expectedOffset += spec.elements * 4;
    }
    if (expectedOffset !== buffer.length) throw new Error("unclaimed weight bytes");
  }
  get(name: string, shape: number[]): Float32Array {
    const value = this.tensors.get(name);
    if (!value || value.shape.join(",") !== shape.join(",")) throw new Error(`tensor shape mismatch: ${name}`);
    return value.values;
  }
}
export class Dense {
  private weights: Float32Array;
  private bias?: Float32Array;
  constructor(store: TensorStore, name: string, readonly inputSize: number, readonly outputSize: number, bias = true) {
    this.weights = store.get(name + ".weight", [outputSize, inputSize]);
    if (bias) this.bias = store.get(name + ".bias", [outputSize]);
  }
  forward(input: ArrayLike<number>, output?: Float32Array): Float32Array {
    if (input.length !== this.inputSize) throw new Error("dense input shape mismatch");
    const result = output ?? new Float32Array(this.outputSize), size = this.inputSize, weights = this.weights;
    if (result.length !== this.outputSize) throw new Error("dense output shape mismatch");
    for (let row = 0; row < result.length; row++) {
      let total = this.bias?.[row] ?? 0;
      const offset = row * size;
      let col = 0;
      for (; col + 3 < size; col += 4) total += weights[offset + col] * input[col] +
        weights[offset + col + 1] * input[col + 1] + weights[offset + col + 2] * input[col + 2] +
        weights[offset + col + 3] * input[col + 3];
      for (; col < size; col++) total += weights[offset + col] * input[col];
      result[row] = total;
    }
    return result;
  }
}
export const sigmoid = (x: number): number => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
export const softplus = (x: number): number => x > 20 ? x : Math.log1p(Math.exp(x));
export function softmax(input: ArrayLike<number>): number[] {
  let peak = -Infinity;
  for (let i = 0; i < input.length; i++) peak = Math.max(peak, input[i]);
  const exps = Array.from(input, v => Math.exp(v - peak));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / total);
}
export function relu(input: Float32Array): Float32Array {
  for (let i = 0; i < input.length; i++) input[i] = Math.max(0, input[i]);
  return input;
}
export class GRU {
  private layers: { wi: Float32Array; wh: Float32Array; bi: Float32Array; bh: Float32Array; inputSize: number }[];
  constructor(store: TensorStore, name: string, readonly inputSize: number, readonly hiddenSize: number, count: number) {
    this.layers = Array.from({ length: count }, (_, layer) => {
      const size = layer === 0 ? inputSize : hiddenSize;
      return { inputSize: size, wi: store.get(`${name}.weight_ih_l${layer}`, [3 * hiddenSize, size]),
        wh: store.get(`${name}.weight_hh_l${layer}`, [3 * hiddenSize, hiddenSize]),
        bi: store.get(`${name}.bias_ih_l${layer}`, [3 * hiddenSize]),
        bh: store.get(`${name}.bias_hh_l${layer}`, [3 * hiddenSize]) };
    });
  }
  forward(history: Float32Array[]): Float32Array {
    let sequence = history;
    const size = this.hiddenSize, f = Math.fround;
    for (const layer of this.layers) {
      let hidden = new Float32Array(size);
      const outputs: Float32Array[] = [];
      for (const input of sequence) {
        if (input.length !== layer.inputSize) throw new Error("GRU input shape mismatch");
        const gi = new Float32Array(3 * size), gh = new Float32Array(3 * size);
        for (let row = 0; row < 3 * size; row++) {
          let a = layer.bi[row], b = layer.bh[row];
          for (let col = 0; col < input.length; col++) a += layer.wi[row * input.length + col] * input[col];
          for (let col = 0; col < size; col++) b += layer.wh[row * size + col] * hidden[col];
          gi[row] = a; gh[row] = b;
        }
        const next = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          const r = f(sigmoid(f(gi[i] + gh[i]))), z = f(sigmoid(f(gi[size + i] + gh[size + i])));
          const n = f(Math.tanh(f(gi[2 * size + i] + f(r * gh[2 * size + i]))));
          next[i] = f(n + f(z * f(hidden[i] - n)));
        }
        hidden = next; outputs.push(hidden);
      }
      sequence = outputs;
    }
    if (!sequence.length) throw new Error("GRU requires a nonempty sequence");
    return sequence[sequence.length - 1];
  }
}
