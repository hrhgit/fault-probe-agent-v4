/** Task adapters own preprocessing and outputs; the registry owns lazy loading. */
export interface TaskContract { input: unknown; output: unknown }
export interface ModelAdapter<I, O> {
  readonly id: string;
  predict(input: I): O;
}
export class ModelRegistry<Tasks extends { [K in keyof Tasks]: TaskContract }> {
  private entries = new Map<string, {
    task: keyof Tasks;
    factory: () => unknown;
    instance?: unknown;
  }>();

  register<K extends keyof Tasks>(task: K, id: string,
    factory: () => ModelAdapter<Tasks[K]["input"], Tasks[K]["output"]>): void {
    if (!id || this.entries.has(id)) throw new Error(`duplicate/empty model id: ${id}`);
    this.entries.set(id, { task, factory });
  }

  resolve<K extends keyof Tasks>(task: K, id: string): ModelAdapter<Tasks[K]["input"], Tasks[K]["output"]> {
    const entry = this.entries.get(id);
    if (!entry) throw new Error(`unknown model: ${id}`);
    if (entry.task !== task) throw new Error(`${id} does not implement task ${String(task)}`);
    if (!entry.instance) {
      const adapter = entry.factory() as ModelAdapter<Tasks[K]["input"], Tasks[K]["output"]>;
      if (adapter.id !== id) throw new Error(`model id mismatch: ${id}/${adapter.id}`);
      entry.instance = adapter;
    }
    return entry.instance as ModelAdapter<Tasks[K]["input"], Tasks[K]["output"]>;
  }

  predict<K extends keyof Tasks>(task: K, id: string, input: Tasks[K]["input"]): Tasks[K]["output"] {
    return this.resolve(task, id).predict(input);
  }

  list(): { id: string; task: keyof Tasks; loaded: boolean }[] {
    return [...this.entries].map(([id, value]) => ({ id, task: value.task, loaded: !!value.instance }));
  }
}
