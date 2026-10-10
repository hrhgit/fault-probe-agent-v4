/** A persistent local worker; one load per scenario, ordered requests, explicit errors. */
import { spawn, ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import * as path from "node:path";
import { createInterface } from "node:readline";
export class PythonTransport {
  private child?: ChildProcessWithoutNullStreams;
  private sequence = 0;
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
  private failure?: Error;
  private chain: Promise<unknown> = Promise.resolve();
  cpuSeconds = 0;
  constructor(readonly directory = path.resolve(__dirname, "../..")) {}
  private start(): void {
    if (this.failure) throw this.failure;
    if (this.child) return;
    const python = process.env.BASE_MODEL_PYTHON ?? path.join(this.directory, ".runtime/venv/bin/python");
    if (!existsSync(python)) throw new Error(`Model Python missing: ${python}. Run npm run models:setup or set BASE_MODEL_PYTHON.`);
    this.child = spawn(python, ["-u", path.join(this.directory, "runtime/worker.py")], {
      cwd: this.directory, stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1", OMP_NUM_THREADS: "1", MKL_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1" },
    });
    this.child.stderr.on("data", chunk => process.stderr.write(chunk));
    const lines = createInterface({ input: this.child.stdout, crlfDelay: Infinity });
    lines.on("line", line => {
      try {
        const response = JSON.parse(line), callback = this.pending.get(response.id);
        if (!callback) throw new Error("Unexpected model response id");
        this.pending.delete(response.id);
        if (response.error) callback.reject(new Error(response.error));
        else { this.cpuSeconds += Number(response.cpu_seconds ?? 0); callback.resolve(response.result); }
      } catch (error) { this.fail(error instanceof Error ? error : new Error(String(error))); }
    });
    this.child.on("error", error => this.fail(error));
    this.child.on("exit", (code, signal) => this.fail(new Error(`Model worker exited (${code ?? signal})`)));
  }
  private fail(error: Error): void {
    this.failure ??= error;
    for (const callback of this.pending.values()) callback.reject(error);
    this.pending.clear();
    this.child?.stdin.end();
  }
  request<O>(operation: string, input: unknown): Promise<O> {
    const job = this.chain.then(() => new Promise<O>((resolve, reject) => {
      try {
        this.start();
        const id = ++this.sequence;
        const line = JSON.stringify({ id, operation, input }) + "\n";
        this.pending.set(id, { resolve: value => resolve(value as O), reject });
        this.child!.stdin.write(line, error => { if (error) this.fail(error); });
      } catch (error) { reject(error); }
    }));
    this.chain = job.catch(() => undefined);
    return job;
  }
  async close(): Promise<void> {
    await this.chain;
    const child = this.child;
    if (!child || child.exitCode !== null || child.signalCode !== null) return;
    const exited = new Promise<void>(resolve => child.once("exit", () => resolve()));
    child.stdin.end();
    await exited;
  }
}
