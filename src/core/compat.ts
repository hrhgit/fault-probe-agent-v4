/** Small numeric helpers preserve the Python 3.12 baseline's arithmetic and tie order. */
export type Payload = Record<string, any>;
export type Vec = [number, number, number];
export const mod = (a: number, b: number): number => {
  let r = a % b;
  if (r !== 0 && r < 0 !== b < 0) r += b;
  return r === 0 ? (b < 0 ? -0 : 0) : r;
};
export function floorDiv(a: number, b: number): number {
  let r = a % b,
    d = (a - r) / b;
  if (r !== 0 && r < 0 !== b < 0) {
    r += b;
    d -= 1;
  }
  if (d === 0) return a < 0 !== b < 0 ? -0 : 0;
  const f = Math.floor(d);
  return d - f > 0.5 ? f + 1 : f;
}
export function sum(values: Iterable<number>): number {
  let s = 0,
    c = 0;
  for (const x of values) {
    const t = s + x;
    c += Math.abs(s) >= Math.abs(x) ? s - t + x : x - t + s;
    s = t;
  }
  return c !== 0 && Number.isFinite(c) ? s + c : s;
}
export function median(values: Iterable<number>): number {
  const a = [...values].sort((x, y) => x - y),
    k = Math.floor(a.length / 2);
  if (!a.length) throw new Error("median of empty sequence");
  return a.length % 2 ? a[k] : 0.5 * (a[k - 1] + a[k]);
}
export function tupleCompare(a: number[], b: number[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++)
    if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  return a.length - b.length;
}
export function appendBounded<T>(a: T[], value: T, capacity: number): void {
  a.push(value);
  if (a.length > capacity) a.splice(0, a.length - capacity);
}
export function asciiJson(value: unknown): string {
  return JSON.stringify(value).replace(
    /[\u007f-\uffff]/g,
    (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"),
  );
}
