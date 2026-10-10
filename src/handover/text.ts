/**
 * Deterministic clean-up and decoding of handover notes before the model reads them.
 * The model does the language understanding; character-level work (zero-width characters,
 * emoticons inside numbers, Morse, Caesar shifts, sol-fa digits) is done here because
 * models are unreliable at it and it would cost an extra round trip.
 */

const ZERO_WIDTH = /[​-‍⁠﻿]/g;
/** Emoticon bodies never contain CJK ideographs, kana words or digits; "(UTC)" style tags are plain ASCII words. */
const EMOTICON = /[(（][^()（）一-鿿぀-ゟ0-9０-９]{1,14}[)）]/g;

export function stripEmoticons(text: string): string {
  let out = text.replace(/m[(（]_ _[)）]m/g, "");
  out = out.replace(EMOTICON, (group) => (/^[(（][A-Za-z .~,:-]+[)）]$/.test(group) ? group : ""));
  return out.replace(/[ヽﾉ┐┌╮╭]/g, "");
}

/** The text the model is given: one line per speaker entry, ASCII digits and punctuation. */
export function cleanText(raw: string): string {
  const stripped = stripEmoticons(raw.replace(ZERO_WIDTH, "")).normalize("NFKC");
  return stripped
    .split(/\r?\n/)
    .map((line) => line.replace(/[ \t]+/g, " ").trim())
    .filter((line) => line.length > 0)
    .join("\n");
}

const MORSE: Record<string, string> = {
  ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F", "--.": "G", "....": "H", "..": "I",
  ".---": "J", "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
  "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y", "--..": "Z",
  "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4", ".....": "5", "-....": "6",
  "--...": "7", "---..": "8", "----.": "9", "-..-.": "/", "-....-": "-", "---...": ":", ".-.-.-": ".",
  "--..--": ",",
};

function decodeMorseWith(words: string[][], dot: string, dash: string): string | null {
  const out: string[] = [];
  for (const word of words) {
    let text = "";
    for (const token of word) {
      let code = "";
      for (const ch of token) code += ch === dot ? "." : ch === dash ? "-" : "?";
      const letter = MORSE[code];
      if (letter === undefined) return null;
      text += letter;
    }
    out.push(text);
  }
  return out.join(" ");
}

/** Decode a run written in any two-symbol alphabet (●○, ·—, 嘀嗒, tiTA, ♪♫ ...); null when it is not Morse. */
export function decodeMorse(segment: string): string | null {
  let body = segment.replace(/ti/g, "\u0001").replace(/TA/g, "\u0002");
  const words = body
    .split("/")
    .map((word) => word.trim().split(/\s+/).filter(Boolean))
    .filter((word) => word.length > 0);
  if (!words.length) return null;
  const symbols = new Set<string>();
  for (const word of words) for (const token of word) for (const ch of token) symbols.add(ch);
  if (symbols.size !== 2) return null;
  const [a, b] = [...symbols];
  const first = decodeMorseWith(words, a, b),
    second = decodeMorseWith(words, b, a);
  if (first !== null && second !== null) {
    const score = (s: string) => (s.match(/[0-9/]/g) ?? []).length;
    return score(first) >= score(second) ? first : second;
  }
  return first ?? second;
}

const ENGLISH = new Set(
  ("the sky thin thick cloud clouds haze rain wind windy closed close dome local update latest still observable not " +
    "night all whole under this long range from to and is of on in no for north south east west sector has have " +
    "maintenance stop stopped fault throughput report ticket engineer engineers until after before will be it " +
    "observe observing cannot can shoot keep time utc forecast tonight tomorrow heavy light clear open").split(" "),
);

function shiftText(text: string, shift: number): string {
  let out = "";
  for (const ch of text) {
    const code = ch.charCodeAt(0);
    if (code >= 65 && code <= 90) out += String.fromCharCode(((code - 65 - shift + 2600) % 26) + 65);
    else if (code >= 97 && code <= 122) out += String.fromCharCode(((code - 97 - shift + 2600) % 26) + 97);
    else out += ch;
  }
  return out;
}

function englishScore(text: string): number {
  let hits = 0;
  for (const word of text.toLowerCase().match(/[a-z]+/g) ?? []) if (ENGLISH.has(word)) hits++;
  return hits;
}

/** Undo a Caesar shift on an English run; returns null when the run already reads as English or nothing fits. */
export function decodeCaesar(segment: string): { text: string; shift: number } | null {
  const letters = (segment.match(/[A-Za-z]/g) ?? []).length;
  if (letters < 12) return null;
  let best = 0,
    best_score = englishScore(segment);
  const plain = best_score;
  for (let shift = 1; shift < 26; shift++) {
    const score = englishScore(shiftText(segment, shift));
    if (score > best_score) {
      best_score = score;
      best = shift;
    }
  }
  if (best === 0 || best_score < 3 || best_score < plain + 2) return null;
  return { text: shiftText(segment, best), shift: best };
}

const SOLFA: [RegExp, string][] = [
  [/高音\s*do/gi, "8"],
  [/高音\s*re/gi, "9"],
  [/休止/g, "0"],
  [/sol/gi, "5"],
  [/do/gi, "1"],
  [/re/gi, "2"],
  [/mi/gi, "3"],
  [/fa/gi, "4"],
  [/la/gi, "6"],
  [/si/gi, "7"],
];

/** "休止 休止 : fa sol" -> "00:45"; null when the run holds anything but sol-fa syllables. */
export function decodeSolfa(segment: string): string | null {
  let body = segment;
  for (const [pattern, digit] of SOLFA) body = body.replace(pattern, digit);
  body = body.replace(/\s+/g, "");
  return /^[0-9]{1,2}:[0-9]{2}$/.test(body) || /^[0-9]{3,4}$/.test(body) ? body : null;
}

const MORSE_HINT = /摩斯|摩尔斯|换个写法|換個寫法|敲了一遍|morse/i;
const SOLFA_RUN = /((?:(?:高音\s*)?(?:do|re|mi|fa|sol|la|si)|休止)(?:\s*:?\s*(?:(?:高音\s*)?(?:do|re|mi|fa|sol|la|si)|休止)){2,})/gi;

/**
 * Clean a note and append a decoded reading after each encoded run, e.g. `… TAti 〔解码: 10/10 2345-2145 OK THIN〕`.
 * Returns the annotated text and how many runs were decoded.
 */
export function annotate(raw: string): { text: string; decoded: number } {
  let decoded = 0;
  const lines = cleanText(raw).split("\n").map((line) => {
    let out = line;
    const speaker = /^【[^】]*】/.exec(out)?.[0] ?? "";
    const rest = out.slice(speaker.length);
    // Morse: the encoded run is whatever follows the last colon or closing bracket of a line that looks like one.
    const body = rest.replace(/(\s*\([^()]*\))+\s*$/, ""),
      cut = Math.max(body.lastIndexOf(":"), body.lastIndexOf(")"));
    const tail = body.slice(cut + 1).trim();
    if (tail.length >= 8 && (MORSE_HINT.test(rest) || /^[^一-鿿A-Za-z0-9]{8,}$/.test(tail.replace(/ti|TA/g, "·")))) {
      const morse = decodeMorse(tail);
      if (morse !== null && /[0-9A-Z]/.test(morse)) {
        decoded++;
        return `${speaker}${rest.slice(0, cut + 1)} 〔摩斯解码: ${morse}〕`;
      }
    }
    // Caesar: the longest ASCII run of the line.
    const ascii = rest.match(/[A-Za-z][A-Za-z0-9 ,.:;()\/\-~'%]{14,}/g) ?? [];
    for (const run of ascii) {
      const caesar = decodeCaesar(run);
      if (caesar !== null) {
        decoded++;
        out = out.replace(run, `${run} 〔凯撒解码(位移${caesar.shift}): ${caesar.text.trim()}〕`);
      }
    }
    out = out.replace(SOLFA_RUN, (run) => {
      const digits = decodeSolfa(run);
      if (digits === null) return run;
      decoded++;
      return `${run} 〔唱名解码: ${digits}〕`;
    });
    return out;
  });
  return { text: lines.join("\n"), decoded };
}
