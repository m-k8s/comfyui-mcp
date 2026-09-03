/**
 * Reader for the pseudo-Python that `to_code` renders. It recognises exactly
 * two statement shapes and refuses everything else, naming the line:
 *
 *   a, b = Type(input=variable, value=literal)   # assignment
 *   Type(input=variable, value=literal)          # bare call
 *
 * A value is a variable (an identifier) or a constant: a string, a number,
 * True, False, None, a list or a dict of constants. No expression is
 * evaluated, no attribute is looked up, no call is made. The code is data.
 *
 * Why hand-written rather than a Python parser: the fragment is small, the
 * accepted grammar is two lines, and the refusal messages are the product —
 * a model that gets "line 3: all arguments must be named" fixes line 3.
 */

/** What a constant can be: exactly what a widget value can hold. */
export type Literal = string | number | boolean | null | Literal[] | { [key: string]: Literal };

export type CodeValue = { kind: "var"; name: string } | { kind: "literal"; value: Literal };

export interface CodeStatement {
  /** 1-based line of the statement's first token. */
  line: number;
  /** Variables on the left of `=`; empty for a bare call. */
  targets: string[];
  classType: string;
  args: Array<{ name: string; value: CodeValue }>;
}

export interface ParsedCode {
  statements: CodeStatement[];
  problems: string[];
}

type Token =
  | { kind: "ident"; text: string }
  | { kind: "number"; value: number }
  | { kind: "string"; value: string }
  | { kind: "punct"; text: string };

const KEYWORDS_REFUSED = new Set([
  "import", "from", "def", "class", "for", "while", "if", "elif", "else", "try",
  "except", "finally", "with", "return", "yield", "lambda", "global", "nonlocal",
  "assert", "del", "pass", "raise", "async", "await", "print",
]);

const ONLY_TWO_SHAPES =
  "only assignments `x = Type(...)` and bare calls `Type(...)` are read; nothing else is interpreted.";

/** Strip a `#` comment that is outside any string literal. */
function stripComment(line: string): string {
  let quote: string | null = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quote) {
      if (c === "\\") i++;
      else if (c === quote) quote = null;
    } else if (c === '"' || c === "'") {
      quote = c;
    } else if (c === "#") {
      return line.slice(0, i);
    }
  }
  return line;
}

function decodeEscapes(raw: string): string {
  return raw.replace(/\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)/g, (_m, e: string) => {
    switch (e[0]) {
      case "n": return "\n";
      case "t": return "\t";
      case "r": return "\r";
      case "\\": return "\\";
      case '"': return '"';
      case "'": return "'";
      case "0": return "\0";
      case "u": return String.fromCharCode(parseInt(e.slice(1), 16));
      case "x": return String.fromCharCode(parseInt(e.slice(1), 16));
      default: return e;
    }
  });
}

class ParseRefusal extends Error {}

function tokenize(text: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      i++;
      continue;
    }
    if (c === '"' || c === "'") {
      // Triple quotes are refused rather than half-read.
      if (text.startsWith(c.repeat(3), i)) {
        throw new ParseRefusal("triple-quoted strings are not read; write the value on one line with \\n escapes.");
      }
      let j = i + 1;
      let raw = "";
      while (j < text.length && text[j] !== c) {
        if (text[j] === "\\") {
          raw += text[j] + (text[j + 1] ?? "");
          j += 2;
        } else {
          raw += text[j];
          j++;
        }
      }
      if (j >= text.length) throw new ParseRefusal("a string never closes.");
      tokens.push({ kind: "string", value: decodeEscapes(raw) });
      i = j + 1;
      continue;
    }
    const num = /^-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?/.exec(text.slice(i));
    if (num && (c !== "-" || /\d|\./.test(text[i + 1] ?? ""))) {
      tokens.push({ kind: "number", value: Number(num[0]) });
      i += num[0].length;
      continue;
    }
    const ident = /^[A-Za-z_][A-Za-z0-9_]*/.exec(text.slice(i));
    if (ident) {
      tokens.push({ kind: "ident", text: ident[0] });
      i += ident[0].length;
      continue;
    }
    tokens.push({ kind: "punct", text: c });
    i++;
  }
  return tokens;
}

/**
 * Logical statements: a statement continues onto the next physical line
 * while a bracket is open, as Python itself reads it.
 */
function logicalLines(code: string): Array<{ line: number; text: string; unbalanced: boolean }> {
  const out: Array<{ line: number; text: string; unbalanced: boolean }> = [];
  const physical = code.split(/\r?\n/);
  let buffer = "";
  let start = 0;
  let depth = 0;
  for (let n = 0; n < physical.length; n++) {
    const text = stripComment(physical[n]);
    if (!buffer && !text.trim()) continue;
    if (!buffer) start = n + 1;
    buffer += (buffer ? "\n" : "") + text;
    for (const ch of text) {
      if (ch === "(" || ch === "[" || ch === "{") depth++;
      else if (ch === ")" || ch === "]" || ch === "}") depth--;
    }
    if (depth <= 0) {
      out.push({ line: start, text: buffer, unbalanced: depth < 0 });
      buffer = "";
      depth = 0;
    }
  }
  if (buffer) out.push({ line: start, text: buffer, unbalanced: true });
  return out;
}

class Cursor {
  private i = 0;
  constructor(private readonly tokens: Token[]) {}
  peek(offset = 0): Token | undefined {
    return this.tokens[this.i + offset];
  }
  next(): Token | undefined {
    return this.tokens[this.i++];
  }
  isPunct(text: string, offset = 0): boolean {
    const t = this.peek(offset);
    return t?.kind === "punct" && t.text === text;
  }
  done(): boolean {
    return this.i >= this.tokens.length;
  }
}

function parseLiteral(cur: Cursor, argName: string): Literal {
  const t = cur.next();
  const refuse = (): never => {
    throw new ParseRefusal(`"${argName}" must be a variable or a constant; no expression is evaluated.`);
  };
  if (!t) return refuse();
  if (t.kind === "string") return t.value;
  if (t.kind === "number") return t.value;
  if (t.kind === "ident") {
    if (t.text === "True") return true;
    if (t.text === "False") return false;
    if (t.text === "None") return null;
    return refuse();
  }
  if (t.text === "[") {
    const items: Literal[] = [];
    while (!cur.isPunct("]")) {
      items.push(parseLiteral(cur, argName));
      if (cur.isPunct(",")) cur.next();
      else if (!cur.isPunct("]")) refuse();
    }
    cur.next();
    return items;
  }
  if (t.text === "{") {
    const obj: { [key: string]: Literal } = {};
    while (!cur.isPunct("}")) {
      const key = cur.next();
      if (key?.kind !== "string") return refuse();
      if (!cur.isPunct(":")) return refuse();
      cur.next();
      obj[key.value] = parseLiteral(cur, argName);
      if (cur.isPunct(",")) cur.next();
      else if (!cur.isPunct("}")) refuse();
    }
    cur.next();
    return obj;
  }
  return refuse();
}

function parseStatement(text: string): CodeStatement {
  const tokens = tokenize(text);
  const cur = new Cursor(tokens);
  const first = cur.peek();
  if (!first || first.kind !== "ident" || KEYWORDS_REFUSED.has(first.text)) {
    throw new ParseRefusal(ONLY_TWO_SHAPES);
  }

  // Targets: `a = ...` or `a, b = ...`; otherwise a bare call.
  const targets: string[] = [];
  let k = 0;
  while (cur.peek(k)?.kind === "ident" && cur.isPunct(",", k + 1)) k += 2;
  if (cur.peek(k)?.kind === "ident" && cur.isPunct("=", k + 1)) {
    for (let j = 0; j <= k; j += 2) {
      const t = cur.next();
      if (t?.kind === "ident") targets.push(t.text);
      cur.next(); // the comma, or the `=` on the last one
    }
  }

  const callee = cur.next();
  if (callee?.kind !== "ident" || KEYWORDS_REFUSED.has(callee.text)) throw new ParseRefusal(ONLY_TWO_SHAPES);
  if (!cur.isPunct("(")) {
    throw new ParseRefusal("the right-hand side must be a call of the form `Type(input=..., value=...)`.");
  }
  cur.next();

  const args: CodeStatement["args"] = [];
  while (!cur.isPunct(")")) {
    if (cur.isPunct("*")) {
      const name = cur.isPunct("*", 1) ? `**${(cur.peek(2) as { text?: string } | undefined)?.text ?? ""}` : "*args";
      throw new ParseRefusal(`\`${name}\` is not interpreted.`);
    }
    const name = cur.peek();
    if (name?.kind !== "ident" || !cur.isPunct("=", 1)) {
      throw new ParseRefusal(
        `all arguments must be named, "${callee.text}(model=..., steps=12)", so each one reaches a known input.`,
      );
    }
    cur.next();
    cur.next();
    const v = cur.peek();
    let value: CodeValue;
    if (v?.kind === "ident" && !["True", "False", "None"].includes(v.text) && !cur.isPunct("(", 1) && !cur.isPunct(".", 1)) {
      cur.next();
      value = { kind: "var", name: v.text };
    } else {
      value = { kind: "literal", value: parseLiteral(cur, name.text) };
    }
    if (!cur.isPunct(",") && !cur.isPunct(")")) {
      throw new ParseRefusal(`"${name.text}" must be a variable or a constant; no expression is evaluated.`);
    }
    args.push({ name: name.text, value });
    if (cur.isPunct(",")) cur.next();
  }
  cur.next();
  if (!cur.done()) throw new ParseRefusal(ONLY_TWO_SHAPES);

  return { line: 0, targets, classType: callee.text, args };
}

export function parseCode(code: string): ParsedCode {
  const statements: CodeStatement[] = [];
  const problems: string[] = [];
  for (const logical of logicalLines(code)) {
    if (logical.unbalanced) {
      problems.push(`line ${logical.line}: unbalanced brackets, the statement never closes.`);
      continue;
    }
    try {
      statements.push({ ...parseStatement(logical.text), line: logical.line });
    } catch (err) {
      if (err instanceof ParseRefusal) problems.push(`line ${logical.line}: ${err.message}`);
      else throw err;
    }
  }
  return { statements, problems };
}
