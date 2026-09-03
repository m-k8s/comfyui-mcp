// Incognito turns — a conversation the user asked NOT to keep. Four traces the
// orchestrator leaves of a normal turn, and what each does under incognito:
//
//   the Claude session file  ~/.claude/projects/<cwd>/<session>.jsonl, which the
//                            Agent SDK writes and history.ts lists for the
//                            mobile client → deleted when the turn ends;
//   the durable resume index (SessionStore)          → not written;
//   the orchestrator log line quoting the message    → text withheld;
//   the Ollama transcript dump (COMFYUI_MCP_TRANSCRIPT_DIR) → skipped.
//
// Langfuse, on the LiteLLM proxy of the custom lane, is deliberately NOT in
// this list: it is the user's own observability, switched on by them.

import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { claudeSessionFile, forgetClaudeSession } from "../../orchestrator/history.js";
import { PanelAgent } from "../../orchestrator/panel-agent.js";
import type {
  AgentBackend,
  AgentCapabilities,
  AgentEvent,
  BackendStartOptions,
  NeutralTurn,
} from "../../orchestrator/agent-backend.js";

const BASE_CAPS: AgentCapabilities = {
  persistentChannel: true,
  streamingDeltas: true,
  interruptMidTurn: true,
  forkAtAnchor: false,
  inProcessMcp: false,
  modelEnumeration: false,
  slashCommands: false,
  hooks: false,
  vision: true,
  turnMarkers: true,
};

describe("forgetClaudeSession", () => {
  it("deletes the session's JSONL under the encoded cwd, and says so", () => {
    const home = mkdtempSync(join(tmpdir(), "cmcp-home-"));
    const cwd = "/home/someone/dev/comfyui-mcp";
    const file = claudeSessionFile("3594458b-50a6-4ca9-bd73-6674c111fa32", cwd, home);
    expect(file).toBe(join(home, ".claude", "projects", "-home-someone-dev-comfyui-mcp", "3594458b-50a6-4ca9-bd73-6674c111fa32.jsonl"));
    mkdirSync(join(home, ".claude", "projects", "-home-someone-dev-comfyui-mcp"), { recursive: true });
    writeFileSync(file, '{"type":"user"}\n');
    expect(forgetClaudeSession("3594458b-50a6-4ca9-bd73-6674c111fa32", cwd, home)).toBe(true);
    expect(existsSync(file)).toBe(false);
    // A second call has nothing to delete and says so instead of throwing.
    expect(forgetClaudeSession("3594458b-50a6-4ca9-bd73-6674c111fa32", cwd, home)).toBe(false);
    rmSync(home, { recursive: true, force: true });
  });

  it("refuses a session id that is not a plain id (never a path)", () => {
    const home = mkdtempSync(join(tmpdir(), "cmcp-home-"));
    expect(() => forgetClaudeSession("../../etc/passwd", "/x", home)).toThrow(/session id/);
    rmSync(home, { recursive: true, force: true });
  });
});

function backendWith(events: (turn: NeutralTurn, n: number) => AgentEvent[]) {
  const seen: NeutralTurn[] = [];
  const backend: AgentBackend = {
    id: "claude",
    capabilities: BASE_CAPS,
    async *run(opts: BackendStartOptions): AsyncIterable<AgentEvent> {
      yield { type: "session", sessionId: "s1" };
      for await (const turn of opts.channel) {
        seen.push(turn);
        for (const ev of events(turn, seen.length)) yield ev;
        if (seen.length >= 2) break;
      }
    },
    async interrupt() {},
    async listModels() {
      return [];
    },
  };
  return { backend, seen };
}

describe("PanelAgent incognito turns", () => {
  it("passes `incognito` to the backend and forgets the Claude session when the turn ends", async () => {
    const forgotten: string[] = [];
    const { backend, seen } = backendWith((_t, n) => [
      { type: "assistant", text: "ok", turn: n },
      { type: "result", ok: true, turn: n },
    ]);
    const agent = new PanelAgent(
      "tab1",
      { mcpServers: undefined, systemAppend: "", model: "m", onSay: () => {}, forgetSession: (sid) => forgotten.push(sid) },
      backend,
    );
    // Two messages queued before the drain are BATCHED into one turn (and one
    // incognito item makes the whole batch incognito), so space them out: the
    // second is queued once the first turn has been handed to the backend.
    const running = agent.start();
    agent.send("a normal question");
    await new Promise((r) => setTimeout(r, 60));
    agent.send("a private question", { incognito: true });
    await Promise.race([running, new Promise((r) => setTimeout(r, 400))]);
    expect(seen.map((t) => t.incognito === true)).toEqual([false, true]);
    // Forgotten exactly once, after the incognito turn, for the live session.
    expect(forgotten).toEqual(["s1"]);
  });

  it("does not forget anything for ordinary turns", async () => {
    const forgotten: string[] = [];
    const { backend } = backendWith((_t, n) => [{ type: "result", ok: true, turn: n }]);
    const agent = new PanelAgent(
      "tab1",
      { mcpServers: undefined, systemAppend: "", model: "m", onSay: () => {}, forgetSession: (sid) => forgotten.push(sid) },
      backend,
    );
    const running = agent.start();
    agent.send("one");
    await new Promise((r) => setTimeout(r, 60));
    agent.send("two");
    await Promise.race([running, new Promise((r) => setTimeout(r, 400))]);
    expect(forgotten).toEqual([]);
  });
});
