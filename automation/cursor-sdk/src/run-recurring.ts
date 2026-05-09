import { Agent, CursorAgentError } from "@cursor/sdk";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

type CliArgs = {
  cadence: "daily" | "weekly";
  trigger: string;
  configPath: string;
};

function parseArgs(argv: string[]): CliArgs {
  let cadence: "daily" | "weekly" = "daily";
  let trigger = "cursor-sdk";
  let configPath = "configs/recurring_runs.json";

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--cadence") {
      const value = argv[i + 1];
      if (value === "daily" || value === "weekly") {
        cadence = value;
        i += 1;
      }
    } else if (arg === "--trigger") {
      const value = argv[i + 1];
      if (value) {
        trigger = value;
        i += 1;
      }
    } else if (arg === "--config") {
      const value = argv[i + 1];
      if (value) {
        configPath = value;
        i += 1;
      }
    }
  }

  return { cadence, trigger, configPath };
}

function buildPrompt(args: CliArgs): string {
  const cmd = [
    "python3",
    "scripts/hf/recurring_runner.py",
    "--cadence",
    args.cadence,
    "--trigger",
    args.trigger,
    "--config",
    args.configPath,
  ].join(" ");
  return [
    "Execute exactly one command in the repo root and report the result.",
    "Do not edit files.",
    "Command:",
    cmd,
    "Return:",
    "1) Exit status",
    "2) Key summary lines",
    "3) Any actionable failure if non-zero",
  ].join("\n");
}

async function main(): Promise<void> {
  const apiKey = process.env.CURSOR_API_KEY;
  if (!apiKey) {
    console.error("Missing CURSOR_API_KEY.");
    process.exit(1);
  }

  const args = parseArgs(process.argv.slice(2));
  const modelId = process.env.CURSOR_MODEL_ID ?? "composer-2";

  const thisDir = path.dirname(fileURLToPath(import.meta.url));
  const repoRoot = path.resolve(thisDir, "..", "..", "..");
  const prompt = buildPrompt(args);

  try {
    const result = await Agent.prompt(prompt, {
      apiKey,
      model: { id: modelId },
      local: { cwd: repoRoot },
    });

    const output =
      typeof result.result === "string"
        ? result.result
        : JSON.stringify(result.result, null, 2);
    console.log(output);

    if (result.status === "error") {
      process.exit(2);
    }
  } catch (error) {
    if (error instanceof CursorAgentError) {
      console.error(`Cursor SDK startup failed: ${error.message}`);
      process.exit(error.isRetryable ? 75 : 1);
    }
    throw error;
  }
}

await main();
