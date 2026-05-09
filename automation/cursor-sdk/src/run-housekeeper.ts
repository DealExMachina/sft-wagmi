import { Agent, CursorAgentError } from "@cursor/sdk";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

type CliArgs = {
  area: string;
  extraInstruction: string | null;
};

function parseArgs(argv: string[]): CliArgs {
  let area = ".";
  let extraInstruction: string | null = null;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--area") {
      const value = argv[i + 1];
      if (value) {
        area = value;
        i += 1;
      }
    } else if (arg === "--instruction") {
      const value = argv[i + 1];
      if (value) {
        extraInstruction = value;
        i += 1;
      }
    }
  }

  return { area, extraInstruction };
}

function buildPrompt(args: CliArgs): string {
  const lines = [
    "You are the repository housekeeper for sft-wagmi.",
    "Goal: keep docs current, reduce duplication, and remove stale redundant documents when safe.",
    "",
    "Execution rules:",
    `- Work in this path scope first: ${args.area}`,
    "- Prioritize README consistency with current scripts, config files, and actual repo behavior.",
    "- Remove only documents that are clearly redundant or obsolete. If deleting, update inbound references.",
    "- Prefer small, reviewable edits over broad rewrites.",
    "- Do not run destructive git commands.",
    "- Do not commit or push.",
    "",
    "Required output at the end:",
    "1) Files changed",
    "2) Why each change was needed",
    "3) Any follow-up manual actions",
  ];

  if (args.extraInstruction) {
    lines.push("", `Extra instruction: ${args.extraInstruction}`);
  }

  return lines.join("\n");
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
