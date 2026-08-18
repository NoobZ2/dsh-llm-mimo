#!/bin/bash
# dsh-llm-mimo build: create the dependency junctions into the DSH checkout's
# pnpm store (mirroring the dsh-super-injector build), then validate
# lib/index.js. lib/index.js itself is hand-written ESM — no tsc step.
#
# Usage: DSH_CHECKOUT=<path-to-dsh-checkout> bash scripts/build.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKOUT="${DSH_CHECKOUT:-}"
if [ -z "$CHECKOUT" ] || [ ! -d "$CHECKOUT/node_modules/.pnpm" ]; then
  echo "build: cannot locate the dsh checkout (set DSH_CHECKOUT)" >&2
  exit 1
fi

node -e '
  const fs = require("fs");
  const path = require("path");
  const checkout = process.argv[1];
  const pnpm = path.join(checkout, "node_modules", ".pnpm");
  const entries = fs.readdirSync(pnpm);
  // pnpm truncates long store dir names, so probe each entry for the real
  // package directory instead of matching prefixes. Prefer canonical entries
  // (the package directory is a real dir, not another entry scope junction).
  const resolveEntry = (scope, name) => {
    const candidates = entries.filter((e) => {
      if (!e.startsWith(scope)) return false;
      const pkg = path.join(pnpm, e, "node_modules", scope, name);
      try {
        return fs.lstatSync(pkg).isDirectory();
      } catch {
        return false;
      }
    });
    const canonical = candidates.find((e) => {
      try {
        return !fs.lstatSync(path.join(pnpm, e, "node_modules", scope, name)).isSymbolicLink();
      } catch {
        return false;
      }
    });
    const hit = canonical ?? candidates[0];
    if (!hit) throw new Error(`build: missing pnpm store entry for ${scope}${name}`);
    return path.join(pnpm, hit, "node_modules", scope, name);
  };
  const junction = (link, target) => {
    fs.rmSync(link, { recursive: true, force: true });
    fs.mkdirSync(path.dirname(link), { recursive: true });
    fs.symlinkSync(target, link, process.platform === "win32" ? "junction" : "dir");
    console.log(`linked ${path.relative(process.cwd(), link)} -> ${target}`);
  };
  const scoped = [
    "cordis",
    "dsh-anonymous-user-id",
    "dsh-credentials",
    "dsh-invariants",
    "dsh-launch-environment",
    "dsh-llm",
    "dsh-settings",
    "dsh-timeout",
    "schemastery",
  ];
  fs.mkdirSync("node_modules/@deepseek-ai", { recursive: true });
  for (const name of scoped) {
    junction(path.join("node_modules/@deepseek-ai", name), resolveEntry("@deepseek-ai", name));
  }
  const flatHit = entries.find((e) => e.startsWith("eventsource-parser@") && fs.existsSync(path.join(pnpm, e, "node_modules", "eventsource-parser")));
  if (!flatHit) throw new Error("build: missing pnpm store entry for eventsource-parser");
  junction(path.join("node_modules", "eventsource-parser"), path.join(pnpm, flatHit, "node_modules", "eventsource-parser"));
' "$CHECKOUT"

if [ ! -f lib/index.js ]; then
  echo "build: lib/index.js missing" >&2
  exit 1
fi
node --check lib/index.js
echo "=== dsh-llm-mimo build complete (junctions linked, lib/index.js validated) ==="
