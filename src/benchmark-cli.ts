#!/usr/bin/env node
/**
 * sec-mem Benchmark CLI
 */

import { spawn } from "child_process";
import { join } from "path";

console.log(`
╔════════════════════════════════════════════════════════════╗
║                  sec-mem Benchmark                         ║
║         High-Performance Memory for OpenClaw               ║
╚════════════════════════════════════════════════════════════╝
`);

const scriptPath = join(__dirname, "..", "quick_benchmark.py");
const python = spawn("python3", [scriptPath], {
  stdio: "inherit"
});

python.on("exit", (code) => {
  process.exit(code || 0);
});
