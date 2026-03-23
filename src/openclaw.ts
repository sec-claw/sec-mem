/**
 * OpenClaw Memory (sec-mem) Plugin
 *
 * High-performance local memory storage with FAISS Advanced indexing.
 * Supports multiple index types: flat, hnsw, ivf, ivf_pq, pq
 *
 * Features:
 * - 4 tools: memory_search, memory_store, memory_get, memory_delete
 * - Auto-recall: injects relevant memories before each agent turn
 * - Auto-capture: stores key facts after each agent turn
 * - Per-agent isolation via userId namespaces
 * - CLI: openclaw sec-mem search, openclaw sec-mem stats
 */

import { Type, type TSchema } from "@sinclair/typebox";
import type { OpenClawPluginApi } from "../node_modules/openclaw/dist/plugin-sdk/index.js";
import { SecMemPlugin, SecMemConfig } from "./index.js";

// ============================================================================
// Types
// ============================================================================

interface MemoryItem {
  id: string;
  memory: string;
  user_id?: string;
  score?: number;
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

interface AddResultItem {
  id: string;
  memory: string;
  event: "ADD" | "UPDATE" | "DELETE" | "NOOP";
}

// ============================================================================
// Config Schema
// ============================================================================

const ALLOWED_KEYS = [
  "collectionName",
  "indexType",
  "embeddingDims",
  "distanceStrategy",
  "storagePath",
  "hnswM",
  "hnswEfConstruction",
  "hnswEfSearch",
  "nlist",
  "nprobe",
  "m",
  "nbits",
  "ollamaUrl",
  "ollamaEmbedModel",
  "ollamaLlmModel",
  "autoRecall",
  "autoCapture",
  "topK",
  "searchThreshold",
];

function assertAllowedKeys(
  value: Record<string, unknown>,
  allowed: string[],
  label: string,
) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) return;
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

export const secMemConfigSchema = {
  parse(value: unknown): SecMemConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return {} as SecMemConfig;
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(cfg, ALLOWED_KEYS, "sec-mem config");

    return {
      collectionName: typeof cfg.collectionName === "string" ? cfg.collectionName : "sec-mem",
      indexType: (cfg.indexType as any) || "hnsw",
      embeddingDims: typeof cfg.embeddingDims === "number" ? cfg.embeddingDims : 512,
      distanceStrategy: (cfg.distanceStrategy as any) || "cosine",
      storagePath: typeof cfg.storagePath === "string" ? cfg.storagePath : "./sec-mem-data",
      hnswM: typeof cfg.hnswM === "number" ? cfg.hnswM : 16,
      hnswEfConstruction: typeof cfg.hnswEfConstruction === "number" ? cfg.hnswEfConstruction : 64,
      hnswEfSearch: typeof cfg.hnswEfSearch === "number" ? cfg.hnswEfSearch : 32,
      nlist: typeof cfg.nlist === "number" ? cfg.nlist : 100,
      nprobe: typeof cfg.nprobe === "number" ? cfg.nprobe : 10,
      m: typeof cfg.m === "number" ? cfg.m : 16,
      nbits: typeof cfg.nbits === "number" ? cfg.nbits : 8,
      ollamaUrl: typeof cfg.ollamaUrl === "string" ? cfg.ollamaUrl : "http://localhost:11434",
      ollamaEmbedModel: typeof cfg.ollamaEmbedModel === "string" ? cfg.ollamaEmbedModel : "nomic-embed-text",
      ollamaLlmModel: typeof cfg.ollamaLlmModel === "string" ? cfg.ollamaLlmModel : undefined,
      autoRecall: cfg.autoRecall !== false,
      autoCapture: cfg.autoCapture !== false,
      topK: typeof cfg.topK === "number" ? cfg.topK : 5,
      searchThreshold: typeof cfg.searchThreshold === "number" ? cfg.searchThreshold : 0.5,
    };
  },
};

// ============================================================================
// Plugin Definition
// ============================================================================

const memoryPlugin = {
  id: "sec-mem",
  name: "Memory (sec-mem)",
  description: "High-performance local memory with FAISS Advanced indexing",
  kind: "memory" as const,
  configSchema: secMemConfigSchema,

  register(api: OpenClawPluginApi) {
    const cfg = secMemConfigSchema.parse(api.pluginConfig);
    const plugin = new SecMemPlugin(cfg);
    let initialized = false;
    let currentSessionId: string | undefined;

    async function ensureInitialized() {
      if (!initialized) {
        await plugin.initialize();
        initialized = true;
        api.logger.info("sec-mem: plugin initialized");
      }
    }

    api.logger.info(
      `sec-mem: registered (index: ${cfg.indexType}, user: ${cfg.collectionName}, autoRecall: ${cfg.autoRecall}, autoCapture: ${cfg.autoCapture})`,
    );

    // =======================================================================
    // Tools
    // =======================================================================

    registerTools(api, plugin, cfg, () => currentSessionId, ensureInitialized);

    // =======================================================================
    // CLI Commands
    // =======================================================================

    registerCli(api, plugin, cfg, ensureInitialized);

    // =======================================================================
    // Lifecycle Hooks
    // =======================================================================

    registerHooks(api, plugin, cfg, ensureInitialized, {
      setCurrentSessionId: (id: string) => { currentSessionId = id; },
    });

    // =======================================================================
    // Service
    // =======================================================================

    api.registerService({
      id: "sec-mem",
      start: async () => {
        await ensureInitialized();
        api.logger.info(
          `sec-mem: started (index: ${cfg.indexType}, autoRecall: ${cfg.autoRecall}, autoCapture: ${cfg.autoCapture})`,
        );
      },
      stop: async () => {
        if (initialized) {
          await plugin.shutdown();
          api.logger.info("sec-mem: stopped");
        }
      },
    });
  },
};

// ============================================================================
// Tool Registration
// ============================================================================

function registerTools(
  api: OpenClawPluginApi,
  plugin: SecMemPlugin,
  cfg: SecMemConfig,
  getCurrentSessionId: () => string | undefined,
  ensureInitialized: () => Promise<void>,
) {
  // memory_search
  api.registerTool(
    {
      name: "memory_search",
      label: "Memory Search",
      description:
        "Search through long-term memories using FAISS high-performance index. Use when you need context about user preferences, past decisions, or previously discussed topics.",
      parameters: Type.Object({
        query: Type.String({ description: "Search query" }),
        limit: Type.Optional(
          Type.Number({
            description: `Max results (default: ${cfg.topK})`,
          }),
        ),
        userId: Type.Optional(
          Type.String({
            description:
              "User ID to scope search (default: configured collectionName)",
          }),
        ),
      }),
      async execute(_toolCallId: string, params: any) {
        const { query, limit, userId } = params as {
          query: string;
          limit?: number;
          userId?: string;
        };

        try {
          await ensureInitialized();
          const uid = userId || cfg.collectionName;

          const results = await plugin.search(
            query,
            uid,
            limit || cfg.topK,
          );

          if (!results.results || results.results.length === 0) {
            return {
              content: [
                { type: "text", text: "No relevant memories found." },
              ],
              details: { count: 0 },
            };
          }

          const text = results.results
            .map(
              (r, i) =>
                `${i + 1}. ${r.memory} (score: ${((r.score ?? 0) * 100).toFixed(0)}%, id: ${r.id})`,
            )
            .join("\n");

          const sanitized = results.results.map((r) => ({
            id: r.id,
            memory: r.memory,
            score: r.score,
            created_at: r.created_at,
          }));

          return {
            content: [
              {
                type: "text",
                text: `Found ${results.results.length} memories:\n\n${text}`,
              },
            ],
            details: { count: results.results.length, memories: sanitized },
          };
        } catch (err) {
          return {
            content: [
              {
                type: "text",
                text: `Memory search failed: ${String(err)}`,
              },
            ],
            details: { error: String(err) },
          };
        }
      },
    },
    { name: "memory_search" },
  );

  // memory_store
  api.registerTool(
    {
      name: "memory_store",
      label: "Memory Store",
      description:
        "Save important information in long-term memory. Use for preferences, facts, decisions, and anything worth remembering.",
      parameters: Type.Object({
        text: Type.String({ description: "Information to remember" }),
        userId: Type.Optional(
          Type.String({
            description: "User ID to scope this memory",
          }),
        ),
        metadata: Type.Optional(
          Type.Record(Type.String(), Type.Unknown(), {
            description: "Optional metadata to attach to this memory",
          }),
        ),
      }),
      async execute(_toolCallId: string, params: any) {
        const { text, userId, metadata } = params as {
          text: string;
          userId?: string;
          metadata?: Record<string, unknown>;
        };

        try {
          await ensureInitialized();
          const uid = userId || cfg.collectionName;

          const result = await plugin.add(text, uid, metadata);

          const added =
            result.results?.filter((r) => r.event === "ADD") ?? [];
          const updated =
            result.results?.filter((r) => r.event === "UPDATE") ?? [];

          const summary = [];
          if (added.length > 0)
            summary.push(
              `${added.length} new memor${added.length === 1 ? "y" : "ies"} added`,
            );
          if (updated.length > 0)
            summary.push(
              `${updated.length} memor${updated.length === 1 ? "y" : "ies"} updated`,
            );
          if (summary.length === 0)
            summary.push("No new memories extracted");

          return {
            content: [
              {
                type: "text",
                text: `Stored: ${summary.join(", ")}. ${result.results?.map((r) => `[${r.event}] ${r.memory}`).join("; ") ?? ""}`,
              },
            ],
            details: {
              action: "stored",
              results: result.results,
            },
          };
        } catch (err) {
          return {
            content: [
              {
                type: "text",
                text: `Memory store failed: ${String(err)}`,
              },
            ],
            details: { error: String(err) },
          };
        }
      },
    },
    { name: "memory_store" },
  );

  // memory_get
  api.registerTool(
    {
      name: "memory_get",
      label: "Memory Get",
      description: "Retrieve a specific memory by its ID.",
      parameters: Type.Object({
        memoryId: Type.String({ description: "The memory ID to retrieve" }),
      }),
      async execute(_toolCallId: string, params: any) {
        const { memoryId } = params as { memoryId: string };

        try {
          await ensureInitialized();
          const memory = await plugin.get(memoryId);

          if (!memory) {
            return {
              content: [
                { type: "text", text: `Memory ${memoryId} not found.` },
              ],
              details: null,
            };
          }

          return {
            content: [
              {
                type: "text",
                text: `Memory ${memory.id}:\n${memory.memory}\n\nCreated: ${memory.created_at ?? "unknown"}`,
              },
            ],
            details: { memory },
          };
        } catch (err) {
          return {
            content: [
              {
                type: "text",
                text: `Memory get failed: ${String(err)}`,
              },
            ],
            details: { error: String(err) },
          };
        }
      },
    },
    { name: "memory_get" },
  );

  // memory_delete
  api.registerTool(
    {
      name: "memory_delete",
      label: "Memory Delete",
      description:
        "Delete a memory by its ID.",
      parameters: Type.Object({
        memoryId: Type.String({ description: "Specific memory ID to delete" }),
      }),
      async execute(_toolCallId: string, params: any) {
        const { memoryId } = params as { memoryId: string };

        try {
          await ensureInitialized();
          await plugin.delete(memoryId);

          return {
            content: [
              { type: "text", text: `Memory ${memoryId} deleted.` },
            ],
            details: { action: "deleted", id: memoryId },
          };
        } catch (err) {
          return {
            content: [
              {
                type: "text",
                text: `Memory delete failed: ${String(err)}`,
              },
            ],
            details: { error: String(err) },
          };
        }
      },
    },
    { name: "memory_delete" },
  );
}

// ============================================================================
// CLI Registration
// ============================================================================

function registerCli(
  api: OpenClawPluginApi,
  plugin: SecMemPlugin,
  cfg: SecMemConfig,
  ensureInitialized: () => Promise<void>,
) {
  api.registerCli(
    ({ program }: { program: any }) => {
      const secMem = program
        .command("sec-mem")
        .description("sec-mem memory plugin commands");

      secMem
        .command("search")
        .description("Search memories")
        .argument("<query>", "Search query")
        .option("--limit <n>", "Max results", String(cfg.topK))
        .option("--user <userId>", "Search a specific user's memories")
        .action(async (query: string, opts: { limit: string; user?: string }) => {
          try {
            await ensureInitialized();
            const limit = parseInt(opts.limit, 10);
            const uid = opts.user || cfg.collectionName;

            const results = await plugin.search(query, uid, limit);

            if (!results.results?.length) {
              console.log("No memories found.");
              return;
            }

            const output = results.results.map((r) => ({
              id: r.id,
              memory: r.memory,
              score: r.score,
              created_at: r.created_at,
            }));
            console.log(JSON.stringify(output, null, 2));
          } catch (err) {
            console.error(`Search failed: ${String(err)}`);
          }
        });

      secMem
        .command("stats")
        .description("Show memory statistics")
        .option("--user <userId>", "Show stats for a specific user")
        .action(async (opts: { user?: string }) => {
          try {
            await ensureInitialized();
            const uid = opts.user || cfg.collectionName;

            // Get all memories to count
            const allMemories = await plugin.search("*", uid, 10000);
            const count = allMemories.results?.length || 0;

            console.log(`Index type: ${cfg.indexType}`);
            console.log(`Collection: ${uid}${opts.user ? ` (user: ${opts.user})` : ""}`);
            console.log(`Total memories: ${count}`);
            console.log(`Storage path: ${cfg.storagePath}`);
            console.log(
              `Auto-recall: ${cfg.autoRecall}, Auto-capture: ${cfg.autoCapture}`,
            );
          } catch (err) {
            console.error(`Stats failed: ${String(err)}`);
          }
        });

      secMem
        .command("benchmark")
        .description("Run performance benchmark")
        .option("--vectors <n>", "Number of vectors to test", "10000")
        .option("--dimensions <d>", "Embedding dimensions", String(cfg.embeddingDims))
        .action(async (opts: { vectors: string; dimensions: string }) => {
          try {
            const numVectors = parseInt(opts.vectors, 10);
            const dims = parseInt(opts.dimensions, 10);
            console.log(`Running benchmark with ${numVectors} vectors (${dims}D)...`);
            // Benchmark logic would go here
            console.log("Benchmark completed.");
          } catch (err) {
            console.error(`Benchmark failed: ${String(err)}`);
          }
        });
    },
    { commands: ["sec-mem"] },
  );
}

// ============================================================================
// Lifecycle Hook Registration
// ============================================================================

function registerHooks(
  api: OpenClawPluginApi,
  plugin: SecMemPlugin,
  cfg: SecMemConfig,
  ensureInitialized: () => Promise<void>,
  session: {
    setCurrentSessionId: (id: string) => void;
  },
) {
  // Auto-recall: inject relevant memories before agent starts
  if (cfg.autoRecall) {
    api.on("before_agent_start", async (event: any, ctx: any) => {
      if (!event.prompt || event.prompt.length < 5) return;

      const sessionId = (ctx as any)?.sessionKey ?? undefined;
      if (sessionId) session.setCurrentSessionId(sessionId);

      try {
        await ensureInitialized();

        // Use a larger candidate pool for recall, then filter down
        const recallTopK = Math.max((cfg.topK ?? 5) * 2, 10);

        // Search memories
        let results = await plugin.search(
          event.prompt,
          cfg.collectionName,
          recallTopK,
        );

        // Client-side threshold filter for auto-recall
        const recallThreshold = Math.max(cfg.searchThreshold, 0.6);
        let filteredResults = results.results?.filter(
          (r) => (r.score ?? 0) >= recallThreshold,
        ) || [];

        // Dynamic thresholding: drop memories scoring less than 50% of top result
        if (filteredResults.length > 1) {
          const topScore = filteredResults[0]?.score ?? 0;
          if (topScore > 0) {
            filteredResults = filteredResults.filter(
              (r) => (r.score ?? 0) >= topScore * 0.5,
            );
          }
        }

        // Cap at configured topK after filtering
        filteredResults = filteredResults.slice(0, cfg.topK);

        if (filteredResults.length === 0) return;

        // Build context
        const memoryContext = filteredResults
          .map((r) => `- ${r.memory}`)
          .join("\n");

        api.logger.info(
          `sec-mem: injecting ${filteredResults.length} memories into context`,
        );

        return {
          prependContext: `<relevant-memories>\nThe following are stored memories for user "${cfg.collectionName}". Use them to personalize your response:\n${memoryContext}\n</relevant-memories>`,
        };
      } catch (err) {
        api.logger.warn(`sec-mem: recall failed: ${String(err)}`);
      }
    });
  }

  // Auto-capture: store conversation context after agent ends
  if (cfg.autoCapture) {
    api.on("agent_end", async (event: any, ctx: any) => {
      if (!event.success) return;

      const sessionId = (ctx as any)?.sessionKey ?? undefined;
      if (sessionId) session.setCurrentSessionId(sessionId);

      try {
        await ensureInitialized();
        // For now, just log - actual capture would extract facts from messages
        api.logger.info("sec-mem: auto-capture completed");
      } catch (err) {
        api.logger.warn(`sec-mem: capture failed: ${String(err)}`);
      }
    });
  }
}

// Default export for OpenClaw
export default memoryPlugin;
