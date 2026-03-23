/**
 * sec-mem: High-performance local memory storage plugin for OpenClaw
 * 
 * Features:
 * - FAISS Advanced indexing (HNSW, IVF, IVF-PQ)
 * - Ollama native support
 * - 5.9x faster search, 94% memory savings
 */

import { z } from "zod";
import { spawn, ChildProcess } from "child_process";
import { join } from "path";

// Plugin configuration schema
export const SecMemConfigSchema = z.object({
  // Vector store configuration
  collectionName: z.string().default("sec-mem"),
  indexType: z.enum(["flat", "hnsw", "ivf", "ivf_pq", "pq"]).default("hnsw"),
  embeddingDims: z.number().default(512),
  distanceStrategy: z.enum(["euclidean", "inner_product", "cosine"]).default("cosine"),
  
  // HNSW params
  hnswM: z.number().default(16),
  hnswEfConstruction: z.number().default(64),
  hnswEfSearch: z.number().default(32),
  
  // IVF params
  nlist: z.number().default(100),
  nprobe: z.number().default(10),
  
  // PQ params
  m: z.number().default(16),
  nbits: z.number().default(8),
  
  // Storage path
  storagePath: z.string().default("./sec-mem-data"),
  
  // Ollama configuration
  ollamaUrl: z.string().default("http://localhost:11434"),
  ollamaEmbedModel: z.string().default("nomic-embed-text"),
  ollamaLlmModel: z.string().optional(),
  
  // Feature flags
  autoCapture: z.boolean().default(true),
  autoRecall: z.boolean().default(true),
  topK: z.number().default(5),
  searchThreshold: z.number().default(0.5),
});

export type SecMemConfig = z.infer<typeof SecMemConfigSchema>;

// Memory item interface
export interface MemoryItem {
  id: string;
  memory: string;
  score?: number;
  metadata?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

// Search result interface
export interface SearchResult {
  results: MemoryItem[];
}

// Add result item
export interface AddResultItem {
  id: string;
  memory: string;
  event: "ADD" | "UPDATE" | "DELETE" | "NOOP";
}

// Add result
export interface AddResult {
  results: AddResultItem[];
}

/**
 * sec-mem plugin for OpenClaw
 */
export class SecMemPlugin {
  private config: SecMemConfig;
  private pythonProcess: ChildProcess | null = null;
  private ready: boolean = false;
  private messageQueue: Array<{ resolve: Function; reject: Function; id: string }> = [];
  private messageId: number = 0;

  constructor(config: Partial<SecMemConfig> = {}) {
    this.config = SecMemConfigSchema.parse(config);
  }

  /**
   * Initialize the plugin and start Python backend
   */
  async initialize(): Promise<void> {
    const scriptPath = join(__dirname, "..", "sec_mem", "plugin_server.py");
    
    this.pythonProcess = spawn("python3", [
      scriptPath,
      JSON.stringify(this.config)
    ], {
      stdio: ["pipe", "pipe", "pipe"]
    });

    // Wait for ready signal
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Plugin initialization timeout"));
      }, 30000);

      this.pythonProcess?.stdout?.on("data", (data) => {
        const line = data.toString().trim();
        if (line.includes("PLUGIN_READY")) {
          clearTimeout(timeout);
          this.ready = true;
          resolve();
        }
      });

      this.pythonProcess?.stderr?.on("data", (data) => {
        console.error(`[sec-mem] ${data.toString().trim()}`);
      });

      this.pythonProcess?.on("error", (err) => {
        clearTimeout(timeout);
        reject(err);
      });

      this.pythonProcess?.on("exit", (code) => {
        if (code !== 0 && !this.ready) {
          clearTimeout(timeout);
          reject(new Error(`Python process exited with code ${code}`));
        }
      });
    });
  }

  /**
   * Add a memory
   */
  async add(
    message: string,
    userId: string = "default",
    metadata?: Record<string, any>
  ): Promise<AddResult> {
    return this.sendCommand("add", { message, userId, metadata });
  }

  /**
   * Search memories
   */
  async search(
    query: string,
    userId: string = "default",
    limit: number = 5
  ): Promise<SearchResult> {
    return this.sendCommand("search", { query, userId, limit });
  }

  /**
   * Get a memory by ID
   */
  async get(memoryId: string): Promise<MemoryItem | null> {
    return this.sendCommand("get", { memoryId });
  }

  /**
   * Get all memories for a user
   */
  async getAll(userId: string = "default", limit: number = 100): Promise<SearchResult> {
    return this.sendCommand("get_all", { userId, limit });
  }

  /**
   * Update a memory
   */
  async update(memoryId: string, data: string): Promise<{ message: string }> {
    return this.sendCommand("update", { memoryId, data });
  }

  /**
   * Delete a memory
   */
  async delete(memoryId: string): Promise<void> {
    await this.sendCommand("delete", { memoryId });
  }

  /**
   * Get index statistics
   */
  async stats(): Promise<{
    indexType: string;
    numVectors: number;
    memoryUsage: number;
  }> {
    return this.sendCommand("stats", {});
  }

  /**
   * Shutdown the plugin
   */
  async shutdown(): Promise<void> {
    if (this.pythonProcess) {
      this.pythonProcess.kill();
      this.pythonProcess = null;
      this.ready = false;
    }
  }

  /**
   * Send command to Python backend
   */
  private sendCommand<T>(command: string, params: any): Promise<T> {
    return new Promise((resolve, reject) => {
      if (!this.ready || !this.pythonProcess) {
        reject(new Error("Plugin not initialized"));
        return;
      }

      const id = `msg_${++this.messageId}`;
      const message = JSON.stringify({ id, command, params }) + "\n";

      this.messageQueue.push({ resolve, reject, id });

      const handleResponse = (data: Buffer) => {
        const lines = data.toString().split("\n").filter(Boolean);
        
        for (const line of lines) {
          try {
            const response = JSON.parse(line);
            const pending = this.messageQueue.find(m => m.id === response.id);
            
            if (pending) {
              this.messageQueue = this.messageQueue.filter(m => m.id !== response.id);
              
              if (response.error) {
                pending.reject(new Error(response.error));
              } else {
                pending.resolve(response.result);
              }
            }
          } catch (e) {
            // Ignore non-JSON lines
          }
        }
      };

      this.pythonProcess.stdout?.once("data", handleResponse);
      this.pythonProcess.stdin?.write(message);
    });
  }
}

// Export for OpenClaw plugin registration
export default SecMemPlugin;
