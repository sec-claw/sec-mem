/**
 * Shared type definitions for the OpenClaw Mem0 plugin.
 */

export type Mem0Mode = "platform" | "open-source";

/**
 * Configuration for FAISS Advanced vector store.
 * 
 * Available index types:
 * - flat: Exact search, most accurate but slowest (default)
 * - hnsw: Hierarchical Navigable Small World, fast approximate search (best for < 1M vectors)
 * - ivf: Inverted File Index, memory efficient for large datasets (best for > 1M vectors)
 * - ivf_pq: IVF with Product Quantization, highly compressed storage (best for memory-constrained)
 * - pq: Product Quantization only, extreme compression
 * 
 * Example configurations:
 * 
 * ```json
 * // HNSW for fast search (recommended for < 1M vectors)
 * {
 *   "provider": "faiss_advanced",
 *   "config": {
 *     "collection_name": "mem0",
 *     "index_type": "hnsw",
 *     "hnsw_m": 16,
 *     "hnsw_ef_construction": 64,
 *     "hnsw_ef_search": 32,
 *     "embedding_model_dims": 1536
 *   }
 * }
 * 
 * // IVF-PQ for compressed storage (recommended for memory-constrained environments)
 * {
 *   "provider": "faiss_advanced",
 *   "config": {
 *     "collection_name": "mem0",
 *     "index_type": "ivf_pq",
 *     "nlist": 100,
 *     "nprobe": 10,
 *     "m": 16,
 *     "nbits": 8,
 *     "embedding_model_dims": 1536
 *   }
 * }
 * ```
 */
export interface FAISSAdvancedConfig {
  provider: "faiss_advanced";
  config: {
    collection_name?: string;
    path?: string;
    distance_strategy?: "euclidean" | "inner_product" | "cosine";
    normalize_L2?: boolean;
    embedding_model_dims?: number;
    index_type?: "flat" | "hnsw" | "ivf" | "ivf_pq" | "pq";
    // HNSW parameters
    hnsw_m?: number;
    hnsw_ef_construction?: number;
    hnsw_ef_search?: number;
    // IVF parameters
    nlist?: number;
    nprobe?: number;
    // PQ parameters
    m?: number;
    nbits?: number;
    // Training parameters
    auto_train_threshold?: number;
    // GPU acceleration
    use_gpu?: boolean;
    gpu_id?: number;
  };
}

/**
 * Standard vector store configuration.
 */
export interface StandardVectorStoreConfig {
  provider: string;
  config: Record<string, unknown>;
}

export type VectorStoreConfig = StandardVectorStoreConfig | FAISSAdvancedConfig;

export type Mem0Config = {
  mode: Mem0Mode;
  // Platform-specific
  apiKey?: string;
  orgId?: string;
  projectId?: string;
  customInstructions: string;
  customCategories: Record<string, string>;
  enableGraph: boolean;
  // OSS-specific
  customPrompt?: string;
  oss?: {
    embedder?: { provider: string; config: Record<string, unknown> };
    vectorStore?: VectorStoreConfig;
    llm?: { provider: string; config: Record<string, unknown> };
    historyDbPath?: string;
    disableHistory?: boolean;
  };
  // Shared
  userId: string;
  autoCapture: boolean;
  autoRecall: boolean;
  searchThreshold: number;
  topK: number;
};

export interface AddOptions {
  user_id: string;
  run_id?: string;
  custom_instructions?: string;
  custom_categories?: Array<Record<string, string>>;
  enable_graph?: boolean;
  output_format?: string;
  source?: string;
}

export interface SearchOptions {
  user_id: string;
  run_id?: string;
  top_k?: number;
  threshold?: number;
  limit?: number;
  keyword_search?: boolean;
  reranking?: boolean;
  source?: string;
}

export interface ListOptions {
  user_id: string;
  run_id?: string;
  page_size?: number;
  source?: string;
}

export interface MemoryItem {
  id: string;
  memory: string;
  user_id?: string;
  score?: number;
  categories?: string[];
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface AddResultItem {
  id: string;
  memory: string;
  event: "ADD" | "UPDATE" | "DELETE" | "NOOP";
}

export interface AddResult {
  results: AddResultItem[];
}

export interface Mem0Provider {
  add(
    messages: Array<{ role: string; content: string }>,
    options: AddOptions,
  ): Promise<AddResult>;
  search(query: string, options: SearchOptions): Promise<MemoryItem[]>;
  get(memoryId: string): Promise<MemoryItem>;
  getAll(options: ListOptions): Promise<MemoryItem[]>;
  delete(memoryId: string): Promise<void>;
}
