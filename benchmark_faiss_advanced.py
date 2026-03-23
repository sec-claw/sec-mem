#!/usr/bin/env python3
"""
Benchmark script for FAISS Advanced vector store with Ollama.
Compares different index types: flat, hnsw, ivf, ivf_pq

Requirements:
    pip install faiss-cpu ollama sec_memai

Ollama models needed:
    - Embedding: nomic-embed-text (default) or others
    - LLM: qwen3:4b-instruct-2507-q4_K_M (or your preferred model)

Usage:
    python benchmark_faiss_advanced.py
"""

import os
import sys
import time
import tempfile
import shutil
import json
import psutil
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check dependencies
try:
    import faiss
except ImportError:
    print("❌ FAISS not installed. Run: pip install faiss-cpu")
    sys.exit(1)

try:
    from ollama import Client
except ImportError:
    print("❌ Ollama client not installed. Run: pip install ollama")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("⚠️  psutil not installed. Memory monitoring disabled. Run: pip install psutil")
    psutil = None


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""
    index_type: str
    num_vectors: int
    embedding_dim: int
    
    # Timing (seconds)
    index_build_time: float = 0.0
    avg_search_time_ms: float = 0.0
    
    # Memory (MB)
    memory_usage_mb: float = 0.0
    
    # Quality
    recall_at_10: float = 0.0  # vs flat index
    
    # Throughput
    vectors_per_second: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class OllamaEmbedding:
    """Simple Ollama embedding wrapper."""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.client = Client(host=base_url)
        self._ensure_model()
        
    def _ensure_model(self):
        """Check if model exists."""
        try:
            models = self.client.list()["models"]
            model_names = [m.get("name", m.get("model", "")) for m in models]
            if self.model not in model_names and f"{self.model}:latest" not in model_names:
                print(f"⚠️  Model {self.model} not found. Attempting to pull...")
                self.client.pull(self.model)
                print(f"✓ Model {self.model} pulled successfully")
        except Exception as e:
            print(f"⚠️  Could not verify model: {e}")
    
    def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        response = self.client.embed(model=self.model, input=text)
        embeddings = response.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        raise ValueError(f"No embeddings returned for: {text[:50]}...")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        # Ollama embed API supports batch
        response = self.client.embed(model=self.model, input=texts)
        return response.get("embeddings", [])


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    if psutil is None:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def generate_test_memories(count: int) -> List[Dict[str, str]]:
    """Generate diverse test memories."""
    templates = [
        "User prefers {language} for {task} projects",
        "User likes to work during {time_of_day}",
        "User's favorite {category} is {item}",
        "User prefers {style} communication style",
        "User uses {tool} for {purpose}",
        "User lives in {city}, {country}",
        "User works as a {job_title} at {company}",
        "User enjoys {hobby} in free time",
        "User prefers {food} over {other_food}",
        "User's goal is to {goal} by {timeframe}",
    ]
    
    fillers = {
        "language": ["Python", "JavaScript", "Rust", "Go", "TypeScript", "C++", "Java", "Ruby"],
        "task": ["backend", "frontend", "data science", "ML", "automation", "scripting"],
        "time_of_day": ["morning", "afternoon", "evening", "late night"],
        "category": ["color", "movie genre", "music genre", "book genre"],
        "item": ["blue", "sci-fi", "jazz", "mystery", "red", "action", "rock", "fantasy"],
        "style": ["concise", "detailed", "technical", "conversational"],
        "tool": ["VS Code", "Vim", "Emacs", "PyCharm", "Cursor", "Zed"],
        "purpose": ["coding", "debugging", "testing", "deployment"],
        "city": ["Beijing", "Shanghai", "Shenzhen", "Hangzhou", "Chengdu", "Guangzhou"],
        "country": ["China", "USA", "UK", "Japan", "Singapore", "Germany"],
        "job_title": ["engineer", "manager", "director", "architect", "developer", "researcher"],
        "company": ["TechCorp", "StartupXYZ", "BigTech", "Freelance", "Research Lab"],
        "hobby": ["hiking", "reading", "gaming", "cooking", "traveling", "photography"],
        "food": ["Italian", "Chinese", "Japanese", "Mexican", "Indian", "Thai"],
        "other_food": ["fast food", "processed food", "junk food"],
        "goal": ["learn Rust", "build a startup", "get promoted", "publish a paper"],
        "timeframe": ["end of year", "next quarter", "2026", "next month"],
    }
    
    memories = []
    for i in range(count):
        template = templates[i % len(templates)]
        memory = template.format(**{k: np.random.choice(v) for k, v in fillers.items()})
        memories.append({
            "id": f"mem_{i}",
            "text": memory,
            "metadata": {"user_id": "test_user", "index": i}
        })
    
    return memories


def benchmark_index(
    index_type: str,
    config: Dict[str, Any],
    memories: List[Dict],
    embedder: OllamaEmbedding,
    flat_results: List[Tuple[str, float]] = None,
) -> BenchmarkResult:
    """Benchmark a single index type."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {index_type.upper()}")
    print(f"{'='*60}")
    
    dim = config["embedding_model_dims"]
    result = BenchmarkResult(
        index_type=index_type,
        num_vectors=len(memories),
        embedding_dim=dim,
    )
    
    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix=f"faiss_{index_type}_")
    config["path"] = test_dir
    
    try:
        # Import FAISSAdvanced
        from sec_mem.vector_stores.faiss_advanced import FAISSAdvanced
        
        # Measure initial memory
        mem_before = get_memory_usage()
        
        # Create index
        print(f"Creating {index_type} index...")
        start_time = time.time()
        store = FAISSAdvanced(**config)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(memories)} memories...")
        texts = [m["text"] for m in memories]
        
        # Batch embedding
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                embeddings = embedder.embed_batch(batch)
                all_embeddings.extend(embeddings)
                print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)}", end="\r")
            except Exception as e:
                print(f"\n⚠️  Batch embedding failed, falling back to single: {e}")
                for text in batch:
                    all_embeddings.append(embedder.embed(text))
        print()
        
        # Insert into index
        print("Inserting vectors...")
        vectors = all_embeddings
        payloads = [m["metadata"] for m in memories]
        ids = [m["id"] for m in memories]
        
        insert_start = time.time()
        store.insert(vectors, payloads, ids)
        insert_time = time.time() - insert_start
        
        build_time = time.time() - start_time
        result.index_build_time = build_time
        result.vectors_per_second = len(memories) / build_time if build_time > 0 else 0
        
        # Measure memory
        mem_after = get_memory_usage()
        result.memory_usage_mb = mem_after - mem_before
        
        print(f"✓ Index built in {build_time:.2f}s ({result.vectors_per_second:.1f} vec/s)")
        print(f"✓ Memory usage: {result.memory_usage_mb:.2f} MB")
        
        # Get index info
        info = store.col_info()
        print(f"✓ Index info: {json.dumps(info, indent=2, default=str)}")
        
        # Benchmark search
        print("\nBenchmarking search...")
        
        # Use a few test queries
        test_queries = [
            "What programming languages does the user prefer?",
            "Where does the user live?",
            "What are the user's hobbies?",
            "What tools does the user use?",
            "What are the user's goals?",
        ]
        
        search_times = []
        all_results = []
        
        for query in test_queries:
            query_vec = embedder.embed(query)
            
            # Warm up
            _ = store.search(query, [query_vec], limit=10)
            
            # Timed search
            times = []
            for _ in range(5):  # 5 iterations
                start = time.perf_counter()
                results = store.search(query, [query_vec], limit=10)
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
            
            avg_time = np.mean(times[1:])  # Exclude first (cache)
            search_times.append(avg_time)
            all_results.append((query, results))
        
        result.avg_search_time_ms = np.mean(search_times)
        print(f"✓ Average search time: {result.avg_search_time_ms:.2f} ms")
        
        # Calculate recall if flat results provided
        if flat_results is not None and index_type != "flat":
            recalls = []
            for (query, results), (flat_query, flat_res) in zip(all_results, flat_results):
                if len(results) == 0 or len(flat_res) == 0:
                    continue
                flat_ids = {r.id for r in flat_res[:10]}
                retrieved_ids = {r.id for r in results[:10]}
                intersection = flat_ids & retrieved_ids
                recall = len(intersection) / len(flat_ids) if flat_ids else 0
                recalls.append(recall)
            
            result.recall_at_10 = np.mean(recalls) if recalls else 0
            print(f"✓ Recall@10 (vs flat): {result.recall_at_10:.2%}")
        
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
        del store
        
        return result, all_results if index_type == "flat" else None
        
    except Exception as e:
        print(f"❌ Error benchmarking {index_type}: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(test_dir, ignore_errors=True)
        return result, None


def print_comparison_table(results: List[BenchmarkResult]):
    """Print a nice comparison table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*100)
    
    # Header
    print(f"{'Index Type':<15} {'Vectors':<10} {'Build Time':<12} {'Search (ms)':<12} {'Memory (MB)':<12} {'Recall@10':<10} {'Vec/s':<10}")
    print("-"*100)
    
    # Rows
    for r in results:
        print(f"{r.index_type:<15} {r.num_vectors:<10} {r.index_build_time:<12.2f} {r.avg_search_time_ms:<12.2f} {r.memory_usage_mb:<12.2f} {r.recall_at_10:<10.1%} {r.vectors_per_second:<10.1f}")
    
    print("="*100)
    
    # Find best for each metric
    if len(results) > 1:
        print("\n🏆 BEST PERFORMERS:")
        
        # Fastest search (excluding flat)
        approx_results = [r for r in results if r.index_type != "flat"]
        if approx_results:
            fastest = min(approx_results, key=lambda x: x.avg_search_time_ms)
            print(f"   Fastest Search:    {fastest.index_type.upper()} ({fastest.avg_search_time_ms:.2f} ms)")
        
        # Lowest memory
        smallest = min(results, key=lambda x: x.memory_usage_mb)
        print(f"   Lowest Memory:     {smallest.index_type.upper()} ({smallest.memory_usage_mb:.2f} MB)")
        
        # Best recall
        if approx_results:
            best_recall = max(approx_results, key=lambda x: x.recall_at_10)
            print(f"   Best Recall:       {best_recall.index_type.upper()} ({best_recall.recall_at_10:.1%})")
        
        # Fastest build
        fastest_build = min(results, key=lambda x: x.index_build_time)
        print(f"   Fastest Build:     {fastest_build.index_type.upper()} ({fastest_build.index_build_time:.2f}s)")


def main():
    """Run the benchmark."""
    print("="*60)
    print("FAISS Advanced Benchmark with Ollama")
    print("="*60)
    
    # Configuration
    OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    
    print(f"\nConfiguration:")
    print(f"  Ollama URL:     {OLLAMA_URL}")
    print(f"  Embedding:      {EMBEDDING_MODEL}")
    print(f"  LLM:            {LLM_MODEL}")
    
    # Check Ollama connection
    print("\nChecking Ollama connection...")
    try:
        client = Client(host=OLLAMA_URL)
        models = client.list()["models"]
        print(f"✓ Connected to Ollama")
        print(f"  Available models: {len(models)}")
        for m in models[:5]:
            print(f"    - {m.get('name', m.get('model', 'unknown'))}")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return 1
    
    # Initialize embedder
    print(f"\nInitializing embedding model: {EMBEDDING_MODEL}")
    try:
        embedder = OllamaEmbedding(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
        # Test embedding
        test_emb = embedder.embed("Hello, world!")
        EMBEDDING_DIM = len(test_emb)
        print(f"✓ Embedding dimension: {EMBEDDING_DIM}")
    except Exception as e:
        print(f"❌ Failed to initialize embedding: {e}")
        return 1
    
    # Test sizes
    TEST_SIZES = [100, 500, 1000]
    
    # Index configurations
    INDEX_CONFIGS = {
        "flat": {
            "collection_name": "benchmark_flat",
            "embedding_model_dims": EMBEDDING_DIM,
            "index_type": "flat",
            "distance_strategy": "cosine",
        },
        "hnsw": {
            "collection_name": "benchmark_hnsw",
            "embedding_model_dims": EMBEDDING_DIM,
            "index_type": "hnsw",
            "distance_strategy": "cosine",
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 32,
        },
        "ivf": {
            "collection_name": "benchmark_ivf",
            "embedding_model_dims": EMBEDDING_DIM,
            "index_type": "ivf",
            "distance_strategy": "euclidean",
            "nlist": 100,
            "nprobe": 10,
            "auto_train_threshold": 1000,
        },
    }
    
    # Add IVF-PQ if dimensions are compatible
    if EMBEDDING_DIM % 16 == 0:
        INDEX_CONFIGS["ivf_pq"] = {
            "collection_name": "benchmark_ivf_pq",
            "embedding_model_dims": EMBEDDING_DIM,
            "index_type": "ivf_pq",
            "distance_strategy": "euclidean",
            "nlist": 100,
            "nprobe": 10,
            "m": 16,
            "nbits": 8,
            "auto_train_threshold": 1000,
        }
    else:
        print(f"\n⚠️  EMBEDDING_DIM ({EMBEDDING_DIM}) not divisible by 16, skipping IVF-PQ")
        # Try with m=8
        if EMBEDDING_DIM % 8 == 0:
            INDEX_CONFIGS["ivf_pq"] = {
                "collection_name": "benchmark_ivf_pq",
                "embedding_model_dims": EMBEDDING_DIM,
                "index_type": "ivf_pq",
                "distance_strategy": "euclidean",
                "nlist": 100,
                "nprobe": 10,
                "m": 8,
                "nbits": 8,
                "auto_train_threshold": 1000,
            }
    
    # Run benchmarks for each size
    all_results = {}
    
    for size in TEST_SIZES:
        print(f"\n\n{'#'*60}")
        print(f"# BENCHMARKING WITH {size} VECTORS")
        print(f"{'#'*60}")
        
        # Generate test data
        print(f"\nGenerating {size} test memories...")
        memories = generate_test_memories(size)
        
        results = []
        flat_results_cache = None
        
        for index_type, config in INDEX_CONFIGS.items():
            result, search_results = benchmark_index(
                index_type=index_type,
                config=config.copy(),
                memories=memories,
                embedder=embedder,
                flat_results=flat_results_cache,
            )
            results.append(result)
            
            # Cache flat results for recall calculation
            if index_type == "flat" and search_results:
                flat_results_cache = search_results
        
        all_results[size] = results
        
        # Print comparison for this size
        print_comparison_table(results)
    
    # Final summary
    print("\n\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    
    for size, results in all_results.items():
        print(f"\n{size} vectors:")
        flat_result = next((r for r in results if r.index_type == "flat"), None)
        
        for r in results:
            if r.index_type == "flat":
                continue
            
            speedup = flat_result.avg_search_time_ms / r.avg_search_time_ms if flat_result else 0
            mem_ratio = r.memory_usage_mb / flat_result.memory_usage_mb if flat_result and flat_result.memory_usage_mb > 0 else 0
            
            print(f"  {r.index_type.upper():10}:")
            print(f"    Search speedup: {speedup:.2f}x vs flat")
            print(f"    Memory ratio:   {mem_ratio:.2f}x of flat")
            print(f"    Recall@10:      {r.recall_at_10:.1%}")
    
    # Recommendations
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)
    
    print("""
Based on your benchmark results:

For SMALL datasets (< 10K vectors):
  → Use HNSW for best speed/recall tradeoff
  
For MEDIUM datasets (10K - 100K vectors):
  → Use HNSW if memory permits
  → Use IVF if memory is constrained
  
For LARGE datasets (> 100K vectors):
  → Use IVF for balanced performance
  → Use IVF-PQ for memory-constrained environments
  
For MEMORY-CONSTRAINED environments:
  → Use IVF-PQ with higher 'm' value for better recall
  → Or use IVF-PQ with nbits=4 for extreme compression
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
