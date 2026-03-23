#!/usr/bin/env python3
"""
Quick benchmark for FAISS Advanced with Ollama.
Tests with smaller dataset for faster execution.
"""

import os
import sys
import time
import tempfile
import shutil
import numpy as np

# 确保能正确导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入，绕过 mem0 包结构
import importlib.util

# 加载 faiss_advanced 模块
spec = importlib.util.spec_from_file_location(
    "faiss_advanced", 
    os.path.join(os.path.dirname(__file__), "mem0/vector_stores/faiss_advanced.py")
)

# 设置必要的依赖模块
import types

# mem0 包结构
mem0 = types.ModuleType('mem0')
mem0.__version__ = "0.0.0"
sys.modules['mem0'] = mem0

# vector_stores 子包
vs = types.ModuleType('mem0.vector_stores')
vs.__path__ = []
sys.modules['mem0.vector_stores'] = vs

# base 模块
base_mod = types.ModuleType('mem0.vector_stores.base')
from abc import ABC, abstractmethod

class VectorStoreBase(ABC):
    @abstractmethod
    def create_col(self, name, vector_size, distance): pass
    @abstractmethod
    def insert(self, vectors, payloads=None, ids=None): pass
    @abstractmethod
    def search(self, query, vectors, limit=5, filters=None): pass
    @abstractmethod
    def delete(self, vector_id): pass
    @abstractmethod
    def update(self, vector_id, vector=None, payload=None): pass
    @abstractmethod
    def get(self, vector_id): pass
    @abstractmethod
    def list_cols(self): pass
    @abstractmethod
    def delete_col(self): pass
    @abstractmethod
    def col_info(self): pass
    @abstractmethod
    def list(self, filters=None, limit=None): pass
    @abstractmethod
    def reset(self): pass

base_mod.VectorStoreBase = VectorStoreBase
sys.modules['mem0.vector_stores.base'] = base_mod

# 现在加载 faiss_advanced
faiss_advanced = importlib.util.module_from_spec(spec)
spec.loader.exec_module(faiss_advanced)
FAISSAdvanced = faiss_advanced.FAISSAdvanced


def generate_random_vectors(n, dim):
    """Generate random normalized vectors."""
    vecs = np.random.randn(n, dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.tolist()


def benchmark_index(index_type, config, vectors, payloads, ids):
    """Benchmark a single index configuration."""
    print(f"\n{'='*50}")
    print(f"Testing: {index_type.upper()}")
    print(f"{'='*50}")
    
    test_dir = tempfile.mkdtemp(prefix=f"faiss_{index_type}_")
    config["path"] = test_dir
    
    try:
        # Create and populate index
        store = FAISSAdvanced(**config)
        
        start = time.time()
        store.insert(vectors, payloads, ids)
        build_time = time.time() - start
        
        # Benchmark search
        query_vec = generate_random_vectors(1, config["embedding_model_dims"])[0]
        
        # Warm up
        _ = store.search("test", [query_vec], limit=10)
        
        # Timed searches
        times = []
        for _ in range(20):
            start = time.perf_counter()
            results = store.search("test", [query_vec], limit=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        
        # Get info
        info = store.col_info()
        
        print(f"  Vectors:      {info['count']}")
        print(f"  Build time:   {build_time:.2f}s")
        print(f"  Search time:  {avg_time:.2f} ms (avg of 20)")
        print(f"  Throughput:   {1000/avg_time:.0f} queries/sec")
        
        # Estimate memory
        mem_size = 0
        try:
            import faiss
            with tempfile.NamedTemporaryFile(delete=False) as f:
                faiss.write_index(store.index, f.name)
                mem_size = os.path.getsize(f.name) / (1024 * 1024)
                os.remove(f.name)
        except:
            pass
        
        import pickle
        doc_size = len(pickle.dumps(store.docstore)) / (1024 * 1024)
        total_mem = mem_size + doc_size
        
        print(f"  Index size:   {mem_size:.2f} MB")
        print(f"  Docstore:     {doc_size:.2f} MB")
        print(f"  Total memory: {total_mem:.2f} MB")
        
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
        del store
        
        return {
            "index_type": index_type,
            "build_time": build_time,
            "search_ms": avg_time,
            "memory_mb": total_mem,
            "count": info['count'],
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(test_dir, ignore_errors=True)
        return None


def main():
    print("="*50)
    print("FAISS Advanced Quick Benchmark")
    print("="*50)
    
    # Configuration
    DIM = 512  # nomic-embed-text dimension
    N_VECTORS = 1000
    
    print(f"\nConfig:")
    print(f"  Vectors: {N_VECTORS}")
    print(f"  Dimension: {DIM}")
    
    # Generate test data
    print("\nGenerating test data...")
    vectors = generate_random_vectors(N_VECTORS, DIM)
    payloads = [{"data": f"Memory {i}", "user_id": "user1"} for i in range(N_VECTORS)]
    ids = [f"vec_{i}" for i in range(N_VECTORS)]
    
    # Index configurations
    configs = {
        "flat": {
            "collection_name": "test_flat",
            "embedding_model_dims": DIM,
            "index_type": "flat",
            "distance_strategy": "cosine",
        },
        "hnsw": {
            "collection_name": "test_hnsw",
            "embedding_model_dims": DIM,
            "index_type": "hnsw",
            "distance_strategy": "cosine",
            "hnsw_m": 16,
            "hnsw_ef_search": 32,
        },
        "ivf": {
            "collection_name": "test_ivf",
            "embedding_model_dims": DIM,
            "index_type": "ivf",
            "distance_strategy": "euclidean",
            "nlist": 100,
            "nprobe": 10,
            "auto_train_threshold": 1000,
        },
    }
    
    # Add IVF-PQ if compatible
    if DIM % 16 == 0:
        configs["ivf_pq"] = {
            "collection_name": "test_ivf_pq",
            "embedding_model_dims": DIM,
            "index_type": "ivf_pq",
            "distance_strategy": "euclidean",
            "nlist": 100,
            "nprobe": 10,
            "m": 16,
            "nbits": 8,
            "auto_train_threshold": 1000,
        }
    elif DIM % 8 == 0:
        configs["ivf_pq"] = {
            "collection_name": "test_ivf_pq",
            "embedding_model_dims": DIM,
            "index_type": "ivf_pq",
            "distance_strategy": "euclidean",
            "nlist": 100,
            "nprobe": 10,
            "m": 8,
            "nbits": 8,
            "auto_train_threshold": 1000,
        }
    
    # Run benchmarks
    results = []
    for name, config in configs.items():
        result = benchmark_index(name, config, vectors, payloads, ids)
        if result:
            results.append(result)
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Index':<12} {'Build (s)':<12} {'Search (ms)':<12} {'Memory (MB)':<12} {'QPS':<10}")
    print("-"*80)
    
    flat_result = None
    for r in results:
        if r["index_type"] == "flat":
            flat_result = r
        qps = 1000 / r["search_ms"] if r["search_ms"] > 0 else 0
        print(f"{r['index_type']:<12} {r['build_time']:<12.2f} {r['search_ms']:<12.2f} {r['memory_mb']:<12.2f} {qps:<10.0f}")
    
    print("="*80)
    
    # Performance comparison
    if flat_result:
        print("\nPerformance vs Flat Index (Original):")
        for r in results:
            if r["index_type"] == "flat":
                continue
            speedup = flat_result["search_ms"] / r["search_ms"]
            mem_ratio = r["memory_mb"] / flat_result["memory_mb"] if flat_result["memory_mb"] > 0 else 0
            print(f"  {r['index_type'].upper():10}: {speedup:.1f}x faster, {mem_ratio:.2f}x memory ({'+' if mem_ratio > 1 else ''}{(mem_ratio-1)*100:.0f}%)")
    
    print("\n✓ Benchmark complete!")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR YOUR SETUP")
    print("="*80)
    print("""
Based on benchmark results:

1. For FASTEST search (recommended for < 10K vectors):
   → Use HNSW index
   
2. For LARGE datasets (> 100K vectors):
   → Use IVF index
   
3. For MEMORY-CONSTRAINED environments:
   → Use IVF-PQ index (10-20x memory savings!)
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
