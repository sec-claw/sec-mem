#!/usr/bin/env python3
"""
Large-scale benchmark for sec-mem FAISS Advanced indexing.

Tests performance with 100K+ vectors across different index types:
- flat: Exact search baseline
- hnsw: Fast approximate search
- ivf: Balanced speed/accuracy
- ivf_pq: Compressed storage (94% savings)
- pq: Product quantization

Usage:
    python benchmark_large_scale.py [--vectors N] [--dimensions D] [--queries Q]
"""

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add sec_mem to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sec_mem.vector_stores.faiss_advanced import FAISSAdvanced


class LargeScaleBenchmark:
    """Large-scale FAISS benchmark."""
    
    def __init__(self, num_vectors: int, dimensions: int, num_queries: int = 100):
        self.num_vectors = num_vectors
        self.dimensions = dimensions
        self.num_queries = num_queries
        self.results: Dict[str, Dict] = {}
        
    def generate_data(self, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random vector data."""
        np.random.seed(seed)
        
        print(f"  Generating {self.num_vectors:,} vectors ({self.dimensions}D)...")
        vectors = np.random.randn(self.num_vectors, self.dimensions).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        print(f"  Generating {self.num_queries} query vectors...")
        queries = np.random.randn(self.num_queries, self.dimensions).astype('float32')
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        
        return vectors, queries
    
    def benchmark_index(
        self,
        index_type: str,
        vectors: np.ndarray,
        queries: np.ndarray,
        **index_kwargs
    ) -> Dict:
        """Benchmark a specific index type."""
        print(f"\n{'='*60}")
        print(f"Testing: {index_type.upper()}")
        print(f"{'='*60}")
        
        collection_name = f"bench_{index_type}_{self.num_vectors}"
        storage_path = f"./benchmark_data_{index_type}"
        
        # Cleanup previous benchmark data
        import shutil
        if os.path.exists(storage_path):
            shutil.rmtree(storage_path)
        
        # Create index
        start_time = time.time()
        store = FAISSAdvanced(
            collection_name=collection_name,
            index_type=index_type,
            embedding_model_dims=self.dimensions,
            path=storage_path,
            **index_kwargs
        )
        init_time = time.time() - start_time
        print(f"  Init time: {init_time:.2f}s")
        
        # Add vectors
        print(f"  Adding {self.num_vectors:,} vectors...")
        start_time = time.time()
        
        # Add in batches to avoid memory issues
        batch_size = min(10000, self.num_vectors)
        for i in range(0, self.num_vectors, batch_size):
            end_idx = min(i + batch_size, self.num_vectors)
            batch = vectors[i:end_idx]
            ids = [f"vec_{j}" for j in range(i, end_idx)]
            store.insert(vectors=batch.tolist(), ids=ids)
            if (i // batch_size) % 10 == 0:
                print(f"    Added {end_idx:,} / {self.num_vectors:,}")
        
        add_time = time.time() - start_time
        print(f"  Add time: {add_time:.2f}s ({self.num_vectors/add_time:,.0f} vectors/s)")
        
        # Get index info
        info = store.col_info()
        memory_mb = info.get('memory_estimate_mb', 0)
        print(f"  Memory usage: {memory_mb:.2f} MB")
        
        # Search benchmark
        print(f"  Running {self.num_queries} searches...")
        start_time = time.time()
        
        latencies = []
        for i, query in enumerate(queries):
            query_start = time.time()
            query_vec = query.reshape(1, -1).astype('float32')
            scores, indices = store.index.search(query_vec, 10)
            query_time = time.time() - query_start
            latencies.append(query_time * 1000)  # Convert to ms
            
            if (i + 1) % 20 == 0:
                print(f"    Completed {i + 1} / {self.num_queries}")
        
        total_search_time = time.time() - start_time
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        qps = self.num_queries / total_search_time
        
        print(f"  Search time: {total_search_time:.2f}s")
        print(f"  Throughput: {qps:,.0f} QPS")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  P95 latency: {p95_latency:.2f}ms")
        print(f"  P99 latency: {p99_latency:.2f}ms")
        
        # Save index to measure disk size
        store._save()
        
        # Get disk size
        disk_size_mb = 0
        for dirpath, dirnames, filenames in os.walk(storage_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                disk_size_mb += os.path.getsize(fp)
        disk_size_mb = disk_size_mb / (1024 * 1024)
        print(f"  Disk size: {disk_size_mb:.2f} MB")
        
        return {
            "index_type": index_type,
            "num_vectors": self.num_vectors,
            "dimensions": self.dimensions,
            "init_time_s": init_time,
            "add_time_s": add_time,
            "add_throughput": self.num_vectors / add_time,
            "search_time_s": total_search_time,
            "qps": qps,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "memory_mb": memory_mb,
            "disk_mb": disk_size_mb,
        }
    
    def run(self) -> Dict[str, Dict]:
        """Run full benchmark suite."""
        print("\n" + "="*60)
        print(f"Large-Scale FAISS Benchmark")
        print(f"Vectors: {self.num_vectors:,} | Dimensions: {self.dimensions}")
        print("="*60)
        
        vectors, queries = self.generate_data()
        
        # Test configurations
        configs = [
            ("flat", {}),
            ("hnsw", {"hnsw_m": 16, "hnsw_ef_search": 32}),
            ("ivf", {"nlist": min(100, self.num_vectors // 10), "nprobe": 10}),
        ]
        
        # Add IVF-PQ for larger datasets
        if self.num_vectors >= 10000:
            configs.append(("ivf_pq", {
                "nlist": min(100, self.num_vectors // 10),
                "nprobe": 10,
                "m": 16,
                "nbits": 8
            }))
        
        # Add PQ for very large datasets
        if self.num_vectors >= 50000:
            configs.append(("pq", {"m": 16, "nbits": 8}))
        
        for index_type, kwargs in configs:
            try:
                self.results[index_type] = self.benchmark_index(
                    index_type, vectors, queries, **kwargs
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                self.results[index_type] = {"error": str(e)}
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Filter out errors
        valid_results = {k: v for k, v in self.results.items() if "error" not in v}
        
        if not valid_results:
            print("No successful benchmarks!")
            return
        
        # Header
        print(f"\n{'Index Type':<12} {'Add (s)':<10} {'QPS':<10} {'Latency (ms)':<15} {'Memory (MB)':<12} {'Disk (MB)':<10}")
        print("-" * 80)
        
        # Sort by QPS
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1].get("qps", 0), reverse=True)
        
        for index_type, result in sorted_results:
            print(f"{index_type:<12} "
                  f"{result['add_time_s']:<10.1f} "
                  f"{result['qps']:<10,.0f} "
                  f"{result['avg_latency_ms']:<15.2f} "
                  f"{result['memory_mb']:<12.1f} "
                  f"{result['disk_mb']:<10.1f}")
        
        # Speedup comparison
        if "flat" in valid_results:
            print("\n" + "-" * 80)
            print("Speedup vs Flat (exact search):")
            flat_qps = valid_results["flat"]["qps"]
            flat_memory = valid_results["flat"]["memory_mb"]
            flat_disk = valid_results["flat"]["disk_mb"]
            
            for index_type, result in sorted_results:
                if index_type != "flat":
                    speedup = result["qps"] / flat_qps
                    memory_str = f"{(1 - result['memory_mb'] / flat_memory) * 100:.0f}%" if flat_memory > 0 else "N/A"
                    disk_reduction = (1 - result["disk_mb"] / flat_disk) * 100
                    print(f"  {index_type}: {speedup:.1f}x faster, {disk_reduction:.0f}% less disk, memory: {memory_str}")
        
        print("\n" + "="*80)
    
    def save_results(self, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            filename = f"benchmark_results_{self.num_vectors}.json"
        
        output = {
            "config": {
                "num_vectors": self.num_vectors,
                "dimensions": self.dimensions,
                "num_queries": self.num_queries,
            },
            "results": self.results
        }
        
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Large-scale FAISS benchmark")
    parser.add_argument("--vectors", type=int, default=100000, help="Number of vectors (default: 100000)")
    parser.add_argument("--dimensions", type=int, default=512, help="Vector dimensions (default: 512)")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries (default: 100)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Check if Ollama is running for embedding generation
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print("✓ Ollama is running")
    except:
        print("⚠ Ollama not detected at localhost:11434 - benchmark will use random vectors")
    
    benchmark = LargeScaleBenchmark(
        num_vectors=args.vectors,
        dimensions=args.dimensions,
        num_queries=args.queries
    )
    
    benchmark.run()
    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
