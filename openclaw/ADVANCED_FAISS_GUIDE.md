# Advanced FAISS Configuration Guide

The `faiss_advanced` vector store provider offers optimized indexing strategies for different use cases, including HNSW for fast search and Product Quantization (PQ) for memory compression.

## Index Type Comparison

| Index Type | Best For | Memory Usage | Search Speed | Recall@10 |
|------------|----------|--------------|--------------|-----------|
| `flat` | Small datasets (< 100K) | 100% | Slowest | 100% |
| `hnsw` | Medium datasets (< 1M) | ~150% | Fast | ~95% |
| `ivf` | Large datasets (> 1M) | ~110% | Medium | ~90% |
| `ivf_pq` | Memory-constrained | ~10-25% | Medium | ~85% |
| `pq` | Extreme compression | ~5-10% | Medium | ~80% |

## Configuration Examples

### 1. HNSW (Recommended for Most Use Cases)

Best for datasets under 1 million vectors where search speed is important.

```json
{
  "mode": "open-source",
  "userId": "user",
  "oss": {
    "vectorStore": {
      "provider": "faiss_advanced",
      "config": {
        "collection_name": "mem0",
        "index_type": "hnsw",
        "embedding_model_dims": 1536,
        "distance_strategy": "cosine",
        "path": "./mem0_data",
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 32
      }
    }
  }
}
```

**Parameters:**
- `hnsw_m`: Number of neighbors per layer (default: 16, range: 2-64)
  - Higher = better recall, but slower build and more memory
  - 16-32 is good for most cases
  - 64 for maximum accuracy
- `hnsw_ef_construction`: Build-time search depth (default: 64)
  - Higher = better index quality
  - Should be >= hnsw_m
- `hnsw_ef_search`: Query-time search depth (default: 32)
  - Higher = better recall, but slower search
  - Can be adjusted per-query if needed

### 2. IVF-PQ (Memory-Constrained Environments)

Best when memory is limited and you have > 100K vectors.

```json
{
  "mode": "open-source",
  "userId": "user",
  "oss": {
    "vectorStore": {
      "provider": "faiss_advanced",
      "config": {
        "collection_name": "mem0",
        "index_type": "ivf_pq",
        "embedding_model_dims": 1536,
        "distance_strategy": "euclidean",
        "path": "./mem0_data",
        "nlist": 100,
        "nprobe": 10,
        "m": 16,
        "nbits": 8,
        "auto_train_threshold": 1000
      }
    }
  }
}
```

**Parameters:**
- `nlist`: Number of clusters (default: 100)
  - Should be roughly sqrt(expected_vectors)
  - 100 for up to 100K vectors
  - 1000 for up to 1M vectors
- `nprobe`: Clusters to search (default: 10)
  - Higher = better recall, slower search
  - 1/10th of nlist is a good starting point
- `m`: Number of subquantizers (default: 16)
  - Must divide embedding_model_dims
  - For 1536 dims: try 16, 12, or 8
  - Higher = better accuracy, less compression
- `nbits`: Bits per subquantizer (default: 8)
  - 8: Standard compression (256 centroids per subquantizer)
  - 4: Extreme compression (16 centroids per subquantizer)

**Memory Compression:**
- With `m=16`, `nbits=8`: ~16x compression (vectors use 16 bytes instead of 256)
- With `m=16`, `nbits=4`: ~32x compression (vectors use 8 bytes)

### 3. IVF Only (Large Datasets)

Best for > 1M vectors where memory is not as constrained.

```json
{
  "mode": "open-source",
  "userId": "user",
  "oss": {
    "vectorStore": {
      "provider": "faiss_advanced",
      "config": {
        "collection_name": "mem0",
        "index_type": "ivf",
        "embedding_model_dims": 1536,
        "nlist": 1000,
        "nprobe": 50
      }
    }
  }
}
```

### 4. PQ Only (Extreme Compression)

Best when memory is extremely limited.

```json
{
  "mode": "open-source",
  "userId": "user",
  "oss": {
    "vectorStore": {
      "provider": "faiss_advanced",
      "config": {
        "collection_name": "mem0",
        "index_type": "pq",
        "embedding_model_dims": 1536,
        "m": 16,
        "nbits": 4
      }
    }
  }
}
```

## Choosing the Right Index

### Decision Flowchart

```
Number of vectors?
├── < 100K
│   └── Use hnsw (fast, simple)
├── 100K - 1M
│   └── Memory constrained?
│       ├── Yes → Use ivf_pq
│       └── No → Use hnsw
└── > 1M
    └── Memory constrained?
        ├── Yes → Use ivf_pq with higher nlist
        └── No → Use ivf
```

### Common Scenarios

| Scenario | Recommended Index | Why |
|----------|------------------|-----|
| Personal assistant (< 50K memories) | `hnsw` | Fast, accurate, simple |
| Team collaboration (< 500K) | `hnsw` or `ivf` | Balance of speed and memory |
| Enterprise (1M+) | `ivf` | Scalable, memory efficient |
| Edge device / limited RAM | `ivf_pq` | 10-20x memory reduction |
| Archive / long-term storage | `ivf_pq` with nbits=4 | Maximum compression |

## GPU Acceleration

For even faster search on compatible hardware:

```json
{
  "provider": "faiss_advanced",
  "config": {
    "collection_name": "mem0",
    "index_type": "hnsw",
    "use_gpu": true,
    "gpu_id": 0
  }
}
```

**Note:** GPU acceleration requires `faiss-gpu` package.

## Migration from Standard FAISS

To migrate from the standard `faiss` provider to `faiss_advanced`:

1. Change provider name:
   ```json
   "vectorStore": {
     "provider": "faiss_advanced",
     "config": {
       "collection_name": "mem0",
       "index_type": "flat"  // Same behavior as standard faiss
     }
   }
   ```

2. The data format is compatible - your existing memories will be loaded.

3. To optimize existing data, use the `rebuild_index()` API (if available in your client).

## Performance Tuning Tips

1. **Start with defaults**: The default parameters work well for most cases.

2. **Tune nprobe for IVF**: If recall is too low, increase `nprobe`. If search is too slow, decrease it.

3. **Tune efSearch for HNSW**: Similar to nprobe - increase for better recall, decrease for speed.

4. **Monitor training**: IVF and PQ indexes require training. The index will collect data until `auto_train_threshold` vectors are available, then automatically train.

5. **Distance strategy**:
   - Use `cosine` or `inner_product` for semantic similarity (most embedding models)
   - Use `euclidean` only if specifically needed

## Troubleshooting

### "Index not trained" warning
- Normal for IVF/PQ indexes on first startup
- The index will auto-train when enough data is collected
- Or manually trigger training by inserting `auto_train_threshold` vectors

### High memory usage with HNSW
- HNSW uses more memory than flat for small datasets (< 10K)
- Benefits only appear at larger scales
- Consider `flat` for very small datasets

### Low recall with IVF-PQ
- Increase `nprobe` (IVF) or decrease `m` (PQ)
- Higher values = better accuracy, slower search

### Slow search after migration
- PQ-based indexes are optimized for memory, not speed
- Consider `hnsw` if speed is more important than memory
