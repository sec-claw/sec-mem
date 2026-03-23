"""
Advanced FAISS vector store with support for:
- IVF (Inverted File Index): Efficient for large-scale data
- HNSW (Hierarchical Navigable Small World): Fast approximate search
- PQ (Product Quantization): Memory compression
- IVF-PQ: Combined approach for large-scale compressed storage

Usage:
    # HNSW for fast search (recommended for < 1M vectors)
    config = {
        "collection_name": "mem0",
        "index_type": "hnsw",
        "embedding_model_dims": 1536,
        "path": "/path/to/storage"
    }
    
    # IVF for large-scale data (recommended for > 1M vectors)
    config = {
        "collection_name": "mem0",
        "index_type": "ivf",
        "nlist": 100,  # Number of clusters
        "embedding_model_dims": 1536,
        "path": "/path/to/storage"
    }
    
    # IVF-PQ for compressed storage (recommended for memory-constrained environments)
    config = {
        "collection_name": "mem0",
        "index_type": "ivf_pq",
        "nlist": 100,
        "m": 16,  # Number of subquantizers
        "nbits": 8,  # Bits per subquantizer
        "embedding_model_dims": 1536,
        "path": "/path/to/storage"
    }
"""

import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from pydantic import BaseModel

import warnings

# Suppress FAISS warnings
try:
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")
    
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)
    
    import faiss
except ImportError:
    raise ImportError(
        "Could not import faiss python package. "
        "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
        "or `pip install faiss-cpu` (depending on Python version)."
    )

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[Dict]


class FAISSAdvanced(VectorStoreBase):
    """
    Advanced FAISS vector store supporting multiple index types:
    - flat: Exact search (default, slowest, most accurate)
    - hnsw: Hierarchical Navigable Small World (fast, good accuracy)
    - ivf: Inverted File Index (memory efficient for large datasets)
    - ivf_pq: IVF with Product Quantization (highly compressed)
    - pq: Product Quantization only (extreme compression)
    """
    
    SUPPORTED_INDEX_TYPES = ["flat", "hnsw", "ivf", "ivf_pq", "pq"]
    
    def __init__(
        self,
        collection_name: str,
        path: Optional[str] = None,
        distance_strategy: str = "euclidean",
        normalize_L2: bool = False,
        embedding_model_dims: int = 1536,
        index_type: str = "flat",
        # HNSW parameters
        hnsw_m: int = 16,  # Number of neighbors per layer
        hnsw_ef_construction: int = 64,  # Build-time search depth
        hnsw_ef_search: int = 32,  # Query-time search depth
        # IVF parameters
        nlist: int = 100,  # Number of clusters (should be ~sqrt(n_vectors))
        nprobe: int = 10,  # Number of clusters to search
        # PQ parameters
        m: int = 16,  # Number of subquantizers (must divide embedding_model_dims)
        nbits: int = 8,  # Bits per subquantizer
        # Auto-training parameters
        auto_train_threshold: int = 1000,  # Min vectors before training IVF/PQ
        # Performance tuning
        use_gpu: bool = False,  # Use GPU if available
        gpu_id: int = 0,
    ):
        """
        Initialize the advanced FAISS vector store.
        
        Args:
            collection_name: Name of the collection
            path: Path for local storage
            distance_strategy: "euclidean", "inner_product", or "cosine"
            normalize_L2: Whether to L2 normalize vectors
            embedding_model_dims: Dimension of embedding vectors
            index_type: "flat", "hnsw", "ivf", "ivf_pq", or "pq"
            
            HNSW params:
                hnsw_m: Number of neighbors (higher = more accurate, slower)
                hnsw_ef_construction: Build-time accuracy (higher = better index)
                hnsw_ef_search: Query-time accuracy (higher = better recall)
            
            IVF params:
                nlist: Number of clusters (sqrt of expected vectors is good)
                nprobe: Clusters to search (higher = better recall, slower)
            
            PQ params:
                m: Subquantizers (must divide embedding_model_dims)
                nbits: Bits per code (8 is standard, 4 for extreme compression)
        """
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValueError(f"index_type must be one of {self.SUPPORTED_INDEX_TYPES}")
        
        if index_type in ["pq", "ivf_pq"] and embedding_model_dims % m != 0:
            raise ValueError(f"embedding_model_dims ({embedding_model_dims}) must be divisible by m ({m})")
        
        self.collection_name = collection_name
        self.path = path or f"/tmp/faiss/{collection_name}"
        self.distance_strategy = distance_strategy.lower()
        self.normalize_L2 = normalize_L2
        self.embedding_model_dims = embedding_model_dims
        self.index_type = index_type
        
        # HNSW params
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        
        # IVF params
        self.nlist = nlist
        self.nprobe = nprobe
        
        # PQ params
        self.m = m
        self.nbits = nbits
        
        self.auto_train_threshold = auto_train_threshold
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        
        # Storage structures
        self.index = None
        self.docstore: Dict[str, Dict] = {}
        self.index_to_id: Dict[int, str] = {}
        self.id_to_index: Dict[str, int] = {}
        self.is_trained = False
        self._training_data: List[np.ndarray] = []
        
        # Create directory
        os.makedirs(self.path, exist_ok=True)
        
        # Load existing or create new
        index_path = f"{self.path}/{collection_name}.faiss"
        docstore_path = f"{self.path}/{collection_name}.pkl"
        meta_path = f"{self.path}/{collection_name}_meta.pkl"
        
        if os.path.exists(index_path) and os.path.exists(docstore_path):
            self._load(index_path, docstore_path, meta_path)
        else:
            self._create_index()
    
    def _create_index(self) -> None:
        """Create a new FAISS index based on configuration."""
        metric = self._get_metric()
        d = self.embedding_model_dims
        
        if self.index_type == "flat":
            # Exact search - most accurate, slowest
            if metric == faiss.METRIC_INNER_PRODUCT:
                self.index = faiss.IndexFlatIP(d)
            else:
                self.index = faiss.IndexFlatL2(d)
            self.is_trained = True
            
        elif self.index_type == "hnsw":
            # HNSW - approximate search, fast
            self.index = faiss.IndexHNSWFlat(d, self.hnsw_m, metric)
            self.index.hnsw.efConstruction = self.hnsw_ef_construction
            self.index.hnsw.efSearch = self.hnsw_ef_search
            self.is_trained = True
            
        elif self.index_type == "ivf":
            # IVF - coarse quantization
            quantizer = faiss.IndexFlat(d, metric)
            self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, metric)
            self.is_trained = False
            
        elif self.index_type == "pq":
            # PQ only - extreme compression
            self.index = faiss.IndexPQ(d, self.m, self.nbits, metric)
            self.is_trained = False
            
        elif self.index_type == "ivf_pq":
            # IVF + PQ - compressed approximate search
            quantizer = faiss.IndexFlat(d, metric)
            self.index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, self.nbits, metric)
            self.is_trained = False
        
        # Move to GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
                logger.info(f"Moved index to GPU {self.gpu_id}")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
        
        logger.info(
            f"Created {self.index_type} index with dim={d}, "
            f"metric={'cosine/IP' if metric == faiss.METRIC_INNER_PRODUCT else 'L2'}"
        )
    
    def _get_metric(self) -> int:
        """Get FAISS metric constant."""
        if self.distance_strategy in ["inner_product", "cosine"]:
            return faiss.METRIC_INNER_PRODUCT
        return faiss.METRIC_L2
    
    def _load(self, index_path: str, docstore_path: str, meta_path: str) -> None:
        """Load index and metadata from disk."""
        try:
            self.index = faiss.read_index(index_path)
            
            with open(docstore_path, "rb") as f:
                self.docstore, self.index_to_id = pickle.load(f)
            
            # Build reverse mapping
            self.id_to_index = {v: k for k, v in self.index_to_id.items()}
            
            # Load metadata if exists
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                    self.is_trained = meta.get("is_trained", True)
                    self.index_type = meta.get("index_type", self.index_type)
                    # Restore IVF params
                    if hasattr(self.index, "nprobe"):
                        self.index.nprobe = meta.get("nprobe", self.nprobe)
            else:
                self.is_trained = True  # Loaded index is always trained
            
            logger.info(
                f"Loaded {self.index_type} index from {index_path} "
                f"with {self.index.ntotal} vectors"
            )
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._create_index()
    
    def _save(self) -> None:
        """Save index and metadata to disk."""
        if not self.path or self.index is None:
            return
        
        try:
            os.makedirs(self.path, exist_ok=True)
            index_path = f"{self.path}/{self.collection_name}.faiss"
            docstore_path = f"{self.path}/{self.collection_name}.pkl"
            meta_path = f"{self.path}/{self.collection_name}_meta.pkl"
            
            # Move to CPU for saving if on GPU
            index_to_save = self.index
            if self.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                try:
                    index_to_save = faiss.index_gpu_to_cpu(self.index)
                except:
                    index_to_save = self.index
            
            faiss.write_index(index_to_save, index_path)
            
            with open(docstore_path, "wb") as f:
                pickle.dump((self.docstore, self.index_to_id), f)
            
            # Save metadata
            meta = {
                "is_trained": self.is_trained,
                "index_type": self.index_type,
                "nprobe": getattr(self.index, "nprobe", self.nprobe),
                "hnsw_ef_search": self.hnsw_ef_search,
            }
            with open(meta_path, "wb") as f:
                pickle.dump(meta, f)
                
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _train_if_needed(self, vectors: np.ndarray) -> None:
        """Train the index if needed and sufficient data is available."""
        if self.is_trained:
            return
        
        n_vectors = len(vectors)
        min_train_vectors = max(self.nlist * 10, self.auto_train_threshold)
        
        # Collect training data
        if not hasattr(self, '_training_data'):
            self._training_data = []
        
        self._training_data.extend(vectors)
        
        if len(self._training_data) >= min_train_vectors:
            train_vectors = np.array(self._training_data[:min_train_vectors], dtype=np.float32)
            
            logger.info(f"Training {self.index_type} index with {len(train_vectors)} vectors...")
            self.index.train(train_vectors)
            self.is_trained = True
            self._training_data = []  # Clear training data
            logger.info("Training completed")
        else:
            logger.debug(f"Collecting training data: {len(self._training_data)}/{min_train_vectors}")
    
    def create_col(self, name: str, distance: str = None) -> "FAISSAdvanced":
        """Create a new collection."""
        if distance:
            self.distance_strategy = distance.lower()
        self.collection_name = name
        self._create_index()
        self._save()
        return self
    
    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Insert vectors into the collection."""
        if self.index is None:
            raise ValueError("Index not initialized")
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]
        
        if len(vectors) != len(ids) or len(vectors) != len(payloads):
            raise ValueError("Vectors, payloads, and IDs must have the same length")
        
        vectors_np = np.array(vectors, dtype=np.float32)
        
        # Normalize if needed
        if self.normalize_L2 and self.distance_strategy == "euclidean":
            faiss.normalize_L2(vectors_np)
        
        # Train if needed (for IVF/PQ based indexes)
        if not self.is_trained and self.index_type in ["ivf", "pq", "ivf_pq"]:
            self._train_if_needed(vectors_np)
            if not self.is_trained:
                # Store vectors for later insertion after training
                if not hasattr(self, '_pending_vectors'):
                    self._pending_vectors = []
                for vec, vid, payload in zip(vectors, ids, payloads):
                    self._pending_vectors.append((vec, vid, payload))
                logger.info(f"Storing {len(vectors)} vectors for later insertion (waiting for training)")
                return
        
        # Insert pending vectors first if we just got trained
        if hasattr(self, '_pending_vectors') and self.is_trained:
            pending = self._pending_vectors
            self._pending_vectors = []
            if pending:
                pending_vecs = np.array([p[0] for p in pending], dtype=np.float32)
                pending_ids = [p[1] for p in pending]
                self._insert_vectors(pending_vecs, pending_ids)
                for vid, payload in [(p[1], p[2]) for p in pending]:
                    self.docstore[vid] = payload
                    self.id_to_index[vid] = len(self.id_to_index)
                    self.index_to_id[len(self.index_to_id)] = vid
        
        # Insert current vectors
        self._insert_vectors(vectors_np, ids)
        
        # Update docstore
        for vid, payload in zip(ids, payloads):
            self.docstore[vid] = payload.copy()
            idx = len(self.id_to_index)
            self.id_to_index[vid] = idx
            self.index_to_id[idx] = vid
        
        self._save()
        logger.info(f"Inserted {len(vectors)} vectors. Total: {self.index.ntotal}")
    
    def _insert_vectors(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Insert vectors into FAISS index."""
        if not self.is_trained and self.index_type in ["ivf", "pq", "ivf_pq"]:
            logger.warning("Index not trained yet, vectors will be stored for later insertion")
            return
        
        # For indexes that don't support ID mapping directly
        if self.index_type in ["flat", "hnsw"]:
            self.index.add(vectors)
        else:
            # IVF and PQ based indexes
            self.index.add(vectors)
    
    def search(
        self,
        query: str,
        vectors: List[List[float]],
        limit: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        """Search for similar vectors."""
        if self.index is None:
            raise ValueError("Index not initialized")
        
        if self.index.ntotal == 0:
            return []
        
        # Set nprobe for IVF indexes
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        
        # Set efSearch for HNSW
        if self.index_type == "hnsw" and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = self.hnsw_ef_search
        
        query_vectors = np.array(vectors, dtype=np.float32)
        
        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        # Normalize if needed
        if self.normalize_L2 and self.distance_strategy == "euclidean":
            faiss.normalize_L2(query_vectors)
        
        # Calculate fetch count
        fetch_k = limit * 3 if filters else limit
        fetch_k = min(fetch_k, self.index.ntotal)
        
        # Search
        scores, indices = self.index.search(query_vectors, fetch_k)
        
        # Parse results
        results = self._parse_output(scores[0], indices[0], limit, filters)
        
        return results
    
    def _parse_output(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
        limit: int,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        """Parse search output with optional filtering."""
        results = []
        
        for i in range(len(indices)):
            if indices[i] == -1:
                continue
            
            idx = int(indices[i])
            
            # Handle ID mapping
            if self.index_type in ["flat", "hnsw"]:
                vector_id = self.index_to_id.get(idx)
            else:
                # For IVF/PQ, indices might be internal
                vector_id = self.index_to_id.get(idx)
            
            if vector_id is None:
                continue
            
            payload = self.docstore.get(vector_id)
            if payload is None:
                continue
            
            # Apply filters
            if filters and not self._apply_filters(payload, filters):
                continue
            
            # Convert distance to similarity score
            score = float(scores[i])
            if self.distance_strategy == "euclidean":
                # Convert L2 distance to similarity (0-1 range)
                score = 1.0 / (1.0 + score)
            else:
                # Inner product / cosine similarity is already in right range
                score = max(0.0, min(1.0, (score + 1.0) / 2.0))
            
            results.append(OutputData(
                id=vector_id,
                score=score,
                payload=payload.copy(),
            ))
            
            if len(results) >= limit:
                break
        
        return results
    
    def _apply_filters(self, payload: Dict, filters: Dict) -> bool:
        """Apply filters to a payload."""
        if not filters or not payload:
            return True
        
        for key, value in filters.items():
            if key.startswith("$"):
                # Handle special operators
                if key == "$or":
                    return any(self._apply_filters(payload, cond) for cond in value)
                elif key == "$and":
                    return all(self._apply_filters(payload, cond) for cond in value)
                continue
            
            if key not in payload:
                return False
            
            if isinstance(value, dict):
                # Handle comparison operators
                for op, val in value.items():
                    if op == "eq" and payload[key] != val:
                        return False
                    elif op == "ne" and payload[key] == val:
                        return False
                    elif op in ["gt", "gte", "lt", "lte"]:
                        try:
                            pval = float(payload[key])
                            vval = float(val)
                            if op == "gt" and not (pval > vval):
                                return False
                            elif op == "gte" and not (pval >= vval):
                                return False
                            elif op == "lt" and not (pval < vval):
                                return False
                            elif op == "lte" and not (pval <= vval):
                                return False
                        except (ValueError, TypeError):
                            return False
                    elif op == "in" and payload[key] not in val:
                        return False
                    elif op == "nin" and payload[key] in val:
                        return False
                    elif op == "contains" and val not in str(payload[key]):
                        return False
                    elif op == "icontains" and val.lower() not in str(payload[key]).lower():
                        return False
            elif isinstance(value, list):
                if payload[key] not in value:
                    return False
            elif payload[key] != value:
                return False
        
        return True
    
    def delete(self, vector_id: str) -> None:
        """Delete a vector by ID."""
        if vector_id not in self.docstore:
            logger.warning(f"Vector {vector_id} not found")
            return
        
        # FAISS doesn't support direct deletion for most indexes
        # We mark as deleted in docstore and rebuild index periodically
        self.docstore.pop(vector_id, None)
        
        # Update mappings
        if vector_id in self.id_to_index:
            idx = self.id_to_index.pop(vector_id)
            self.index_to_id.pop(idx, None)
        
        self._save()
        logger.info(f"Deleted vector {vector_id} (soft delete)")
    
    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ) -> None:
        """Update a vector and/or its payload."""
        if vector_id not in self.docstore:
            raise ValueError(f"Vector {vector_id} not found")
        
        current_payload = self.docstore[vector_id].copy()
        
        if payload is not None:
            self.docstore[vector_id] = payload.copy()
        
        # For vector updates, we need to delete and re-insert
        if vector is not None:
            self.delete(vector_id)
            self.insert(
                vectors=[vector],
                payloads=[self.docstore.get(vector_id, current_payload)],
                ids=[vector_id],
            )
        else:
            self._save()
    
    def get(self, vector_id: str) -> Optional[OutputData]:
        """Retrieve a vector by ID."""
        if vector_id not in self.docstore:
            return None
        
        return OutputData(
            id=vector_id,
            score=None,
            payload=self.docstore[vector_id].copy(),
        )
    
    def list_cols(self) -> List[str]:
        """List all collections."""
        if not self.path:
            return [self.collection_name] if self.index else []
        
        try:
            path = Path(self.path)
            return [f.stem for f in path.glob("*.faiss")]
        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
            return [self.collection_name] if self.index else []
    
    def delete_col(self) -> None:
        """Delete the collection."""
        if self.path:
            for ext in [".faiss", ".pkl", "_meta.pkl"]:
                file_path = f"{self.path}/{self.collection_name}{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        self.index = None
        self.docstore = {}
        self.index_to_id = {}
        self.id_to_index = {}
        self.is_trained = False
    
    def col_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if self.index is None:
            return {
                "name": self.collection_name,
                "count": 0,
                "index_type": self.index_type,
            }
        
        info = {
            "name": self.collection_name,
            "count": self.index.ntotal,
            "dimension": self.embedding_model_dims,
            "index_type": self.index_type,
            "distance": self.distance_strategy,
            "is_trained": self.is_trained,
        }
        
        # Add index-specific info
        if self.index_type == "hnsw":
            info["hnsw_m"] = self.hnsw_m
            info["hnsw_ef_search"] = self.hnsw_ef_search
        elif self.index_type in ["ivf", "ivf_pq"]:
            info["nlist"] = self.nlist
            info["nprobe"] = self.nprobe
        
        if self.index_type in ["pq", "ivf_pq"]:
            info["m"] = self.m
            info["nbits"] = self.nbits
            info["compression_ratio"] = (32 * self.embedding_model_dims) / (self.m * self.nbits)
        
        return info
    
    def list(
        self,
        filters: Optional[Dict] = None,
        limit: int = 100,
    ) -> List[List[OutputData]]:
        """List all vectors with optional filtering."""
        results = []
        count = 0
        
        for vector_id, payload in self.docstore.items():
            if filters and not self._apply_filters(payload, filters):
                continue
            
            results.append(OutputData(
                id=vector_id,
                score=None,
                payload=payload.copy(),
            ))
            
            count += 1
            if count >= limit:
                break
        
        return [results]
    
    def reset(self) -> None:
        """Reset the collection."""
        logger.warning(f"Resetting collection {self.collection_name}")
        self.delete_col()
        self._create_index()
    
    def rebuild_index(self) -> None:
        """Rebuild the index (useful after many deletions or for optimization)."""
        if not self.docstore:
            logger.info("No vectors to rebuild")
            return
        
        logger.info(f"Rebuilding {self.index_type} index with {len(self.docstore)} vectors...")
        
        # Collect all vectors
        vectors = []
        ids = []
        payloads = []
        
        # Try to reconstruct vectors from index
        for idx, vector_id in self.index_to_id.items():
            if vector_id in self.docstore:
                try:
                    # Attempt to reconstruct vector
                    vec = self.index.reconstruct(int(idx))
                    vectors.append(vec)
                    ids.append(vector_id)
                    payloads.append(self.docstore[vector_id])
                except:
                    pass
        
        if not vectors:
            logger.warning("Could not reconstruct vectors, keeping current index")
            return
        
        # Save old data
        old_docstore = self.docstore.copy()
        
        # Reset and rebuild
        self.delete_col()
        self._create_index()
        
        # Re-insert
        self.insert(vectors=vectors, payloads=payloads, ids=ids)
        
        logger.info(f"Index rebuilt with {len(vectors)} vectors")
    
    def get_index_size(self) -> Dict[str, Any]:
        """Get detailed size information about the index."""
        info = {
            "num_vectors": self.index.ntotal if self.index else 0,
            "docstore_entries": len(self.docstore),
            "memory_estimate_mb": 0,
        }
        
        if self.index:
            # Estimate FAISS index size
            # Write to temp buffer to get size
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = f.name
            try:
                faiss.write_index(self.index, temp_path)
                info["memory_estimate_mb"] = os.path.getsize(temp_path) / (1024 * 1024)
                os.remove(temp_path)
            except:
                pass
        
        # Add docstore estimate
        try:
            docstore_size = len(pickle.dumps(self.docstore)) / (1024 * 1024)
            info["docstore_size_mb"] = docstore_size
            info["memory_estimate_mb"] += docstore_size
        except:
            pass
        
        return info
