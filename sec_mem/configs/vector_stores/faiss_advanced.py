from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FAISSAdvancedConfig(BaseModel):
    """Configuration for Advanced FAISS vector store.
    
    Supports multiple index types optimized for different use cases:
    - flat: Exact search, most accurate but slowest
    - hnsw: Approximate search, fast with good accuracy (best for < 1M vectors)
    - ivf: Inverted file index, memory efficient for large datasets
    - ivf_pq: IVF with product quantization, highly compressed storage
    - pq: Product quantization only, extreme compression
    """
    
    collection_name: str = Field("sec_mem", description="Name of the collection")
    path: Optional[str] = Field(None, description="Path for local FAISS database storage")
    distance_strategy: str = Field(
        "euclidean",
        description="Distance metric: 'euclidean', 'inner_product', or 'cosine'"
    )
    normalize_L2: bool = Field(
        False,
        description="Whether to L2 normalize vectors (for euclidean distance)"
    )
    embedding_model_dims: int = Field(
        1536,
        description="Dimension of embedding vectors"
    )
    
    # Index type selection
    index_type: str = Field(
        "flat",
        description="Index type: 'flat', 'hnsw', 'ivf', 'ivf_pq', 'pq'"
    )
    
    # HNSW parameters
    hnsw_m: int = Field(
        16,
        description="HNSW: Number of neighbors per layer (higher = more accurate, slower)"
    )
    hnsw_ef_construction: int = Field(
        64,
        description="HNSW: Build-time search depth (higher = better index quality)"
    )
    hnsw_ef_search: int = Field(
        32,
        description="HNSW: Query-time search depth (higher = better recall, slower)"
    )
    
    # IVF parameters
    nlist: int = Field(
        100,
        description="IVF: Number of clusters (should be ~sqrt of expected vectors)"
    )
    nprobe: int = Field(
        10,
        description="IVF: Number of clusters to search (higher = better recall, slower)"
    )
    
    # PQ parameters
    m: int = Field(
        16,
        description="PQ: Number of subquantizers (must divide embedding_model_dims)"
    )
    nbits: int = Field(
        8,
        description="PQ: Bits per subquantizer (8 is standard, 4 for extreme compression)"
    )
    
    # Training parameters
    auto_train_threshold: int = Field(
        1000,
        description="Minimum vectors before auto-training IVF/PQ indexes"
    )
    
    # GPU acceleration
    use_gpu: bool = Field(
        False,
        description="Use GPU acceleration if available"
    )
    gpu_id: int = Field(
        0,
        description="GPU device ID to use"
    )
    
    @model_validator(mode="before")
    @classmethod
    def validate_index_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        valid_types = ["flat", "hnsw", "ivf", "ivf_pq", "pq"]
        index_type = values.get("index_type", "flat")
        if index_type not in valid_types:
            raise ValueError(f"index_type must be one of {valid_types}")
        return values
    
    @model_validator(mode="before")
    @classmethod
    def validate_distance_strategy(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        valid_strategies = ["euclidean", "inner_product", "cosine"]
        strategy = values.get("distance_strategy", "euclidean")
        if strategy not in valid_strategies:
            raise ValueError(f"distance_strategy must be one of {valid_strategies}")
        return values
    
    @model_validator(mode="before")
    @classmethod
    def validate_pq_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that embedding dimensions are compatible with PQ subquantizers."""
        index_type = values.get("index_type", "flat")
        dims = values.get("embedding_model_dims", 1536)
        m = values.get("m", 16)
        
        if index_type in ["pq", "ivf_pq"]:
            if dims % m != 0:
                raise ValueError(
                    f"embedding_model_dims ({dims}) must be divisible by m ({m}) "
                    f"for PQ-based indexes. Try m={dims // 16}, m={dims // 12}, or m={dims // 8}"
                )
        return values
    
    @model_validator(mode="before")
    @classmethod
    def validate_hnsw_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HNSW parameters."""
        index_type = values.get("index_type")
        if index_type == "hnsw":
            m = values.get("hnsw_m", 16)
            if m < 2 or m > 64:
                raise ValueError("hnsw_m must be between 2 and 64")
            
            ef_construction = values.get("hnsw_ef_construction", 64)
            if ef_construction < m:
                raise ValueError("hnsw_ef_construction should be >= hnsw_m")
        return values
    
    @model_validator(mode="before")
    @classmethod
    def validate_ivf_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate IVF parameters."""
        index_type = values.get("index_type")
        if index_type in ["ivf", "ivf_pq"]:
            nlist = values.get("nlist", 100)
            nprobe = values.get("nprobe", 10)
            if nprobe > nlist:
                raise ValueError("nprobe should not exceed nlist")
        return values
    
    def get_compression_ratio(self) -> float:
        """Calculate the compression ratio for PQ-based indexes."""
        if self.index_type not in ["pq", "ivf_pq"]:
            return 1.0
        # Original: 32 bits per dimension
        # Compressed: nbits per subquantizer
        original_bits = 32 * self.embedding_model_dims
        compressed_bits = self.m * self.nbits
        return original_bits / compressed_bits
    
    def estimate_memory_mb(self, num_vectors: int) -> float:
        """Estimate memory usage in MB for given number of vectors."""
        if self.index_type == "flat":
            # 4 bytes per float
            return (num_vectors * self.embedding_model_dims * 4) / (1024 * 1024)
        elif self.index_type == "hnsw":
            # Vectors + graph structure
            vector_size = num_vectors * self.embedding_model_dims * 4
            graph_size = num_vectors * self.hnsw_m * 8  # 8 bytes per connection
            return (vector_size + graph_size) / (1024 * 1024)
        elif self.index_type == "ivf":
            # Vectors + cluster centroids
            vector_size = num_vectors * self.embedding_model_dims * 4
            centroid_size = self.nlist * self.embedding_model_dims * 4
            return (vector_size + centroid_size) / (1024 * 1024)
        elif self.index_type in ["pq", "ivf_pq"]:
            # PQ codes + codebook
            code_size = num_vectors * self.m * self.nbits / 8
            codebook_size = (2 ** self.nbits) * self.m * self.embedding_model_dims // self.m * 4
            if self.index_type == "ivf_pq":
                code_size += self.nlist * self.embedding_model_dims * 4  # IVF centroids
            return (code_size + codebook_size) / (1024 * 1024)
        return 0.0
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
