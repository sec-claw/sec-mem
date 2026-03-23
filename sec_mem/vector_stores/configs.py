from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    provider: str = Field("faiss_advanced", description="Vector store provider")
    config: Optional[Dict[str, Any]] = Field({}, description="Provider-specific config")
