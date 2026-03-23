from typing import Optional
from pydantic import BaseModel, Field


class RerankerConfig(BaseModel):
    """Reranker configuration (minimal)."""
    provider: Optional[str] = Field(None, description="Reranker provider")
    config: Optional[dict] = Field({}, description="Reranker configuration")
