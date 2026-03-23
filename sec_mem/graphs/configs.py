from typing import Optional
from pydantic import BaseModel, Field


class GraphStoreConfig(BaseModel):
    """Graph store configuration (minimal implementation)."""
    provider: Optional[str] = Field(None, description="Graph store provider")
    config: Optional[dict] = Field({}, description="Graph store configuration")
