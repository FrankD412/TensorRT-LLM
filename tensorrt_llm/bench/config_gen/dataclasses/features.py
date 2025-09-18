from pydantic import BaseModel, Field


class FeatureConfig(BaseModel):
    """Features that are supported by the model."""
    chunked_prefill: bool = Field(default=False,
                                  description="Enable/disable chunked prefill.")
    kv_cache_reuse: bool = Field(default=False,
                                 description="Enable/disable kv cache reuse.")
