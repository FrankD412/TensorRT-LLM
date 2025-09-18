from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator
from typing_extensions import Self


class FeatureConfig(BaseModel):
    enable_chunked_context: bool = Field(
        default=True, description="Whether to enable chunking.")
    enable_cache_reuse: bool = Field(
        default=True, description="Whether to enable cache reuse.")


class WorldConfig(BaseModel):
    tp: int = Field(
        default=None,
        description="The tensor parallelism size to use for tuning.")
    pp: int = Field(
        default=None,
        description="The pipeline parallelism size to use for tuning.")
    ep: Optional[int] = Field(
        default=None,
        description="The expert parallelism size to use for tuning.")
    cluster_size: Optional[int] = Field(
        default=None, description="The expert cluster size to use for tuning.")
    gpus_per_node: Optional[int] = Field(
        default=None,
        description="The number of GPUs per node to use for tuning.")

    @computed_field
    def world_size(self) -> int:
        return int(self.tp * self.pp)


class TuningConstraints(BaseModel):
    target_input_len: Optional[int] = Field(
        description="The target input length to use for tuning.")
    target_output_len: Optional[int] = Field(
        description="The target output length to use for tuning.")
    max_input_len: int = Field(
        description="The maximum input length to use for tuning.")
    max_output_len: int = Field(
        description="The maximum output length to use for tuning.")
    concurrency: Optional[int] = Field(
        default=None,
        description="The number of concurrent requests to use for tuning.")

    class Config:
        extra = "ignore"

    @model_validator(mode="after")
    def validate_target_input_output_len(self) -> Self:
        """Set the target input and output lengths if unspecified."""
        self.target_input_len = self.target_input_len or self.max_input_len
        self.target_output_len = self.target_output_len or self.max_output_len

        return self

    @computed_field
    def max_sequence_len(self) -> int:
        """Compute the maximum sequence length."""
        return max(self.max_input_len, self.max_output_len)
