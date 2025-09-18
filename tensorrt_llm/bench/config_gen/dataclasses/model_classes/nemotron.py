from typing import Optional

from pydantic import AliasChoices, Field, model_validator

from tensorrt_llm.bench.dataclasses.model_classes.model import ModelConfig


class NemotronHybridConfig(ModelConfig):
    hybrid_override_pattern: str
    d_state: int = Field(validation_alias=AliasChoices(
        "d_state",
        "mamba_d_state",
        "ssm_state_size",
    ))
    d_conv: int = Field(validation_alias=AliasChoices(
        "d_conv",
        "mamba_d_conv",
        "conv_kernel",
    ))
    expand: int = Field(validation_alias=AliasChoices(
        "expand",
        "mamba_expand",
    ))
    n_groups: int
    mamba_head_dim: int
    d_inner: Optional[int] = Field(default=None)
    mamba_num_heads: Optional[int] = Field(default=None)
    num_mamba_layers: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def set_values_if_none(self):
        """Set the values if cannot get values from HF config.json."""
        if not self.d_inner:
            self.d_inner = self.hidden_size * self.expand
        if not self.mamba_num_heads:
            self.mamba_num_heads = self.d_inner // self.mamba_head_dim
        if self.num_mamba_layers is None:
            self.num_mamba_layers = self.hybrid_override_pattern.count("M")
        if self.num_attention_layers is None:
            self.num_attention_layers = self.hybrid_override_pattern.count("*")

        super().set_values_if_none()
        return self

    def extra_model_cache_in_gb(self, bytes_per_elem, target_seq_len=None):
        conv_dim = self.d_inner + 2 * self.n_groups * self.d_state
        conv_state_elems = conv_dim * (self.d_conv - 1)
        ssm_state_elems = self.mamba_num_heads * self.mamba_head_dim * self.d_state
        gb_per_mamba_cache = bytes_per_elem * self.num_mamba_layers * (
            conv_state_elems + ssm_state_elems) / (1024**3)
        return gb_per_mamba_cache

    def cache_memory_fraction(self, cache_memory_fraction):
        # Each mamba cache entry is pretty large (~50MB for 8B model), so we are more conservative when estimating the max batch size
        return cache_memory_fraction**2
