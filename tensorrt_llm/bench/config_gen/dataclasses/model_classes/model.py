from typing import Literal, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, Field, model_validator
from transformers import AutoConfig

from tensorrt_llm.bench.build.utils import get_safetensors_metadata


class ModelConfig(BaseModel):
    """ Model specific configurations. The parameters are needed in engine
        setting calculation.
    """
    name: str
    model_type: str
    param_count: int
    num_hidden_layers: int = Field(validation_alias=AliasChoices(
        "num_hidden_layers",
        "n_layer",
        AliasPath("text_config", "num_hidden_layers"),
        AliasPath("language_config", "num_hidden_layers"),
    ))
    num_attention_layers: Optional[int] = Field(default=None)
    num_attention_heads: int = Field(validation_alias=AliasChoices(
        "num_attention_heads",
        "n_head",
        AliasPath("text_config", "num_attention_heads"),
        AliasPath("language_config", "num_attention_heads"),
    ))
    num_key_value_heads: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "num_key_value_heads",
            "num_kv_heads",
            AliasPath("text_config", "num_key_value_heads"),
            AliasPath("language_config", "num_key_value_heads"),
        ),
    )
    hidden_size: int = Field(validation_alias=AliasChoices(
        "hidden_size",
        "n_embd",
        AliasPath("text_config", "hidden_size"),
    ))
    head_size: Optional[int] = Field(default=None,
                                     validation_alias=AliasChoices(
                                         "head_size",
                                         "head_dim",
                                         "attention_head_dim",
                                         AliasPath("text_config", "head_dim"),
                                     ))
    max_position_embeddings: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "max_position_embeddings",
            "n_positions",
            AliasPath("text_config", "max_position_embeddings"),
        ))
    dtype: Literal["float16", "bfloat16", "float32",
                   None] = Field(default="float16",
                                 validation_alias=AliasChoices(
                                     "dtype", "torch_dtype"))

    @model_validator(mode="after")
    def set_values_if_none(self):
        """ Set the values if cannot get values from HF config.json. """
        if not self.dtype:  # for GPT-J
            self.dtype = "float16"
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_size is None:
            self.head_size = self.hidden_size // self.num_attention_heads
        if self.num_attention_layers is None:
            self.num_attention_layers = self.num_hidden_layers
        return self

    @classmethod
    def get_param_count(cls, model_hf_name, hf_model_path):
        """ Read the parameter count from HF safetensor metadata. """
        if model_hf_name == "EleutherAI/gpt-j-6b":  # GPT-J repo doesn't use safetensor format.
            param_count = 6053381344
        else:
            model_name_or_path = hf_model_path or model_hf_name
            metadata = get_safetensors_metadata(model_name_or_path)
            param_count = sum(metadata.parameter_count.values())
        assert param_count, f"Can't get valid parameter count for model: {model_name_or_path}."

        return param_count

    @classmethod
    def from_hf(cls, model_hf_name, hf_model_path):
        model_name_or_path = hf_model_path or model_hf_name
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True).to_dict()
        param_count = cls.get_param_count(model_hf_name, hf_model_path)

        return cls(name=model_hf_name, param_count=param_count, **hf_config)

    def extra_model_cache_in_gb(self, bytes_per_elem, target_seq_len=None):
        return 0

    def cache_memory_fraction(self, cache_memory_fraction):
        return cache_memory_fraction
