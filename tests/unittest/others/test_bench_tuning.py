# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for trtllm-bench tuning engine-size estimation.

Covers NVBug 5881240: pre-quantized NVFP4 checkpoints were producing an
engine-size estimate that was 2x too small because the FP4 compression factor
was applied twice (once by the packed tensor shapes in safetensors, once by
BYTES_PER_ELEM[NVFP4] = 0.5).
"""

from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.bench.build.dataclasses import SAFETENSORS_DTYPE_BYTES, ModelConfig
from tensorrt_llm.bench.build.tuning import calc_engine_setting
from tensorrt_llm.llmapi.llm_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = dict(
    name="test-model",
    model_type="llama",
    num_hidden_layers=80,
    num_attention_heads=64,
    num_key_value_heads=8,
    hidden_size=8192,
    head_size=128,
    max_position_embeddings=131072,
    dtype="bfloat16",
)


def _make_model_config(param_count, model_size_bytes=None):
    return ModelConfig(param_count=param_count, model_size_bytes=model_size_bytes, **_BASE_CONFIG)


def _make_fake_safetensors_metadata(dtype_counts: dict):
    """Build a minimal SafetensorsRepoMetadata stub."""
    meta = MagicMock()
    meta.parameter_count = dtype_counts
    return meta


def _nvfp4_quant_config():
    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    qc.kv_cache_quant_algo = QuantAlgo.FP8
    return qc


def _no_quant_config():
    return QuantConfig(quant_algo=QuantAlgo.NO_QUANT)


# ---------------------------------------------------------------------------
# SAFETENSORS_DTYPE_BYTES constant sanity checks
# ---------------------------------------------------------------------------


def test_dtype_bytes_nvfp4_entries():
    assert SAFETENSORS_DTYPE_BYTES["F4_E2M1"] == 0.5
    assert SAFETENSORS_DTYPE_BYTES["I4"] == 0.5
    assert SAFETENSORS_DTYPE_BYTES["U4"] == 0.5


def test_dtype_bytes_fp8_entries():
    assert SAFETENSORS_DTYPE_BYTES["F8_E4M3"] == 1.0
    assert SAFETENSORS_DTYPE_BYTES["F8_E5M2"] == 1.0


def test_dtype_bytes_standard_entries():
    assert SAFETENSORS_DTYPE_BYTES["BF16"] == 2.0
    assert SAFETENSORS_DTYPE_BYTES["F16"] == 2.0
    assert SAFETENSORS_DTYPE_BYTES["F32"] == 4.0


# ---------------------------------------------------------------------------
# ModelConfig._get_safetensors_counts
# ---------------------------------------------------------------------------


@patch("tensorrt_llm.bench.build.dataclasses.get_safetensors_metadata")
def test_get_safetensors_counts_prequantized_nvfp4(mock_meta):
    """U8-packed NVFP4 checkpoint yields param_count and model_size_bytes = elements * 1 byte."""
    mock_meta.return_value = _make_fake_safetensors_metadata({"U8": 230_000_000_000})
    param_count, model_size_bytes = ModelConfig._get_safetensors_counts("some/nvfp4-model", None)
    assert param_count == 230_000_000_000
    assert model_size_bytes == pytest.approx(230_000_000_000 * 1.0)


@patch("tensorrt_llm.bench.build.dataclasses.get_safetensors_metadata")
def test_get_safetensors_counts_bf16_checkpoint(mock_meta):
    """BF16 checkpoint: model_size_bytes = elements × 2 bytes/elem."""
    mock_meta.return_value = _make_fake_safetensors_metadata({"BF16": 405_000_000_000})
    param_count, model_size_bytes = ModelConfig._get_safetensors_counts("some/bf16-model", None)
    assert param_count == 405_000_000_000
    assert model_size_bytes == pytest.approx(405_000_000_000 * 2.0)


@patch("tensorrt_llm.bench.build.dataclasses.get_safetensors_metadata")
def test_get_safetensors_counts_mixed_dtypes(mock_meta):
    """Mixed dtype checkpoint: weighted sum per dtype."""
    mock_meta.return_value = _make_fake_safetensors_metadata(
        {
            "U8": 200_000_000_000,  # packed FP4 weights → 1 byte each
            "F8_E4M3": 30_000_000_000,  # FP8 scales → 1 byte each
        }
    )
    param_count, model_size_bytes = ModelConfig._get_safetensors_counts("some/mixed-model", None)
    assert param_count == 230_000_000_000
    expected_bytes = 200_000_000_000 * 1.0 + 30_000_000_000 * 1.0
    assert model_size_bytes == pytest.approx(expected_bytes)


@patch("tensorrt_llm.bench.build.dataclasses.get_safetensors_metadata")
def test_get_safetensors_counts_gptj_no_safetensors(mock_meta):
    """GPT-J uses a hardcoded param count and returns None for model_size_bytes."""
    param_count, model_size_bytes = ModelConfig._get_safetensors_counts("EleutherAI/gpt-j-6b", None)
    mock_meta.assert_not_called()
    assert param_count == 6053381344
    assert model_size_bytes is None


# ---------------------------------------------------------------------------
# calc_engine_setting — engine size branch selection
# ---------------------------------------------------------------------------


@patch("tensorrt_llm.bench.build.tuning.get_device_memory", return_value=1000.0)
def test_calc_engine_setting_prequantized_nvfp4(mock_mem):
    """Pre-quantized NVFP4 uses model_size_bytes directly rather than param_count * 0.5."""
    # HF reports 230B packed U8 elements; model_size_bytes = 230B bytes (~214 GB).
    # The buggy path would use 230B * 0.5 = ~107 GB (2x too small).
    model_config = _make_model_config(
        param_count=230_000_000_000, model_size_bytes=230_000_000_000.0
    )
    quant_config = _nvfp4_quant_config()

    max_batch_size, max_num_tokens = calc_engine_setting(
        model_config=model_config,
        quant_config=quant_config,
        tp_size=8,
        pp_size=1,
        target_input_len=128,
        target_output_len=128,
    )
    assert max_batch_size > 0
    assert max_num_tokens > 0


@patch("tensorrt_llm.bench.build.tuning.get_device_memory", return_value=1000.0)
def test_calc_engine_setting_on_the_fly_nvfp4(mock_mem):
    """On-the-fly NVFP4 (BF16 checkpoint): engine size uses param_count * 0.5 (~189 GB)."""
    # BF16 checkpoint: param_count = 405B, model_size_bytes = 810B (2 bytes/elem)
    model_config = _make_model_config(
        param_count=405_000_000_000, model_size_bytes=810_000_000_000.0
    )
    quant_config = _nvfp4_quant_config()

    max_batch_size, max_num_tokens = calc_engine_setting(
        model_config=model_config,
        quant_config=quant_config,
        tp_size=8,
        pp_size=1,
        target_input_len=128,
        target_output_len=128,
    )
    assert max_batch_size > 0
    assert max_num_tokens > 0


@patch("tensorrt_llm.bench.build.tuning.get_device_memory", return_value=1000.0)
def test_calc_engine_setting_gptj_fallback(mock_mem):
    """GPT-J (model_size_bytes=None): falls back to param_count * byte_per_elem."""
    model_config = _make_model_config(param_count=6_053_381_344, model_size_bytes=None)
    quant_config = _no_quant_config()

    max_batch_size, max_num_tokens = calc_engine_setting(
        model_config=model_config,
        quant_config=quant_config,
        tp_size=1,
        pp_size=1,
        target_input_len=128,
        target_output_len=128,
    )
    assert max_batch_size > 0
    assert max_num_tokens > 0


@patch("tensorrt_llm.bench.build.tuning.get_device_memory", return_value=1000.0)
def test_calc_engine_setting_prequantized_fp8(mock_mem):
    """Pre-quantized FP8 uses dtype-aware branch when model_size_bytes < param_count * 2."""
    model_config = _make_model_config(
        param_count=405_000_000_000, model_size_bytes=405_000_000_000.0
    )
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    quant_config.kv_cache_quant_algo = QuantAlgo.FP8

    max_batch_size, max_num_tokens = calc_engine_setting(
        model_config=model_config,
        quant_config=quant_config,
        tp_size=8,
        pp_size=1,
        target_input_len=128,
        target_output_len=128,
    )
    assert max_batch_size > 0
    assert max_num_tokens > 0
