from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Protocol, Union

from tensorrt_llm.bench.config_gen.dataclasses.features import FeatureConfig
from tensorrt_llm.bench.dataclasses.scenario import (TuningConstraints,
                                                     WorldConfig)


class ScenarioProtocol(Protocol):

    def get_settings(
        self,
        model: Union[str, Path],
        world: WorldConfig,
        constraints: TuningConstraints,
    ) -> Dict[str, Any]:
        ...

    def apply_features(self, llm_kwargs: Dict[str, Any],
                       feature_config: FeatureConfig) -> Dict[str, Any]:
        ...


class DefaultScenario(ScenarioProtocol):

    def get_settings(
        self,
        model: Union[str, Path],
        world: WorldConfig,
        _: TuningConstraints,
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "pipeline_parallel_size": world.pp_size,
            "tensor_parallel_size": world.tp_size,
            "gpus_per_node": world.gpus_per_node,
            "moe_expert_parallel_size": world.ep_size,
            "moe_cluster_parallel_size": world.cluster_size,
            "trust_remote_code": True,
            "batching_type": "INFLIGHT",
        }
