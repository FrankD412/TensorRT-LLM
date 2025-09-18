class ScenarioProtocol(Protocol):

    @classmethod
    def get_heuristics(
        cls,
        model: Union[str, Path],
        world: WorldConfig,
        constraints: TuningConstraints,
    ) -> Dict[str, Any]:
        ...

    def apply_features(self, llm_kwargs: Dict[str, Any],
                       feature_config: FeatureConfig) -> Dict[str, Any]:
        ...


class PytMaxThroughputScenario(ScenarioProtocol):
    """Maximum throughput heuristic tuning for the PyTorch backend."""

    @classmethod
    def get_settings(cls, scenario: ScenarioSpecification) -> Dict[str, Any]:
        # Collect reference information.
        # Configuration classes
        tllm_model_config = scenario.environment.get_tllm_model_config()
        bench_model_config = scenario.environment.get_bench_model_config()

        # World, dataset, and settings information.
        world = scenario.world
        dataset_metadata = scenario.dataset_metadata
        kv_cache_dtype = \
            scenario.llm_config.extra_llm_api_options.get(
                "kv_cache_dtype", "auto"
            )
        chunked_prefill = \
            scenario.llm_config.extra_llm_api_options.get(
                "enable_chunked_prefill", False
            )

        # Update the KV cache settings.
        validate_and_set_kv_cache_quant(tllm_model_config, kv_cache_dtype)

        # Find tuned parameters
        max_batch_size, max_num_tokens = get_benchmark_engine_settings(
            bench_model_config,
            tllm_model_config.quant_config,
            world.tp,
            world.pp,
            dataset_metadata.avg_isl,
            dataset_metadata.avg_osl,
        )

        max_batch_size = scenario.batching_config.max_batch_size or max_batch_size
        max_num_tokens = scenario.batching_config.max_num_tokens or max_num_tokens

        # Update CUDA graph settings.
        cuda_graph_config = {
            "enable_padding": True,
            "max_batch_size": max_batch_size,
        }

        # Get the initial settings from the parent class (absolute default settings)
        llm_args = super().get_settings(scenario)
        # Update scheduler settings for scheduling in the IFB scheduler.
        llm_args |= {
            "scheduler_config": {
                "capacity_scheduler_policy": "GUARANTEED_NO_EVICT",
                "context_chunking_policy": "FIRST_COME_FIRST_SERVED",
            },
            "cuda_graph_config": cuda_graph_config,
            "enable_chunked_prefill": chunked_prefill,
            "kv_cache_dtype": kv_cache_dtype,
            "max_seq_len": scenario.batching_config.max_seq_len,
            "max_batch_size": max_batch_size,
            "max_num_tokens": max_num_tokens,
        }

        return llm_args
