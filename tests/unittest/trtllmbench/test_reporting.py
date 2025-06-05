from math import ceil
from typing import List, Tuple
import pytest
from tensorrt_llm.bench.dataclasses.configuration import ExecutorSettingsConfig, ExecutorWorldConfig, PerformanceOptions, RuntimeConfig
from tensorrt_llm.bench.dataclasses.general import DatasetMetadata
from tensorrt_llm.bench.dataclasses.reporting import PerfItemTuple, StatsKeeper, ReportUtility
from tensorrt_llm.bench.dataclasses.statistics import PercentileStats
from tensorrt_llm.llmapi.llm_args import CapacitySchedulerPolicy


def get_synthetic_request_info(num_requests: int) -> List[Tuple[int, int, int, int]]:
    request_info = []
    r_id = 1
    isl = 10
    osl = isl
    for _ in range(num_requests):
        request_info.append(
            (
                r_id,
                isl,
                [x for x in range(1, osl + 1)],
                isl + osl,
            )
        )
        r_id += 1
        isl += 5
        osl += 10

    return request_info


@pytest.fixture(scope="session")
def get_mock_runtime_configuration() -> RuntimeConfig:
    return RuntimeConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        sw_version="0.0.0",
        world_config=ExecutorWorldConfig(
            pp_size=1,
            tp_size=1,
            gpus_per_node=8,
        ),
        settings_config=ExecutorSettingsConfig(
            chunking=True,
            scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION,
            max_batch_size=1,
            max_num_tokens=1024,
            kv_cache_percent=0.90,
            kv_cache_reuse=False,
        ),
        performance_options=PerformanceOptions(
            cuda_graphs=True,
            pytorch_config={
                "use_cuda_graph": True,
                "cuda_graph_padding_enabled": True,
                "cuda_graph_max_batch_size": 1,
            },
        ),
        backend="pytorch",
    )


@pytest.fixture(scope="session")
def synthetic_perf_items() -> List[PerfItemTuple]:
    perf_items = []
    request_info = get_synthetic_request_info(10)
    start_time = 0
    # Each request lasts for 10 iterations
    end_time = (start_time + 10) * 1e9 # convert to nanoseconds

    for req in request_info:
        ttft_time = start_time + ceil((end_time - start_time) * .15)
        perf_items.append(
            PerfItemTuple(
                start_timestamp=start_time,
                end_timestamp=end_time,
                request_id=req[0],
                num_input_tokens=req[1],
                response_is_final=True,  # Fixed value
                error=False,             # Fixed value
                tokens=req[2],
                decoding_iteration=len(req[2]),
                time_on_first_token=ttft_time,
            )
        )

        start_time += (5 * 1e9)  # Increment start time for the next request
        # Increment the end time for the next request and offset by the number
        # of requests to make it so that requests take a different amount of time.
        end_time = start_time + ((10 + len(perf_items)) * 1e9)

    return perf_items


def test_percentile_stats() -> None:
    mock_items = list(range(1000))
    # Test percentile stats
    stats = PercentileStats.from_iterable(mock_items)
    assert stats.average == (mock_items[-1] + mock_items[0]) / 2
    assert stats.minimum == mock_items[0]
    assert stats.maximum == mock_items[-1]
    assert stats.p50 == mock_items[int(len(mock_items) * 0.50)]
    assert stats.p90 == mock_items[int(len(mock_items) * 0.90)]
    assert stats.p95 == mock_items[int(len(mock_items) * 0.95)]
    assert stats.p99 == mock_items[int(len(mock_items) * 0.99)]


def test_dataset_metadata() -> None:
    mock_items = list(range(1000))
    seq_lens = [2 * x for x in mock_items]
    stats = PercentileStats.from_iterable(mock_items)
    seq_len_stats = PercentileStats.from_iterable(seq_lens)
    dataset_metadata = DatasetMetadata(
        isl_stats=stats,
        osl_stats=stats,
        seq_len_stats=PercentileStats.from_iterable(seq_lens),
        num_requests=len(mock_items),
    )

    # Number of requests
    assert dataset_metadata.num_requests == len(mock_items)

    # Averages
    assert dataset_metadata.isl_stats.average == stats.average
    assert dataset_metadata.osl_stats.average == stats.average
    assert dataset_metadata.seq_len_stats.average == seq_len_stats.average

    # Percentiles
    # P50
    assert dataset_metadata.isl_stats.p50 == stats.p50
    assert dataset_metadata.osl_stats.p50 == stats.p50
    assert dataset_metadata.seq_len_stats.p50 == seq_len_stats.p50

    # P90
    assert dataset_metadata.isl_stats.p90 == stats.p90
    assert dataset_metadata.osl_stats.p90 == stats.p90
    assert dataset_metadata.seq_len_stats.p90 == seq_len_stats.p90

    # P95
    assert dataset_metadata.isl_stats.p95 == stats.p95
    assert dataset_metadata.osl_stats.p95 == stats.p95
    assert dataset_metadata.seq_len_stats.p95 == seq_len_stats.p95

    # P99
    assert dataset_metadata.isl_stats.p99 == stats.p99
    assert dataset_metadata.osl_stats.p99 == stats.p99
    assert dataset_metadata.seq_len_stats.p99 == seq_len_stats.p99

    # Minimums
    assert dataset_metadata.isl_stats.minimum == stats.minimum
    assert dataset_metadata.osl_stats.minimum == stats.minimum
    assert dataset_metadata.seq_len_stats.minimum == seq_len_stats.minimum

    # Maximums
    assert dataset_metadata.isl_stats.maximum == stats.maximum
    assert dataset_metadata.osl_stats.maximum == stats.maximum
    assert dataset_metadata.seq_len_stats.maximum == seq_len_stats.maximum


def test_stats_keeper(synthetic_perf_items) -> None:
    perf_items = synthetic_perf_items

    stats_keeper = StatsKeeper()
    for perf_item in perf_items:
        stats_keeper.register_request_perf_item(perf_item)
    statistics = stats_keeper.generate_statistics_summary()

    assert statistics.num_requests == len(perf_items)

    # Baseline metrics
    request_latencies = [x.end_timestamp - x.start_timestamp for x in perf_items]
    generation_times = [x.end_timestamp - x.time_on_first_token for x in perf_items]
    time_to_first_tokens = [x.time_on_first_token - x.start_timestamp for x in perf_items]
    generation_tokens = [len(x.tokens) - 1 for x in perf_items]
    output_tokens = [len(x.tokens) for x in perf_items]
    input_tokens = [x.num_input_tokens for x in perf_items]

    # Calculated metrics
    output_throughput_per_user = [osl / t_e2e for osl, t_e2e in zip(output_tokens, request_latencies)]
    output_speed_per_user = [gen_osl / t_gen for gen_osl, t_gen in zip(generation_tokens, generation_times)]
    intertoken_latencies_per_user = [t_gen / gen_osl for t_gen, gen_osl in zip(generation_times, generation_tokens)]

    # Sort lists
    start = min([x.start_timestamp for x in perf_items])
    end = max([x.end_timestamp for x in perf_items])

    # Aggregate Statistics
    assert statistics.total_output_tokens == sum(output_tokens)
    assert statistics.total_input_tokens == sum(input_tokens)
    assert statistics.total_latency_ns == end - start

    # Percentile statistics
    percent_output_throughput = PercentileStats.from_iterable(output_throughput_per_user)
    percent_generation_throughput = PercentileStats.from_iterable(output_speed_per_user)
    percent_intertoken_latencies = PercentileStats.from_iterable(intertoken_latencies_per_user)
    percent_request_latencies = PercentileStats.from_iterable(request_latencies)
    percent_time_to_first_tokens = PercentileStats.from_iterable(time_to_first_tokens)
    percent_output_tokens = PercentileStats.from_iterable(output_tokens)


    # Output Tokens
    assert statistics.token_percentiles.average == percent_output_tokens.average
    assert statistics.token_percentiles.p50 == percent_output_tokens.p50
    assert statistics.token_percentiles.p90 == percent_output_tokens.p90
    assert statistics.token_percentiles.p95 == percent_output_tokens.p95
    assert statistics.token_percentiles.p99 == percent_output_tokens.p99
    assert statistics.token_percentiles.minimum == percent_output_tokens.minimum
    assert statistics.token_percentiles.maximum == percent_output_tokens.maximum

    # Output Throughput Per User
    assert statistics.output_throughput_percentiles.average == percent_output_throughput.average
    assert statistics.output_throughput_percentiles.p50 == percent_output_throughput.p50
    assert statistics.output_throughput_percentiles.p90 == percent_output_throughput.p90
    assert statistics.output_throughput_percentiles.p95 == percent_output_throughput.p95
    assert statistics.output_throughput_percentiles.p99 == percent_output_throughput.p99
    assert statistics.output_throughput_percentiles.minimum == percent_output_throughput.minimum
    assert statistics.output_throughput_percentiles.maximum == percent_output_throughput.maximum

    # Generation Throughput Per User
    assert statistics.generation_tp_percentiles.average == percent_generation_throughput.average
    assert statistics.generation_tp_percentiles.p50 == percent_generation_throughput.p50
    assert statistics.generation_tp_percentiles.p90 == percent_generation_throughput.p90
    assert statistics.generation_tp_percentiles.p95 == percent_generation_throughput.p95
    assert statistics.generation_tp_percentiles.p99 == percent_generation_throughput.p99
    assert statistics.generation_tp_percentiles.minimum == percent_generation_throughput.minimum
    assert statistics.generation_tp_percentiles.maximum == percent_generation_throughput.maximum

    # Intertoken Latencies Per User
    assert statistics.tpot_percentiles.average == percent_intertoken_latencies.average
    assert statistics.tpot_percentiles.p50 == percent_intertoken_latencies.p50
    assert statistics.tpot_percentiles.p90 == percent_intertoken_latencies.p90
    assert statistics.tpot_percentiles.p95 == percent_intertoken_latencies.p95
    assert statistics.tpot_percentiles.p99 == percent_intertoken_latencies.p99
    assert statistics.tpot_percentiles.minimum == percent_intertoken_latencies.minimum

    # Request Latencies Per User
    assert statistics.request_latency_percentiles.average == percent_request_latencies.average
    assert statistics.request_latency_percentiles.p50 == percent_request_latencies.p50
    assert statistics.request_latency_percentiles.p90 == percent_request_latencies.p90
    assert statistics.request_latency_percentiles.p95 == percent_request_latencies.p95
    assert statistics.request_latency_percentiles.p99 == percent_request_latencies.p99
    assert statistics.request_latency_percentiles.minimum == percent_request_latencies.minimum

    # Time to First Token Per User
    assert statistics.ttft_percentiles.average == percent_time_to_first_tokens.average
    assert statistics.ttft_percentiles.p50 == percent_time_to_first_tokens.p50
    assert statistics.ttft_percentiles.p90 == percent_time_to_first_tokens.p90
    assert statistics.ttft_percentiles.p95 == percent_time_to_first_tokens.p95
    assert statistics.ttft_percentiles.p99 == percent_time_to_first_tokens.p99
    assert statistics.ttft_percentiles.minimum == percent_time_to_first_tokens.minimum






