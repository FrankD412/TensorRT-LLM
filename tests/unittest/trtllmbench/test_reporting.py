from math import ceil
from typing import List, Tuple
import pytest
from tensorrt_llm.bench.dataclasses.general import DatasetMetadata
from tensorrt_llm.bench.dataclasses.reporting import PerfItemTuple, StatsKeeper, ReportUtility
from tensorrt_llm.bench.dataclasses.statistics import PercentileStats


def get_synthetic_dataset_metadata(perf_items: List[PerfItemTuple]) -> DatasetMetadata:
    isls = [x.num_input_tokens for x in perf_items]
    osls = [len(x.tokens) for x in perf_items]
    sqls = [isl + osl for isl, osl in zip(isls, osls)]

    return DatasetMetadata(
        isl_stats=PercentileStats.from_iterable(isls),
        osl_stats=PercentileStats.from_iterable(osls),
        seq_len_stats=PercentileStats.from_iterable(sqls),
        num_requests=len(perf_items),
    )


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
def synthetic_perf_items() -> List[PerfItemTuple]:
    perf_items = []
    request_info = get_synthetic_request_info(10)
    start_time = 0
    # Each request lasts for 10 iterations
    end_time = (start_time + 10) * 1e9 # convert to nanoseconds

    for req in request_info:
        ttft_time = ceil((end_time - start_time) * .15)
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


def test_stats_keeper(synthetic_perf_items) -> None:
    perf_items = synthetic_perf_items

    stats_keeper = StatsKeeper()
    for perf_item in perf_items:
        stats_keeper.register_request_perf_item(perf_item)
    statistics = stats_keeper.generate_statistics_summary()

    assert statistics.num_complete == len(perf_items)

    # Baseline metrics
    request_latencies = [x.end_timestamp - x.start_timestamp for x in perf_items]
    generation_times = [x.end_timestamp - x.time_on_first_token for x in perf_items]
    time_to_first_tokens = [x.time_on_first_token for x in perf_items]
    generation_tokens = [len(x.tokens) - 1 for x in perf_items]
    output_tokens = [len(x.tokens) for x in perf_items]
    input_tokens = [x.num_input_tokens for x in perf_items]

    # Calculated metrics
    output_throughput_per_user = [osl / t_e2e for osl, t_e2e in zip(output_tokens, request_latencies)]
    output_speed_per_user = [gen_osl / t_gen for gen_osl, t_gen in zip(generation_tokens, generation_times)]
    intertoken_latencies_per_user = [t_gen / gen_osl for gen_osl, t_gen in zip(output_tokens, generation_times)]

    # Aggregate Statistics
    assert statistics.total_output_tokens == sum(output_tokens)
    assert statistics.total_input_tokens == sum(input_tokens)
    assert statistics.total_latency_ns == sum(x.end_timestamp - x.start_timestamp for x in perf_items)

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






