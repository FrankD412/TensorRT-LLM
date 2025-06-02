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

