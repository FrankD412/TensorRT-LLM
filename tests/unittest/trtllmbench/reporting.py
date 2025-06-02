import pytest
from tensorrt_llm.bench.dataclasses.reporting import PerfItemTuple


@pytest.fixture(scope="session")
def synthetic_report_data():
    tuples = []
    start_time = 0
    for i in range(10):
        end_time = start_time + 10  # Each request lasts for 10 iterations
        request_id = i + 1
        num_input_tokens = 5  # Fixed number of input tokens
        tokens = [j for j in range(2, 12)]  # Tokens with a length of at least 2
        decoding_iteration = 10  # Fixed number of iterations
        time_on_first_token = 1  # Arbitrary fixed value

        tuples.append(
            PerfItemTuple(
                start_timestamp=start_time,
                end_timestamp=end_time,
                request_id=request_id,
                num_input_tokens=num_input_tokens,
                response_is_final=True,  # Fixed value
                error=False,             # Fixed value
                tokens=tokens,
                decoding_iteration=decoding_iteration,
                time_on_first_token=time_on_first_token,
            )
        )
        start_time += 5  # Increment start time for the next request
    return tuples
