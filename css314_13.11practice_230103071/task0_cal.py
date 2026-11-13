
import multiprocessing as mp
import time

from harness import emit_verify_tag


# Replace this with your actual Student ID
STUDENT_ID = "230103071"

NUMBER_OF_READS = 1_000_000


def main():

    print("=" * 55)
    print("PHASE 0: HARDWARE CALIBRATION")
    print("=" * 55)

    # 1. Detect logical processors
    logical_processors = mp.cpu_count()

    print(f"Logical processors: {logical_processors}")

    # 2. Measure timer latency
    start = time.perf_counter_ns()

    last_value = 0

    for _ in range(NUMBER_OF_READS):
        last_value = time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_time = end - start

    average_latency = total_time / NUMBER_OF_READS

    total_time_ms = total_time / 1_000_000

    # 3. Display results
    print(f"Timer reads: {NUMBER_OF_READS:,}")
    print(f"Total time: {total_time:,} ns")
    print(f"Average timer latency: {average_latency:.4f} ns")

    # Prevent the last reading from being unused
    assert last_value > 0

    # 4. Verification hash
    print()

    print(
        emit_verify_tag(
            "PHASE0",
            STUDENT_ID,
            average_latency,
            total_time_ms
        )
    )


if __name__ == "__main__":
    main()

