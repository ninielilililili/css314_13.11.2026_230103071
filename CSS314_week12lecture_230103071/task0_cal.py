
import multiprocessing as mp
import time

from harness import emit_verify_tag


def main():
    student_id = 230103071

    # Number of logical processors
    logical_processors = mp.cpu_count()

    # Number of timer reads
    iterations = 2_000_000

    # Measure timer overhead
    start = time.perf_counter_ns()

    for _ in range(iterations):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_ns = total_ns / iterations

    # Display results
    print("===== PHASE 0: CALIBRATION =====")
    print(f"Logical processors: {logical_processors}")
    print(f"Timer reads: {iterations:,}")
    print(f"Total time: {total_ns:,} ns")
    print(f"Average timer latency: {average_ns:.4f} ns")

    # Generate verification tag
    emit_verify_tag(
        "PHASE0",
        student_id,
        logical_processors,
        logical_processors,
        average_ns
    )


if __name__ == "__main__":
    main()
