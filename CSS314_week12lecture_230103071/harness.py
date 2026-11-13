
from dataclasses import dataclass
import time
import statistics


@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float


def fnv1a_hash(text):
    """Calculate a 64-bit FNV-1a hash."""
    offset_basis = 14695981039346656037
    prime = 1099511628211
    mask = 0xFFFFFFFFFFFFFFFF

    hash_value = offset_basis

    for byte in text.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * prime) & mask

    return hash_value


def run_benchmark(fn, warmup_runs=3, timed_runs=9):
    """Warm up, time repeated executions, and calculate statistics."""

    for _ in range(warmup_runs):
        fn()

    times = []

    for _ in range(timed_runs):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()

        elapsed_ms = (end - start) / 1_000_000
        times.append(elapsed_ms)

    times.sort()

    median_ms = statistics.median(times)
    q1 = times[len(times) // 4]
    q3 = times[(3 * len(times)) // 4]

    return BenchStats(
        min_ms=times[0],
        median_ms=median_ms,
        iqr_ms=q3 - q1,
        max_ms=times[-1]
    )


def emit_verify_tag(task_id, student_seed, hw,
                    primary_metric, secondary_metric):

    payload = (
        f"{task_id}|{student_seed}|{hw}|"
        f"{primary_metric:.4f}|{secondary_metric:.4f}"
    )

    signature = fnv1a_hash(payload)

    result = (
        f"VERIFY_HASH [{task_id}]: "
        f"{signature:016X} (P_log={hw})"
    )

    print(result)
    return result
