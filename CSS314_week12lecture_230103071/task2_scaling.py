
import multiprocessing as mp
import math
import time
import os
import statistics

# ==========================================
# PHASE 2: STRONG AND WEAK SCALING
# ==========================================

STUDENT_ID = "230103071"

P_PHYS = 16
P_LOG = 24

N_STRONG = 1_600_000
N0_WEAK = 250_000

WARMUP_RUNS = 2
TIMED_RUNS = 7

MASK = 0xFFFFFFFFFFFFFFFF


# ==========================================
# STAGE A: STRICTLY SERIAL COMPUTATION
# ==========================================

def stage_a(n):
    total = 0.0

    for i in range(n):
        x = (i + 1) * 0.00001

        for j in range(12):
            if j % 2 == 0:
                x = math.sin(x) + 0.000001
            else:
                x = math.sqrt(abs(x) + 0.000001)

        total += x

    return total


# ==========================================
# STAGE B: PARALLEL COMPUTATION
# ==========================================

def stage_b_worker(args):
    start, end = args

    result = 0.0

    for i in range(start, end):
        x = (i + 1) * 0.00001

        for j in range(35):
            x = x * 1.0000001 + math.sin(x + j * 0.001)

        result += x

    return result


def make_chunks(n, p):
    chunks = []

    for worker in range(p):
        start = worker * n // p
        end = (worker + 1) * n // p
        chunks.append((start, end))

    return chunks


# ==========================================
# ONE COMPLETE PIPELINE TRIAL
# ==========================================

def pipeline_trial(pool, n, p):

    serial_n = int(n * 0.05)

    start_total = time.perf_counter_ns()

    # Stage A runs in the parent process
    start_a = time.perf_counter_ns()
    serial_result = stage_a(serial_n)
    end_a = time.perf_counter_ns()

    stage_a_ms = (end_a - start_a) / 1_000_000

    # Stage B runs in the persistent process pool
    chunks = make_chunks(n, p)

    results = pool.map(stage_b_worker, chunks)

    # Consume results to ensure completed computation
    checksum = serial_result + sum(results)

    end_total = time.perf_counter_ns()

    total_ms = (end_total - start_total) / 1_000_000

    return total_ms, stage_a_ms, checksum


# ==========================================
# BENCHMARK FUNCTION
# ==========================================

def run_benchmark(pool, n, p):

    # Warmup runs
    for _ in range(WARMUP_RUNS):
        pipeline_trial(pool, n, p)

    total_times = []
    stage_a_times = []

    # Timed runs
    for run in range(TIMED_RUNS):

        total_ms, stage_a_ms, checksum = pipeline_trial(
            pool, n, p
        )

        total_times.append(total_ms)
        stage_a_times.append(stage_a_ms)

    median_total = statistics.median(total_times)
    median_stage_a = statistics.median(stage_a_times)

    return median_total, median_stage_a


# ==========================================
# FNV-1a VERIFICATION HASH
# ==========================================

def fnv1a_hash(data):

    value = 0xCBF29CE484222325

    for byte in data.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & MASK

    return f"{value:016X}"


def emit_verify_tag(task_id, student_id, p_log, metric1, metric2):

    data = (
        f"{task_id}|{student_id}|{p_log}|"
        f"{metric1:.9f}|{metric2:.9f}"
    )

    digest = fnv1a_hash(data)

    print(f"VERIFY_HASH[{task_id}]: {digest}")


# ==========================================
# STRONG SCALING
# ==========================================

def strong_scaling():

    print("\n" + "=" * 65)
    print("EXPERIMENT 2A: STRONG SCALING")
    print("=" * 65)

    results = {}

    for p in range(1, P_LOG + 1):

        print(f"\nTesting {p} processes...")

        # Pool is created outside timed region
        with mp.Pool(processes=p) as pool:

            median_ms, stage_a_ms = run_benchmark(
                pool, N_STRONG, p
            )

        results[p] = (median_ms, stage_a_ms)

        print(f"Median time: {median_ms:.3f} ms")
        print(f"Stage A time: {stage_a_ms:.3f} ms")

    t1 = results[1][0]

    print("\nSTRONG SCALING RESULTS")
    print("-" * 75)
    print(
        f"{'P':>4} {'Time(ms)':>14} "
        f"{'Speedup':>14} {'Efficiency':>14}"
    )

    for p in range(1, P_LOG + 1):

        tp = results[p][0]

        speedup = t1 / tp
        efficiency = speedup / p

        print(
            f"{p:4d} {tp:14.3f} "
            f"{speedup:14.4f} {efficiency:13.2%}"
        )

    return results


# ==========================================
# WEAK SCALING
# ==========================================

def weak_scaling():

    print("\n" + "=" * 65)
    print("EXPERIMENT 2B: WEAK SCALING")
    print("=" * 65)

    results = {}

    for p in range(1, P_LOG + 1):

        n = p * N0_WEAK

        print(f"\nTesting {p} processes, N = {n:,}...")

        with mp.Pool(processes=p) as pool:

            median_ms, stage_a_ms = run_benchmark(
                pool, n, p
            )

        results[p] = (median_ms, stage_a_ms)

        print(f"Median time: {median_ms:.3f} ms")
        print(f"Stage A time: {stage_a_ms:.3f} ms")

    t1 = results[1][0]

    print("\nWEAK SCALING RESULTS")
    print("-" * 85)

    print(
        f"{'P':>4} {'Time(ms)':>14} "
        f"{'Efficiency':>14} {'Gustafson':>14}"
    )

    for p in range(1, P_LOG + 1):

        tp, stage_a_ms = results[p]

        efficiency = t1 / tp

        alpha = stage_a_ms / tp

        gustafson = p - alpha * (p - 1)

        print(
            f"{p:4d} {tp:14.3f} "
            f"{efficiency:13.2%} {gustafson:14.4f}"
        )

    return results


# ==========================================
# ANALYTICAL CALCULATIONS
# ==========================================

def analytical_analysis(strong, weak):

    print("\n" + "=" * 65)
    print("ANALYTICAL ANALYSIS: AMDAHL AND GUSTAFSON")
    print("=" * 65)

    t1 = strong[1][0]
    stage_a = strong[1][1]

    serial_fraction = stage_a / t1

    theoretical_max = 1 / serial_fraction

    p = P_LOG

    amdahl_speedup = 1 / (
        serial_fraction +
        (1 - serial_fraction) / p
    )

    observed_speedup = t1 / strong[p][0]

    weak_p_time, weak_p_stage_a = weak[p]

    alpha = weak_p_stage_a / weak_p_time

    gustafson_speedup = p - alpha * (p - 1)

    print(f"Serial fraction: {serial_fraction:.6f}")
    print(f"Amdahl theoretical maximum: {theoretical_max:.4f}")
    print(f"Amdahl speedup at {p} processes: {amdahl_speedup:.4f}")
    print(f"Observed strong speedup: {observed_speedup:.4f}")

    print(f"\nWeak scaling serial fraction: {alpha:.6f}")
    print(f"Gustafson speedup at {p}: {gustafson_speedup:.4f}")

    # Physical-to-logical core boundary
    before = t1 / strong[P_PHYS][0]
    after = t1 / strong[P_PHYS + 1][0]

    marginal_before = before - (t1 / strong[P_PHYS - 1][0])
    marginal_after = after - before

    if marginal_before != 0:
        drop = (
            (marginal_before - marginal_after)
            / abs(marginal_before)
        ) * 100
    else:
        drop = float("nan")

    print("\nCPU ARCHITECTURE ANALYSIS")
    print(f"Speedup at 16 processes: {before:.4f}")
    print(f"Speedup at 17 processes: {after:.4f}")
    print(f"Marginal speedup change: {drop:.2f}%")

    emit_verify_tag(
        "PHASE2",
        STUDENT_ID,
        P_LOG,
        observed_speedup,
        gustafson_speedup
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    mp.freeze_support()

    print("PARALLEL COMPUTING - PHASE 2")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical processors: {P_LOG}")
    print(f"Student ID: {STUDENT_ID}")

    strong_results = strong_scaling()

    weak_results = weak_scaling()

    analytical_analysis(
        strong_results,
        weak_results
    )

    print("\nPHASE 2 COMPLETED.")
