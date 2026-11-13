
import multiprocessing as mp
import math
import time

from harness import emit_verify_tag


# =========================================
# CONFIGURATION
# =========================================

STUDENT_ID = "230103071"

LOGICAL_CORES = mp.cpu_count()

PROCESS_COUNTS = list(range(1, LOGICAL_CORES + 1))

STRONG_N = 1_200_000
WEAK_N0 = 200_000

# =========================================
# STAGE A: SERIAL WORK
# =========================================

def stage_a(n):

    count = int(0.05 * n)

    total = 0.0

    for i in range(count):

        x = (i + 1) * 0.00001 + 0.5

        # 5 rounds = 10 math operations
        for _ in range(5):
            x = math.sin(x)
            x = math.sqrt(abs(x) + 1.0)

        total += x

    return total


# =========================================
# STAGE B: PARALLEL WORKER
# =========================================

def stage_b_worker(args):

    start, end = args

    total = 0.0

    for i in range(start, end):

        x = (i + 1) * 0.000001 + 0.5

        # 10 rounds:
        # sin + cos + FMA-like arithmetic
        # 30 conceptual operations per element
        for _ in range(10):

            a = math.sin(x)
            b = math.cos(x)

            x = a * 0.5 + b * 0.5 + x * 0.1

        total += x

    return total


# =========================================
# PARALLEL PIPELINE
# =========================================

def run_pipeline(pool, p, n):

    # Stage A: serial, in parent process
    stage_a_start = time.perf_counter()

    serial_result = stage_a(n)

    stage_a_time = (
        time.perf_counter() - stage_a_start
    )

    # Divide all N elements among p workers
    chunk = n // p

    tasks = []

    for worker_id in range(p):

        start = worker_id * chunk

        if worker_id == p - 1:
            end = n
        else:
            end = start + chunk

        tasks.append((start, end))

    # Stage B: parallel
    parallel_results = pool.map(
        stage_b_worker,
        tasks
    )

    # Keep results observable
    checksum = serial_result + sum(parallel_results)

    return stage_a_time, checksum


# =========================================
# STRONG SCALING
# =========================================

def strong_scaling():

    print("\n" + "=" * 70)
    print("PHASE 2A: STRONG SCALING")
    print("=" * 70)

    results = {}

    for p in PROCESS_COUNTS:

        # Pool creation is outside timed pipeline
        with mp.Pool(processes=p) as pool:

            start = time.perf_counter()

            stage_a_time, checksum = run_pipeline(
                pool,
                p,
                STRONG_N
            )

            total_time = time.perf_counter() - start

        results[p] = {
            "time": total_time,
            "stage_a": stage_a_time,
            "checksum": checksum
        }

        print(
            f"Processes: {p:2d} | "
            f"Time: {total_time:.4f} s | "
            f"Stage A: {stage_a_time:.4f} s"
        )

    # Calculate strong scaling metrics
    t1 = results[1]["time"]

    for p in PROCESS_COUNTS:

        tp = results[p]["time"]

        speedup = t1 / tp
        efficiency = speedup / p

        results[p]["speedup"] = speedup
        results[p]["efficiency"] = efficiency

    # Amdahl's serial fraction
    serial_fraction = (
        results[1]["stage_a"] / t1
    )

    amdahl_cap = (
        1 / serial_fraction
        if serial_fraction > 0
        else float("inf")
    )

    print("\nSTRONG SCALING SUMMARY")
    print(
        "Processes | Time(s) | Speedup | Efficiency"
    )

    for p in PROCESS_COUNTS:

        r = results[p]

        print(
            f"{p:9d} | "
            f"{r['time']:7.4f} | "
            f"{r['speedup']:7.3f} | "
            f"{r['efficiency']:10.3f}"
        )

    print(
        f"\nSerial fraction: {serial_fraction:.6f}"
    )

    print(
        f"Amdahl theoretical cap: {amdahl_cap:.4f}"
    )

    return results, amdahl_cap


# =========================================
# WEAK SCALING
# =========================================

def weak_scaling():

    print("\n" + "=" * 70)
    print("PHASE 2B: WEAK SCALING")
    print("=" * 70)

    results = {}

    for p in PROCESS_COUNTS:

        n = p * WEAK_N0

        with mp.Pool(processes=p) as pool:

            start = time.perf_counter()

            stage_a_time, checksum = run_pipeline(
                pool,
                p,
                n
            )

            total_time = time.perf_counter() - start

        results[p] = {
            "time": total_time,
            "stage_a": stage_a_time,
            "checksum": checksum,
            "n": n
        }

        print(
            f"Processes: {p:2d} | "
            f"N: {n:9,d} | "
            f"Time: {total_time:.4f} s"
        )

    # Weak scaling metrics
    t1 = results[1]["time"]

    print("\nWEAK SCALING SUMMARY")

    print(
        "Processes | Time(s) | Weak Efficiency | "
        "Gustafson Speedup"
    )

    for p in PROCESS_COUNTS:

        r = results[p]

        tp = r["time"]

        efficiency = t1 / tp

        alpha = r["stage_a"] / tp

        gustafson = p - alpha * (p - 1)

        r["efficiency"] = efficiency
        r["gustafson"] = gustafson

        print(
            f"{p:9d} | "
            f"{tp:7.4f} | "
            f"{efficiency:15.4f} | "
            f"{gustafson:17.4f}"
        )

    return results


# =========================================
# ARCHITECTURAL INFLECTION
# =========================================

def architectural_analysis(strong):

    print("\n" + "=" * 70)
    print("ARCHITECTURAL INFLECTION")
    print("=" * 70)

    p1 = 16
    p2 = 17

    s16 = strong[p1]["speedup"]
    s17 = strong[p2]["speedup"]

    marginal = s17 - s16

    print(f"Speedup at 16 processes: {s16:.4f}")
    print(f"Speedup at 17 processes: {s17:.4f}")
    print(f"Marginal speedup change: {marginal:.4f}")

    percentage = (
        (s17 - s16) / s16 * 100
    )

    print(
        f"Percentage change: {percentage:.2f}%"
    )


# =========================================
# MAIN
# =========================================

def main():

    print("=" * 70)
    print("PHASE 2: STRONG AND WEAK SCALING")
    print("=" * 70)

    print(f"Logical processors: {LOGICAL_CORES}")
    print(f"Strong workload: {STRONG_N:,}")
    print(f"Weak workload per process: {WEAK_N0:,}")

    strong, amdahl_cap = strong_scaling()

    weak = weak_scaling()

    architectural_analysis(strong)

    # Verification uses results at P_log = 24
    strong_log = strong[LOGICAL_CORES]["speedup"]

    gustafson_log = weak[LOGICAL_CORES]["gustafson"]

    print("\n===== PHASE 2 VERIFICATION =====")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            strong_log,
            gustafson_log
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
