
import multiprocessing as mp
import math
import time

from harness import emit_verify_tag


# ============================================================
# CONFIGURATION
# ============================================================

STUDENT_ID = "230103071"

N = 8000
PHYSICAL_CORES = 16

CHUNK_SIZES = [1, 8, 64, 512]


# ============================================================
# SKEWED WORKLOAD
# ============================================================

def calculate_item(i):
    """
    K(i) = 1 + floor((i / 40)^2)
         = 1 + floor(i^2 / 1600)

    Each iteration performs sin, cos, and sqrt.
    """

    k = 1 + (i * i) // 1600

    x = 0.5 + (i % 97) * 0.001

    for _ in range(k):
        x = (
            math.sin(x)
            + math.cos(x)
            + math.sqrt(x + 1.0)
        )

    return x


# ============================================================
# STATIC SCHEDULING WORKER
# ============================================================

def static_worker(worker_id, mode, worker_times, checksums):

    active_start = time.perf_counter()
    checksum = 0.0

    if mode == "block":

        start = worker_id * N // PHYSICAL_CORES
        end = (worker_id + 1) * N // PHYSICAL_CORES

        for i in range(start, end):
            checksum += calculate_item(i)

    elif mode == "cyclic":

        for i in range(worker_id, N, PHYSICAL_CORES):
            checksum += calculate_item(i)

    active_time = time.perf_counter() - active_start

    worker_times[worker_id] = active_time
    checksums[worker_id] = checksum


# ============================================================
# DYNAMIC QUEUE WORKER
# ============================================================

def dynamic_worker(
    worker_id,
    next_index,
    chunk_size,
    worker_times,
    checksums
):

    active_time = 0.0
    checksum = 0.0

    while True:

        # Claim the next chunk using a synchronized counter.
        with next_index.get_lock():

            start = next_index.value

            if start >= N:
                break

            end = min(start + chunk_size, N)

            next_index.value = end

        # Process the claimed chunk.
        # Queue waiting time is excluded from active time.
        chunk_start = time.perf_counter()

        for i in range(start, end):
            checksum += calculate_item(i)

        active_time += time.perf_counter() - chunk_start

    worker_times[worker_id] = active_time
    checksums[worker_id] = checksum


# ============================================================
# RUN ONE SCHEDULING EXPERIMENT
# ============================================================

def run_experiment(mode, chunk_size=None):

    ctx = mp.get_context("spawn")

    worker_times = ctx.Array(
        "d",
        PHYSICAL_CORES,
        lock=False
    )

    checksums = ctx.Array(
        "d",
        PHYSICAL_CORES,
        lock=False
    )

    processes = []

    if mode == "dynamic":

        next_index = ctx.Value("i", 0)

    else:

        next_index = None

    # Create all worker processes.
    for worker_id in range(PHYSICAL_CORES):

        if mode == "dynamic":

            process = ctx.Process(
                target=dynamic_worker,
                args=(
                    worker_id,
                    next_index,
                    chunk_size,
                    worker_times,
                    checksums
                )
            )

        else:

            process = ctx.Process(
                target=static_worker,
                args=(
                    worker_id,
                    mode,
                    worker_times,
                    checksums
                )
            )

        processes.append(process)

    # Measure total wall-clock time.
    wall_start = time.perf_counter()

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    wall_time = time.perf_counter() - wall_start

    # Check that every process completed successfully.
    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker process failed with exit code {process.exitcode}"
            )

    # Read per-worker active runtimes.
    times = list(worker_times)

    average_time = sum(times) / PHYSICAL_CORES

    maximum_time = max(times)

    if average_time > 0:
        imbalance = maximum_time / average_time
    else:
        imbalance = 0.0

    total_checksum = sum(checksums)

    return wall_time, imbalance, times, total_checksum


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4: SKEWED WORKLOADS AND DYNAMIC CHUNK QUEUE")
    print("=" * 70)

    print(f"Number of items: {N}")
    print(f"Physical cores / workers: {PHYSICAL_CORES}")
    print(f"Logical processors: {mp.cpu_count()}")

    print("\nWorkload:")
    print("K(i) = 1 + floor((i / 40)^2)")
    print("Operations: sin, cos, sqrt")

    # --------------------------------------------------------
    # STRATEGY 1: CONTIGUOUS BLOCK STATIC
    # --------------------------------------------------------

    print("\n===== STRATEGY 1: CONTIGUOUS BLOCK STATIC =====")

    block_wall, block_lambda, block_times, block_sum = (
        run_experiment("block")
    )

    print(f"Wall time: {block_wall:.4f} seconds")
    print(f"Load imbalance lambda: {block_lambda:.4f}")
    print(f"Checksum: {block_sum:.6f}")

    print("Worker active times:")

    for i, worker_time in enumerate(block_times):
        print(f"Worker {i:2d}: {worker_time:.4f} s")

    # --------------------------------------------------------
    # STRATEGY 2: CYCLIC ROUND-ROBIN STATIC
    # --------------------------------------------------------

    print("\n===== STRATEGY 2: CYCLIC ROUND-ROBIN STATIC =====")

    cyclic_wall, cyclic_lambda, cyclic_times, cyclic_sum = (
        run_experiment("cyclic")
    )

    print(f"Wall time: {cyclic_wall:.4f} seconds")
    print(f"Load imbalance lambda: {cyclic_lambda:.4f}")
    print(f"Checksum: {cyclic_sum:.6f}")

    print("Worker active times:")

    for i, worker_time in enumerate(cyclic_times):
        print(f"Worker {i:2d}: {worker_time:.4f} s")

    # --------------------------------------------------------
    # STRATEGY 3: DYNAMIC QUEUE
    # --------------------------------------------------------

    print("\n===== STRATEGY 3: DYNAMIC CHUNK QUEUE =====")

    dynamic_results = {}

    for chunk_size in CHUNK_SIZES:

        print(f"\n--- Chunk size C = {chunk_size} ---")

        wall, imbalance, worker_times, checksum = (
            run_experiment("dynamic", chunk_size)
        )

        dynamic_results[chunk_size] = {
            "wall": wall,
            "lambda": imbalance,
            "times": worker_times,
            "checksum": checksum
        }

        print(f"Wall time: {wall:.4f} seconds")
        print(f"Load imbalance lambda: {imbalance:.4f}")
        print(f"Checksum: {checksum:.6f}")

        print("Worker active times:")

        for i, worker_time in enumerate(worker_times):
            print(f"Worker {i:2d}: {worker_time:.4f} s")

    # --------------------------------------------------------
    # FIND BEST DYNAMIC CHUNK SIZE
    # --------------------------------------------------------

    best_chunk = min(
        dynamic_results,
        key=lambda c: dynamic_results[c]["wall"]
    )

    best_dynamic_wall = dynamic_results[best_chunk]["wall"]
    best_dynamic_lambda = dynamic_results[best_chunk]["lambda"]

    print("\n===== PHASE 4 SUMMARY =====")

    print(
        f"Block static wall time: "
        f"{block_wall:.4f} seconds"
    )

    print(
        f"Cyclic static wall time: "
        f"{cyclic_wall:.4f} seconds"
    )

    print("\nDynamic scheduling results:")

    for chunk_size in CHUNK_SIZES:

        result = dynamic_results[chunk_size]

        print(
            f"C = {chunk_size:3d} | "
            f"Wall time = {result['wall']:.4f} s | "
            f"Lambda = {result['lambda']:.4f}"
        )

    print(f"\nBest dynamic chunk size: {best_chunk}")
    print(f"Best dynamic wall time: {best_dynamic_wall:.4f} seconds")
    print(f"Best dynamic lambda: {best_dynamic_lambda:.4f}")

    # --------------------------------------------------------
    # VERIFICATION HASH
    # --------------------------------------------------------

    print("\n===== PHASE 4 VERIFICATION =====")

    emit_verify_tag(
        "PHASE4",
        STUDENT_ID,
        block_wall,
        best_dynamic_wall
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
