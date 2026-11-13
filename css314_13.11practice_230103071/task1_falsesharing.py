
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import statistics

from harness import emit_verify_tag


# ---------------- CONFIGURATION ----------------

M = 16_000_000
SHM_SIZE = 4096
PHYSICAL_CORES = 16
LOGICAL_CORES = mp.cpu_count()

STRIDES = [1, 2, 4, 8, 16, 32]
PROCESS_COUNTS = [1, 2, 4, 8, 16, 24]

RUNS = 3

MASK = 0xFFFFFFFFFFFFFFFF
MULTIPLIER = 6364136223846793005


# ---------------- WORKER ----------------

def fs_worker(shm_name, slot_idx, iters):

    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    for i in range(iters):
        acc = (
            acc * MULTIPLIER + (i | 1)
        ) & MASK

        arr[slot_idx] = acc

    del arr
    shm.close()


# ---------------- SINGLE TRIAL ----------------

def run_trial(p, stride):

    shm = shared_memory.SharedMemory(
        create=True,
        size=SHM_SIZE
    )

    processes = []

    base_iters = M // p
    remainder = M % p

    start = time.perf_counter_ns()

    try:
        for worker_id in range(p):

            slot_idx = worker_id * stride

            iters = base_iters + (
                1 if worker_id < remainder else 0
            )

            process = mp.Process(
                target=fs_worker,
                args=(
                    shm.name,
                    slot_idx,
                    iters
                )
            )

            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

        end = time.perf_counter_ns()

        return (end - start) / 1e6

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

        shm.close()
        shm.unlink()


# ---------------- BENCHMARK ----------------

def benchmark(p, stride):

    samples = []

    for _ in range(RUNS):
        elapsed = run_trial(p, stride)
        samples.append(elapsed)

    samples.sort()

    median = statistics.median(samples)

    iqr = (
        samples[(3 * RUNS) // 4]
        - samples[RUNS // 4]
    )

    throughput = M / (median * 1000)

    return {
        "median": median,
        "iqr": iqr,
        "throughput": throughput
    }


# ---------------- METRICS ----------------

def calculate_metrics(t1, tp, p):

    speedup = t1 / tp
    efficiency = speedup / p

    if p == 1:
        serial_fraction = None
    else:
        serial_fraction = (
            (1 / speedup) - (1 / p)
        ) / (1 - (1 / p))

    return speedup, efficiency, serial_fraction


# ---------------- MAIN ----------------

def main():

    print("=" * 65)
    print("PHASE 1: FALSE SHARING AND KARp-FLATT DIAGNOSTICS")
    print("=" * 65)

    print(f"Logical processors: {LOGICAL_CORES}")
    print(f"Physical cores: {PHYSICAL_CORES}")
    print(f"Total operations: {M:,}")
    print(f"Benchmark repetitions: {RUNS}")

    # =========================================
    # PART A: STRIDE SWEEP
    # =========================================

    print("\n===== STRIDE SWEEP (P = 16) =====")

    stride_results = {}

    for stride in STRIDES:

        result = benchmark(
            PHYSICAL_CORES,
            stride
        )

        stride_results[stride] = result

        print(
            f"Stride: {stride:2d} | "
            f"Offset: {stride * 8:3d} bytes | "
            f"Median: {result['median']:.3f} ms | "
            f"IQR: {result['iqr']:.3f} ms | "
            f"Throughput: {result['throughput']:.3f} Mops/s"
        )

    # =========================================
    # PART B: PROCESS SCALING
    # =========================================

    scaling_results = {}

    for stride in [1, 16]:

        print(
            f"\n===== PROCESS SCALING: STRIDE {stride} ====="
        )

        scaling_results[stride] = {}

        for p in PROCESS_COUNTS:

            result = benchmark(p, stride)

            scaling_results[stride][p] = result

        t1 = scaling_results[stride][1]["median"]

        print(
            "\nProcesses | Median ms | Speedup | "
            "Efficiency | Karp-Flatt"
        )

        for p in PROCESS_COUNTS:

            tp = scaling_results[stride][p]["median"]

            speedup, efficiency, serial = (
                calculate_metrics(t1, tp, p)
            )

            serial_text = (
                "N/A" if serial is None
                else f"{serial:.5f}"
            )

            print(
                f"{p:9d} | "
                f"{tp:9.3f} | "
                f"{speedup:7.3f} | "
                f"{efficiency:10.3f} | "
                f"{serial_text:>10}"
            )

    # =========================================
    # PART C: VERIFICATION HASH
    # =========================================

    stride1_time = stride_results[1]["median"]
    stride16_time = stride_results[16]["median"]

    print("\n===== PHASE 1 VERIFICATION =====")

    print(
        emit_verify_tag(
            "PHASE1",
            "230103071",
            stride1_time,
            stride16_time
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
