
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import statistics
import math

from harness import emit_verify_tag


# =====================================================
# CONFIGURATION
# =====================================================

STUDENT_ID = "230103071"

PHYSICAL_CORES = 16
N = 256

MEMORY_SIZE = 64 * 10**6

FLOPS_PER_WORKER = 60_000_000

BLOCK_SIZES = [8, 16, 32, 64, 96, 128]

RUNS = 3

MASK = 0xFFFFFFFFFFFFFFFF


# =====================================================
# PART 3A: PARALLEL MEMORY BANDWIDTH
# =====================================================

def memory_copy_worker(src_name, dst_name, start, end):

    src_shm = shared_memory.SharedMemory(name=src_name)
    dst_shm = shared_memory.SharedMemory(name=dst_name)

    src = src_shm.buf
    dst = dst_shm.buf

    dst[start:end] = src[start:end]

    src.release()
    dst.release()

    src_shm.close()
    dst_shm.close()


def memory_bandwidth_trial():

    src_shm = shared_memory.SharedMemory(
        create=True,
        size=MEMORY_SIZE
    )

    dst_shm = shared_memory.SharedMemory(
        create=True,
        size=MEMORY_SIZE
    )

    processes = []

    p = PHYSICAL_CORES

    chunk = MEMORY_SIZE // p

    try:

        start = time.perf_counter()

        for worker_id in range(p):

            left = worker_id * chunk

            right = (
                MEMORY_SIZE
                if worker_id == p - 1
                else left + chunk
            )

            process = mp.Process(
                target=memory_copy_worker,
                args=(
                    src_shm.name,
                    dst_shm.name,
                    left,
                    right
                )
            )

            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        elapsed = time.perf_counter() - start

        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    "Memory worker failed"
                )

        return elapsed

    finally:

        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

        src_shm.close()
        dst_shm.close()

        src_shm.unlink()
        dst_shm.unlink()


def measure_memory_bandwidth():

    print("\n===== PART 3A: DRAM BANDWIDTH =====")

    samples = []

    for run in range(RUNS):

        elapsed = memory_bandwidth_trial()

        samples.append(elapsed)

        print(
            f"Run {run + 1}: {elapsed:.4f} seconds"
        )

    median_time = statistics.median(samples)

    # 64 MB read + 64 MB written
    bandwidth = (
        2 * MEMORY_SIZE / median_time / 1e9
    )

    print(f"Median time: {median_time:.6f} seconds")
    print(f"Sustained bandwidth: {bandwidth:.4f} GB/s")

    return bandwidth


# =====================================================
# PART 3A: COMPUTE CEILING
# =====================================================

def compute_worker(worker_id, iterations, output):

    # 16 independent scalar accumulators
    x0 = 1.01
    x1 = 1.02
    x2 = 1.03
    x3 = 1.04
    x4 = 1.05
    x5 = 1.06
    x6 = 1.07
    x7 = 1.08
    x8 = 1.09
    x9 = 1.10
    x10 = 1.11
    x11 = 1.12
    x12 = 1.13
    x13 = 1.14
    x14 = 1.15
    x15 = 1.16

    a = 1.00000001
    b = 0.00000001

    for _ in range(iterations):

        x0 = x0 * a + b
        x1 = x1 * a + b
        x2 = x2 * a + b
        x3 = x3 * a + b
        x4 = x4 * a + b
        x5 = x5 * a + b
        x6 = x6 * a + b
        x7 = x7 * a + b

        x8 = x8 * a + b
        x9 = x9 * a + b
        x10 = x10 * a + b
        x11 = x11 * a + b
        x12 = x12 * a + b
        x13 = x13 * a + b
        x14 = x14 * a + b
        x15 = x15 * a + b

    output[worker_id] = (
        x0 + x1 + x2 + x3 +
        x4 + x5 + x6 + x7 +
        x8 + x9 + x10 + x11 +
        x12 + x13 + x14 + x15
    )


def compute_trial():

    p = PHYSICAL_CORES

    # 16 variables, 2 FLOPs each per iteration
    iterations = FLOPS_PER_WORKER // 32

    output = mp.RawArray("d", p)

    processes = []

    start = time.perf_counter()

    for worker_id in range(p):

        process = mp.Process(
            target=compute_worker,
            args=(worker_id, iterations, output)
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    elapsed = time.perf_counter() - start

    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError("Compute worker failed")

    checksum = sum(output)

    assert math.isfinite(checksum)

    return elapsed


def measure_compute_ceiling():

    print("\n===== PART 3A: COMPUTE CEILING =====")

    samples = []

    for run in range(RUNS):

        elapsed = compute_trial()

        samples.append(elapsed)

        print(
            f"Run {run + 1}: {elapsed:.4f} seconds"
        )

    median_time = statistics.median(samples)

    total_flops = FLOPS_PER_WORKER * PHYSICAL_CORES

    # GFLOP/s
    compute_gflops = (
        total_flops / median_time / 1e9
    )

    print(f"Median time: {median_time:.4f} seconds")
    print(f"Compute ceiling: {compute_gflops:.4f} GFLOP/s")

    return compute_gflops


# =====================================================
# PART 3B: MATRIX INITIALIZATION
# =====================================================

def initialize_matrices(a_shm, b_shm):

    total = N * N

    A = (ctypes.c_double * total).from_buffer(a_shm.buf)
    B = (ctypes.c_double * total).from_buffer(b_shm.buf)

    for i in range(total):

        A[i] = ((i % 17) + 1) / 17.0
        B[i] = ((i % 13) + 1) / 13.0

    del A
    del B


# =====================================================
# GEMM WORKER
# =====================================================

def gemm_worker(
    a_name,
    b_name,
    c_name,
    n,
    row_start,
    row_end,
    variant,
    block_size
):

    a_shm = shared_memory.SharedMemory(name=a_name)
    b_shm = shared_memory.SharedMemory(name=b_name)
    c_shm = shared_memory.SharedMemory(name=c_name)

    total = n * n

    A = (ctypes.c_double * total).from_buffer(a_shm.buf)
    B = (ctypes.c_double * total).from_buffer(b_shm.buf)
    C = (ctypes.c_double * total).from_buffer(c_shm.buf)

    # ---------------------------------------------
    # VARIANT 1: NAIVE I-J-K
    # ---------------------------------------------

    if variant == 1:

        for i in range(row_start, row_end):

            for j in range(n):

                total_sum = 0.0

                for k in range(n):

                    total_sum += (
                        A[i * n + k] *
                        B[k * n + j]
                    )

                C[i * n + j] = total_sum

    # ---------------------------------------------
    # VARIANT 2: REORDERED I-K-J
    # ---------------------------------------------

    elif variant == 2:

        for i in range(row_start, row_end):

            for k in range(n):

                a_ik = A[i * n + k]

                for j in range(n):

                    C[i * n + j] += (
                        a_ik * B[k * n + j]
                    )

    # ---------------------------------------------
    # VARIANT 3: CACHE-BLOCKED I-K-J
    # ---------------------------------------------

    elif variant == 3:

        bs = block_size

        for ii in range(row_start, row_end, bs):

            i_end = min(ii + bs, row_end)

            for jj in range(0, n, bs):

                j_end = min(jj + bs, n)

                for kk in range(0, n, bs):

                    k_end = min(kk + bs, n)

                    for i in range(ii, i_end):

                        for k in range(kk, k_end):

                            a_ik = A[i * n + k]

                            for j in range(jj, j_end):

                                C[i * n + j] += (
                                    a_ik * B[k * n + j]
                                )

    else:

        raise ValueError("Invalid GEMM variant")

    # Release memoryviews before closing
    del A
    del B
    del C

    a_shm.close()
    b_shm.close()
    c_shm.close()


# =====================================================
# GEMM EXECUTION
# =====================================================

def gemm_trial(
    a_shm,
    b_shm,
    c_shm,
    variant,
    block_size=32
):

    p = PHYSICAL_CORES

    chunk = N // p

    processes = []

    # Reset C
    C = (ctypes.c_double * (N * N)).from_buffer(c_shm.buf)

    for i in range(N * N):
        C[i] = 0.0

    del C

    start = time.perf_counter()

    for worker_id in range(p):

        row_start = worker_id * chunk

        row_end = (
            N
            if worker_id == p - 1
            else row_start + chunk
        )

        process = mp.Process(
            target=gemm_worker,
            args=(
                a_shm.name,
                b_shm.name,
                c_shm.name,
                N,
                row_start,
                row_end,
                variant,
                block_size
            )
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    elapsed = time.perf_counter() - start

    for process in processes:
        if process.exitcode != 0:
            raise RuntimeError(
                f"GEMM worker failed: {process.exitcode}"
            )

    C = (ctypes.c_double * (N * N)).from_buffer(c_shm.buf)

    result = list(C)

    del C

    return elapsed, result


def benchmark_gemm(
    a_shm,
    b_shm,
    c_shm,
    variant,
    block_size=32
):

    samples = []
    last_result = None

    for run in range(RUNS):

        elapsed, result = gemm_trial(
            a_shm,
            b_shm,
            c_shm,
            variant,
            block_size
        )

        samples.append(elapsed)
        last_result = result

    median_time = statistics.median(samples)

    operations = 2 * (N ** 3)

    gflops = (
        operations / median_time / 1e9
    )

    return median_time, gflops, last_result


# =====================================================
# PART 3B: GEMM EXPERIMENTS
# =====================================================

def run_gemm_experiments():

    print("\n===== PART 3B: GEMM EXPERIMENTS =====")

    total = N * N * ctypes.sizeof(ctypes.c_double)

    a_shm = shared_memory.SharedMemory(
        create=True,
        size=total
    )

    b_shm = shared_memory.SharedMemory(
        create=True,
        size=total
    )

    c_shm = shared_memory.SharedMemory(
        create=True,
        size=total
    )

    try:

        initialize_matrices(a_shm, b_shm)

        # -----------------------------------------
        # VARIANT 1
        # -----------------------------------------

        print("\n--- VARIANT 1: NAIVE I-J-K ---")

        t1, g1, result1 = benchmark_gemm(
            a_shm,
            b_shm,
            c_shm,
            1
        )

        print(f"Median time: {t1:.4f} seconds")
        print(f"Performance: {g1:.6f} GFLOP/s")

        # -----------------------------------------
        # VARIANT 2
        # -----------------------------------------

        print("\n--- VARIANT 2: REORDERED I-K-J ---")

        t2, g2, result2 = benchmark_gemm(
            a_shm,
            b_shm,
            c_shm,
            2
        )

        print(f"Median time: {t2:.4f} seconds")
        print(f"Performance: {g2:.6f} GFLOP/s")

        # -----------------------------------------
        # VARIANT 3: TILE SWEEP
        # -----------------------------------------

        print("\n--- VARIANT 3: CACHE-BLOCKED ---")

        tile_results = {}

        for bs in BLOCK_SIZES:

            elapsed, gflops, result3 = benchmark_gemm(
                a_shm,
                b_shm,
                c_shm,
                3,
                bs
            )

            tile_footprint = 3 * bs * bs * 8

            intensity = bs / 12.0

            tile_results[bs] = {
                "time": elapsed,
                "gflops": gflops,
                "footprint": tile_footprint,
                "intensity": intensity
            }

            print(
                f"BS: {bs:3d} | "
                f"Time: {elapsed:.4f} s | "
                f"GFLOP/s: {gflops:.6f} | "
                f"Tile footprint: {tile_footprint:,} B | "
                f"Intensity: {intensity:.4f}"
            )

        # -----------------------------------------
        # BEST TILE
        # -----------------------------------------

        best_bs = max(
            tile_results,
            key=lambda bs: tile_results[bs]["gflops"]
        )

        best_gflops = tile_results[best_bs]["gflops"]

        print(f"\nBest block size: {best_bs}")
        print(f"Best blocked performance: {best_gflops:.6f} GFLOP/s")

        # -----------------------------------------
        # NUMERICAL CORRECTNESS
        # -----------------------------------------

        print("\n--- NUMERICAL VERIFICATION ---")

        max_error = max(
            abs(x - y)
            for x, y in zip(result2, result3)
        )

        print(f"Maximum absolute error: {max_error:.12e}")

        if max_error >= 1e-9:
            raise AssertionError(
                "GEMM numerical verification failed"
            )

        print("GEMM verification: PASSED")

        return g1, best_gflops, best_bs, tile_results

    finally:

        a_shm.close()
        b_shm.close()
        c_shm.close()

        a_shm.unlink()
        b_shm.unlink()
        c_shm.unlink()


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 70)
    print("PHASE 3: EMPIRICAL ROOFLINE AND CACHE-BLOCKED GEMM")
    print("=" * 70)

    print(f"Physical cores: {PHYSICAL_CORES}")
    print(f"Logical processors: {mp.cpu_count()}")
    print(f"Matrix size: {N} x {N}")
    print(f"Memory buffer: {MEMORY_SIZE:,} bytes")

    # PART 3A
    bandwidth = measure_memory_bandwidth()

    compute = measure_compute_ceiling()

    ridge_point = compute / bandwidth

    print("\n===== ROOFLINE CEILINGS =====")

    print(f"Memory bandwidth: {bandwidth:.4f} GB/s")
    print(f"Compute ceiling: {compute:.4f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.4f} FLOP/byte")

    # PART 3B
    g1, g3, best_bs, tile_results = (
        run_gemm_experiments()
    )

    # -----------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------

    print("\n===== PHASE 3 SUMMARY =====")

    print(f"Variant 1 GFLOP/s: {g1:.6f}")
    print(f"Best Variant 3 GFLOP/s: {g3:.6f}")
    print(f"Best block size: {best_bs}")

    print(f"Ridge point: {ridge_point:.6f}")

    print("\n===== PHASE 3 VERIFICATION =====")

    print(
        emit_verify_tag(
            "PHASE3",
            STUDENT_ID,
            g1,
            g3
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
