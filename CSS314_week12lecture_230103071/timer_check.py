import time

iterations = 2_000_000

start = time.perf_counter_ns()

for _ in range(iterations):
    time.perf_counter_ns()

end = time.perf_counter_ns()

total_time = end - start
average_time = total_time / iterations

print("Number of clock reads:", iterations)
print("Total time:", total_time, "ns")
print("Average call latency:", average_time, "ns")
