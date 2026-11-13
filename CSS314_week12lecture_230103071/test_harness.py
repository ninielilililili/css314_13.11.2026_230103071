from harness import BenchStats

stats = BenchStats(10.0, 12.0, 2.0, 15.0)

print("Minimum:", stats.min_ms)
print("Median:", stats.median_ms)
print("IQR:", stats.iqr_ms)
print("Maximum:", stats.max_ms)
