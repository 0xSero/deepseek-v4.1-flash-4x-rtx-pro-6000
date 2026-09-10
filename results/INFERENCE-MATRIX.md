# Prefill and total decode speed

Total decode is the sum of actual token deliveries across all requests over the same shared wall-clock interval. Per-request decode is the median over that identical interval. Prefill is effective burst prefill including queueing and mixed decode. A dash means the wave never had all requested streams decoding simultaneously; it is not zero throughput.

| Input tokens/request | Requested C | Prefill tok/s | Total decode tok/s | Decode tok/s/request | Shared seconds | Peak overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 1070.03 | 152.30 | 152.30 | 6.72 | 1 |
| 512 | 2 | 1848.85 | 259.44 | 129.72 | 7.60 | 2 |
| 512 | 4 | 2976.91 | 409.56 | 102.80 | 9.82 | 4 |
| 512 | 6 | 3216.80 | 564.41 | 94.66 | 10.69 | 6 |
| 512 | 8 | 3477.04 | 582.04 | 72.89 | 13.12 | 8 |
| 2048 | 1 | 5875.07 | 160.26 | 160.26 | 6.38 | 1 |
| 2048 | 2 | 6205.04 | 323.78 | 161.89 | 6.25 | 2 |
| 2048 | 4 | 6227.07 | 491.53 | 121.34 | 7.95 | 4 |
| 2048 | 6 | 6359.51 | 594.52 | 100.80 | 10.02 | 6 |
| 2048 | 8 | 1169.06 | — | — | 0.00 | 7 |
| 8192 | 1 | 7457.54 | 157.92 | 157.92 | 6.48 | 1 |
| 8192 | 2 | 7614.83 | 312.88 | 156.44 | 6.43 | 2 |
| 8192 | 4 | 7500.59 | 449.20 | 112.35 | 9.03 | 4 |
| 8192 | 6 | 7580.93 | 646.41 | 107.66 | 9.10 | 6 |
| 8192 | 8 | 3166.70 | — | — | 0.00 | 7 |
| 32768 | 1 | 7816.60 | 187.42 | 187.42 | 5.46 | 1 |
| 32768 | 2 | 7771.20 | 321.50 | 160.75 | 6.24 | 2 |
| 32768 | 4 | 7704.31 | 500.76 | 123.59 | 7.67 | 4 |
| 32768 | 6 | 7614.22 | 589.67 | 98.57 | 9.82 | 6 |
| 32768 | 8 | 5947.31 | — | — | 0.00 | 7 |
| 65536 | 1 | 7447.35 | 169.18 | 169.18 | 6.05 | 1 |
| 65536 | 2 | 7449.80 | 309.88 | 154.94 | 6.54 | 2 |
| 65536 | 4 | 7427.78 | 502.34 | 125.84 | 7.86 | 4 |
| 65536 | 6 | 7402.73 | 565.61 | 95.99 | 10.17 | 6 |
| 65536 | 8 | 6462.15 | — | — | 0.00 | 7 |
| 131072 | 1 | 7151.56 | 174.05 | 174.05 | 5.88 | 1 |
| 131072 | 2 | 7133.28 | 305.92 | 152.96 | 6.60 | 2 |
| 131072 | 4 | 7137.90 | 492.31 | 120.94 | 7.72 | 4 |
| 131072 | 6 | 7128.33 | 617.78 | 102.36 | 9.44 | 6 |
| 131072 | 8 | 6663.41 | — | — | 0.00 | 7 |
| 200000 | 1 | 6860.88 | 190.73 | 190.73 | 5.36 | 1 |
| 200000 | 2 | 6906.85 | 309.87 | 154.93 | 6.46 | 2 |
| 200000 | 4 | 6861.24 | 473.63 | 119.09 | 8.41 | 4 |
| 200000 | 6 | 6901.21 | 586.75 | 97.82 | 10.24 | 6 |
| 200000 | 8 | 6549.36 | — | — | 0.00 | 7 |
| 400000 | 1 | 6128.53 | 166.75 | 166.75 | 6.14 | 1 |
| 400000 | 2 | 6171.34 | 248.73 | 124.37 | 8.13 | 2 |
| 400000 | 4 | 6168.66 | 425.04 | 106.29 | 9.54 | 4 |
| 400000 | 6 | 6178.22 | 500.57 | 83.36 | 11.69 | 6 |
| 400000 | 8 | 6045.83 | — | — | 0.00 | 7 |

## Sustained populated KV capacity

A separate synthetic stress run used seven distinct-prefix 400,000-token inputs and forced 8,192 output tokens each (`ignore_eos=true`). All seven completed without request errors. Runtime logs recorded 2,853,120 populated full tokens with seven requests decoding and CUDA graphs active. Allocated pool: 2,865,664 logical tokens. Native checkpoint and KV formats were unchanged; this is not an all-FP8 weight conversion.

| Input tokens/request | Concurrent requests | Effective prefill tok/s | Total decode tok/s | Median decode tok/s/request | Shared decode seconds |
|---:|---:|---:|---:|---:|---:|
| 400,000 | 7 | 6,352.59 | 721.21 | 103.69 | 74.48 |

See [machine-readable result](capacity-2800000.json). Decode counts actual delivered token IDs over the same overlapping interval for all seven streams. This establishes capacity and synthetic throughput, not response quality; its forced-length workload differs from the matrix above.
