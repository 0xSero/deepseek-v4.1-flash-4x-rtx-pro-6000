# Study protocol and upstream guidance

This study applies Local Inference Lab's published guidance to DeepSeek-V4.1-Flash on four RTX PRO 6000 Blackwell GPUs. Installation is complete; performance and quality qualification remain ongoing. No global SOTA claim is made.

## Sources and scope

| Source | Pinned revision | Application |
|---|---|---|
| [SGLang skills](https://github.com/local-inference-lab/sglang/tree/c662a9fd6e35c2548c45c09ece8d3f47846d61f2/.claude/skills) | c662a9fd6e35c2548c45c09ece8d3f47846d61f2 | Installed sglang-sota-performance, llm-serving-auto-benchmark, llm-torch-profiler-analysis with their helpers and references. Apache-2.0 license retained. These are skills distributed in the fork; authorship is not attributed exclusively to the fork owner. |
| [B12x guidance](https://github.com/local-inference-lab/b12x/blob/01ac763bef7503d934c4c2874bfd005799ca24a1/AGENTS.md) | 01ac763bef7503d934c4c2874bfd005799ca24a1 | Correctness before timings, real production paths, graph replay, fixed workspace, exact hardware and artifact identity, bounded experiments and structural profiling. Repository LICENSE contains Apache-2.0. |
| [Inference benchmark](https://github.com/local-inference-lab/llm-inference-bench/tree/ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f) | ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f | Distinguish sustained, burst and end-to-end throughput; verify actual concurrency; inspect repetition; paired quality comparisons. README declares MIT. |
| [Archived KLD skill](https://github.com/local-inference-lab/rtx6kpro/blob/b0fd4061de01c1542d8535a2c68074327482a3af/scripts/kld-eval-skill.md) | b0fd4061de01c1542d8535a2c68074327482a3af | Reviewed, excluded from execution: explicitly Qwen3.5-specific, not a general route-controlled MoE protocol. Its numerical thresholds do not apply to DeepSeek. |

The B12x vLLM capture helper was reviewed but not installed: it assumes a different runtime, endpoint and another operator's trace destination. Generic launch/cleanup examples and default model choices are not applied to this live service. User requirements retain DSpark, CUDA graphs, DeepSeek only and the single-host storage scope.

## Fixed experiment contract

- Checkpoint revision fb2764a5cf321eaa5070ca8f9e892818f477c16d, native Engram FP8/E8M0; no new quantization arm.
- Four GPUs, TP4/EP4, 275 W per GPU, DSpark block 5, eight slots, memory fraction 0.95, context 524288 and pool cap 4200000 for the current candidate. Capacity allocation and populated capacity remain separate facts.
- Preserve the original 45-cell workload: 512, 2048, 8192, 32768, 65536, 131072, 200000, 400000 and 500000 input tokens at C1/2/4/6/8, each with 8192 forced output tokens. Synthetic forced-output throughput is not a quality evaluation.
- Keep model/tokenizer, endpoint, input IDs, sampling, output length, cache budget and scheduler settings identical between storage arms. Record prompt-cache state separately from DDR5 Engram-cache warmth and operating-system page-cache state.
- Save sanitized launch commands, image digest, source and workload hashes, GPU UUIDs, clocks, power/throttle state, exact CLI help, raw per-request timings, errors, observed overlap, allocator evidence, retractions and cache counters. Missing measurements stay unknown.

## Bounded next experiments

| Order | Experiment | Budget and promotion gate |
|---|---|---|
| 1 | Finish existing native baseline and queued diagnostics | Preserve the running sweep; require all 45 receipts. Check 8 x 500k populated KV with engine evidence. Inspect video, real-checkpoint byte parity and GPU graph replay separately. |
| 2 | B12x io_uring with the same 64 GiB DDR5 budget | Byte parity and dynamic graph replay first, then real-model quality smoke. Screen 512/C1, 8192/C8, 400k/C1 and 400k/C8. No reader speedup claim from CPU tests. |
| 3 | Repeat matched storage comparisons | Three repetitions on screening cells, balanced A/B order where restart cost permits, both arms warmed by the same procedure. Treat differences below 5% as inconclusive pending repeatability. Preserve cold-start measurements separately. |
| 4 | Increase prefill chunk | Test 4096, then 8192, then 16384 only after the preceding candidate passes long-context/C8 memory and correctness checks. One variable at a time; known 4096 failure remains in the ledger. |
| 5 | DDR5 cache sizing | Hold reader and chunk fixed; screen 32 versus 64 GiB. Larger caches require measured host-memory headroom and no active swap growth. Full 188.83 GiB native Engram cannot fit 125 GiB host RAM. |
| 6 | Profile the limiting cases | Separate prefill/decode traces, bounded active steps, actual long-context and concurrency shapes. Inspect kernel time, CPU/GPU idle gaps, collectives and I/O waits before further kernel changes. Never include profiler-overhead runs in speed tables. |
| 7 | Full acceptance sweep | Only qualifying configurations receive the full 45-cell matrix, quality comparison, 4M populated-capacity proof, API/UI tests and soak. |

At most ten new configurations in the initial tuning round. A failed configuration ends that branch, not the overall goal. Two successive knob changes without a repeatable gain trigger profiling rather than a wider blind sweep. Profile decode with populated long-context KV as well as the helper's one-token diagnostic; the latter cannot explain long-context attention costs alone.

## Ranking and quality

Rank only accepted candidates, prioritizing total shared-window decode tok/s as requested, while retaining per-request rates, prefill, TTFT and burst gaps separately. Reject regressions exceeding 5% in any tested concurrency/length bucket unless explicitly presented as a tradeoff; repeat noisy comparisons. No absolute latency SLA has been specified, so do not manufacture SLA-passing goodput or p99 confidence from a handful of requests.

Native storage changes require byte-identical lookup outputs, finite/nonzero real-model results, dynamic-ID CUDA graph replay and paired semantic checks. Quality runs use natural termination and fixed prompts; forced 8192-token throughput outputs are unsuitable as quality scores. Compare identical text, reasoning, image and original-frame video tasks across arms. Repeated/incorrect video frames remain an unresolved quality failure, not a supported feature. Confidence intervals and paired item flips belong to sufficiently sized quality evaluations, not tiny smoke checks.

## Evidence audit applied to the published snapshot

See [protocol-audit.json](results/protocol-audit.json). The audit validates existing result structure, full client overlap, common manifest identity, positive shared measurement windows and explicit missing acceptance gates. It does not upgrade incomplete results to production acceptance. Raw observations remain in the README tables; no synthetic metric is substituted for missing p99, goodput, quality or populated-capacity evidence.
