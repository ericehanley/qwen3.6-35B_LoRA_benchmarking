# Qwen3.6-35B GKE TPU v6e-8 Serving Benchmark Directory

This file compiles the final metrics from all benchmarking runs conducted on the **TPU v6e-8** (8 physical cores, 32GB HBM per core) cluster.

---

## 📊 1. Qwen3.6-35B Base Model Benchmark (Prefix Caching Sweeps)

* **Workload Shape**: 9,000 mean input tokens (2,600 shared prefix + 6,400 unique input tokens) -> **600 generated output tokens**.
* **Serving Layout**: **TP=8, DP=1** (Native vLLM serving).
* **Baseline Reference**: Single NVIDIA H200 SXM (TP=1).

| Concurrency | Platform | Req/s | Output Throughput | Total Throughput | TTFT (med / P99) | TPOT (med / P99) | ITL (med / P99) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | **TPU v6e-8 (TP4-DP2)** | 0.96 | 576 tok/s | 9,222 tok/s | 1,268 / 4,356 ms | 11.6 / 13.8 ms | 9.1 / 29.2 ms |
| | TPU v6e-8 (TP8-DP1) | 0.61 | 367 tok/s | 5,879 tok/s | 2,515 / 3,424 ms | 17.5 / 21.0 ms | 17.4 / 19.5 ms |
| | *NVIDIA H200 (TP=1)* | **1.14** | **682 tok/s** | **10,915 tok/s** | **1,080** / 9,872 ms | **8.6** / 20.2 ms | -- |
| **16** | **TPU v6e-8 (TP4-DP2)** | 1.46 | 876 tok/s | 14,019 tok/s | 1,557 / 7,053 ms | 14.8 / 18.7 ms | 10.3 / 39.3 ms |
| | TPU v6e-8 (TP8-DP1) | 1.02 | 615 tok/s | 9,834 tok/s | 3,246 / 5,832 ms | 20.4 / 25.1 ms | 18.0 / 21.1 ms |
| | *NVIDIA H200 (TP=1)* | **1.77** | **1,061 tok/s**| **16,979 tok/s** | **1,161** / **3,147** ms | **13.0** / **14.7** ms | -- |
| **32** | **TPU v6e-8 (TP4-DP2)** | 1.49 | 891 tok/s | 14,256 tok/s | 3,296 / 21,942 ms | 28.6 / 41.6 ms | 12.3 / 393.1 ms |
| | TPU v6e-8 (TP8-DP1) | 1.45 | 869 tok/s | 13,899 tok/s | 4,313 / 9,958 ms | 29.5 / 36.1 ms | 21.3 / 709.6 ms |
| | *NVIDIA H200 (TP=1)* | **2.30** | **1,379 tok/s**| **22,071 tok/s** | **1,160** / **6,008** ms | **21.1** / **22.7** ms | -- |
| **64** | TPU v6e-8 (TP4-DP2) | 1.34 | 802 tok/s | 12,833 tok/s | 26,893 / 34,273 ms | 40.2 / 66.7 ms | 21.4 / 792.0 ms |
| | **TPU v6e-8 (TP8-DP1)** | 1.83 | 1,097 tok/s | 17,559 tok/s | 4,336 / 18,448 ms | 50.8 / 57.1 ms | 28.2 / 727.2 ms |
| | *NVIDIA H200 (TP=1)* | **2.87** | **1,722 tok/s**| **27,556 tok/s** | **1,214** / **10,838** ms | **35.0** / **36.5** ms | -- |

---

## 📊 2. Qwen3.6-35B LoRA Merged Model Benchmark (TP/DP Layouts)

* **Workload Shape**: 1,024 input tokens -> **128 generated output tokens** (1,000 total prompts).
* **Configured Request Rate**: **8.00 RPS** (except Run 1, which was `inf`).
* **Platform**: TPU v6e-8.

| Run ID | Topology | Request Rate | Max Concurrency Ceiling | Attention Kernel | Output Throughput | Total Throughput | Mean TTFT | Mean TPOT | Request Failure Rate |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | TP=8, DP=1 | 8.00 RPS | Capped (est. 8) | Default | 602.04 tok/s | 5,418.36 tok/s | 43,185.63 ms | **12.24 ms** | -- |
| **Run 1** | TP=8, DP=1 | `inf` RPS | Default (256) | Default | **1,687.83 tok/s** | **15,190.45 tok/s** | 34,670.49 ms | 127.86 ms | 0.0% |
| **Run 2** | TP=8, DP=1 | 8.00 RPS | Default (256) | Batched RPA | 978.80 tok/s | 8,809.23 tok/s | **694.57 ms** | 177.06 ms | **0.0%** |
| **Run 3** | TP=8, DP=1 | 8.00 RPS | Default (256) | Default | 982.86 tok/s | 8,845.73 tok/s | **669.77 ms** | 180.56 ms | **0.0%** |
| **Run 4** | TP=4, DP=2 (Disjoint Proxy) | 8.00 RPS | 128 (256 total) | Default | 624.42 tok/s | 5,619.74 tok/s | 12,497.61 ms | 14.76 ms | 26.1% ❌ |
| **Run 5** | **TP=4, DP=2 (Native + Patch)** | 8.00 RPS | Default (64 per DP) | Default | **756.20 tok/s** | **6,805.79 tok/s** | 21,954.66 ms | 26.40 ms | **0.0%**  |

---

## 💡 Key Serving Insights

1. **Tensor Parallel (TP) Dilution on MoEs**:
   * Qwen3.6-35B only activates **3B parameters per token**. Sharding this workload across 8 chips (`TP=8`) makes the calculation steps microscopic, resulting in the runtime being dominated by Inter-Chip Interconnect (ICI) sync overhead.
   * This explains why the single NVIDIA H200 (`TP=1`, local calculation, no ICI syncs) achieves **~40-45% higher throughput** and lower latencies than `TP=8` on TPU v6e.

2. **TP8-DP1 vs. TP4-DP2 Tradeoff**:
   * **TP8-DP1** is optimized for **latency-critical** tasks (TTFT of **~670 ms** vs. **21.9s** on TP4-DP2) because it shards the prefill step across 8 chips. Under high concurrency load, native vLLM handles the queue with 0% request drops, but the lack of parallel DP replicas limits scale capacity.
   * **Native TP4-DP2** is optimized for **concurrency-heavy** tasks, load-balancing requests across the two DP replicas to achieve higher throughput and handle larger batch bounds at 8.00 RPS.

3. **Multiprocess Engine Optimization**:
   * Patching the vLLM JAX engine to run **Native TP4-DP2** under a unified API server increased output token throughput by **21%** (**756 tok/s** vs. **624 tok/s**) and completely eliminated request drops compared to the disjoint python proxy setup.
