# Qwen3.6-35B MoE LoRA TPU v6e Serving Benchmark Report

This document compiles the benchmarking results and serving profile analysis for the merged Qwen3.6-35B MoE LoRA model served on a Google Cloud GKE **TPU v6e (2x4 topology, 8 physical cores, 32GB HBM per core)** cluster.

---

## 📊 Summary of Benchmark Runs

All benchmarks were run with a **1024 input tokens -> 128 output tokens** workload (1,000 total prompts).

| Run Setup | Tensor Parallel (TP) / Data Parallel (DP) | Configured Request Rate | Max Concurrency Ceiling (`--max-num-seqs`) | Attention Kernel Type | Output Token Throughput | Total Token Throughput | Mean Time-To-First-Token (TTFT) | Mean Time-Per-Output-Token (TPOT) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | TP=8, DP=1 | 8.00 RPS | Capped (est. `--max-num-seqs 8`) | Default | **602.04 tok/s** | **5,418.36 tok/s** | 43,185.63 ms (43.19s) | **12.24 ms** |
| **Run 1** | TP=8, DP=1 | Infinite (`inf`) | Default (256) | Default | **1,687.83 tok/s** | **15,190.45 tok/s** | 34,670.49 ms (34.67s) | 127.86 ms |
| **Run 2** | TP=8, DP=1 | 8.00 RPS | Default (256) | Batched RPA | **978.80 tok/s** | **8,809.23 tok/s** | 694.57 ms (0.69s) | 177.06 ms |
| **Run 3** | TP=8, DP=1 | 8.00 RPS | Default (256) | Default | **982.86 tok/s** | **8,845.73 tok/s** | 669.77 ms (0.67s) | 180.56 ms |
| **Run 4** | TP=4, DP=2 (Disjoint) | 8.00 RPS | 128 (256 total) | Default | **624.42 tok/s** | **5,619.74 tok/s** | 12,497.61 ms (12.50s) | 14.76 ms |


---

## 🧠 Architectural Insights & Performance Analysis

### 1. The TPOT Discrepancy & Batch size Trade-off
* **Why Baseline TPOT is 12ms but ours is ~180ms**:
  * Time-Per-Output-Token (TPOT) is heavily governed by the active execution batch size.
  * In the **Baseline** run, the active execution batch size was restricted (likely capped to **8**). Running small batches minimizes the physical execution time per decode step (yielding a low **12.24 ms** latency) but severely limits hardware efficiency.
  * In **Runs 2 & 3**, the server was permitted to batch up to **256** concurrent requests. Under load, this dynamic batch size increases the computation step time (to **~180 ms**) but yields much higher token throughput.

### 2. Throughput & Queueing Behavior
* **Throughput Scaling**:
  * Leaving the batch size uncapped at **8.00 RPS** load (**Run 3**) yields **982.86 tokens/s** (+62.5% higher throughput compared to the baseline's 602.04 tokens/s).
  * Saturation at infinite rate (**Run 1**) scales the output throughput to **1,687.83 tokens/s** (2.8x higher than the baseline).
* **Queueing Overhead (TTFT)**:
  * Because the Baseline restricted its batch size to 8, it could only process a maximum of **4.88 requests per second**.
  * When fed a load of **8.00 RPS**, the arrival rate exceeded the processing speed, building a massive queue. This explains their extremely high **43.19-second average TTFT**.
  * By allowing larger batches, our server processed the 8.00 RPS load with almost zero queueing, bringing average TTFT down to **669.77 ms**.

### 3. Batched RPA Kernel vs. Default
* Comparing **Run 2** (Batched RPA attention) and **Run 3** (Default attention) at 8.00 RPS shows that the custom RPA kernel is slightly faster on step latency (177ms vs 180ms) but yields almost identical overall throughput (~980 tokens/s). It is not the source of the baseline performance gap.

### 4. Data Parallel (DP) Incompatibility on Multimodal Models
* **Why TP=4 / DP=2 fails on Qwen3.6-35B MoE**:
  * During the multimodal precompilation/warmup phase, the vLLM engine initializes and compiles the visual encoder (vision tower).
  * The vision tower processes one image item per step, meaning its active query/key/value batch dimension is `1` (shape `bf16[1, 16, 128, 72]`).
  * The current JAX `shard_map` implementation for the vision attention kernel (`_flash_attention` in `mm_encoder_attention.py`) maps this batch dimension to the `'data'` mesh axis.
  * When DP > 1 (e.g. DP=2), JAX attempts to divide this batch dimension of `1` by the DP mesh size of `2`. Since 2 does not evenly divide 1, JAX throws a `ValueError` during warmup and crashes.
  * Therefore, **multimodal models containing vision towers are restricted to DP=1 on vLLM TPU** until the vision attention layer supports padding/sharding configurations that decouple from the DP axis. To use an 8-chip topology, you must run **TP=8, DP=1**.

### 5. Disjoint Processes TP=4 / DP=2 vs. Single Process TP=8 / DP=1
* **Why TP4-DP2 has 36% lower throughput and high latency**:
  * To bypass the DP axis mismatch on the vision tower, we split the pod's 8 chips into two separate, independent JAX engine processes (each running `TP=4, DP=1` on 4 chips) and load-balanced them using a local Python proxy.
  * While this configuration bypassed the vision tower compiler crash, it introduced severe **host-level contention**:
    * **CPU Thread Contention**: Concurrently running two separate JAX compilers/runtime schedulers on the same physical CPU node causes intense orchestration and thread-context-switching overhead.
    * **Memory Bandwidth & Bus Saturation**: The two processes compete for host memory bus and PCI lanes during model loading, compiling, and execution.
  * **The Performance Penalty**:
    * Under 8.00 RPS load, the combined throughput of the two TP=4 replicas was **624.42 tokens/sec** — a **36.5% reduction** compared to the single TP=8 process (**982.86 tokens/sec**).
    * Due to this capacity reduction, the server was unable to sustain the 8.00 RPS load (which requires at least 1,024 tokens/s output capacity), leading to queue saturation, high TTFT (12.5 seconds), and **26.1% request failures** due to connection/queue timeouts.
    * **Step Latency (TPOT)**: When requests successfully executed, the TPOT was **14.76 ms** (as opposed to 180 ms in TP=8). This low TPOT is an artifact of smaller active batch sizes on the independent replicas, but the throughput loss from host contention makes it a negative overall tradeoff.
  * **Recommendation**: For TPU v6e-8 topologies, running a single unified **TP=8, DP=1** instance is highly recommended over splitting the hardware slice into disjoint processes.

