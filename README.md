# 🌌 Osmosis: Qwen3.6-35B MoE LoRA TPU serving & Benchmarking

This repository contains Kubernetes manifests, automation scripts, and serving profiles for merging, serving, and benchmarking the **Qwen3.6-35B MoE LoRA** model on Google Cloud GKE **TPU v6e** clusters.

---

## 📁 Repository Structure

```tree
.
├── README.md                      # Project overview, layout description, and guide
├── .gitignore                     # Git ignore rules (excl. adapter weights)
├── adapter_config.json            # LoRA adapter configuration metadata
├── merge-job.yaml                 # GKE Job for base model + LoRA weight merging
├── create_gke_tpu_cluster.sh      # Bash script to create standard GKE cluster with TPU v6e
├── qwen-base-bench-tp8-dp1.yaml   # Benchmark client + vLLM serving pod manifest for base model (TP=8, DP=1)
├── qwen-lora-bench-tp8-dp1.yaml   # Benchmark client + vLLM serving pod manifest for LoRA merged model (TP=8, DP=1)
├── qwen-lora-bench.yaml           # Baseline benchmark manifest
├── benchmark_results_summary.md   # Finalized benchmarking report & latency sweeps analysis
└── tp4_dp2_experiment/            # Disjoint process load-balanced TP=4, DP=2 serve experiment
    ├── qwen-lora-bench-tp4-dp2.yaml  # Manifest launching dual JAX replicas and proxy container
    ├── proxy.py                      # FastAPI reverse proxy to load-balance between engine replicas
    ├── native_tp4_dp2_results.json   # Non-streaming benchmark JSON output for TP4-DP2
    └── native_tp4_dp2_results_streaming.json # Streaming benchmark JSON output for TP4-DP2
```

---

## 🛠️ Overview of Core Components

### 1. TPU Cluster Creation
* **`create_gke_tpu_cluster.sh`**: Automates provisioning GKE Standard clusters, setting up required subnetworks, and bootstrapping a TPU v6e flex nodepool (`tpu-v6e-flex-pool` with `2x4` topology, i.e., 8 TPU v6e cores).

### 2. LoRA Weight Merging
* **`merge-job.yaml`**: Mounts an SSD persistent volume, downloads the base Qwen3.6 weights and LoRA adapters from GCS, installs PEFT dependencies, and runs an in-memory python script to merge the weights. The final merged weights are saved on the shared volume for local serving.

### 3. Serving Benchmarks (TP=8, DP=1)
* **`qwen-base-bench-tp8-dp1.yaml`**: Manifest to run the benchmark sweep (concurrency 8, 16, 32, 64) for the untuned/base model `Qwen/Qwen3.6-35B-A3B` on TPU v6e-8.
* **`qwen-lora-bench-tp8-dp1.yaml`**: The primary recommended manifest for merged model. Launches the unified vLLM server with TP=8 sharding alongside a benchmark client container running native `vllm bench serve` against the local API server endpoint.

### 4. Disjoint TP=4, DP=2 Load-Balanced Serving
* **`tp4_dp2_experiment/`**: Standard JAX data parallelism (DP > 1) crashes during compilation for multimodal models because JAX sharding maps the visual encoder's batch dimension of `1` onto the DP axis. This subproject bypasses that compile error by splitting the 8-chip TPU slice into two independent vLLM JAX processes (`TP=4, DP=1` each) and load balancing them using **`proxy.py`**.
  > [!WARNING]
  > Our results show that running disjoint processes on a single physical host incurs a **36.5% throughput performance penalty** due to heavy CPU context-switching and memory-bus contention. Running unified **TP=8, DP=1** is strongly recommended.

---

## 📊 Benchmarking & Results
For detailed throughput scaling charts, latency sweeps, queueing analysis, and architectural findings, please consult the finalized benchmark report:
👉 [benchmark_results_summary.md](file:///Users/ericehanley/code/osmosis/benchmark_results_summary.md)
