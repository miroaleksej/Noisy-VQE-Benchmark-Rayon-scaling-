## 🔬 Noisy-VQE Benchmark (Rayon scaling)

This project includes a small benchmark script to measure **wall-clock scaling of noisy VQE** with respect to the number of Rayon worker threads.

### What is being benchmarked

* **Algorithm:** Noisy VQE with quantum trajectories
* **Parallelism:** Independent noisy trajectories (embarrassingly parallel)
* **RNG:** OND-RNG with deterministic seeding
* **Backend:** MPS + SVD (faer)
* **Metric:** Real elapsed time (`/usr/bin/time`)

The benchmark measures **end-to-end runtime**, including:

* MPS evolution,
* SVD truncation,
* noisy channels,
* shot-based energy estimation.

---

### Benchmark script

The script is located at the repository root:

```bash
bench_noisy.sh
```

Make sure it is executable:

```bash
chmod +x bench_noisy.sh
```

Run the benchmark:

```bash
./bench_noisy.sh
```

---

### Default benchmark parameters

The script uses the following fixed parameters:

```text
mode          = noisy
trajectories  = 40
shots         = 200
noise (p)     = 0.01
seed          = bench-seed
threads       = 1, 2, 4, 8
```

Each run is executed with:

```bash
cargo run -p emulator --release -- \
  --mode noisy \
  --threads <T> \
  --trajectories 40 \
  --shots 200 \
  --p 0.01 \
  --seed bench-seed
```

Output is discarded; only timing is measured.

---

### Expected scaling behavior

On Apple Silicon (M-series) and typical laptops:

* Near-linear speedup up to the number of **performance cores**
* Diminishing returns beyond that due to:

  * memory bandwidth,
  * SVD cost,
  * thermal limits

Typical results look like:

```
threads = 1  →  ~1.6 s
threads = 2  →  ~0.9 s
threads = 4  →  ~0.5 s
threads = 8  →  ~0.4 s
```

The **recommended default** for MacBooks is:

```bash
--threads 4
```

---

### Determinism and reproducibility

* All runs are **fully deterministic** given the same `--seed`
* Parallel execution does **not** affect numerical results
* Determinism is enforced by:

  * per-trajectory derived seeds,
  * ordered collection before reduction

You can verify this manually:

```bash
cargo run -p emulator -- --mode noisy --threads 1 --seed test
cargo run -p emulator -- --mode noisy --threads 8 --seed test
```

Both runs produce identical energies and CSV output.

---

### Notes

* This benchmark focuses on **trajectory-level parallelism**
* Shot-level parallelism is intentionally left serial to preserve estimator structure
* The benchmark reflects realistic hardware-like noisy VQE workloads

---

## Architecture overview

The simulator is built around a tensor-network (MPS) core with explicit control
over truncation and deterministic stochasticity. The key building blocks are:

* **MPS state + SVD truncation (faer)** for 1D circuits with controllable bond
  dimension (`max_bond`, `cutoff`).
* **Observables** (`Z`, `ZZ`, `XX`, `YY`) and **energies** (Ising, Heisenberg),
  computed without measurement collapse.
* **VQE modes**: analytic, shot-based, and noisy trajectories.
* **Deterministic OND-RNG** for reproducible shots and noise.
* **Parallel trajectories** (Rayon) with `--threads` control.

## Scaling experiments

Two standalone binaries support accuracy and entanglement scaling studies.

### 1) Bond-dimension growth vs depth

```bash
cargo run -p chi_sweep --release -- \
  --n 64 --depth-max 200 --depth-step 5 \
  --max-bond 16,32,64 --cutoff 1e-8 \
  --seed chi-64 --out chi_64.csv
```

CSV columns:
```
max_bond,depth,chi_max,layer_ms
```

### 2) Energy error vs bond dimension

```bash
cargo run -p error_sweep --release -- \
  --n 40 --depth 50 --chi-test 8,16,32 \
  --chi-ref 128 --chi-ref-check 192 \
  --h heisenberg --heisenberg-jx 1.0 --heisenberg-jy 1.0 --heisenberg-jz 1.0 \
  --cutoff 1e-8 --seed err-40 --out error_40.csv
```

CSV columns:
```
chi,energy,error_energy
```

### 3) Fidelity vs bond dimension (n <= 30)

```bash
cargo run -p fidelity_sweep --release -- \
  --n 24 --depth 30 --chi-test 4,8,16,32 \
  --chi-ref 64 --cutoff 1e-8 \
  --seed fid-24 --out fidelity_24.csv
```

CSV columns:
```
chi,fidelity,one_minus_fidelity
```

## Reproducibility

All stochastic components are fully deterministic under a fixed `--seed`:

* OND-RNG is the only source of randomness (shots and noise).
* Parallel execution does not change results (per-trajectory seeds are derived
  deterministically and results are reduced in a stable order).

For example, the following two runs must produce identical outputs:

```bash
cargo run -p emulator -- --mode noisy --threads 1 --seed test
cargo run -p emulator -- --mode noisy --threads 8 --seed test
```

---
