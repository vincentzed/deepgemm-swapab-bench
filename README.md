# The Geometry of Blackwell GEMMs: Why SwapAB ($C^T = B^T \cdot A^T$) Wins Decode and Is Mandatory for MoE

## The Upshot

On the NVIDIA Blackwell Ultra B300 (sm_103, an SM100-family part), computing matrix multiplications in transposed form ($C^T = B^T \cdot A^T$) transforms an otherwise severe hardware under-utilization problem into near-optimal execution during low-batch inference.

* **Dense decode speedup:** At small batch sizes between 1 and 16 tokens, SwapAB accelerates dense matrix multiplications by up to 1.53x by ensuring all 128 physical Tensor Memory lanes remain saturated.
* **Prefill acceleration through multicast:** In Mixture of Experts prefill, SwapAB aligns thread block clusters to enable asynchronous Tensor Memory Accelerator multicast, reducing kernel execution times by up to 24 percent.
* **Bandwidth savings in cold decode:** When paired with FP4 quantized weights during memory-bound decode, SwapAB achieves a 1.65x speedup over FP8 by halving the volume of streamed weight bytes.
* **A structural requirement for MoE:** For Mixture of Experts grouped GEMM operations, SwapAB is not merely a performance enhancement. It is a strict architectural dependency of the underlying scheduler and memory layout. Disabling it produces incorrect numerical results or uncompilable configurations.

---

## Introduction

At its core, SwapAB is a straightforward identity of matrix transposition:

$$C = A \cdot B \iff C^T = B^T \cdot A^T$$

Mathematically, both formulations produce the exact same output. On earlier GPU generations, transposing operands in this way made little difference to runtime performance. On NVIDIA Blackwell GPUs, however, SwapAB addresses a fundamental asymmetry in how the hardware allocates on-chip memory during small-batch decode.

In modern inference libraries such as [DeepGEMM](https://github.com/sgl-project/DeepGEMM), SwapAB serves as the default layout strategy for low-latency execution. In this repository, we examine the mechanics of SwapAB and benchmark its impact on the complete shape catalog of [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) across FP8, FP4 with quantized weights, and BF16. Every measurement was gathered directly on an NVIDIA B300 GPU using CUPTI hardware counters.

<p align="center"><img src="media/dense-speedup-table.png" width="92%" alt="terminal style results table showing speedups for five GEMM shapes across fp8, fp4, and bf16"></p>
<p align="center"><em>Speedup delivered by the swapped layout at batch size M=1, and the performance inversion point beyond batch size M=64.</em></p>

<p align="center"><img src="media/dense-kernel-times.png" width="88%" alt="stacked bar charts showing absolute kernel execution time in microseconds across fp8, fp4, and bf16"></p>
<p align="center"><em>Absolute kernel execution time across data types. The swapped kernel maintains flat latency from 1 to 16 tokens, whereas the standard non-swapped kernel catches up only when batch sizes become large.</em></p>

---

## Microarchitectural Background: Tensor Memory and UMMA

During the decode phase of large language models, matrix operations are heavily skewed. The activation matrix ($A$) contains only a few token rows ($M \le 16$), whereas the weight matrix ($B$) spans thousands of channels ($N, K \ge 1024$).

Blackwell executes matrix operations through the `tcgen05.mma` instruction, referred to in CUTLASS as UMMA. These instructions stage operands in Shared Memory and accumulate outputs directly into Tensor Memory. Tensor Memory is a dedicated 256 KB on-chip storage system located on each Streaming Multiprocessor, organized physically as 128 rows (known as lanes) by 512 columns of 32 bits, with columns allocated dynamically per kernel.

The hardware treats matrix dimensions with strict structural asymmetry:

* **The M dimension is rigid:** It maps directly across Tensor Memory lanes and requires an allocation of 128 lanes per thread block, or 256 lanes across a paired two-CTA cluster. The instruction set also defines a 64-lane variant, which these kernels do not use.
* **The N dimension is flexible:** It maps across Tensor Memory columns and can be allocated dynamically in variable multiples of 16 columns up to 256.

This constraint is asserted directly within DeepGEMM in [`sm100_fp8_fp4_gemm_1d1d.cuh:297-299`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh#L297-L299):

```cpp
DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                 (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                 (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                 "Invalid MMA instruction shape");
```

### The Inefficiency of Default Decode Layouts

When a standard decode matrix multiplication ($C = A \cdot B$) runs without SwapAB, the small token dimension lands on the rigid M axis. The hardware is forced to allocate the entire block of 128 Tensor Memory lanes. At a batch size of 16 tokens, only 16 of the 128 allocated lanes perform useful computation. The remaining 112 lanes sit idle, creating an eightfold compute under-utilization.

### How SwapAB Restores Hardware Efficiency

We can think of SwapAB as re-aligning the problem with the physical geometry of the hardware. By formulating the operation as $C^T = B^T \cdot A^T$, the operand axes are reversed:

* The large weight dimension maps to the rigid 128-lane M axis, ensuring that every physical lane is fully utilized.
* The small token dimension moves to the flexible N axis (`UMMA_N`), as configured in [`sm100_fp8_fp4_gemm_1d1d.cuh:50-51`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh#L50-L51):

```cpp
constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;   // Rigid allocation: 128 or 256
constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;   // Flexible allocation: multiples of 16
```

The instruction shape is fixed at JIT compilation time: the layout heuristic enumerates `BLOCK_M` — the value that becomes `UMMA_N` — in multiples of 16, as implemented in [`sm100.hpp:47-58`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/csrc/jit_kernels/heuristics/sm100.hpp#L47-L58). When processing 7 tokens, the heuristic selects `BLOCK_M = 16`, which is the minimum instruction step size, reducing unused compute overhead from roughly 18x (7 useful lanes out of 128) down to approximately 2.3x (7 useful columns out of 16). A runtime resize path also exists, as shown in [`sm100_fp8_fp4_gemm_1d1d.cuh:329-331`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh#L329-L331), but the value it receives differs from `BLOCK_M` only for the tail blocks of the partial-sum grouped layout ([`scheduler/gemm.cuh:162-169`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/deep_gemm/include/deep_gemm/scheduler/gemm.cuh#L162-L169)); dense kernels always execute at their compiled `BLOCK_M`.

---

## The Five Implementation Steps of SwapAB

Executing SwapAB requires coordinated transformations across descriptors, instruction dispatch, and memory epilogues.

### Step 1: Heuristic Layout Selection
DeepGEMM evaluates dense matrix shapes and scores both standard and swapped configurations, as implemented in [`sm100.hpp:47`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/csrc/jit_kernels/heuristics/sm100.hpp#L47). For grouped GEMMs, the heuristic selects SwapAB unconditionally, as defined in [`sm100.hpp:31-43`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/csrc/jit_kernels/heuristics/sm100.hpp#L31-L43):

```cpp
// Always enable swap A/B (and multicasting if possible) for m-grouped GEMMs
if (desc.gemm_type == GemmType::MGroupedContiguous or ...) {
    const bool swap_ab = true;
```

### Step 2: Descriptor Inversion
Data types and layout majorness are inverted within the Cute instruction descriptor, as defined in [`sm100_fp8_fp4_gemm_1d1d.cuh:283-286`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh#L283-L286). In FP4 recipes where activations are FP8 and weights are FP4, the descriptor exposes the FP4 weights as Operand A after the swap:

```cpp
auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc_block_scaled<b_dtype_t, a_dtype_t, ..., kMajorB, kMajorA>()
                          : cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, ..., kMajorA, kMajorB>();
```

### Step 3: Reversed Instruction Dispatch
Although operand B is loaded into shared memory buffer `smem_b`, the issue stage passes the descriptor of B into the operand A slot. Scale factor columns in Tensor Memory swap accordingly, as specified in [`sm100_fp8_fp4_gemm_1d1d.cuh:375-386`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh#L375-L386):

```cpp
if constexpr (kSwapAB) {
    mma_t::fma(b_desc, a_desc, ...);   // B is fed as operand A
}
```

Following this dispatch, Tensor Memory accumulates the transposed result $C^T$ rather than $C$.

### Step 4: Epilogue Transposition Store
Tensor Memory cannot write directly to Global Memory through asynchronous copy engines. Output tiles must first move into registers using `tcgen05.ld`, pass into Shared Memory, and then write to Global Memory using the Tensor Memory Accelerator.

The epilogue in [`sm100_store_cd_swap_ab.cuh`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/deep_gemm/include/deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh) performs an in-flight transposition during the Shared Memory store. FP32 accumulators are converted to BF16 and packed:

```cpp
ptx::SM90_U32x4_STSM_T<int>::copy(
    math::cast_into_bf16_and_pack(values[0], values[1]), ..., smem_ptr);
```

This compiles directly into the transposing PTX instruction:

```ptx
stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1, %2, %3, %4};
```

The `.trans` modifier transposes each 8x8 sub-tile during the Shared Memory write. After this transposition, a standard Tensor Memory Accelerator store transfers the final matrix $C$ into Global Memory. This extra transposition step represents the primary computational trade-off of SwapAB.

### Step 5: Cluster Axis Alignment and TMA Multicast
DeepGEMM couples Cooperative Thread Array cluster pairing to the chosen layout, as shown in [`sm100.hpp:76-88`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/csrc/jit_kernels/heuristics/sm100.hpp#L76-L88). Standard layouts pair thread blocks along the M axis, while swapped layouts pair thread blocks along the N axis.

During low-batch decode, the token axis (M) never produces enough blocks to pair across a cluster. The weight axis (N) is the only dimension with enough blocks to pair on — and pairing along N is exactly what the swapped layout permits. Paired thread blocks then share the token activation tile through hardware multicast while streaming independent weight tiles.

---

## Serving Context and Evaluated Layer Shapes

To evaluate these mechanics under production conditions, we benchmarked the complete matrix operation suite of the 671B parameter DeepSeek-V3.2 architecture under Tensor Parallelism 8 and Expert Parallelism 8 sharding.

* **Multi-Head Latent Attention projections:** Projections compress activations into low-dimensional query and key-value latents (`fused_qkv_a_proj`), followed by head up-projections (`q_b_proj`).
* **DeepSeek Sparse Attention indexer projections:** Projections score history tokens to enforce sparsity across the context (`wq_b`, `wk`, `weights_proj`).
* **Feed-forward networks:** Operations include dense Multi-Layer Perceptrons in early layers, shared experts, and 256 routed Mixture of Experts projections sharded across GPUs.

| Operation (TP8 Per-GPU) | Shape ($N \times K$) | Serving Precision | Architectural Provenance |
|---|---|---|---|
| Fused `q_a` + `kv_a` | $2112 \times 7168$ | FP8 | Combined `q_a_proj` and `kv_a_proj_with_mqa`, replicated across ranks |
| `q_b_proj` | $3072 \times 1536$ | FP8 | Column-parallel slice of full head projection |
| `o_proj` | $7168 \times 2048$ | FP8 | Row-parallel attention output projection |
| DSA Indexer `wq_b` | $8192 \times 1536$ | FP8 | Indexer query projection, replicated across ranks |
| DSA Indexer `wk` + `weights_proj` | $192 \times 7168$ | BF16 | Fused key projection and gating weights |
| Dense MLP `gate` + `up` | $4608 \times 7168$ | FP8 | Merged dense feed-forward layer for layers 0 through 2 |
| Dense MLP `down` | $7168 \times 2304$ | FP8 | Down-projection for initial dense layers |
| Shared Expert `gate` + `up` | $4096 \times 7168$ | FP8 | Fused gate and up-projection for the shared expert |
| `kv_b_proj` (Prefill path) | $4096 \times 512$ | FP8 | Key-value projection used during prefill |
| LM Head | $16160 \times 7168$ | BF16 | Vocabulary output projection sharded across ranks |
| MTP `eh_proj` | $7168 \times 14336$ | BF16 | Multi-Token Prediction draft layer projection |
| MoE Experts (EP8, 32 groups) | Gate+Up: $4096 \times 7168$, Down: $7168 \times 2048$ | FP8 / FP4 | Routed expert feed-forward projections |

### Precision Configurations
* FP8 uses 128x128 block-scaled weights with UE8M0 scale factors.
* FP4 uses FP8 activations multiplied by packed E2M1 weights with 1x32 UE8M0 scale factors.
* BF16 uses unscaled floating-point matrix multiplication.

---

## Performance Results

### Result 1: Dense Decode Speedup and Batch Size Crossover

The table below reports warm L2 execution times in microseconds for swapped versus non-swapped configurations, along with the resulting speedup ratio.

| Operation Shape | Batch M=1 | Batch M=8 | Batch M=64 |
|---|---:|---:|---:|
| MTP `eh_proj` | 17.5 / 23.8 (**1.36x**) | 18.7 / 21.3 (1.14x) | 22.4 / 19.1 (0.85x) |
| Dense MLP `gate` + `up` | 8.1 / 10.4 (**1.29x**) | 8.1 / 9.2 (1.14x) | 9.2 / 8.3 (0.90x) |
| `o_proj` | 5.0 / 5.7 (1.15x) | 4.8 / 5.4 (1.11x) | 5.3 / 4.6 (0.87x) |
| Indexer `wq_b` | 4.4 / 5.0 (1.12x) | 4.4 / 4.7 (1.07x) | 4.8 / 4.1 (0.86x) |
| LM Head | 20.2 / 22.0 (1.09x) | 20.5 / 21.1 (1.03x) | 21.6 / 22.1 (1.02x) |

Three clear trends emerge from these empirical measurements:

1. **Small batch sizes achieve significant gains:** Across batch sizes from 1 to 16 tokens, SwapAB delivers a 1.10x to 1.53x speedup. For example, the LM Head in FP4 at batch size 1 executes in 9.9 microseconds swapped versus 15.3 microseconds non-swapped, achieving a 1.53x speedup. Dense MLP Gate and Up in BF16 at batch size 1 executes in 12.3 microseconds swapped versus 18.0 microseconds non-swapped, achieving a 1.47x speedup.
2. **Performance inverts as batch sizes scale:** Beyond a batch size of approximately 32 to 64, standard matrix multiplications become faster. In this regime, the Tensor Memory lanes are naturally saturated by the larger token count, while the swapped layout continues to pay the transposition overhead during the epilogue. This crossover point aligns with the dynamic selection threshold used in DeepGEMM heuristics.
3. **Cold L2 conditions become memory bandwidth bound:** When weights are not resident in the L2 cache and must be fetched directly from High Bandwidth Memory, execution latency is dominated by memory bus traffic. Under these conditions, swapped and non-swapped variants perform within 0.94x to 1.05x of one another.

---

### Result 2: TMA Multicast and FP4 Streaming Benefits

Because SwapAB enables thread block clustering along the N dimension, it allows kernels to leverage asynchronous Tensor Memory Accelerator multicast.

<p align="center"><img src="media/moe-multicast-times.png" width="85%" alt="bar chart comparing execution times of MoE prefill kernels with and without multicast"></p>
<p align="center"><em>Execution time of MoE prefill expert kernels with and without TMA multicast. Standard layouts may cluster only along M, where these shapes never have two blocks to pair; the N-axis clustering that multicast depends on here exists only in the swapped layout.</em></p>

* In MoE prefill operations with 512 tokens per expert, TMA multicast delivers a 13 to 24 percent reduction in total kernel execution time across FP8, FP4, and BF16.
* In single-token decode operations, multicast has a neutral effect because token counts per expert are insufficient to form multi-block clusters.

The table below presents execution times for cold L2 decode operations across 8 tokens per expert:

| Precision Format | Gate and Up Latency | Down Projection Latency |
|---|---:|---:|
| FP8 | 160.5 µs | 90.7 µs |
| FP4 Quantized Weights | 96.0 µs | 55.2 µs |

Here, FP4 weights provide an overall 1.65x speedup compared to FP8. Because decode operations are bottlenecked by the rate at which weights can be streamed from High Bandwidth Memory, reducing the weight footprint by half translates directly into reduced latency.

---

### Result 3: Grouped GEMM Structural Dependencies

When SwapAB is forcibly disabled for M-grouped GEMMs, the operations fail completely rather than merely exhibiting reduced performance:

<p align="center"><img src="media/moe-swap-mandatory.png" width="88%" alt="terminal style output matrix showing correctness failures when SwapAB is disabled for grouped GEMMs"></p>
<p align="center"><em>Behavior of grouped GEMMs when SwapAB is forcibly disabled across FP8, FP4, and BF16.</em></p>

* **Masked decode layouts:** Kernels compile and run, but produce incorrect numerical results with relative errors near 0.5 across all data types. DeepGEMM's grouped scheduler computes each block's effective M — the quantity that feeds the runtime UMMA-N update — using logic its own comment marks as valid "for swap A/B and psum layout only," as seen in [`scheduler/gemm.cuh:162-169`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/deep_gemm/include/deep_gemm/scheduler/gemm.cuh#L162-L169).
* **Contiguous prefill layouts:** Kernels fail to construct a valid execution layout. For M-grouped GEMMs, the heuristic generates exactly one layout candidate — the swapped one ([`sm100.hpp:31-43`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/csrc/jit_kernels/heuristics/sm100.hpp#L31-L43)) — so filtering it out leaves an empty candidate list. The memory format agrees with this design: our serving configuration aligns expert group boundaries to multiples of 240 rows (the alignment is runtime-configurable through `set_mk_alignment_for_contiguous_layout`; the library default is 128, per [`runtime.hpp:10`](https://github.com/sgl-project/DeepGEMM/blob/a348cd95a8ddfd3ef337b595d8e8b841c0d9cf0f/csrc/jit_kernels/heuristics/runtime.hpp#L10)), and 240 matches the 16-row granularity of the swapped UMMA instruction while being divisible by none of the standard non-swapped tile sizes of 32, 64, or 128 rows.

Because Mixture of Experts operations account for the vast majority of total arithmetic work in architectures like DeepSeek-V3.2, SwapAB serves as a mandatory structural foundation for DeepGEMM on SM100 hardware.

---

## Experimental Methodology

To evaluate these operations with high precision, our benchmarking harness implements a controlled experimental pipeline:

* **Surgical kernel instrumentation:** DeepGEMM was instrumented using the patch in [`patches/`](patches/) to expose the runtime environment variables `DG_FORCE_SWAP_AB` and `DG_FORCE_MULTICAST`. All other heuristic routines, comparators, and epilogues remain stock `sgl-project/DeepGEMM @ a348cd9`.
* **Serving-accurate inputs:** Input tensors were constructed with quantization scale factors pre-packed in advance, matching the data structures used by production serving engines.
* **Pre-timing verification:** Every benchmark configuration was validated for numerical correctness against an FP32 reference implementation before profiling. Configurations that failed numerical checks were recorded and excluded from timing sweeps.
* **CUPTI timing harness:** Timing was performed using CUPTI hardware counters through `flashinfer.testing.utils.bench_gpu_time_with_cupti` with CUDA Graph replay. Latencies represent the median of 30 replayed iterations following 10 warmup iterations.
* **Cache management:** L2 cache lines were flushed outside measured timing windows to evaluate realistic cold-decode memory access patterns, while dedicated warm-L2 passes isolated compute efficiency.

Raw profiling logs and layout selections are available in [`logs/`](logs/).

---

## Reproducing the Results

### Environment Requirements
* An NVIDIA GPU of the SM100 family, such as a B300 (sm_103).
* A CUDA 13 development container with PyTorch and FlashInfer, such as `lmsysorg/sglang:nightly-dev-cu13`.
* CUPTI Python bindings installed via `pip install cupti-python`.

### Running Sweeps
```bash
git clone https://github.com/vincentzed/deepgemm-swapab-bench
cd deepgemm-swapab-bench

# Clones DeepGEMM @ a348cd9, applies the evaluation patch, compiles, and runs sweeps
./run.sh
```

### Regenerating Figures
```bash
uv run scripts/make_visuals.py
uv run scripts/make_plots.py
./scripts/svg_to_png.sh
```

The patch in [`patches/`](patches/) is designed strictly for benchmarking and comparative evaluation. It should not be deployed to production environments, as disabling SwapAB causes grouped GEMM operations to return incorrect outputs.

---

## Repository Structure

```text
bench/
  shapes.py             # Model shape catalog and sharding configurations
  bench_swapab.py       # CUPTI benchmarking harness across data types and layouts
  probe_auto_choice.py  # Inspection script for default heuristic decisions
  inspect_st_headers.py # HTTP range request parser for Safetensors headers
scripts/
  make_visuals.py       # Terminal visualization generator using Rich
  make_plots.py         # Plot generator using Vega-Lite
  thread_visuals.py     # Plot formatting helpers
  svg_to_png.sh         # Headless browser rendering script for PNG export
patches/
  0001-dg-force-swap-ab-multicast-knobs.patch
logs/                   # Profiling logs and raw CSV output
media/                  # Benchmark plots and visual summaries
run.sh                  # Automation script for building and running sweeps
```

---

## References

* Implementation verified against [sgl-project/DeepGEMM @ a348cd9](https://github.com/sgl-project/DeepGEMM).
* Model dimensions extracted from [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) and [SGLang](https://github.com/sgl-project/sglang).
* Hardware instruction definitions referenced from the [NVIDIA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/).

## License

This repository is licensed under the MIT License. DeepGEMM is licensed under the MIT License by its authors.
