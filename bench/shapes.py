"""DeepSeek-V3.2 production GEMM shapes, from the checkpoint itself.

Every (N, K) below was read from the safetensors headers of
https://huggingface.co/deepseek-ai/DeepSeek-V3.2 (see `inspect_st_headers.py`;
no weight download needed — the shapes live in each shard's JSON header) and
cross-checked against how sglang shards each layer at TP8
(`python/sglang/srt/models/deepseek_v2.py`, `layers/attention/dsa/dsa_indexer.py`):

- `fused_qkv_a_proj_with_mqa`, and the whole DSA indexer, are ReplicatedLinear
  (full checkpoint shape on every GPU).
- `q_b_proj` / `kv_b_proj` are ColumnParallel (N / 8), `o_proj` and the MLP
  `down_proj`s are RowParallel (K / 8).
- Routed experts run as m-grouped GEMMs; at EP8 each GPU owns 256/8 = 32 experts.
- The indexer's `wk` + `weights_proj` are fused by sglang into one bf16
  `wk_weights_proj` GEMM of N = 128 + 64 = 192.

M is the number of tokens in the forward batch: decode M = per-GPU batch size
(1-16 is the latency-sensitive regime SwapAB exists for), prefill M = chunk size.
"""

# name, N, K, provenance (checkpoint tensor -> per-GPU shape at TP8)
DENSE_TP8 = [
    ("fused_qkv_a", 2112, 7168, "q_a_proj[1536,7168] + kv_a_proj_with_mqa[576,7168], replicated"),
    ("q_b_proj", 3072, 1536, "q_b_proj[24576,1536] column-parallel / 8"),
    ("o_proj", 7168, 2048, "o_proj[7168,16384] row-parallel / 8 (same N,K as shared-expert down_proj)"),
    ("indexer_wq_b", 8192, 1536, "indexer.wq_b[8192,1536], replicated (DSA)"),
    ("indexer_wk_wproj", 192, 7168, "indexer.wk[128,7168] + weights_proj[64,7168] fused, replicated, bf16"),
    ("dense_mlp_gate_up", 4608, 7168, "layers 0-2 gate/up[18432,7168] merged column-parallel: 2*18432/8"),
    ("dense_mlp_down", 7168, 2304, "layers 0-2 down[7168,18432] row-parallel / 8"),
    ("shared_expert_gate_up", 4096, 7168, "shared expert gate/up[2048,7168] merged, unsharded (= per-expert MoE shape)"),
    ("kv_b_proj", 4096, 512, "kv_b_proj[32768,512] column-parallel / 8 (MHA prefill path)"),
    ("lm_head", 16160, 7168, "lm_head[129280,7168] column-parallel / 8, bf16"),
    ("mtp_eh_proj", 7168, 14336, "MTP layer 61 eh_proj[7168,14336], bf16"),
]

# Full-checkpoint (TP1) shapes; also DeepGEMM's own test shapes (tests/generators.py)
DENSE_TP1 = [
    ("q_b_proj_tp1", 24576, 1536, "q_b_proj[24576,1536] full"),
    ("kv_b_proj_tp1", 32768, 512, "kv_b_proj[32768,512] full"),
    ("o_proj_tp1", 7168, 16384, "o_proj[7168,16384] full"),
]

# MoE m-grouped GEMMs at EP8: 32 local experts per GPU.
# expected_m_per_group for decode = global_batch * topk(8) / 256 experts
#   -> global decode batch 32..1024 maps to expected_m 1..32.
GROUPED_EP8 = [
    ("moe_gate_up", 32, 4096, 7168, "experts.*.gate/up[2048,7168] merged -> N=4096, 32 groups at EP8"),
    ("moe_down", 32, 7168, 2048, "experts.*.down[7168,2048], 32 groups at EP8"),
]

DENSE_M_LIST = [1, 4, 8, 16, 64, 128]
DENSE_M_LIST_TP1 = [1, 8, 64]
MASKED_EXPECTED_M_LIST = [1, 4, 8, 16, 32]  # decode, CUDA-graph masked layout
CONTIGUOUS_EXPECTED_M_LIST = [32, 128, 512]  # prefill, contiguous layout
