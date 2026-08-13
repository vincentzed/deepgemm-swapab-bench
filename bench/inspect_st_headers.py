#!/usr/bin/env python3
"""Inspect safetensors tensor shapes of a HF repo WITHOUT downloading weights.

Fetches only the safetensors JSON header of each requested shard via HTTP
Range requests (8-byte length prefix, then the header itself).
"""
import json
import struct
import subprocess
import sys

REPO = "deepseek-ai/DeepSeek-V3.2"
BASE = f"https://huggingface.co/{REPO}/resolve/main"


def fetch_range(url: str, start: int, end: int) -> bytes:
    return subprocess.run(
        ["curl", "-sL", "-r", f"{start}-{end}", url],
        check=True, capture_output=True,
    ).stdout


def shard_header(shard: str) -> dict:
    url = f"{BASE}/{shard}"
    n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    return json.loads(fetch_range(url, 8, 8 + n - 1))


def main() -> None:
    idx = json.loads(subprocess.run(
        ["curl", "-sL", f"{BASE}/model.safetensors.index.json"],
        check=True, capture_output=True).stdout)["weight_map"]

    # Tensor names covering every distinct GEMM weight kind:
    # dense layer (0), first MoE layer (3), incl. DSA indexer + shared expert + MTP (61)
    wanted = [k for k in idx if (
        k.startswith("model.layers.0.")
        or (k.startswith("model.layers.3.") and (".experts.0." in k or ".experts." not in k))
        or k.startswith("model.layers.61.")
        or "embed" in k or "lm_head" in k
    )]
    shards = sorted(set(idx[k] for k in wanted))
    print(f"# fetching headers of {len(shards)} shards", file=sys.stderr)

    headers = {}
    for s in shards:
        headers[s] = shard_header(s)

    rows = []
    for name in sorted(wanted):
        h = headers[idx[name]].get(name)
        if h:
            rows.append((name, h["dtype"], h["shape"]))
    for name, dtype, shape in rows:
        print(f"{name:80s} {dtype:12s} {shape}")


if __name__ == "__main__":
    main()
