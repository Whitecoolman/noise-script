#!/usr/bin/env python3
"""
Add small Gaussian noise to float weights in Hugging Face-style safetensors checkpoints.

Use only on model checkpoints **you own and have the right to modify**. This is a
common technique to break byte-identical duplicates and (when scaled appropriately)
reduce activation-fingerprint similarity between two legitimately distinct uploads.
See ``scripts/security/copy_detection_threat_probe.py`` for how cosine similarity
responds to noise at a toy scale — you may need to tune ``--relative`` or ``--sigma``.

Examples (from repo root with PYTHONPATH including this repo if needed):

  # Noise scaled to a fraction of each tensor's standard deviation (recommended start)
  python3 scripts/add_weight_noise.py /path/to/model --out /path/to/model_noisy \\
      --relative 1e-4 --seed 42

  # Fixed absolute noise (same std for every parameter, in float space before cast)
  python3 scripts/add_weight_noise.py /path/to/model --out /path/to/model_noisy \\
      --sigma 1e-4 --seed 42

  # Single shard
  python3 scripts/add_weight_noise.py weights.safetensors --out weights_noisy.safetensors \\
      --relative 5e-5 --seed 0

  # GPU (e.g. RTX Pro 6000 / large VRAM): faster noise + ``safe_open`` loads shards onto CUDA
  python3 affine/add_weight_noise.py /path/to/model --out /path/to/out \\
      --relative 5e-5 --seed 42 --device cuda --verbose

Large checkpoints (multi‑tens of GB): files at or above ``WEIGHT_NOISE_STREAM_MIN_BYTES`` (default
2GiB) use a **streaming** reader/writer so RAM stays near **one tensor at a time**, not the whole
model. You need enough **disk** for a temporary ``*.blob.tmp`` (~same size as the shard).
Set ``WEIGHT_NOISE_STREAM_MIN_BYTES=0`` to always stream.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import struct
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

try:
    from safetensors.torch import _tobytes  # type: ignore[attr-defined]
except ImportError:
    # safetensors >= 0.5 removed the private ``_tobytes`` helper. Reproduce its
    # output (raw little-endian bytes for one tensor, exactly as written into a
    # safetensors data blob) via the public ``save`` API, stripping the 8-byte
    # length prefix and JSON header to recover only the data portion.
    from safetensors.torch import save as _st_save

    def _tobytes(tensor: "torch.Tensor", name: str) -> bytes:  # type: ignore[name-defined]
        blob = _st_save({name: tensor})
        header_len = struct.unpack("<Q", blob[:8])[0]
        return bytes(blob[8 + header_len:])

# Default chunk size (elements) for std/noise loops — override with --chunk-elements.
# On GPU, larger chunks (e.g. 16M–64M) often improve throughput if VRAM allows.
_DEFAULT_CHUNK_ELEMS = 4_000_000
_SMALL_TENSOR_ELEMS = 2_000_000

# Above this file size, stream tensors to disk instead of ``load_file`` (avoids ~1× model RAM).
# Override with env ``WEIGHT_NOISE_STREAM_MIN_BYTES`` (integer bytes); ``0`` = always stream.
def _streaming_threshold_bytes() -> int:
    raw = os.environ.get("WEIGHT_NOISE_STREAM_MIN_BYTES")
    if raw is not None:
        return int(raw)
    return 2 * 1024 * 1024 * 1024  # 2 GiB


def _max_data_offset(header: dict) -> int:
    m = 0
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        off = entry.get("data_offsets")
        if isinstance(off, list) and len(off) >= 2:
            m = max(m, int(off[1]))
    return m


def _resolve_device(s: str) -> torch.device:
    d = torch.device(s)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"Requested device {s!r} but CUDA is not available.")
    return d


def _tensor_for_safetensors_bytes(t: torch.Tensor) -> torch.Tensor:
    """safetensors byte packing expects CPU contiguous tensors."""
    if t.is_cuda:
        return t.contiguous().cpu()
    return t.contiguous()


def process_safetensors_file_streaming(
    src: Path,
    dst: Path,
    *,
    sigma: float | None,
    relative: float | None,
    generator: torch.Generator,
    device: torch.device,
    chunk_elems: int,
) -> None:
    """
    Read tensors one at a time via ``safe_open``, write a temp blob, then reassemble the file.

    Peak host RAM is low; with ``--device cuda``, peak **VRAM** is roughly the largest single
    tensor (FP32 scratch during noise) plus overhead — fine for large GPUs (e.g. RTX Pro 6000).
    Requires free disk space about the size of the checkpoint for the temporary ``.blob.tmp`` file.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob_tmp = dst.with_suffix(dst.suffix + ".blob.tmp")

    with open(src, "rb") as sf:
        header_len = struct.unpack("<Q", sf.read(8))[0]
        header_bytes = sf.read(header_len)

    header = json.loads(header_bytes.decode("utf-8"))
    expected_blob_len = _max_data_offset(header)

    written = 0
    try:
        open_dev = "cpu" if device.type != "cuda" else str(device)
        with open(blob_tmp, "wb") as out_blob, safe_open(
            str(src), framework="pt", device=open_dev
        ) as f:
            for key in f.offset_keys():
                t = f.get_tensor(key)
                noisy = _perturb_tensor(
                    t,
                    sigma=sigma,
                    relative=relative,
                    generator=generator,
                    chunk_elems=chunk_elems,
                )
                payload = _tobytes(_tensor_for_safetensors_bytes(noisy), key)
                out_blob.write(payload)
                written += len(payload)
                del t, noisy, payload

        if expected_blob_len and written != expected_blob_len:
            raise RuntimeError(
                f"Streaming noise wrote {written} bytes but header expects {expected_blob_len} "
                f"in the data blob (file may use tensor sharing or an unusual layout)."
            )

        with open(dst, "wb") as out:
            out.write(struct.pack("<Q", len(header_bytes)))
            out.write(header_bytes)
            with open(blob_tmp, "rb") as bf:
                shutil.copyfileobj(bf, out, length=-1)
    except Exception:
        if dst.is_file():
            try:
                dst.unlink()
            except OSError:
                pass
        raise
    finally:
        if blob_tmp.is_file():
            try:
                blob_tmp.unlink()
            except OSError:
                pass


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _noise_scale_for_tensor(t: torch.Tensor, sigma: float | None, relative: float | None) -> float:
    if sigma is not None and relative is not None:
        raise ValueError("Specify only one of --sigma or --relative")
    if sigma is not None:
        return float(sigma)
    if relative is not None:
        std = float(t.float().std().item())
        # Avoid dead scales on near-constant tensors
        base = max(std, 1e-12)
        return float(relative) * base
    raise ValueError("Provide --sigma or --relative")


def _std_chunked(flat: torch.Tensor, *, chunk_elems: int) -> float:
    """Population std over all elements without materializing full FP32 views where avoidable."""
    n = flat.numel()
    if n == 0:
        return 0.0
    if flat.device.type == "cuda":
        total = torch.zeros((), dtype=torch.float64, device=flat.device)
        total_sq = torch.zeros((), dtype=torch.float64, device=flat.device)
        for i in range(0, n, chunk_elems):
            end = min(i + chunk_elems, n)
            c = flat[i:end].float()
            total += c.sum(dtype=torch.float64)
            cd = c.double()
            total_sq += (cd * cd).sum()
            del c, cd
        mean = total / n
        var = torch.clamp(total_sq / n - mean * mean, min=0.0)
        return float(torch.sqrt(var).item())

    s = 0.0
    ss = 0.0
    for i in range(0, n, chunk_elems):
        end = min(i + chunk_elems, n)
        c = flat[i:end].float()
        s += float(c.sum())
        ss += float((c * c).sum())
        del c
    mean = s / n
    var = max(ss / n - mean * mean, 0.0)
    return math.sqrt(var)


def _noise_scale_chunked(
    flat: torch.Tensor,
    sigma: float | None,
    relative: float | None,
    *,
    chunk_elems: int,
) -> float:
    if sigma is not None:
        return float(sigma)
    if relative is not None:
        std = _std_chunked(flat, chunk_elems=chunk_elems)
        base = max(std, 1e-12)
        return float(relative) * base
    raise ValueError("Provide --sigma or --relative")


def _perturb_tensor(
    t: torch.Tensor,
    *,
    sigma: float | None,
    relative: float | None,
    generator: torch.Generator,
    chunk_elems: int,
) -> torch.Tensor:
    """Add Gaussian noise; large tensors use chunked math to limit peak RAM/VRAM."""
    if not t.dtype.is_floating_point:
        return t.clone()

    n = t.numel()
    if n <= _SMALL_TENSOR_ELEMS:
        scale = _noise_scale_for_tensor(t, sigma, relative)
        tf = t.float()
        noise = torch.randn(
            tf.shape,
            generator=generator,
            dtype=torch.float32,
            device=tf.device,
        ) * scale
        out = (tf + noise).to(dtype=t.dtype, device=t.device)
        del tf, noise
        return out

    flat = t.reshape(-1)
    scale = _noise_scale_chunked(flat, sigma, relative, chunk_elems=chunk_elems)
    out = torch.empty_like(t)
    out_flat = out.reshape(-1)
    for i in range(0, n, chunk_elems):
        end = min(i + chunk_elems, n)
        x = flat[i:end].float()
        noise = torch.randn(
            end - i,
            generator=generator,
            dtype=torch.float32,
            device=x.device,
        ) * scale
        out_flat[i:end] = (x + noise).to(dtype=t.dtype)
        del x, noise
    return out


def _generator_on(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def add_noise_to_state_dict(
    tensors: dict[str, torch.Tensor],
    *,
    sigma: float | None,
    relative: float | None,
    generator: torch.Generator,
    compute_device: torch.device,
    chunk_elems: int,
) -> dict[str, torch.Tensor]:
    """
    Add noise in-place on ``tensors`` to avoid holding two full checkpoints (~2× RAM).

    Mutates and returns the same dict (caller must not retain another reference to the
    pre-noise dict while this runs).
    """
    for name in list(tensors.keys()):
        t = tensors.pop(name)
        if compute_device.type == "cuda" and t.dtype.is_floating_point:
            td = t.to(compute_device, non_blocking=True)
            noisy = _perturb_tensor(
                td,
                sigma=sigma,
                relative=relative,
                generator=generator,
                chunk_elems=chunk_elems,
            )
            tensors[name] = noisy.cpu()
            del td
        else:
            tensors[name] = _perturb_tensor(
                t,
                sigma=sigma,
                relative=relative,
                generator=generator,
                chunk_elems=chunk_elems,
            )
        del t
    return tensors


def process_safetensors_file(
    src: Path,
    dst: Path,
    *,
    sigma: float | None,
    relative: float | None,
    generator: torch.Generator,
    compute_device: torch.device,
    chunk_elems: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    threshold = _streaming_threshold_bytes()
    if threshold == 0 or src.stat().st_size >= threshold:
        process_safetensors_file_streaming(
            src,
            dst,
            sigma=sigma,
            relative=relative,
            generator=generator,
            device=compute_device,
            chunk_elems=chunk_elems,
        )
        return

    tensors = load_file(str(src))
    add_noise_to_state_dict(
        tensors,
        sigma=sigma,
        relative=relative,
        generator=generator,
        compute_device=compute_device,
        chunk_elems=chunk_elems,
    )
    save_file(tensors, str(dst))


def copy_non_safetensors(src_dir: Path, dst_dir: Path) -> None:
    """Copy tokenizer/config files; caller overwrites *.safetensors separately."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            continue
        rel = path.relative_to(src_dir)
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)


def process_directory(
    src_dir: Path,
    dst_dir: Path,
    *,
    sigma: float | None,
    relative: float | None,
    seed: int,
    overwrite: bool,
    compute_device: torch.device,
    chunk_elems: int,
    verbose: bool,
) -> None:
    st_files = sorted(src_dir.rglob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files under {src_dir}")

    if dst_dir.exists() and not overwrite:
        raise FileExistsError(f"{dst_dir} exists; pass --overwrite to replace")

    copy_non_safetensors(src_dir, dst_dir)

    # Fresh generator per shard so order is deterministic for a fixed seed
    for i, src_path in enumerate(st_files):
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel
        gen = _generator_on(compute_device, seed + i * 1_000_003)
        t0 = time.perf_counter()
        process_safetensors_file(
            src_path,
            dst_path,
            sigma=sigma,
            relative=relative,
            generator=gen,
            compute_device=compute_device,
            chunk_elems=chunk_elems,
        )
        if verbose:
            if compute_device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(f"  [{i + 1}/{len(st_files)}] {rel}  {dt:.2f}s", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to safetensors weight tensors (float only)."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Directory (HF model folder) or a single .safetensors file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory or output .safetensors file",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--sigma",
        type=float,
        help="Absolute noise std (applied in FP32, same for all tensors)",
    )
    g.add_argument(
        "--relative",
        type=float,
        help="Noise std = relative * std(tensor) per tensor (good default for mixed scales)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Where to run noise math: cpu | cuda | cuda:N (streaming loads shards onto GPU when cuda)",
    )
    parser.add_argument(
        "--chunk-elements",
        type=int,
        default=_DEFAULT_CHUNK_ELEMS,
        metavar="N",
        help=f"Chunk size (elements) for large tensors / std estimate (default {_DEFAULT_CHUNK_ELEMS}). "
        "Increase on large GPUs for throughput.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Per-shard timings on stderr (uses CUDA sync when --device cuda)",
    )
    args = parser.parse_args()

    inp = args.input_path.resolve()
    out = args.out.resolve()

    if args.chunk_elements < 65_536:
        print("--chunk-elements must be at least 65536", file=sys.stderr)
        return 2

    compute_device = _resolve_device(args.device.strip())
    chunk_elems = int(args.chunk_elements)

    _set_global_seed(args.seed)

    if inp.is_dir():
        if out.exists() and not out.is_dir():
            print("--out must be a directory when input is a directory", file=sys.stderr)
            return 2
        n_shards = len(list(inp.rglob("*.safetensors")))
        if args.verbose:
            print(
                f"device={compute_device} chunk_elems={chunk_elems} shards={n_shards}",
                file=sys.stderr,
            )
        process_directory(
            inp,
            out,
            sigma=args.sigma,
            relative=args.relative,
            seed=args.seed,
            overwrite=args.overwrite,
            compute_device=compute_device,
            chunk_elems=chunk_elems,
            verbose=args.verbose,
        )
        print(f"Wrote noisy checkpoint to {out} ({n_shards} shard(s)).")
        return 0

    if inp.is_file() and inp.suffix == ".safetensors":
        if out.suffix != ".safetensors":
            print("--out should be a .safetensors file when input is a single shard", file=sys.stderr)
            return 2
        gen = _generator_on(compute_device, args.seed)
        t0 = time.perf_counter()
        process_safetensors_file(
            inp,
            out,
            sigma=args.sigma,
            relative=args.relative,
            generator=gen,
            compute_device=compute_device,
            chunk_elems=chunk_elems,
        )
        if args.verbose:
            if compute_device.type == "cuda":
                torch.cuda.synchronize()
            print(f"  done in {time.perf_counter() - t0:.2f}s", file=sys.stderr)
        print(f"Wrote {out}")
        return 0

    print("Input must be a directory or a .safetensors file", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
