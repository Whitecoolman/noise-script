#!/usr/bin/env python3
"""
HuggingFace Model Download, Convert, and Upload Bot

USAGE:
------
# Using command-line arguments (recommended):
python hf_convert(basic).py --source_repo ChrisJackieChan/Affine-Vitovvv --target_repo WebScraper991923/Affine-S3

# Or use defaults (modify variables below):
python hf_convert(basic).py

python3 /home/sn38/albedo/hf_convert1.py \
  --source_repo kumaresano/albedo-qwen3.6-35b-bk20 \
  --target_repo samman311/albedo-qwen3.6-35b-ty5 \
  --noise-sigma 1e-4 \
  --seed 24

WHAT IT DOES:
-------------
1. Downloads model from source HuggingFace repository
2. Removes README-related files from the local model root, rewrites safetensors metadata hashes,
   then adds Gaussian noise to float weights in every ``*.safetensors`` shard (see ``add_weight_noise.py``)
   unless ``--skip-noise``
3. Cleans target repository (removes all existing files)
4. Uploads to your target HuggingFace repository

EXAMPLE:
--------
python hf_convert_2.py --source_repo "allforone111/nimbus-V7" --target_repo "Seniordev90101/Test-H1"
python hf_convert.py --source_repo model_name --target_repo "Seniordev90101/Affine-H6-5FcN2yN4685mGzAj63D4VZpzFAidVtc2pv4xepQynfu59D5e"
"""

import os
import sys
import json
import hashlib
import struct
import time
import shutil
import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download, HfApi, create_repo, CommitOperationDelete

# Same-directory import works when running `python hf_convert1.py` from anywhere
_AFFINE_DIR = Path(__file__).resolve().parent
if str(_AFFINE_DIR) not in sys.path:
    sys.path.insert(0, str(_AFFINE_DIR))

from add_weight_noise import (
    _DEFAULT_CHUNK_ELEMS,
    _generator_on,
    _resolve_device,
    process_safetensors_file,
    _set_global_seed,
)

# ============================================
# INPUT VARIABLES - SET THESE BEFORE RUNNING
# ============================================
source_repo = "affinier/affine-star-1"  # Source HuggingFace repo to download from
target_repo = "WebScraper991923/Affine-S9"  # Your target HuggingFace repo to upload to
hf_token = ""  # Set HF_TOKEN env, pass --hf_token, or use cached `hf auth login`
work_dir = "/home/sn38/albedo"   # Working directory for downloaded models


# python hf_convert_bot.py

# ============================================
# HASH MODIFICATION FUNCTIONS (from hash.py)
# ============================================
def generate_new_hash():
    """Generate a new random hash for the safetensor file."""
    timestamp = str(int(time.time() * 1000000))  # microseconds
    random_data = os.urandom(16)  # 16 random bytes
    combined = (timestamp + random_data.hex()).encode()
    return hashlib.sha256(combined).hexdigest()

# Chunk size when copying tensor payload (bounds RAM; avoids OOM on multi-GB shards).
_SAFE_COPY_CHUNK = 256 * 1024 * 1024


def modify_safetensor_hash(file_path, display_name=None):
    """
    Modify metadata hash in a safetensors file.

    Uses a temp file + chunked copy for the tensor blob so multi‑GB models do not
    require loading the whole file into RAM (which previously triggered Linux OOM killer).
    """
    label = display_name if display_name is not None else file_path.name
    path = Path(file_path)
    print(f"  Processing: {label}")

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(path, "rb") as f:
        header_length = struct.unpack("<Q", f.read(8))[0]
        header_data = f.read(header_length)
        tensor_start = 8 + header_length

    header = json.loads(header_data.decode("utf-8"))

    new_hash = generate_new_hash()
    print(f"    New hash: {new_hash}")

    if "__metadata__" not in header:
        header["__metadata__"] = {}

    header["__metadata__"]["hash"] = new_hash
    header["__metadata__"]["modified"] = str(int(time.time()))

    new_header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    new_header_length = len(new_header_json)

    file_size = path.stat().st_size
    tensor_total_len = file_size - tensor_start
    if tensor_total_len < 0:
        raise ValueError(f"Invalid safetensors header length in {label}")

    try:
        with open(tmp_path, "wb") as out:
            out.write(struct.pack("<Q", new_header_length))
            out.write(new_header_json)
            with open(path, "rb") as inp:
                inp.seek(tensor_start)
                remaining = tensor_total_len
                while remaining > 0:
                    n = min(_SAFE_COPY_CHUNK, remaining)
                    buf = inp.read(n)
                    if len(buf) != n:
                        raise ValueError(
                            f"Short read copying tensor payload for {label} "
                            f"(expected {n} bytes, got {len(buf)})"
                        )
                    out.write(buf)
                    remaining -= n
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.is_file():
                tmp_path.unlink()
        except OSError:
            pass
        raise

    print(f"Successfully modified hash for {label}")


def apply_weight_noise_inplace(
    file_path: Path,
    *,
    sigma: float | None,
    relative: float | None,
    shard_index: int,
    noise_seed: int,
    compute_device: torch.device,
    chunk_elems: int,
) -> None:
    """Load shard, add Gaussian noise (same rules as ``add_weight_noise.py``), atomically replace file."""
    tmp_path = file_path.with_suffix(file_path.suffix + ".noise.tmp")
    gen = _generator_on(compute_device, noise_seed + shard_index * 1_000_003)
    try:
        process_safetensors_file(
            file_path,
            tmp_path,
            sigma=sigma,
            relative=relative,
            generator=gen,
            compute_device=compute_device,
            chunk_elems=chunk_elems,
        )
        os.replace(tmp_path, file_path)
    except Exception:
        if tmp_path.is_file():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _is_readme_related_root_filename(name: str) -> bool:
    """True if filename looks like a README / model card at repo root (case-insensitive)."""
    lower = name.lower()
    if lower == "readme":
        return True
    if lower.startswith("readme."):
        return True
    if lower.startswith("readme_"):
        return True
    return False


def remove_readme_related_files_at_root(model_dir: str | Path) -> int:
    """
    Delete README-related files only in the model directory root (not subfolders).
    Run before hash modification so uploads omit original cards/README copies.
    """
    root = Path(model_dir)
    if not root.is_dir():
        return 0
    removed = 0
    for entry in sorted(root.iterdir()):
        if not entry.is_file():
            continue
        if not _is_readme_related_root_filename(entry.name):
            continue
        try:
            entry.unlink()
            print(f"  [REMOVED root] {entry.name}")
            removed += 1
        except OSError as e:
            print(f"  [WARNING] Could not remove {entry.name}: {e}")
    return removed


# ============================================
# MAIN BOT FUNCTIONS
# ============================================
def download_model(source_repo, local_dir, token=None):
    """Download model from HuggingFace repository."""
    print(f"\n{'='*60}")
    print(f"[STEP 1/4] Downloading model from: {source_repo}")
    print(f"{'='*60}")
    
    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download the model
        print(f"Downloading to: {local_dir}")
        snapshot_download(
            repo_id=source_repo,
            local_dir=local_dir,
            token=token,
            resume_download=True
        )
        
        print(f"\n[SUCCESS] Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}")
        return False

def convert_model(
    model_dir,
    *,
    skip_noise: bool = False,
    noise_relative: float | None = 1e-4,
    noise_sigma: float | None = None,
    noise_seed: int = 42,
    noise_device: str = "cpu",
    noise_chunk_elems: int | None = None,
):
    """Rewrite safetensors metadata hashes, then optionally apply weight noise."""
    print(f"\n{'='*60}")
    print(f"[STEP 2/4] Converting safetensor files (metadata hash + noise)")
    print(f"{'='*60}")
    
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print(f"[ERROR] Model directory not found: {model_path}")
        return False

    print("Removing README-related files at model root (before noise/hash pass)...")
    n_rm = remove_readme_related_files_at_root(model_path)
    if n_rm:
        print(f"  Removed {n_rm} file(s).")
    else:
        print("  No README-related root files found.")
    
    # All nested shards (same coverage as ``add_weight_noise.process_directory``)
    safetensor_files = sorted(model_path.rglob("*.safetensors"), key=lambda p: str(p))
    
    if not safetensor_files:
        print("[WARNING] No safetensor files found in the directory")
        print("[INFO] This might be okay if the model uses a different format")
        return True
    
    print(f"Found {len(safetensor_files)} safetensor file(s):")
    for file in safetensor_files:
        try:
            rel = file.relative_to(model_path)
        except ValueError:
            rel = file
        print(f"  - {rel}")

    print("\nModifying safetensors metadata hashes...")
    success_count = 0
    for file_path in safetensor_files:
        try:
            try:
                disp = str(file_path.relative_to(model_path))
            except ValueError:
                disp = file_path.name
            modify_safetensor_hash(file_path, display_name=disp)
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to process {file_path}: {e}")

    if success_count == 0 and len(safetensor_files) > 0:
        return False

    if not skip_noise:
        if noise_sigma is not None and noise_relative is not None:
            print("[ERROR] Pass only one of noise_sigma or noise_relative")
            return False
        sigma_arg = noise_sigma
        relative_arg = None if noise_sigma is not None else noise_relative
        if sigma_arg is None and relative_arg is None:
            relative_arg = 1e-4
        ce = noise_chunk_elems if noise_chunk_elems is not None else _DEFAULT_CHUNK_ELEMS
        if ce < 65_536:
            print("[ERROR] noise_chunk_elems must be >= 65536")
            return False
        compute_device = _resolve_device(noise_device.strip())
        mode = f"sigma={sigma_arg}" if sigma_arg is not None else f"relative={relative_arg}"
        print(
            f"\nAdding Gaussian noise to float weights ({mode}, seed={noise_seed}, "
            f"device={compute_device}, chunk_elems={ce})..."
        )
        _set_global_seed(noise_seed)
        for i, file_path in enumerate(safetensor_files):
            try:
                rel = file_path.relative_to(model_path)
            except ValueError:
                rel = file_path.name
            print(f"  [noise] {rel}")
            apply_weight_noise_inplace(
                file_path,
                sigma=sigma_arg,
                relative=relative_arg,
                shard_index=i,
                noise_seed=noise_seed,
                compute_device=compute_device,
                chunk_elems=ce,
            )
        print("[SUCCESS] Weight noise applied.")
    else:
        print("\n[INFO] Skipping weight noise (--skip-noise).")

    print(f"\n[SUCCESS] Finished step 2: hash pass on {success_count}/{len(safetensor_files)} file(s)"
          + ("" if skip_noise else " + noise"))
    return success_count > 0 or len(safetensor_files) == 0

def clean_repository(target_repo, token=None):
    """Clean target repository by deleting all existing files in one commit."""
    print(f"\n{'='*60}")
    print(f"[STEP 3/4] Cleaning target repository: {target_repo}")
    print(f"{'='*60}")
    
    try:
        api = HfApi(token=token)
        
        # Check if repository exists and has files
        try:
            files = api.list_repo_files(repo_id=target_repo, repo_type="model")
            if len(files) > 0:
                print(f"Found {len(files)} files in target repository")
                print("Deleting all files in one commit...")
                
                # Build delete operations and commit once
                operations = [CommitOperationDelete(path_in_repo=f) for f in files]
                api.create_commit(
                    repo_id=target_repo,
                    repo_type="model",
                    operations=operations,
                    commit_message="Clean repository before upload",
                    token=token,
                )
                print(f"  Deleted {len(files)} files in one commit")
                print(f"[SUCCESS] Repository cleaned successfully!")
            else:
                print(f"[INFO] Repository is already empty")
        except Exception as e:
            # Repository might not exist yet, which is fine
            print(f"[INFO] Repository doesn't exist yet or is empty: {e}")
        
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to clean repository: {e}")
        return False

def upload_model(local_dir, target_repo, token=None):
    """Upload converted model to target HuggingFace repository."""
    print(f"\n{'='*60}")
    print(f"[STEP 4/4] Uploading model to: {target_repo}")
    print(f"{'='*60}")
    
    try:
        api = HfApi(token=token)
        
        # Create repository if it doesn't exist
        try:
            print(f"Creating repository: {target_repo}")
            create_repo(
                repo_id=target_repo,
                token=token,
                private=False,
                exist_ok=True
            )
            print(f"[INFO] Repository ready")
        except Exception as e:
            print(f"[INFO] Repository might already exist: {e}")
        
        # Upload all files
        print(f"Uploading files from: {local_dir}")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=target_repo,
            token=token,
            commit_message=f"Thank you first commit"
        )
        
        print(f"\n[SUCCESS] Model uploaded successfully!")
        print(f"[INFO] View your model at: https://huggingface.co/{target_repo}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to upload model: {e}")
        return False

def cleanup(local_dir, keep_files=False):
    """Clean up temporary files."""
    if keep_files:
        print(f"\n[INFO] Keeping downloaded files at: {local_dir}")
        return
    
    try:
        if os.path.exists(local_dir):
            print(f"\n[INFO] Cleaning up temporary files...")
            shutil.rmtree(local_dir)
            print(f"[SUCCESS] Cleaned up: {local_dir}")
    except Exception as e:
        print(f"[WARNING] Failed to cleanup: {e}")

def run_conversion_bot(
    source_repo,
    target_repo,
    token=None,
    keep_files=False,
    *,
    skip_noise: bool = False,
    noise_relative: float | None = 1e-4,
    noise_sigma: float | None = None,
    noise_seed: int = 42,
    noise_device: str = "cpu",
    noise_chunk_elems: int | None = None,
):
    """Main bot function to download, convert, clean, and upload."""
    print(f"\n{'#'*60}")
    print(f"# HuggingFace Model Conversion Bot")
    print(f"{'#'*60}")
    print(f"Source: {source_repo}")
    print(f"Target: {target_repo}")
    print(f"{'#'*60}\n")
    
    # Resolve auth: explicit arg > HF_TOKEN env > cached `hf auth login`
    if token is None:
        token = os.environ.get('HF_TOKEN')
        if token:
            print("[INFO] Using HF_TOKEN from environment")
        else:
            print("[INFO] Using cached HuggingFace login token (if any)")
    
    # Create working directory with source repo name
    repo_name = source_repo.split('/')[-1]
    local_dir = os.path.join(work_dir, repo_name)
    
    # Step 1: Download
    if not download_model(source_repo, local_dir, token):
        return False
    
    # Step 2: Convert (metadata hash + noise)
    if not convert_model(
        local_dir,
        skip_noise=skip_noise,
        noise_relative=noise_relative,
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        noise_device=noise_device,
        noise_chunk_elems=noise_chunk_elems,
    ):
        return False
    
    # Step 3: Clean target repository
    if not clean_repository(target_repo, token):
        return False
    
    # Step 4: Upload
    if not upload_model(local_dir, target_repo, token):
        return False
    
    # Cleanup
    cleanup(local_dir, keep_files)
    
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Conversion bot finished successfully!")
    print(f"{'='*60}\n")
    return True

# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download, convert, and upload HuggingFace models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hf_convert1.py --source_repo target919/affine-target-6 --target_repo morganbeat/Affine-20253
  python hf_convert1.py --source_repo "meta-llama/Llama-3.2-1B" --target_repo "learydenis/affine-world-200"
  python hf_convert1.py --source_repo "user/repo" --target_repo "target/repo" --hf_token "hf_xxx" --keep
  python hf_convert1.py --source_repo "a/b" --target_repo "c/d" --noise-relative 5e-5 --noise-seed 0
  python hf_convert1.py --source_repo "a/b" --target_repo "c/d" --skip-noise
        """
    )
    
    parser.add_argument(
        "--source_repo",
        type=str,
        default=source_repo,
        help=f"Source repository (default: {source_repo})"
    )
    
    parser.add_argument(
        "--target_repo",
        type=str,
        default=target_repo,
        help=f"Target repository (default: {target_repo})"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (default: from environment variable HF_TOKEN or hardcoded value)"
    )
    
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep downloaded files after completion"
    )

    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Do not add Gaussian noise; only rewrite safetensors metadata hashes",
    )
    parser.add_argument(
        "--noise-relative",
        type=float,
        default=1e-4,
        metavar="R",
        help="Noise std = R * std(tensor) per tensor (default: 1e-4). Ignored if --noise-sigma is set",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=None,
        metavar="S",
        help="Absolute noise std for all float tensors (mutually exclusive with relative scaling)",
    )
    parser.add_argument(
        "--noise-seed",
        "--seed",
        type=int,
        default=42,
        help="RNG seed for weight noise (default: 42). --seed is an alias.",
    )
    parser.add_argument(
        "--noise-device",
        type=str,
        default="cpu",
        help="Device for weight noise (cpu | cuda | cuda:N). Large GPUs speed up big shards.",
    )
    parser.add_argument(
        "--noise-chunk-elements",
        type=int,
        default=_DEFAULT_CHUNK_ELEMS,
        metavar="N",
        help=f"Chunk size for noise/std loops (default {_DEFAULT_CHUNK_ELEMS}). Raise on GPU for throughput.",
    )

    args = parser.parse_args()
    
    # Use command-line arguments or fall back to defaults
    source_repo = args.source_repo
    target_repo = args.target_repo
    hf_token = args.hf_token or os.environ.get('HF_TOKEN') or hf_token
    keep_files = args.keep
    skip_noise = args.skip_noise
    noise_relative = None if args.noise_sigma is not None else args.noise_relative
    noise_sigma = args.noise_sigma
    noise_seed = args.noise_seed

    # Validate required arguments
    if not source_repo or not target_repo:
        parser.error("--source_repo and --target_repo are required")
    
    # Check for HF token
    if not hf_token and not os.environ.get('HF_TOKEN'):
        print("[WARNING] No HuggingFace token provided!")
        print("Set HF_TOKEN environment variable or use --hf_token")
        print("Public models can be downloaded without token, but uploading requires authentication")
    
    # Run the bot
    if args.noise_chunk_elements < 65_536:
        parser.error("--noise-chunk-elements must be >= 65536")

    success = run_conversion_bot(
        source_repo,
        target_repo,
        hf_token,
        keep_files,
        skip_noise=skip_noise,
        noise_relative=noise_relative,
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        noise_device=args.noise_device,
        noise_chunk_elems=args.noise_chunk_elements,
    )
    
    sys.exit(0 if success else 1)
