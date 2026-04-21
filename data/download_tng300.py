#!/usr/bin/env python3
"""Download TNG300-3 snapshot 99 DM coordinates and save as raw f32 files.

Designed to run unattended for hours. Retries indefinitely with 5-minute
backoff until all 16 chunks are downloaded. Already-downloaded chunks are
skipped automatically (resume-safe).

Usage:
    nohup python data/download_tng300.py > tng_download.log 2>&1 &
"""

import os
import subprocess
import time
import numpy as np
import h5py

API_KEY = "d0cd6409c6830ad8b2e7b4c65e6857dc"
BASE_URL = "https://www.tng-project.org/api/TNG300-3/files/snapshot-99"
NUM_CHUNKS = 16
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tng300_raw")
CHUNK_DIR = os.path.join(OUT_DIR, "chunks")
RETRY_WAIT = 300  # 5 minutes between retries


def chunk_is_valid(path):
    """Check if a chunk file exists and is a valid HDF5 with DM coordinates."""
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            n = f["PartType1"]["Coordinates"].shape[0]
        return n > 0
    except Exception:
        return False


def download_one_attempt(chunk_id):
    """Try downloading one chunk once. Returns path on success, None on failure."""
    out_path = os.path.join(CHUNK_DIR, f"chunk_{chunk_id}.hdf5")
    url = f"{BASE_URL}.{chunk_id}.hdf5?dm=Coordinates"
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] Chunk {chunk_id:2d}/15: downloading ...", end=" ", flush=True)

    if os.path.exists(out_path):
        os.remove(out_path)

    result = subprocess.run(
        [
            "curl", "-s", "-L",
            "--retry", "2",
            "--retry-delay", "15",
            "--max-time", "600",
            "-H", f"api-key: {API_KEY}",
            "-o", out_path,
            "-w", "%{http_code}",
            url,
        ],
        capture_output=True, text=True,
    )
    http_code = result.stdout.strip()

    if http_code == "200" and chunk_is_valid(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        with h5py.File(out_path, "r") as f:
            n = f["PartType1"]["Coordinates"].shape[0]
        print(f"HTTP {http_code}, {n:,} particles, {size_mb:.1f} MB", flush=True)
        return out_path

    print(f"HTTP {http_code}", flush=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    return None


def main():
    os.makedirs(CHUNK_DIR, exist_ok=True)

    print(f"TNG300-3 snapshot 99 DM downloader (persistent mode)")
    print(f"Output: {OUT_DIR}")
    print(f"Retry interval: {RETRY_WAIT}s\n", flush=True)

    # Step 1: Download all chunks — loop until all done
    while True:
        missing = []
        for i in range(NUM_CHUNKS):
            path = os.path.join(CHUNK_DIR, f"chunk_{i}.hdf5")
            if not chunk_is_valid(path):
                missing.append(i)

        if not missing:
            break

        print(f"\n  Missing chunks: {missing}", flush=True)
        for i in missing:
            path = download_one_attempt(i)
            if path:
                print(f"           -> OK!", flush=True)
                time.sleep(30)  # polite pause
            else:
                print(f"           -> failed, trying next chunk ...", flush=True)
                time.sleep(10)

        # If still have missing chunks, wait before next round
        remaining = sum(1 for i in range(NUM_CHUNKS)
                       if not chunk_is_valid(os.path.join(CHUNK_DIR, f"chunk_{i}.hdf5")))
        if remaining > 0:
            print(f"\n  {remaining} chunks remaining, waiting {RETRY_WAIT}s ...", flush=True)
            time.sleep(RETRY_WAIT)

    chunk_files = [os.path.join(CHUNK_DIR, f"chunk_{i}.hdf5") for i in range(NUM_CHUNKS)]

    # Step 2: Count total particles
    total = 0
    for path in chunk_files:
        with h5py.File(path, "r") as f:
            total += f["PartType1"]["Coordinates"].shape[0]
    print(f"\nTotal particles: {total:,}")

    # Step 3: Extract coordinates to flat f32 files
    for axis, name in enumerate(["xx.f32", "yy.f32", "zz.f32"]):
        out_path = os.path.join(OUT_DIR, name)
        print(f"Writing {name} ...", end=" ", flush=True)
        with open(out_path, "wb") as fout:
            for path in chunk_files:
                with h5py.File(path, "r") as f:
                    data = f["PartType1"]["Coordinates"][:, axis].astype(np.float32)
                    fout.write(data.tobytes())
        print(f"{os.path.getsize(out_path) / 1e9:.2f} GB")

    print("\nDone! You can delete the chunks dir to save space:")
    print(f"  rm -rf {CHUNK_DIR}")


if __name__ == "__main__":
    main()
