#!/usr/bin/env python3
"""End-to-end pipeline for ParticleGS.

Two modes:
  --mode single   Single-block training (E25 config, ~26 dB masked PSNR)
  --mode block    Multi-block KD-tree + F16 finetune (~27.5 dB masked PSNR)

Handles everything from raw .f32 data to final evaluation.

Usage:
    python end_to_end.py --mode single --name E2E_single
    python end_to_end.py --mode block  --name E2E_block_8 --num_blocks 8
    python end_to_end.py --mode block  --name E2E_block_8 --resume
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARTICLEGS_ROOT = SCRIPT_DIR.parent

PYTHON_BIN = sys.executable

SINGLE_BLOCK_SCRIPT = SCRIPT_DIR / "single_block.py"
BLOCK_PIPELINE_SCRIPT = SCRIPT_DIR / "block_pipeline.py"


def run(cmd, env=None):
    cmd_str = [str(c) for c in cmd]
    print(f"Running: {' '.join(cmd_str[-8:])}")
    result = subprocess.run(cmd_str, cwd=str(PARTICLEGS_ROOT), env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end ParticleGS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python end_to_end.py --mode single --name E2E_single
  python end_to_end.py --mode block  --name E2E_block_8
  python end_to_end.py --mode block  --name E2E_block_4 --num_blocks 4 --resume
""")
    parser.add_argument("--mode", required=True, choices=["single", "block"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--output_root", default=str(PARTICLEGS_ROOT / "runs"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"End-to-End Pipeline: {args.name}")
    print(f"  Mode: {args.mode}")
    print(f"  Output: {Path(args.output_root) / args.name}")
    print(f"{'='*60}")

    t_start = time.time()

    if args.mode == "single":
        cmd = [
            PYTHON_BIN, SINGLE_BLOCK_SCRIPT,
            "--name", args.name,
            "--output_root", args.output_root,
            "--gpu", str(args.gpu),
        ]
        if args.resume:
            cmd.append("--resume")
        rc = run(cmd)
    else:
        nb = args.num_blocks
        if nb < 2 or nb & (nb - 1) != 0:
            print("ERROR: --num_blocks must be a power of 2")
            sys.exit(1)
        cmd = [
            PYTHON_BIN, BLOCK_PIPELINE_SCRIPT,
            "--num_blocks", str(nb),
            "--name", args.name,
            "--output_root", args.output_root,
        ]
        if args.resume:
            cmd.append("--resume")
        rc = run(cmd)

    elapsed = time.time() - t_start
    status = "SUCCESS" if rc == 0 else f"FAILED (exit {rc})"
    print(f"\n{status} — Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
