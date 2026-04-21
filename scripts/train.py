#!/usr/bin/env python3
"""CLI entry point for training. Wraps particlegs.training.train."""
import runpy
import sys

sys.argv[0] = "particlegs.training.train"
runpy.run_module("particlegs.training.train", run_name="__main__", alter_sys=True)
