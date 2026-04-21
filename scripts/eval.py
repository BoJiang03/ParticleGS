#!/usr/bin/env python3
"""CLI entry point for evaluation metrics."""
import runpy
import sys

sys.argv[0] = "particlegs.evaluation.metrics"
runpy.run_module("particlegs.evaluation.metrics", run_name="__main__", alter_sys=True)
