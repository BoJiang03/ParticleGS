#!/usr/bin/env python3
"""CLI entry point for rendering. Wraps particlegs.evaluation.render."""
import runpy
import sys

sys.argv[0] = "particlegs.evaluation.render"
runpy.run_module("particlegs.evaluation.render", run_name="__main__", alter_sys=True)
