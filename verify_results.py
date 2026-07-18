#!/usr/bin/env python3
"""Verify reviewer run results against the authors' reference values.

Reads each experiment's `runs/<exp>/results.json`, extracts the metrics
listed in `reference_results.json`, and prints PASS/FAIL per metric with
the tolerance band applied.

Tolerance classes:
  - hw_independent: absolute or relative tolerance against the reference.
        PSNR, Gaussian counts, model sizes, compression ratios. Must match.
  - hw_dependent: absolute value is GPU-specific (FPS, wall-clock time,
        peak memory). Reported for information only, never fails the run.
  - trend: enforces a relation (e.g. `speedup > 100`). Absolute value free.

Usage:
    python verify_results.py                              # use adae/runs
    python verify_results.py --runs_dir some/other/runs   # custom location
    python verify_results.py --reference some_ref.json    # custom reference
    python verify_results.py --strict                     # exit 1 on HW-indep fails

Exit code is 0 if all hw_independent + trend metrics pass, 1 otherwise
(unless --no-strict, which always returns 0).
"""

import argparse
import json
import os
import sys
from pathlib import Path

ADAE_ROOT = Path(__file__).resolve().parent

# Color PASS/FAIL tags on interactive terminals only — plain when piped/tee'd
# to a log, when NO_COLOR is set, or on a dumb terminal.
_USE_COLOR = (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
              and "NO_COLOR" not in os.environ
              and os.environ.get("TERM") != "dumb")


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


def _ctag(tag):
    return _c({"PASS": "32", "FAIL": "1;31", "INFO": "36"}.get(tag, "33"), tag)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_nested(d, path):
    """Look up a dotted path like 'exp4.blocks_2.finetuned.masked_psnr'.

    The first segment selects which results.json to load (handled by caller);
    this helper walks the remaining segments within a single dict. An
    all-digit segment indexes into a list (e.g. 'exp1.exp1a_sz3.13.cr' is
    the 14th SZ3 rate-distortion point — point order is fixed by the
    experiment's error-bound sweep, so indices are stable across runs).
    Returns None if any segment is missing.
    """
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def format_val(v, unit):
    if v is None:
        return "MISSING"
    if isinstance(v, float):
        s = f"{v:.3f}".rstrip("0").rstrip(".") or "0"
    else:
        s = str(v)
    return f"{s} {unit}".strip() if unit else s


def check_hw_independent(actual, expected, tol_abs, tol_rel):
    """Return (passed, delta_str) for a hw_independent metric."""
    if actual is None:
        return False, "missing"
    diff = actual - expected
    if tol_abs is not None:
        ok = abs(diff) <= tol_abs
        return ok, f"delta={diff:+.3f} (tol=+/-{tol_abs})"
    if tol_rel is not None:
        denom = abs(expected) if expected != 0 else 1.0
        ok = abs(diff) / denom <= tol_rel
        pct = 100 * diff / denom if denom else 0
        return ok, f"delta={diff:+.3f} ({pct:+.1f}%, tol=+/-{tol_rel*100:.0f}%)"
    return False, "reference has no tolerance set"


def check_trend(actual, rule, threshold):
    if actual is None:
        return False, "missing"
    ops = {
        "gt": (actual > threshold, ">"),
        "ge": (actual >= threshold, ">="),
        "lt": (actual < threshold, "<"),
        "le": (actual <= threshold, "<="),
    }
    if rule not in ops:
        return False, f"unknown trend rule {rule!r}"
    ok, op = ops[rule]
    return ok, f"{actual} {op} {threshold}? {'yes' if ok else 'no'}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs_dir", default=str(ADAE_ROOT / "runs"),
                        help="Directory containing <exp>/results.json subdirs")
    parser.add_argument("--reference", default=str(ADAE_ROOT / "reference_results.json"),
                        help="Path to reference_results.json")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True,
                        help="Exit 1 on hw_independent or trend failures (default)")
    parser.add_argument("--no-strict", dest="strict", action="store_false",
                        help="Always exit 0 regardless of failures")
    parser.add_argument("--ae", action="store_true",
                        help="AE fast-path mode: enforce only the reduced set "
                             "(skip EXP-13/FIRE-2, EXP-4 2-block, and the LCP "
                             "baseline, which the full scripts/reproduce.sh run "
                             "still covers). Matches run_all.py --ae / "
                             "scripts/reproduce_ae.sh.")
    args = parser.parse_args()

    # Metric-path prefixes excluded from the AE fast path. See run_all.py
    # AE_EXPERIMENTS. Dropped to fit the SC AE budget: the two render-heaviest
    # units (FIRE-2 full retrain, EXP-4 2-block config) and the LCP baseline
    # (strictly worse than SZ3; the paper's iso-CR comparison only needs SZ3).
    AE_SKIP_PREFIXES = ("exp_fire2.", "exp4.blocks_2.", "exp1.exp1c_lcp.")

    runs_dir = Path(args.runs_dir).resolve()
    ref = load_json(args.reference)

    cached = {}

    def load_exp(exp_name):
        if exp_name not in cached:
            path = runs_dir / exp_name / "results.json"
            cached[exp_name] = load_json(path) if path.exists() else None
        return cached[exp_name]

    counts = {"hw_indep_pass": 0, "hw_indep_fail": 0,
              "trend_pass": 0, "trend_fail": 0,
              "hw_dep_reported": 0, "missing": 0}
    fails = []

    print(f"Verifying results in: {runs_dir}")
    print(f"Against reference:    {args.reference}")
    print(_c("1", "=" * 80))

    for m in ref["metrics"]:
        path = m["path"]
        if args.ae and path.startswith(AE_SKIP_PREFIXES):
            continue  # dropped from the AE fast path (covered by reproduce.sh)
        exp_name, inner_path = path.split(".", 1)
        data = load_exp(exp_name)
        if data is None:
            # A missing results.json means the experiment never ran (or
            # crashed). Scored metrics must FAIL — never silently shrink the
            # denominator, or a partial run could print an all-green OVERALL.
            counts["missing"] += 1
            if m["class"] in ("hw_independent", "trend"):
                key = "hw_indep_fail" if m["class"] == "hw_independent" else "trend_fail"
                counts[key] += 1
                fails.append(path)
                print(f"[{_ctag('FAIL')}] {path}  (no {exp_name}/results.json — experiment did not run)")
            else:
                print(f"[{_ctag('INFO')}] {path}  (no {exp_name}/results.json; hardware-dependent, reported only)")
            continue

        actual = get_nested(data, inner_path)
        expected = m.get("expected")
        unit = m.get("unit", "")
        cls = m["class"]

        if cls == "hw_independent":
            ok, detail = check_hw_independent(
                actual, expected,
                m.get("tolerance_abs"), m.get("tolerance_rel"))
            tag = "PASS" if ok else "FAIL"
            counts["hw_indep_pass" if ok else "hw_indep_fail"] += 1
            print(f"[{_ctag(tag)}] {path}")
            print(f"       expected {format_val(expected, unit)}, "
                  f"actual {format_val(actual, unit)}  ({detail})")
            if not ok:
                fails.append(path)

        elif cls == "trend":
            ok, detail = check_trend(
                actual, m.get("trend_rule", "gt"), m.get("trend_threshold", 0))
            tag = "PASS" if ok else "FAIL"
            counts["trend_pass" if ok else "trend_fail"] += 1
            print(f"[{_ctag(tag)}] {path}  (trend)")
            print(f"       {detail}")
            if not ok:
                fails.append(path)

        elif cls == "hw_dependent":
            counts["hw_dep_reported"] += 1
            ref_gpu = m.get("reference_gpu", "reference GPU")
            print(f"[{_ctag('INFO')}] {path}  (hardware-dependent, reported only)")
            print(f"       reviewer {format_val(actual, unit)}, "
                  f"reference {format_val(expected, unit)} on {ref_gpu}")
            if "note" in m:
                print(f"       note: {m['note']}")

        else:
            print(f"[{_ctag('WARN')}] {path}  (unknown class {cls!r})")

    # Summary
    print(_c("1", "=" * 80))
    total_strict = (counts["hw_indep_pass"] + counts["hw_indep_fail"]
                    + counts["trend_pass"] + counts["trend_fail"])
    strict_pass = counts["hw_indep_pass"] + counts["trend_pass"]
    print(f"Hardware-independent: {counts['hw_indep_pass']}/"
          f"{counts['hw_indep_pass']+counts['hw_indep_fail']} passed")
    print(f"Trend:                {counts['trend_pass']}/"
          f"{counts['trend_pass']+counts['trend_fail']} passed")
    print(f"Hardware-dependent:   {counts['hw_dep_reported']} reported (not scored)")
    if counts["missing"]:
        print(f"Missing results.json: {counts['missing']} metrics "
              f"(scored ones counted as FAIL)")
    print(_c("32" if not fails else "1;31",
             f"OVERALL:              {strict_pass}/{total_strict} "
             f"enforced metrics passed"))
    if fails:
        print(_c("1;31", "\nFailed metrics:"))
        for f in fails:
            print(f"  - {f}")

    if args.strict and fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
