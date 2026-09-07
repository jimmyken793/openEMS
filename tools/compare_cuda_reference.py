#!/usr/bin/env python3
"""Compare stable CPU and CUDA reference-engine probe traces."""

import argparse
import json
import math
import sys
from pathlib import Path

EXPECTED_SAMPLES = 151
COMPLETION_MARKER = "Time for 900 iterations"


def read_probe(path):
    rows = []
    column_count = None
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        row = [float(value) for value in line.split()]
        if len(row) < 2:
            raise ValueError("{}:{} has fewer than two columns".format(
                path, line_number))
        if column_count is None:
            column_count = len(row)
        elif len(row) != column_count:
            raise ValueError("{}:{} has {} columns; expected {}".format(
                path, line_number, len(row), column_count))
        if not all(math.isfinite(value) for value in row):
            raise ValueError("{} contains a non-finite value".format(path))
        rows.append(row)
    if not rows:
        raise ValueError("{} contains no samples".format(path))
    return rows


def compare_probe(reference_path, candidate_path, tolerance):
    reference = read_probe(reference_path)
    candidate = read_probe(candidate_path)
    if len(reference) != EXPECTED_SAMPLES or len(candidate) != EXPECTED_SAMPLES:
        raise ValueError("{} requires {} complete samples; found {} and {}".format(
            reference_path.name, EXPECTED_SAMPLES, len(reference), len(candidate)))
    if len(reference) != len(candidate):
        raise ValueError("{} sample count differs: {} != {}".format(
            reference_path.name, len(reference), len(candidate)))
    if any(len(ref) != len(got) for ref, got in zip(reference, candidate)):
        raise ValueError("{} column counts differ".format(reference_path.name))
    if any(ref[0] != got[0] for ref, got in zip(reference, candidate)):
        raise ValueError("{} sample times differ".format(reference_path.name))

    reference_values = [value for row in reference for value in row[1:]]
    candidate_values = [value for row in candidate for value in row[1:]]
    reference_peak = max(abs(value) for value in reference_values)
    candidate_peak = max(abs(value) for value in candidate_values)
    if reference_peak == 0 or candidate_peak == 0:
        raise ValueError("{} must be nonzero for both engines".format(reference_path.name))

    differences = [
        reference_value - candidate_value
        for reference_value, candidate_value in zip(reference_values, candidate_values)
    ]
    onset_reference = next(
        row[0] for row in reference if any(value != 0 for value in row[1:]))
    onset_candidate = next(
        row[0] for row in candidate if any(value != 0 for value in row[1:]))
    significant = [
        (reference_value, candidate_value)
        for reference_value, candidate_value in zip(reference_values, candidate_values)
        if abs(reference_value) > reference_peak * 1e-12
    ]
    relative_max_error = max(abs(value) for value in differences) / reference_peak
    relative_l2_error = math.sqrt(
        sum(value * value for value in differences)
        / sum(value * value for value in reference_values)
    )
    polarity_mismatches = sum(
        (reference_value > 0) != (candidate_value > 0)
        for reference_value, candidate_value in significant
    )

    passed = (
        onset_reference == onset_candidate
        and polarity_mismatches == 0
        and relative_max_error <= tolerance
        and relative_l2_error <= tolerance
    )
    return {
        "samples": len(reference),
        "reference_peak": reference_peak,
        "candidate_peak": candidate_peak,
        "onset_reference_s": onset_reference,
        "onset_candidate_s": onset_candidate,
        "polarity_mismatches": polarity_mismatches,
        "relative_max_error": relative_max_error,
        "relative_l2_error": relative_l2_error,
        "tolerance": tolerance,
        "passed": passed,
    }


def require_completed_run(output_path, markers):
    log = (output_path / "openems.log").read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in log]
    if missing:
        raise ValueError("{} log is missing markers: {}".format(
            output_path, ", ".join(missing)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu_output", type=Path)
    parser.add_argument("cuda_output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be positive and finite")

    require_completed_run(args.cpu_output, (COMPLETION_MARKER,))
    require_completed_run(args.cuda_output, (
        COMPLETION_MARKER,
        "Create FDTD operator (CUDA reference)",
        "Create FDTD engine (CUDA lifecycle reference)",
        "Running on device",
    ))

    results = {
        name: compare_probe(
            args.cpu_output / name, args.cuda_output / name, args.tolerance)
        for name in ("E_probe", "H_probe")
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(result["passed"] for result in results.values()) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
