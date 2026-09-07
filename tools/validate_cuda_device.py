#!/usr/bin/env python3
"""Run synthetic CPU/CUDA extension regressions and retain every artifact."""

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

SUPPORTED = ('pml', 'pec', 'hard-e-delay', 'soft-h', 'hard-h-delay', 'duplicate-e')
UNSUPPORTED = {'periodic': 'Steady', 'mixed-mur': 'Mur'}
DEVICE_MARKER = 'Create FDTD engine (CUDA device extensions)'
REFERENCE_MARKER = 'Create FDTD engine (CUDA lifecycle reference)'
COMPLETION = re.compile(r'Time for (\d+) iterations with ([\d.eE+-]+) cells : ([\d.eE+-]+) sec')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_probe(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        row = [float(value) for value in line.split()]
        require(len(row) == 4, '{}: expected time and three field components'.format(path))
        require(all(math.isfinite(value) for value in row), '{}: non-finite sample'.format(path))
        if rows:
            require(row[0] > rows[-1][0], '{}: sample times are not increasing'.format(path))
        rows.append(row)
    require(len(rows) > 1, '{}: insufficient samples'.format(path))
    require(any(value != 0 for row in rows for value in row[1:]), '{}: all-zero trace'.format(path))
    return rows


def compare(reference, candidate, tolerance):
    result = {}
    for probe in ('E_probe', 'H_probe'):
        expected = read_probe(reference / probe)
        actual = read_probe(candidate / probe)
        require(len(expected) == len(actual), '{}: sample counts differ'.format(probe))
        require(all(a[0] == b[0] for a, b in zip(expected, actual)), '{}: sample times differ'.format(probe))
        av = [value for row in expected for value in row[1:]]
        bv = [value for row in actual for value in row[1:]]
        peak = max(map(abs, av))
        error = [a - b for a, b in zip(av, bv)]
        maximum = max(map(abs, error)) / peak
        l2 = math.sqrt(sum(x * x for x in error) / sum(x * x for x in av))
        # Floating-point noise near zero must not create a false polarity failure.
        significant = [(a, b) for a, b in zip(av, bv) if abs(a) > peak * tolerance]
        polarity = sum((a > 0) != (b > 0) for a, b in significant)
        onset_a = next(row[0] for row in expected if any(x != 0 for x in row[1:]))
        onset_b = next(row[0] for row in actual if any(x != 0 for x in row[1:]))
        require(maximum <= tolerance and l2 <= tolerance,
                '{}: relative max/L2 errors {:.6g}/{:.6g} exceed {}'.format(probe, maximum, l2, tolerance))
        require(polarity == 0, '{}: significant polarity mismatch'.format(probe))
        require(onset_a == onset_b, '{}: onset differs'.format(probe))
        result[probe] = dict(samples=len(expected), reference_peak=peak,
                             candidate_peak=max(map(abs, bv)), relative_max_error=maximum,
                             relative_l2_error=l2, onset_s=onset_a, polarity_mismatches=polarity)
    return result


def run(binary, engine, xml, output, timeout, marker=None, unsupported=None):
    output.mkdir(parents=True)
    command = [str(binary), str(xml), '--engine=' + engine, '-v']
    start = time.perf_counter()
    with (output / 'openems.log').open('w') as log:
        completed = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    wall = time.perf_counter() - start
    log = (output / 'openems.log').read_text()
    result = dict(command=command, returncode=completed.returncode, process_wall_seconds=wall)
    if unsupported:
        require(completed.returncode != 0, '{}: unsupported device case succeeded'.format(xml.name))
        require('CUDA device extensions do not support active extension:' in log,
                '{}: failure did not identify an unsupported extension'.format(xml.name))
        require(unsupported.lower() in log.lower() and '--engine=cuda-reference' in log,
                '{}: rejection missing extension name or reference guidance'.format(xml.name))
        result['expected_rejection'] = True
        return result
    require(completed.returncode == 0, '{} {}: solver exit {}'.format(xml.name, engine, completed.returncode))
    if marker:
        require(marker in log and 'Running on device' in log, '{}: wrong CUDA execution mode'.format(output))
    else:
        require('Create FDTD engine' in log and 'Running on device' not in log, '{}: wrong CPU engine'.format(output))
    matches = COMPLETION.findall(log)
    require(len(matches) == 1 and int(matches[0][0]) == 900, '{}: did not complete 900 steps'.format(output))
    steps, cells, seconds = matches[0]
    seconds, cells = float(seconds), float(cells)
    require(math.isfinite(seconds) and seconds > 0, '{}: invalid solver timing'.format(output))
    result.update(steps=int(steps), cells=cells, solver_seconds=seconds,
                  solver_mcells_per_second=int(steps) * cells / seconds / 1e6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-bin', required=True, type=Path)
    parser.add_argument('--cuda-bin', required=True, type=Path)
    parser.add_argument('--reference-bin', type=Path, help='Defaults to --cuda-bin')
    parser.add_argument('--reference-engine', default='cuda-reference', choices=('cuda-reference', 'cuda'))
    parser.add_argument('--cases', type=Path, default=Path(__file__).resolve().parents[1] / 'testdata/cuda-device-extensions')
    parser.add_argument('--output', required=True, type=Path, help='New directory; existing outputs are never overwritten')
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    require(math.isfinite(args.tolerance) and args.tolerance > 0, 'Tolerance must be finite and positive')
    require(math.isfinite(args.timeout) and args.timeout > 0, 'Timeout must be finite and positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cpu, cuda = args.cpu_bin.resolve(), args.cuda_bin.resolve()
    reference = (args.reference_bin or args.cuda_bin).resolve()
    results = dict(status='running', evidence='UNRESOLVED/non-SI', tolerance=args.tolerance, cases={})
    summary = args.output / 'summary.json'
    try:
        for name in SUPPORTED + tuple(UNSUPPORTED):
            xml = (args.cases / (name + '.xml')).resolve()
            case = results['cases'][name] = {}
            base = args.output / name
            case['cpu'] = run(cpu, 'basic', xml, base / 'cpu', args.timeout)
            case['reference'] = run(reference, args.reference_engine, xml, base / 'reference', args.timeout, REFERENCE_MARKER)
            case['reference_vs_cpu'] = compare(base / 'cpu', base / 'reference', args.tolerance)
            case['device'] = run(cuda, 'cuda', xml, base / 'device', args.timeout, DEVICE_MARKER, UNSUPPORTED.get(name))
            if name in SUPPORTED:
                case['device_vs_cpu'] = compare(base / 'cpu', base / 'device', args.tolerance)
                case['device_vs_reference'] = compare(base / 'reference', base / 'device', args.tolerance)
            print('{}: passed'.format(name), flush=True)
            summary.write_text(json.dumps(results, indent=2) + '\n')
        results['status'] = 'passed'
    except Exception as error:
        results['status'] = 'failed'
        results['error'] = str(error)
        raise
    finally:
        summary.write_text(json.dumps(results, indent=2) + '\n')
    print('Results: {}'.format(summary))


if __name__ == '__main__':
    main()
