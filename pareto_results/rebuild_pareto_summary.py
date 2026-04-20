"""
Rebuild pareto_summary.json from the individual per-(dataset, model, n_hid) JSON files.

Usage:
    python rebuild_pareto_summary.py
    python rebuild_pareto_summary.py --results_dir /path/to/pareto_results
"""

import argparse
import json
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='pareto_results')
    args = parser.parse_args()

    results_dir = args.results_dir
    summary = {}

    # Pattern: pareto_<Dataset>_<model>_nhid<N>.json
    pattern = re.compile(r'^pareto_(.+)_(ron|sron)_nhid(\d+)\.json$')

    files = sorted(os.listdir(results_dir))
    loaded = 0

    for fname in files:
        m = pattern.match(fname)
        if not m:
            continue

        dataset  = m.group(1)   # e.g. 'sMNIST', 'FordA', 'Adiac'
        model    = m.group(2)   # 'ron' or 'sron'
        n_hid    = m.group(3)   # e.g. '800'

        path = os.path.join(results_dir, fname)
        with open(path) as f:
            result = json.load(f)

        if dataset not in summary:
            summary[dataset] = {}
        if model not in summary[dataset]:
            summary[dataset][model] = {}

        summary[dataset][model][n_hid] = result
        loaded += 1
        print(f"  Loaded: {fname}")

    print(f"\nLoaded {loaded} files.")
    print("Summary structure:")
    for ds, models in summary.items():
        for model, nhids in models.items():
            n_hid_list = sorted(nhids.keys(), key=int)
            print(f"  {ds}/{model}: n_hid = {n_hid_list}")

    out_path = os.path.join(results_dir, 'pareto_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved rebuilt summary to: {out_path}")


if __name__ == '__main__':
    main()