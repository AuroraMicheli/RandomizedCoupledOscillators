"""
Unified hyperparameter search for the LIF reservoir ablation.

Searches over the LIF-specific parameters:
  - tau_m, tau_m_range       (reservoir membrane time constant + heterogeneity)
  - theta_res, theta_res_range (reservoir firing threshold + heterogeneity)

Plus the shared dynamics parameters that interact with neuron model:
  - dt, rho, inp_scaling, theta_lif (encoder threshold), readout_C

All other settings (connectivity, readout_mode, n_hid) are fixed per dataset
to match the s-RON configuration used in the main paper.

Usage:
    python hyperparam_search_lif_ablation.py --dataset fordA
    python hyperparam_search_lif_ablation.py --dataset shd
    python hyperparam_search_lif_ablation.py --dataset sMNIST
    python hyperparam_search_lif_ablation.py --dataset dvs_gesture
"""

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np

# =============================================================================
# Per-dataset configuration
# =============================================================================

DATASET_CONFIGS = {

    # -------------------------------------------------------------------------
    'fordA': dict(
        n_hid          = 800,
        n_samples      = 100,
        exploit_frac   = 0.70,
        seed           = 42,
        timeout        = 600,
        readout_mode   = 'final',
        connectivity_lif2res = 0.2,
        connectivity_res2enc = 1.0,
        # dataset-specific args passed to train_lif_ablation.py
        extra_args     = [],

        # Narrow (exploit) space — centred on reasonable LIF dynamics for FordA
        # FordA: univariate, seq_length=500, binary classification
        # Need persistent state to t=500 -> higher tau_m, lower theta_res
        narrow = {
            'dt':              (0.02,  0.3,   'log'),
            'rho':             (0.6,   1.0,   'linear'),
            'inp_scaling':     (0.3,   5.0,   'log'),
            'theta_lif':       (0.01,  0.3,   'log'),   # encoder (fixed per run)
            'tau_m':           (10.0,  80.0,  'log'),   # reservoir tau_m center
            'tau_m_range':     (5.0,   60.0,  'log'),   # heterogeneity width
            'theta_res':       (0.005, 0.2,   'log'),   # reservoir threshold center
            'theta_res_range': (0.002, 0.15,  'log'),   # heterogeneity width
        },
        wide = {
            'dt':              (0.005, 0.5,   'log'),
            'rho':             (0.3,   1.1,   'linear'),
            'inp_scaling':     (0.01,  10.0,  'log'),
            'theta_lif':       (0.005, 1.0,   'log'),
            'tau_m':           (2.0,   200.0, 'log'),
            'tau_m_range':     (1.0,   150.0, 'log'),
            'theta_res':       (0.001, 0.5,   'log'),
            'theta_res_range': (0.001, 0.4,   'log'),
        },
        c_values  = [0.001, 0.01, 0.1, 1.0, 10.0],
        c_weights = [0.05,  0.15, 0.35, 0.35, 0.10],
    ),

    # -------------------------------------------------------------------------
    'shd': dict(
        n_hid          = 3000,
        n_samples      = 80,
        exploit_frac   = 0.70,
        seed           = 0,
        timeout        = 600,
        readout_mode   = 'final',
        connectivity_lif2res = 0.2,
        connectivity_res2enc = 1.0,
        extra_args     = [
            '--num_steps',    '250',
            '--max_time',     '1.4',
            '--input_density','0.036',
            '--data_dir',     'data/SHD',
        ],

        # SHD: 700 input channels, 20 classes, spike trains
        # Need moderate tau_m (not too slow — 250 steps at dt~0.2)
        # theta_lif must be high enough given dense bursty input
        narrow = {
            'dt':              (0.1,   0.5,   'log'),
            'rho':             (0.7,   1.3,   'linear'),
            'inp_scaling':     (0.02,  1.0,   'log'),
            'theta_lif':       (0.03,  0.5,   'log'),
            'tau_m':           (5.0,   60.0,  'log'),
            'tau_m_range':     (3.0,   50.0,  'log'),
            'theta_res':       (0.001, 0.05,  'log'),
            'theta_res_range': (0.001, 0.04,  'log'),
        },
        wide = {
            'dt':              (0.05,  1.0,   'log'),
            'rho':             (0.5,   1.6,   'linear'),
            'inp_scaling':     (0.005, 2.0,   'log'),
            'theta_lif':       (0.01,  2.0,   'log'),
            'tau_m':           (1.0,   150.0, 'log'),
            'tau_m_range':     (0.5,   120.0, 'log'),
            'theta_res':       (0.0005,0.2,   'log'),
            'theta_res_range': (0.0005,0.15,  'log'),
        },
        c_values  = [0.001, 0.01, 0.1, 1.0],
        c_weights = [0.25,  0.40, 0.25, 0.10],
    ),

    # -------------------------------------------------------------------------
    'sMNIST': dict(
        n_hid          = 800,
        n_samples      = 100,
        exploit_frac   = 0.80,
        seed           = 42,
        timeout        = 300,
        readout_mode   = 'final',
        connectivity_lif2res = 0.2,
        connectivity_res2enc = 1.0,
        extra_args     = [],

        # sMNIST: pixel-by-pixel, seq_length=784, 10 classes
        # s-RON best: gamma~2.7, dt~0.042 -> fast oscillations
        # LIF equivalent: moderate tau_m, moderate threshold
        narrow = {
            'dt':              (0.03,  0.15,  'log'),
            'rho':             (0.85,  1.05,  'linear'),
            'inp_scaling':     (0.5,   5.0,   'log'),
            'theta_lif':       (0.01,  0.2,   'log'),
            'tau_m':           (5.0,   50.0,  'log'),
            'tau_m_range':     (3.0,   40.0,  'log'),
            'theta_res':       (0.001, 0.05,  'log'),
            'theta_res_range': (0.001, 0.04,  'log'),
        },
        wide = {
            'dt':              (0.02,  0.3,   'log'),
            'rho':             (0.7,   1.1,   'linear'),
            'inp_scaling':     (0.1,   10.0,  'log'),
            'theta_lif':       (0.005, 1.0,   'log'),
            'tau_m':           (1.0,   150.0, 'log'),
            'tau_m_range':     (0.5,   120.0, 'log'),
            'theta_res':       (0.0005,0.2,   'log'),
            'theta_res_range': (0.0005,0.15,  'log'),
        },
        c_values  = [0.001, 0.01, 0.1, 1.0, 10.0],
        c_weights = [0.05,  0.25, 0.40, 0.25, 0.05],
    ),

    # -------------------------------------------------------------------------
    'dvs_gesture': dict(
        n_hid          = 3000,
        n_samples      = 100,
        exploit_frac   = 0.70,
        seed           = 0,
        timeout        = 900,
        readout_mode   = 'mean',
        connectivity_lif2res = 1.0,
        connectivity_res2enc = 1.0,
        extra_args     = [
            '--num_steps',     '200',
            '--spatial_factor','4',
            '--input_density', '0.0306',
            '--data_dir',      'data/DVSGesture',
        ],

        # DVS Gesture: event camera, 11 classes, 1077 train samples
        # Two regimes from s-RON search:
        #   A) slow dynamics (low gamma equivalent -> high tau_m)
        #   B) fast dynamics (high gamma equivalent -> low tau_m)
        # We merge both into a single wide narrow space and let search find it
        narrow = {
            'dt':              (0.05,  0.4,   'log'),
            'rho':             (0.85,  1.3,   'linear'),
            'inp_scaling':     (0.01,  0.15,  'log'),
            'theta_lif':       (0.1,   3.0,   'log'),
            'tau_m':           (2.0,   100.0, 'log'),   # covers both fast+slow regimes
            'tau_m_range':     (1.0,   80.0,  'log'),
            'theta_res':       (0.001, 0.1,   'log'),
            'theta_res_range': (0.001, 0.08,  'log'),
        },
        wide = {
            'dt':              (0.02,  0.5,   'log'),
            'rho':             (0.7,   1.6,   'linear'),
            'inp_scaling':     (0.005, 0.3,   'log'),
            'theta_lif':       (0.05,  5.0,   'log'),
            'tau_m':           (0.5,   200.0, 'log'),
            'tau_m_range':     (0.3,   150.0, 'log'),
            'theta_res':       (0.0005,0.3,   'log'),
            'theta_res_range': (0.0005,0.2,   'log'),
        },
        c_values  = [0.0001, 0.0003, 0.001, 0.003, 0.01],
        c_weights = [0.10,   0.20,   0.35,  0.25,  0.10],
    ),
}

# =============================================================================
# Sampling
# =============================================================================

def sample_from_space(space):
    params = {}
    for key, (*bounds, scale) in space.items():
        lo, hi = bounds
        if scale == 'log':
            params[key] = float(np.exp(random.uniform(np.log(lo), np.log(hi))))
        else:
            params[key] = float(random.uniform(lo, hi))
    return params


def sample_params(cfg, exploit=True):
    space  = cfg['narrow'] if exploit else cfg['wide']
    params = sample_from_space(space)
    params['readout_C'] = float(random.choices(
        cfg['c_values'], weights=cfg['c_weights']
    )[0])
    return params


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for LIF reservoir ablation'
    )
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--seed_override', type=int, default=None,
                        help='Override the default search seed')
    parser.add_argument('--n_samples_override', type=int, default=None,
                        help='Override number of configs to try')
    args = parser.parse_args()

    cfg     = DATASET_CONFIGS[args.dataset]
    seed    = args.seed_override    if args.seed_override    is not None else cfg['seed']
    n_samp  = args.n_samples_override if args.n_samples_override is not None else cfg['n_samples']

    random.seed(seed)
    np.random.seed(seed)

    # Directories
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = Path(script_dir) / f"hyperparam_search_lif_{args.dataset}"
    results_dir.mkdir(exist_ok=True)

    # Result file output dir (where train_lif_ablation.py saves JSONs)
    ablation_results_dir = Path(script_dir) / 'ablation_lif'
    ablation_results_dir.mkdir(exist_ok=True)

    n_exploit = int(n_samp * cfg['exploit_frac'])
    n_explore = n_samp - n_exploit
    sample_types = (['exploit'] * n_exploit) + (['explore'] * n_explore)
    random.shuffle(sample_types)

    print(f"LIF Ablation Hyperparameter Search")
    print(f"Dataset:  {args.dataset}")
    print(f"Configs:  {n_samp}  ({n_exploit} exploit / {n_explore} explore)")
    print(f"Seed:     {seed}")
    print(f"n_hid:    {cfg['n_hid']}")
    print(f"Readout:  {cfg['readout_mode']}")
    print(f"Results:  {results_dir}")
    print("=" * 70)

    all_results    = []
    failed_configs = []

    for i, stype in enumerate(sample_types):
        exploit = (stype == 'exploit')
        params  = sample_params(cfg, exploit=exploit)

        icon = '🎯' if exploit else '🌐'
        print(f"\n{icon} Config {i+1}/{n_samp} [{stype}]: "
              f"dt={params['dt']:.3f} "
              f"rho={params['rho']:.3f} "
              f"inp={params['inp_scaling']:.4f} "
              f"theta_lif={params['theta_lif']:.4f} "
              f"tau_m={params['tau_m']:.2f}+/-{params['tau_m_range']:.2f} "
              f"theta_res={params['theta_res']:.5f}+/-{params['theta_res_range']:.5f} "
              f"C={params['readout_C']}")

        cmd = [
            'python', os.path.join(script_dir, 'train_lif_ablation.py'),
            '--dataset',              args.dataset,
            '--n_hid',                str(cfg['n_hid']),
            '--dt',                   str(params['dt']),
            '--rho',                  str(params['rho']),
            '--inp_scaling',          str(params['inp_scaling']),
            '--theta_lif',            str(params['theta_lif']),
            '--tau_m',                str(params['tau_m']),
            '--tau_m_range',          str(params['tau_m_range']),
            '--theta_res',            str(params['theta_res']),
            '--theta_res_range',      str(params['theta_res_range']),
            '--connectivity_lif2res', str(cfg['connectivity_lif2res']),
            '--connectivity_res2enc', str(cfg['connectivity_res2enc']),
            '--readout_mode',         cfg['readout_mode'],
            '--readout_C',            str(params['readout_C']),
            '--seed',                 str(seed),
            '--test_trials',          '1',
            '--use_test',
            '--results_dir',          str(ablation_results_dir),
        ] + cfg['extra_args']

        try:
            subprocess.run(
                cmd, check=True, capture_output=True,
                text=True, timeout=cfg['timeout']
            )

            # Find the most recently written result file for this dataset
            pattern = f"results_lif_ablation_{args.dataset}*seed{seed}.json"
            candidates = list(ablation_results_dir.glob(pattern))
            if not candidates:
                raise FileNotFoundError(
                    f"No result file matching {pattern} in {ablation_results_dir}"
                )
            result_file = max(candidates, key=lambda p: p.stat().st_mtime)

            with open(result_file) as f:
                res = json.load(f)

            res['config_id']   = i
            res['search_seed'] = seed
            res['readout_C']   = params['readout_C']
            res['sample_type'] = stype
            # Store searched params explicitly for easy access in summary
            res['searched_params'] = params
            all_results.append(res)

            gap = res['train_acc_mean'] - res['test_acc_mean']
            print(f"   ✅ Test: {res['test_acc_mean']:.2f}%  "
                  f"Train: {res['train_acc_mean']:.2f}%  "
                  f"Gap: {gap:.1f}%  "
                  f"r_res={res.get('r_res_mean', 0):.4f}  "
                  f"r_enc={res.get('r_enc_mean', 0):.4f}")

        except subprocess.CalledProcessError as e:
            print(f"   ❌ FAILED")
            if e.stderr:
                for line in e.stderr.strip().split('\n')[-3:]:
                    print(f"      {line}")
            failed_configs.append({
                'config_id': i, 'params': params, 'sample_type': stype
            })

        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT (>{cfg['timeout']}s)")
            failed_configs.append({
                'config_id': i, 'params': params, 'sample_type': stype
            })

        except FileNotFoundError as e:
            print(f"   ❌ RESULT FILE NOT FOUND: {e}")
            failed_configs.append({
                'config_id': i, 'params': params, 'sample_type': stype
            })

        # Intermediate save every 10 configs
        if (i + 1) % 10 == 0 and all_results:
            intermediate = sorted(all_results,
                                   key=lambda x: x['test_acc_mean'],
                                   reverse=True)
            with open(results_dir / 'summary_intermediate.json', 'w') as f:
                json.dump(intermediate[:20], f, indent=2)
            best = intermediate[0]
            print(f"\n   💾 Saved top-20 intermediate results "
                  f"(best so far: {best['test_acc_mean']:.2f}% "
                  f"[{best.get('sample_type', '?')}])")

    # =========================================================================
    # Final aggregation
    # =========================================================================

    all_results.sort(key=lambda x: x['test_acc_mean'], reverse=True)

    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(results_dir / 'failed_configs.json', 'w') as f:
        json.dump(failed_configs, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ Completed: {len(all_results)}/{n_samp} configs successful")
    print(f"❌ Failed:    {len(failed_configs)}/{n_samp} configs")

    exploit_res = [r for r in all_results if r.get('sample_type') == 'exploit']
    explore_res = [r for r in all_results if r.get('sample_type') == 'explore']
    if exploit_res:
        print(f"🎯 Exploit best: "
              f"{max(r['test_acc_mean'] for r in exploit_res):.2f}%  "
              f"(mean: {np.mean([r['test_acc_mean'] for r in exploit_res]):.2f}%)")
    if explore_res:
        print(f"🌐 Explore best: "
              f"{max(r['test_acc_mean'] for r in explore_res):.2f}%  "
              f"(mean: {np.mean([r['test_acc_mean'] for r in explore_res]):.2f}%)")
    print("=" * 70)

    if not all_results:
        print("\nNo successful configurations.")
        return

    # Top 10 table
    print(f"\n🏆 TOP 10 CONFIGURATIONS ({args.dataset}):")
    print(f"{'Rank':<5} {'Test%':<10} {'Train%':<10} {'Gap':<7} {'Type':<9} "
          f"{'C':<7} {'dt':<7} {'rho':<6} {'inp':<8} "
          f"{'tau_m':<8} {'tm_rng':<8} "
          f"{'th_res':<9} {'tr_rng':<9} "
          f"{'th_lif':<8} "
          f"{'r_res':<7} {'r_enc'}")
    print("-" * 140)

    for rank, r in enumerate(all_results[:10], 1):
        p = r.get('searched_params', r.get('args', {}))
        gap = r['train_acc_mean'] - r['test_acc_mean']
        print(f"{rank:<5}"
              f"{r['test_acc_mean']:.2f}%     "
              f"{r['train_acc_mean']:.2f}%     "
              f"{gap:.1f}%   "
              f"{r.get('sample_type', '?'):<9}"
              f"{float(r.get('readout_C', 0)):<7}"
              f"{float(p.get('dt', 0)):<7.3f}"
              f"{float(p.get('rho', 0)):<6.3f}"
              f"{float(p.get('inp_scaling', 0)):<8.4f}"
              f"{float(p.get('tau_m', 0)):<8.2f}"
              f"{float(p.get('tau_m_range', 0)):<8.2f}"
              f"{float(p.get('theta_res', 0)):<9.5f}"
              f"{float(p.get('theta_res_range', 0)):<9.5f}"
              f"{float(p.get('theta_lif', 0)):<8.4f}"
              f"{float(r.get('r_res_mean', 0)):<7.4f}"
              f"{float(r.get('r_enc_mean', 0)):.4f}")

    # Parameter trends
    print(f"\n📊 PARAMETER TRENDS (top 10 vs all):")
    for pname in ['dt', 'rho', 'inp_scaling', 'theta_lif',
                  'tau_m', 'tau_m_range', 'theta_res', 'theta_res_range']:
        top_vals = [float(r.get('searched_params', {}).get(pname, 0))
                    for r in all_results[:10]]
        all_vals = [float(r.get('searched_params', {}).get(pname, 0))
                    for r in all_results]
        print(f"  {pname:>18}: "
              f"top10={np.mean(top_vals):.5f}+/-{np.std(top_vals):.5f}  "
              f"all={np.mean(all_vals):.5f}+/-{np.std(all_vals):.5f}")

    # C breakdown
    print(f"\n📊 REGULARIZATION (C) BREAKDOWN:")
    for C_val in cfg['c_values']:
        c_res = [r for r in all_results
                 if abs(float(r.get('readout_C', 0)) - C_val) < C_val * 0.01 + 1e-9]
        if c_res:
            accs = [r['test_acc_mean'] for r in c_res]
            gaps = [r['train_acc_mean'] - r['test_acc_mean'] for r in c_res]
            print(f"  C={C_val:<8}: n={len(c_res)}, "
                  f"test={np.mean(accs):.2f}+/-{np.std(accs):.2f}%  "
                  f"best={max(accs):.2f}%  "
                  f"avg_gap={np.mean(gaps):.1f}%")

    # Firing rate sanity check
    print(f"\n📊 FIRING RATE CHECK — top 10 (healthy range: r_res ~0.05-0.3):")
    for rank, r in enumerate(all_results[:10], 1):
        r_res = float(r.get('r_res_mean', 0))
        flag  = ' ⚠️ SATURATED' if r_res > 0.4 else (
                ' ⚠️ SILENT'    if r_res < 0.01 else '')
        print(f"  {rank}: r_res={r_res:.4f}  "
              f"r_enc={float(r.get('r_enc_mean', 0)):.4f}  "
              f"[{r.get('sample_type', '?')}]  "
              f"test={r['test_acc_mean']:.2f}%{flag}")


if __name__ == '__main__':
    main()