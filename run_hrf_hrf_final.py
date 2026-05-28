"""
Final evaluation of HRF-HRF best configs over 3 seeds.

Runs the best encoder config found by hyperparam_search_hrf_hrf.py for each
dataset with test_trials=3 to get mean ± std for the paper.

Results are saved to ablation_hrf_hrf/final_results/ as JSON files.

Usage:
    python run_hrf_hrf_final.py --dataset sMNIST
    python run_hrf_hrf_final.py --dataset fordA
    python run_hrf_hrf_final.py --dataset shd        --data_dir data/SHD
    python run_hrf_hrf_final.py --dataset dvs_gesture --data_dir data/DVSGesture
"""

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

# =============================================================================
# Best configs from hyperparam search — paste from search output
# =============================================================================

BEST_CONFIGS = {
    'sMNIST': dict(
        # search best: test=95.83%  train=99.94%  gap=4.1%  (single seed)
        dt              = 0.150992,
        rho             = 1.134509,
        inp_scaling     = 4.797460,
        gamma_enc_min   = 6.529243,
        gamma_enc_max   = 18.058043,
        epsilon_enc_min = 0.276429,
        epsilon_enc_max = 6.107177,
        theta_enc       = 0.395823,
        readout_C       = 0.1,
        readout_mode    = 'rms_std_final',
    ),
    'fordA': dict(
        # search best: test=90.53%  train=96.37%  gap=5.8%  (single seed)
        dt              = 0.107118,
        rho             = 0.755791,
        inp_scaling     = 1.371140,
        gamma_enc_min   = 7.151204,
        gamma_enc_max   = 20.839412,
        epsilon_enc_min = 0.699408,
        epsilon_enc_max = 1.362391,
        theta_enc       = 0.044971,
        readout_C       = 0.1,
        readout_mode    = 'mean',
    ),
    'shd': dict(
        # search best: test=87.81%  train=100.00%  gap=12.2%  (single seed)
        # NOTE: r_res=0.457 slightly above healthy range — monitor variance
        dt              = 0.145350,
        rho             = 1.280998,
        inp_scaling     = 0.108700,
        gamma_enc_min   = 0.280994,
        gamma_enc_max   = 0.643239,
        epsilon_enc_min = 0.202003,
        epsilon_enc_max = 0.479238,
        theta_enc       = 0.617473,
        readout_C       = 0.01,
        readout_mode    = 'rms_std_final',
    ),
    'dvs_gesture': dict(
        # search best: test=73.86%  train=99.63%  gap=25.8%  (single seed)
        # WARNING: r_enc=0.691 (ENC_SAT), r_res=0.410 (RES_SAT) in search run.
        # High variance across seeds expected. If std is large, consider
        # using rank-2 config (test=72.35%, r_enc=0.152, r_res=0.438).
        dt              = 0.050328,
        rho             = 1.787278,
        inp_scaling     = 0.082169,
        gamma_enc_min   = 0.001965,
        gamma_enc_max   = 0.011368,
        epsilon_enc_min = 0.088674,
        epsilon_enc_max = 2.088967,
        theta_enc       = 1.295801,
        readout_C       = 0.01,
        readout_mode    = 'rms_std_final',
    ),
}

# Rank-2 DVS config as fallback if rank-1 is too unstable
DVS_RANK2 = dict(
    # test=72.35%  train=99.81%  r_enc=0.152 (healthy)  r_res=0.438
    dt              = 0.086600,
    rho             = 1.254000,
    inp_scaling     = 0.090400,
    gamma_enc_min   = 0.003000,
    gamma_enc_max   = 4.624000,
    epsilon_enc_min = 0.889100,
    epsilon_enc_max = 2.240500,
    theta_enc       = 1.988600,
    readout_C       = 0.003,
    readout_mode    = 'rms_std_final',
)

DATASET_EXTRA_ARGS = {
    'sMNIST':      [],
    'fordA':       [],
    'shd':         ['--data_dir', 'data/SHD'],
    'dvs_gesture': ['--data_dir', 'data/DVSGesture'],
}


def build_parser():
    p = argparse.ArgumentParser(
        description='Run HRF-HRF best config for 3 seeds (final paper results)'
    )
    p.add_argument('--dataset', required=True,
                   choices=['sMNIST', 'fordA', 'shd', 'dvs_gesture'])
    p.add_argument('--data_dir',    type=str, default='data')
    p.add_argument('--test_trials', type=int, default=3)
    p.add_argument('--results_dir', type=str, default=None)
    p.add_argument('--use_rank2_dvs', action='store_true',
                   help='Use rank-2 DVS config (healthier firing rates) '
                        'instead of rank-1')
    return p


def main():
    args = build_parser().parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = (Path(args.results_dir) if args.results_dir
                   else Path(script_dir) / 'ablation_hrf_hrf' / 'final_results')
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = BEST_CONFIGS[args.dataset]
    if args.dataset == 'dvs_gesture' and args.use_rank2_dvs:
        cfg = DVS_RANK2
        print("Using rank-2 DVS config (healthier firing rates)")

    extra_args = DATASET_EXTRA_ARGS[args.dataset]
    if args.data_dir != 'data' and '--data_dir' not in extra_args:
        extra_args = extra_args + ['--data_dir', args.data_dir]

    print('=' * 70)
    print(f'HRF-HRF FINAL EVALUATION  —  {args.dataset}')
    print(f'test_trials: {args.test_trials}')
    print(f'readout_mode: {cfg["readout_mode"]}   C: {cfg["readout_C"]}')
    print(f'Results dir: {results_dir}')
    print('=' * 70)

    cmd = [
        'python', os.path.join(script_dir, 'train_hrf_hrf_ablation.py'),
        '--dataset',         args.dataset,
        '--dt',              str(cfg['dt']),
        '--rho',             str(cfg['rho']),
        '--inp_scaling',     str(cfg['inp_scaling']),
        '--gamma_enc_min',   str(cfg['gamma_enc_min']),
        '--gamma_enc_max',   str(cfg['gamma_enc_max']),
        '--epsilon_enc_min', str(cfg['epsilon_enc_min']),
        '--epsilon_enc_max', str(cfg['epsilon_enc_max']),
        '--theta_enc',       str(cfg['theta_enc']),
        '--readout_C',       str(cfg['readout_C']),
        '--readout_mode',    cfg['readout_mode'],
        '--test_trials',     str(args.test_trials),
        '--use_test',
        '--results_dir',     str(results_dir),
    ] + extra_args

    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"\nFAILED with exit code {result.returncode}")
        return

    # Find and display the result JSON
    pattern = f"results_hrf_hrf_{args.dataset}*trials{args.test_trials}*.json"
    candidates = list(results_dir.glob(pattern))
    if candidates:
        result_file = max(candidates, key=lambda p: p.stat().st_mtime)
        with open(result_file) as f:
            res = json.load(f)
        print(f"\n{'='*70}")
        print(f"FINAL RESULT  —  {args.dataset}")
        print(f"{'='*70}")
        print(f"Test:      {res['test_acc_mean']:.2f}% +/- {res['test_acc_std']:.2f}%")
        print(f"Train:     {res['train_acc_mean']:.2f}% +/- {res['train_acc_std']:.2f}%")
        print(f"Per-trial: {[f'{a:.2f}' for a in res['test_accs_all']]}")
        print(f"r_res:     {res['r_res_mean']:.4f}")
        print(f"r_enc:     {res['r_enc_mean']:.4f}")
        if res['r_enc_mean'] > 0.4:
            print(f"  WARNING: r_enc={res['r_enc_mean']:.4f} — encoder saturated")
        if res['r_res_mean'] > 0.4:
            print(f"  WARNING: r_res={res['r_res_mean']:.4f} — reservoir saturated")
        print(f"Energy:    {res['energy_J_mean']:.3e} J")
        print(f"Saved:     {result_file}")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()