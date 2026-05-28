"""
Hyperparameter search for the HRF-HRF ablation (Option C).

Searched parameters (7+1 per dataset):
  dt, rho, inp_scaling,
  gamma_enc_min/max, epsilon_enc_min/max,
  theta_enc, readout_C, readout_mode

Fixed (from FIXED_RESERVOIR_CONFIGS in train_hrf_hrf_ablation.py):
  gamma, epsilon, theta_rf, connectivity, input_density, n_hid, num_steps, ...

IMPORTANT: --data_dir for SHD and DVS is passed via extra_args inside
DATASET_SEARCH_CONFIGS, NOT as a CLI argument to this script.
Do NOT pass --data_dir when calling this script from sbatch.

Usage:
    python hyperparam_search_hrf_hrf.py --dataset fordA
    python hyperparam_search_hrf_hrf.py --dataset shd
    python hyperparam_search_hrf_hrf.py --dataset sMNIST
    python hyperparam_search_hrf_hrf.py --dataset dvs_gesture

After the search finishes:
  1. Read the top-10 table and firing-rate health check.
  2. Copy the paste-ready snippet into BEST_ENCODER_CONFIGS in
     train_hrf_hrf_ablation.py.
  3. Run train_hrf_hrf_ablation.py --use_best_config --test_trials 3
"""

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np


DATASET_SEARCH_CONFIGS = {

    'sMNIST': dict(
        n_samples    = 80,
        exploit_frac = 0.65,
        seed         = 42,
        # sMNIST: seq=784, n_hid=800. rms_std_final configs take ~7-8 min each.
        # 80 configs x ~5 min avg = ~6-7h. timeout=900 per config (15 min).
        timeout      = 900,
        readout_modes        = ['rms_std_final', 'final', 'mean'],
        readout_mode_weights = [0.5, 0.25, 0.25],
        extra_args   = ['--use_test'],
        c_values  = [0.001, 0.01, 0.1, 1.0, 10.0],
        c_weights = [0.05,  0.25, 0.40, 0.25, 0.05],
        wide = {
            'dt':              (0.01,  0.3,   'log'),
            'rho':             (0.7,   1.3,   'linear'),
            'inp_scaling':     (0.1,   10.0,  'log'),
            'gamma_enc_min':   (0.05,  10.0,  'log'),
            'gamma_enc_max':   (0.1,   20.0,  'log'),
            'epsilon_enc_min': (0.01,  5.0,   'log'),
            'epsilon_enc_max': (0.05,  10.0,  'log'),
            'theta_enc':       (0.01,  5.0,   'log'),
        },
        narrow = {
            'dt':              (0.02,  0.1,   'log'),
            'rho':             (0.85,  1.05,  'linear'),
            'inp_scaling':     (0.5,   5.0,   'log'),
            'gamma_enc_min':   (0.1,   3.0,   'log'),
            'gamma_enc_max':   (0.5,   8.0,   'log'),
            'epsilon_enc_min': (0.02,  1.0,   'log'),
            'epsilon_enc_max': (0.1,   3.0,   'log'),
            'theta_enc':       (0.02,  1.0,   'log'),
        },
    ),

    'fordA': dict(
        n_samples    = 80,
        exploit_frac = 0.65,
        seed         = 42,
        # fordA: seq=500, n_hid=800. ~2-3 min per config. timeout=600.
        timeout      = 600,
        readout_modes        = ['rms_std_final', 'final', 'mean'],
        readout_mode_weights = [0.5, 0.25, 0.25],
        extra_args   = ['--use_test'],
        c_values  = [0.001, 0.01, 0.1, 1.0, 10.0],
        c_weights = [0.05,  0.15, 0.35, 0.35, 0.10],
        wide = {
            'dt':              (0.01,  0.5,   'log'),
            'rho':             (0.5,   1.2,   'linear'),
            'inp_scaling':     (0.05,  10.0,  'log'),
            'gamma_enc_min':   (0.5,   15.0,  'log'),
            'gamma_enc_max':   (1.0,   30.0,  'log'),
            'epsilon_enc_min': (0.01,  5.0,   'log'),
            'epsilon_enc_max': (0.05,  10.0,  'log'),
            'theta_enc':       (0.001, 5.0,   'log'),
        },
        narrow = {
            'dt':              (0.02,  0.15,  'log'),
            'rho':             (0.6,   1.0,   'linear'),
            'inp_scaling':     (0.1,   3.0,   'log'),
            'gamma_enc_min':   (1.0,   8.0,   'log'),
            'gamma_enc_max':   (3.0,   20.0,  'log'),
            'epsilon_enc_min': (0.05,  1.0,   'log'),
            'epsilon_enc_max': (0.2,   3.0,   'log'),
            'theta_enc':       (0.005, 0.5,   'log'),
        },
    ),

    'shd': dict(
        n_samples    = 80,
        exploit_frac = 0.65,
        seed         = 0,
        # shd: seq=250, n_hid=3000, 700-dim input. ~15 min per config. timeout=1200.
        timeout      = 1200,
        readout_modes        = ['rms_std_final', 'final'],
        readout_mode_weights = [0.6, 0.4],
        # data_dir passed here, NOT as CLI arg to this script
        extra_args   = ['--use_test', '--data_dir', 'data/SHD'],
        c_values  = [0.001, 0.01, 0.1, 1.0],
        c_weights = [0.25,  0.40, 0.25, 0.10],
        wide = {
            'dt':              (0.05,  1.0,   'log'),
            'rho':             (0.7,   1.8,   'linear'),
            'inp_scaling':     (0.01,  2.0,   'log'),
            'gamma_enc_min':   (0.001, 1.0,   'log'),
            'gamma_enc_max':   (0.005, 5.0,   'log'),
            'epsilon_enc_min': (0.001, 1.0,   'log'),
            'epsilon_enc_max': (0.005, 5.0,   'log'),
            'theta_enc':       (0.1,   10.0,  'log'),
        },
        narrow = {
            'dt':              (0.1,   0.5,   'log'),
            'rho':             (0.9,   1.4,   'linear'),
            'inp_scaling':     (0.05,  0.8,   'log'),
            'gamma_enc_min':   (0.005, 0.3,   'log'),
            'gamma_enc_max':   (0.02,  1.0,   'log'),
            'epsilon_enc_min': (0.01,  0.3,   'log'),
            'epsilon_enc_max': (0.05,  1.0,   'log'),
            'theta_enc':       (0.3,   5.0,   'log'),
        },
    ),

    'dvs_gesture': dict(
        n_samples    = 80,
        exploit_frac = 0.65,
        seed         = 0,
        # dvs: seq=200, n_hid=3000, 2048-dim input. ~20 min per config. timeout=1500.
        timeout      = 1500,
        readout_modes        = ['rms_std_final', 'mean', 'final'],
        readout_mode_weights = [0.5, 0.3, 0.2],
        # data_dir passed here, NOT as CLI arg to this script
        extra_args   = ['--use_test', '--data_dir', 'data/DVSGesture'],
        c_values  = [0.0001, 0.0003, 0.001, 0.003, 0.01],
        c_weights = [0.10,   0.20,   0.35,  0.25,  0.10],
        wide = {
            'dt':              (0.05,  0.8,   'log'),
            'rho':             (0.8,   2.0,   'linear'),
            'inp_scaling':     (0.005, 1.0,   'log'),
            'gamma_enc_min':   (0.001, 1.0,   'log'),
            'gamma_enc_max':   (0.005, 5.0,   'log'),
            'epsilon_enc_min': (0.001, 1.0,   'log'),
            'epsilon_enc_max': (0.005, 5.0,   'log'),
            'theta_enc':       (0.5,   15.0,  'log'),
        },
        narrow = {
            'dt':              (0.1,   0.6,   'log'),
            'rho':             (1.0,   1.8,   'linear'),
            'inp_scaling':     (0.02,  0.5,   'log'),
            'gamma_enc_min':   (0.001, 0.2,   'log'),
            'gamma_enc_max':   (0.01,  1.0,   'log'),
            'epsilon_enc_min': (0.001, 0.2,   'log'),
            'epsilon_enc_max': (0.01,  1.0,   'log'),
            'theta_enc':       (1.0,   8.0,   'log'),
        },
    ),
}


def _sample_space(space):
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
    params = _sample_space(space)
    for prefix in ('gamma_enc', 'epsilon_enc'):
        lo_key, hi_key = f'{prefix}_min', f'{prefix}_max'
        if params[hi_key] <= params[lo_key] * 1.5:
            params[hi_key] = params[lo_key] * random.uniform(1.5, 3.0)
    params['readout_C'] = float(
        random.choices(cfg['c_values'], weights=cfg['c_weights'])[0]
    )
    params['readout_mode'] = random.choices(
        cfg['readout_modes'], weights=cfg['readout_mode_weights']
    )[0]
    return params


def main():
    parser = argparse.ArgumentParser(
        description='HRF-HRF hyperparam search (Option C). '
                    'Do NOT pass --data_dir here; it is set internally.'
    )
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_SEARCH_CONFIGS.keys()))
    parser.add_argument('--seed_override',      type=int, default=None)
    parser.add_argument('--n_samples_override', type=int, default=None)
    args = parser.parse_args()

    cfg    = DATASET_SEARCH_CONFIGS[args.dataset]
    seed   = args.seed_override      if args.seed_override      is not None else cfg['seed']
    n_samp = args.n_samples_override if args.n_samples_override is not None else cfg['n_samples']

    random.seed(seed); np.random.seed(seed)

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_dir  = Path(script_dir) / f"hyperparam_search_hrf_hrf_{args.dataset}"
    ablation_dir = Path(script_dir) / 'ablation_hrf_hrf'
    results_dir.mkdir(exist_ok=True)
    ablation_dir.mkdir(exist_ok=True)

    n_exploit    = int(n_samp * cfg['exploit_frac'])
    n_explore    = n_samp - n_exploit
    sample_types = (['exploit'] * n_exploit) + (['explore'] * n_explore)
    random.shuffle(sample_types)

    print(f"HRF-HRF Hyperparam Search (Option C)")
    print(f"Dataset: {args.dataset}   Configs: {n_samp}   "
          f"({n_exploit} exploit / {n_explore} explore)   Seed: {seed}")
    print(f"Per-config timeout: {cfg['timeout']}s")
    print(f"Free: dt, rho, inp_scaling, gamma_enc, epsilon_enc, theta_enc, readout_C")
    print(f"Fixed: gamma_res, epsilon_res, theta_rf")
    print("=" * 70)

    all_results    = []
    failed_configs = []

    for i, stype in enumerate(sample_types):
        exploit = (stype == 'exploit')
        params  = sample_params(cfg, exploit=exploit)

        icon = '🎯' if exploit else '🌐'
        print(f"\n{icon} Config {i+1}/{n_samp} [{stype}]  "
              f"dt={params['dt']:.4f}  rho={params['rho']:.3f}  "
              f"inp={params['inp_scaling']:.4f}  "
              f"gEnc=({params['gamma_enc_min']:.3f},{params['gamma_enc_max']:.3f})  "
              f"eEnc=({params['epsilon_enc_min']:.4f},{params['epsilon_enc_max']:.4f})  "
              f"theta_enc={params['theta_enc']:.4f}  "
              f"C={params['readout_C']}  mode={params['readout_mode']}")

        cmd = [
            'python', os.path.join(script_dir, 'train_hrf_hrf_ablation.py'),
            '--dataset',         args.dataset,
            '--dt',              str(params['dt']),
            '--rho',             str(params['rho']),
            '--inp_scaling',     str(params['inp_scaling']),
            '--gamma_enc_min',   str(params['gamma_enc_min']),
            '--gamma_enc_max',   str(params['gamma_enc_max']),
            '--epsilon_enc_min', str(params['epsilon_enc_min']),
            '--epsilon_enc_max', str(params['epsilon_enc_max']),
            '--theta_enc',       str(params['theta_enc']),
            '--readout_C',       str(params['readout_C']),
            '--readout_mode',    params['readout_mode'],
            '--seed',            str(seed),
            '--test_trials',     '1',
            '--results_dir',     str(ablation_dir),
        ] + cfg['extra_args']

        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           text=True, timeout=cfg['timeout'])

            pattern    = f"results_hrf_hrf_{args.dataset}*seed{seed}.json"
            candidates = list(ablation_dir.glob(pattern))
            if not candidates:
                raise FileNotFoundError(f"No result file matching {pattern}")
            result_file = max(candidates, key=lambda p: p.stat().st_mtime)
            with open(result_file) as f:
                res = json.load(f)

            res['config_id'] = i; res['search_seed'] = seed
            res['sample_type'] = stype; res['searched_params'] = params
            all_results.append(res)

            gap   = res['train_acc_mean'] - res['test_acc_mean']
            r_res = float(res.get('r_res_mean', 0))
            r_enc = float(res.get('r_enc_mean', 0))
            enc_flag = ' ⚠️ ENC_SAT'    if r_enc > 0.4  else (' ⚠️ ENC_SILENT' if r_enc < 0.005 else '')
            res_flag = ' ⚠️ RES_SAT'    if r_res > 0.4  else (' ⚠️ RES_SILENT' if r_res < 0.005 else '')
            print(f"   ✅ Test: {res['test_acc_mean']:.2f}%  "
                  f"Train: {res['train_acc_mean']:.2f}%  Gap: {gap:.1f}%  "
                  f"r_res={r_res:.4f}{res_flag}  r_enc={r_enc:.4f}{enc_flag}")

        except subprocess.CalledProcessError as e:
            print(f"   ❌ FAILED")
            if e.stderr:
                for line in e.stderr.strip().split('\n')[-3:]:
                    print(f"      {line}")
            failed_configs.append({'config_id': i, 'params': params, 'sample_type': stype})

        except subprocess.TimeoutExpired:
            print(f"   ⏰ TIMEOUT (>{cfg['timeout']}s)")
            failed_configs.append({'config_id': i, 'params': params, 'sample_type': stype})

        except FileNotFoundError as e:
            print(f"   ❌ RESULT NOT FOUND: {e}")
            failed_configs.append({'config_id': i, 'params': params, 'sample_type': stype})

        if (i + 1) % 10 == 0 and all_results:
            top = sorted(all_results, key=lambda x: x['test_acc_mean'], reverse=True)
            with open(results_dir / 'summary_intermediate.json', 'w') as f:
                json.dump(top[:20], f, indent=2)
            print(f"\n   💾 Top-20 saved  (best: {top[0]['test_acc_mean']:.2f}%)")

    all_results.sort(key=lambda x: x['test_acc_mean'], reverse=True)
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    with open(results_dir / 'failed_configs.json', 'w') as f:
        json.dump(failed_configs, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ {len(all_results)}/{n_samp} successful  ❌ {len(failed_configs)}/{n_samp} failed")
    print("=" * 70)

    if not all_results:
        print("No successful configurations."); return

    print(f"\n🏆 TOP 10  ({args.dataset}):")
    print(f"{'Rk':<4} {'Test%':<9} {'Train%':<9} {'Gap':<6} {'Type':<8} "
          f"{'mode':<16} {'C':<7} {'dt':<7} {'rho':<6} {'inp':<8} "
          f"{'gE_lo':<7} {'gE_hi':<7} {'eE_lo':<8} {'eE_hi':<8} "
          f"{'th_enc':<8} {'r_res':<7} {'r_enc'}")
    print("-" * 155)
    for rank, r in enumerate(all_results[:10], 1):
        p = r.get('searched_params', {})
        gap = r['train_acc_mean'] - r['test_acc_mean']
        print(f"{rank:<4}{r['test_acc_mean']:.2f}%    {r['train_acc_mean']:.2f}%    "
              f"{gap:.1f}%  {r.get('sample_type','?'):<8}"
              f"{p.get('readout_mode','?'):<16}{float(r.get('readout_C',0)):<7}"
              f"{float(p.get('dt',0)):<7.4f}{float(p.get('rho',0)):<6.3f}"
              f"{float(p.get('inp_scaling',0)):<8.4f}"
              f"{float(p.get('gamma_enc_min',0)):<7.3f}{float(p.get('gamma_enc_max',0)):<7.3f}"
              f"{float(p.get('epsilon_enc_min',0)):<8.4f}{float(p.get('epsilon_enc_max',0)):<8.4f}"
              f"{float(p.get('theta_enc',0)):<8.4f}"
              f"{float(r.get('r_res_mean',0)):<7.4f}{float(r.get('r_enc_mean',0)):.4f}")

    print(f"\n📊 FIRING RATE HEALTH — top 10  (healthy: 0.01–0.35)")
    for rank, r in enumerate(all_results[:10], 1):
        r_enc = float(r.get('r_enc_mean', 0)); r_res = float(r.get('r_res_mean', 0))
        ef = ' ⚠️ ENC_SAT' if r_enc > 0.4 else (' ⚠️ ENC_SILENT' if r_enc < 0.005 else '')
        rf = ' ⚠️ RES_SAT' if r_res > 0.4 else (' ⚠️ RES_SILENT' if r_res < 0.005 else '')
        print(f"  {rank}: r_enc={r_enc:.4f}{ef}  r_res={r_res:.4f}{rf}  "
              f"test={r['test_acc_mean']:.2f}%")

    best = all_results[0]; bp = best.get('searched_params', {})
    print(f"\n{'='*70}")
    print(f"📋  PASTE INTO BEST_ENCODER_CONFIGS['{args.dataset}'] in train_hrf_hrf_ablation.py:")
    print(f"{'='*70}")
    print(f"    '{args.dataset}': dict(")
    print(f"        # test={best['test_acc_mean']:.2f}%  "
          f"train={best['train_acc_mean']:.2f}%  "
          f"gap={best['train_acc_mean']-best['test_acc_mean']:.1f}%")
    print(f"        dt              = {float(bp.get('dt',           0)):.6f},")
    print(f"        rho             = {float(bp.get('rho',          0)):.6f},")
    print(f"        inp_scaling     = {float(bp.get('inp_scaling',  0)):.6f},")
    print(f"        gamma_enc_min   = {float(bp.get('gamma_enc_min',0)):.6f},")
    print(f"        gamma_enc_max   = {float(bp.get('gamma_enc_max',0)):.6f},")
    print(f"        epsilon_enc_min = {float(bp.get('epsilon_enc_min',0)):.6f},")
    print(f"        epsilon_enc_max = {float(bp.get('epsilon_enc_max',0)):.6f},")
    print(f"        theta_enc       = {float(bp.get('theta_enc',    0)):.6f},")
    print(f"        readout_C       = {float(best.get('readout_C',  0))},")
    print(f"        readout_mode    = '{bp.get('readout_mode','rms_std_final')}',")
    print(f"    ),")
    print("=" * 70)


if __name__ == '__main__':
    main()