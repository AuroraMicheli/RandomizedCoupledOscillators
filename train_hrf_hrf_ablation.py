"""
Training script for the HRF-HRF ablation of s-RON.

Both encoder and reservoir use HRF (Resonate-and-Fire) neurons.

Parameter fixation strategy (Option C)
---------------------------------------
FIXED from s-RON best configs (same computational role, transferable):
  - gamma, epsilon, theta_rf    reservoir HRF intrinsic oscillator params
  - connectivity_enc2res        structural, matches s-RON lif2hrf
  - connectivity_res2res        structural
  - input_density, num_steps, spatial_factor, n_hid

RE-TUNED by independent random search (interact with new encoder type):
  - dt, rho, inp_scaling
  - gamma_enc, epsilon_enc, theta_enc
  - readout_C, readout_mode

Supports: fordA | shd | sMNIST | dvs_gesture

Usage:
    python train_hrf_hrf_ablation.py --dataset fordA --use_test
    python train_hrf_hrf_ablation.py --dataset fordA --use_best_config --use_test --test_trials 3
    python train_hrf_hrf_ablation.py --dataset shd   --use_best_config --use_test --test_trials 3 --data_dir data/SHD
    python train_hrf_hrf_ablation.py --dataset sMNIST      --use_best_config --use_test --test_trials 3
    python train_hrf_hrf_ablation.py --dataset dvs_gesture --use_best_config --use_test --test_trials 3 --data_dir data/DVSGesture
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from esn import spectral_norm_scaling
from utils_aurora import estimate_snn_energy_sparse
from utils_aurora import spiking_HRF_HRF

from utils          import get_FordA_data, get_mnist_data
from ucr_data_utils import get_SHD_data

try:
    import tonic
    import tonic.transforms as tonic_transforms
    from tonic import DiskCachedDataset
    from torch.utils.data import DataLoader
    TONIC_AVAILABLE = True
except ImportError:
    TONIC_AVAILABLE = False


FIXED_RESERVOIR_CONFIGS = {
    'sMNIST': dict(
        n_hid=800, gamma=(1.7, 3.7), epsilon=(0.001, 0.580), theta_rf=0.005,
        connectivity_enc2res=1.0, connectivity_res2res=1.0,
        input_density=1.0, num_steps=784, max_time=1.4, spatial_factor=4,
    ),
    'fordA': dict(
        n_hid=800, gamma=(5.507, 8.517), epsilon=(0.001, 0.363), theta_rf=0.0010,
        connectivity_enc2res=1.0, connectivity_res2res=1.0,
        input_density=1.0, num_steps=500, max_time=1.4, spatial_factor=4,
    ),
    'shd': dict(
        n_hid=3000, gamma=(0.001, 0.170), epsilon=(0.028, 0.092), theta_rf=0.013,
        connectivity_enc2res=0.2, connectivity_res2res=1.0,
        input_density=0.036, num_steps=250, max_time=1.4, spatial_factor=4,
    ),
    'dvs_gesture': dict(
        n_hid=3000, gamma=(0.001, 0.111), epsilon=(0.001, 0.085), theta_rf=0.03628,
        connectivity_enc2res=1.0, connectivity_res2res=1.0,
        input_density=0.0306, num_steps=200, max_time=1.4, spatial_factor=4,
    ),
}

BEST_ENCODER_CONFIGS = {
    'sMNIST':      None,   # fill after search
    'fordA':       None,   # fill after search
    'shd':         None,   # fill after search
    'dvs_gesture': None,   # fill after search
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_sparse_input_projection(model, input_density, n_inp, n_hid, device):
    if input_density >= 1.0:
        return n_inp * n_hid
    mask = (torch.rand(n_inp, n_hid, device=device) < input_density).float()
    for j in range(n_hid):
        if mask[:, j].sum() == 0:
            mask[torch.randint(0, n_inp, (1,)), j] = 1.0
    for i in range(n_inp):
        if mask[i, :].sum() == 0:
            mask[i, torch.randint(0, n_hid, (1,))] = 1.0
    scale = 1.0 / np.sqrt(input_density)
    model.x2h.data = model.x2h.data * mask * scale
    return int(mask.sum().item())


def load_fordA(args, device):
    n_inp, n_out = 1, 2
    train_loader, _, test_loader = get_FordA_data(args.batch, 120, whole_train=True)
    seq_length = next(iter(train_loader))[0].shape[1]
    print(f"FordA: seq={seq_length}, train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, test_loader, n_inp, n_out, seq_length, 1.0


def load_shd(args, device):
    n_inp, n_out = 700, 20
    train_loader, _, test_loader = get_SHD_data(
        batch_train=args.batch, batch_test=256,
        data_dir=args.data_dir, num_steps=args.num_steps, max_time=args.max_time,
    )
    seq_length = next(iter(train_loader))[0].shape[1]
    print(f"SHD: seq={seq_length}, train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, test_loader, n_inp, n_out, seq_length, args.input_density


def load_smnist(args, device):
    n_inp, n_out = 1, 10
    train_loader, _, test_loader = get_mnist_data(args.batch, 100)
    print(f"sMNIST: seq=784, train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, test_loader, n_inp, n_out, 784, 1.0


def load_dvs_gesture(args, device):
    assert TONIC_AVAILABLE, "pip install tonic"
    sensor_size_orig = tonic.datasets.DVSGesture.sensor_size
    H_orig, W_orig, C = sensor_size_orig[1], sensor_size_orig[0], sensor_size_orig[2]
    sf = args.spatial_factor
    H_ds, W_ds = H_orig // sf, W_orig // sf
    n_inp, n_out = C * H_ds * W_ds, 11
    frame_transform = tonic_transforms.ToFrame(sensor_size=sensor_size_orig, n_time_bins=args.num_steps)

    def collate_fn(batch):
        xs, ys = [], []
        for frames, label in batch:
            t = torch.tensor(frames, dtype=torch.float32)
            if sf > 1:
                T_ = t.size(0)
                t = t.view(T_*C, 1, H_orig, W_orig)
                t = F.avg_pool2d(t, kernel_size=sf, stride=sf)
                t = t.view(T_, C, H_ds, W_ds)
            t = t.reshape(t.size(0), -1)
            t = (t > 0).float()
            xs.append(t); ys.append(label)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    os.makedirs(args.data_dir, exist_ok=True)
    train_ds = DiskCachedDataset(
        tonic.datasets.DVSGesture(save_to=args.data_dir, train=True, transform=frame_transform),
        cache_path=os.path.join(args.data_dir, f'cache_train_T{args.num_steps}_sf{sf}'))
    test_ds = DiskCachedDataset(
        tonic.datasets.DVSGesture(save_to=args.data_dir, train=False, transform=frame_transform),
        cache_path=os.path.join(args.data_dir, f'cache_test_T{args.num_steps}_sf{sf}'))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    seq_length = next(iter(train_loader))[0].shape[1]
    print(f"DVS Gesture: {len(train_ds)} train, {len(test_ds)} test, n_inp={n_inp}, seq={seq_length}")
    return train_loader, test_loader, n_inp, n_out, seq_length, args.input_density


DATASET_LOADERS = {'fordA': load_fordA, 'shd': load_shd,
                   'sMNIST': load_smnist, 'dvs_gesture': load_dvs_gesture}


def build_parser():
    p = argparse.ArgumentParser(description='HRF-HRF ablation (Option C)')
    p.add_argument('--dataset', required=True, choices=['fordA', 'shd', 'sMNIST', 'dvs_gesture'])
    p.add_argument('--data_dir',        type=str,   default='data')
    p.add_argument('--use_best_config', action='store_true')
    p.add_argument('--n_hid',           type=int,   default=800)
    p.add_argument('--batch',           type=int,   default=128)
    p.add_argument('--dt',              type=float, default=0.051)
    p.add_argument('--rho',             type=float, default=0.9)
    p.add_argument('--inp_scaling',     type=float, default=1.0)
    p.add_argument('--gamma_enc_min',   type=float, default=0.5)
    p.add_argument('--gamma_enc_max',   type=float, default=3.0)
    p.add_argument('--epsilon_enc_min', type=float, default=0.1)
    p.add_argument('--epsilon_enc_max', type=float, default=1.0)
    p.add_argument('--theta_enc',       type=float, default=0.5)
    p.add_argument('--readout_C',       type=float, default=0.1)
    p.add_argument('--readout_mode',    type=str,   default='rms_std_final',
                   choices=['final', 'mean', 'rms', 'std', 'rms_std_final',
                            'spikes_mean', 'spikes_rms_std_final'])
    p.add_argument('--input_density',   type=float, default=1.0)
    p.add_argument('--num_steps',       type=int,   default=250)
    p.add_argument('--max_time',        type=float, default=1.4)
    p.add_argument('--spatial_factor',  type=int,   default=4)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--test_trials',     type=int,   default=3)
    p.add_argument('--use_test',        action='store_true')
    p.add_argument('--cpu',             action='store_true')
    p.add_argument('--results_dir',     type=str,   default=None)
    return p


def main():
    args   = build_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ablation_hrf_hrf')

    res_cfg             = FIXED_RESERVOIR_CONFIGS[args.dataset]
    args.n_hid          = res_cfg['n_hid']
    args.input_density  = res_cfg['input_density']
    args.num_steps      = res_cfg['num_steps']
    args.max_time       = res_cfg['max_time']
    args.spatial_factor = res_cfg['spatial_factor']

    if args.use_best_config:
        enc_cfg = BEST_ENCODER_CONFIGS[args.dataset]
        if enc_cfg is None:
            raise ValueError(f"No best config for '{args.dataset}'. Run search first.")
        args.dt = enc_cfg['dt']; args.rho = enc_cfg['rho']; args.inp_scaling = enc_cfg['inp_scaling']
        args.gamma_enc_min = enc_cfg['gamma_enc_min']; args.gamma_enc_max = enc_cfg['gamma_enc_max']
        args.epsilon_enc_min = enc_cfg['epsilon_enc_min']; args.epsilon_enc_max = enc_cfg['epsilon_enc_max']
        args.theta_enc = enc_cfg['theta_enc']
        args.readout_C = enc_cfg['readout_C']; args.readout_mode = enc_cfg.get('readout_mode', 'rms_std_final')

    print('=' * 70)
    print(f'HRF-HRF ABLATION (Option C)  —  {args.dataset}   n_hid={args.n_hid}')
    print(f'dt={args.dt}  rho={args.rho}  inp={args.inp_scaling}')
    print(f'gamma_enc=({args.gamma_enc_min:.4f},{args.gamma_enc_max:.4f})  '
          f'eps_enc=({args.epsilon_enc_min:.4f},{args.epsilon_enc_max:.4f})  '
          f'theta_enc={args.theta_enc}')
    print(f'gamma_res={res_cfg["gamma"]}  eps_res={res_cfg["epsilon"]}  theta_rf={res_cfg["theta_rf"]}')
    print(f'readout={args.readout_mode}  C={args.readout_C}')
    print('=' * 70)

    loader_fn = DATASET_LOADERS[args.dataset]
    train_loader, test_loader, n_inp, n_out, seq_length, input_density = loader_fn(args, device)
    needs_reshape = (args.dataset == 'sMNIST')

    all_test_accs, all_train_accs = [], []
    all_energies, all_sops, all_sops_res, all_sops_enc = [], [], [], []
    all_r_res, all_r_enc, all_n_input_connections = [], [], []

    for trial in range(args.test_trials):
        print(f"\n{'='*70}\nTRIAL {trial+1}/{args.test_trials}\n{'='*70}")
        set_seed(args.seed + trial)

        model = spiking_HRF_HRF(
            n_inp=n_inp, n_hid=args.n_hid, dt=args.dt, rho=args.rho,
            input_scaling=args.inp_scaling,
            gamma_enc=(args.gamma_enc_min, args.gamma_enc_max),
            epsilon_enc=(args.epsilon_enc_min, args.epsilon_enc_max),
            theta_enc=args.theta_enc,
            gamma=res_cfg['gamma'], epsilon=res_cfg['epsilon'], theta_rf=res_cfg['theta_rf'],
            sparse_enc2res=(res_cfg['connectivity_enc2res'] < 1.0),
            connectivity_enc2res=res_cfg['connectivity_enc2res'],
            sparse_res2res=(res_cfg['connectivity_res2res'] < 1.0),
            connectivity_res2res=res_cfg['connectivity_res2res'],
            device=device, readout_mode=args.readout_mode,
        ).to(device)

        n_input_conn = (apply_sparse_input_projection(model, input_density, n_inp, args.n_hid, device)
                        if input_density < 1.0 else n_inp * args.n_hid)
        all_n_input_connections.append(n_input_conn)

        def _extract(loader, split_name):
            model.eval()
            feats, labels_all, r_res_l, r_enc_l = [], [], [], []
            with torch.no_grad():
                for xb, yb in tqdm(loader, ncols=80, desc=f"Extracting {split_name}"):
                    xb = xb.to(device)
                    if needs_reshape:
                        xb = xb.reshape(xb.shape[0], 1, 784).permute(0, 2, 1)
                    features, r = model(xb)
                    feats.append(features.cpu()); r_res_l.append(r['r_hrf'])
                    r_enc_l.append(r['r_lif']); labels_all.append(yb)
            return (torch.cat(feats, dim=0).numpy(),
                    torch.cat(labels_all, dim=0).numpy(),
                    torch.stack(r_res_l).mean().item(),
                    torch.stack(r_enc_l).mean().item())

        train_feats, train_labels, r_res_tr, r_enc_tr = _extract(train_loader, 'train')
        print(f"Features: {train_feats.shape}  r_res={r_res_tr:.4f}  r_enc={r_enc_tr:.4f}")
        test_feats, test_labels = (_extract(test_loader, 'test')[:2]
                                   if args.use_test else (train_feats, train_labels))

        scaler = preprocessing.StandardScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats  = scaler.transform(test_feats)

        # n_jobs=1 prevents OOM: parallel workers each copy the full feature
        # matrix into memory. With rms_std_final (3*n_hid features) on large
        # datasets this causes SIGKILL with n_jobs=-1.
        if args.dataset in ('shd', 'dvs_gesture'):
            clf = LogisticRegression(
                max_iter=2000, verbose=0, n_jobs=1,
                multi_class='multinomial', solver='lbfgs', C=args.readout_C,
            ).fit(train_feats, train_labels)
        else:
            clf = LogisticRegression(
                max_iter=2000, verbose=0, n_jobs=1, C=args.readout_C,
            ).fit(train_feats, train_labels)

        train_acc = clf.score(train_feats, train_labels) * 100
        test_acc  = clf.score(test_feats,  test_labels)  * 100
        print(f"Train: {train_acc:.2f}%   Test: {test_acc:.2f}%")

        snn_energy = estimate_snn_energy_sparse(
            r_hrf=r_res_tr, r_lif=r_enc_tr, n_hid=args.n_hid, T=seq_length,
            lif2hrf_connections=model.n_enc2res_connections, include_lif=True,
        )
        print(f"Energy: {snn_energy['Energy_J']:.3e} J")

        all_test_accs.append(test_acc); all_train_accs.append(train_acc)
        all_energies.append(snn_energy['Energy_J']); all_sops.append(snn_energy['SOPs'])
        all_sops_res.append(snn_energy['HRF_SOPs']); all_sops_enc.append(snn_energy['LIF_SOPs'])
        all_r_res.append(r_res_tr); all_r_enc.append(r_enc_tr)

    mean_test_acc = float(np.mean(all_test_accs)); std_test_acc = float(np.std(all_test_accs))
    print(f"\nFINAL: Test {mean_test_acc:.2f}% +/- {std_test_acc:.2f}%  "
          f"Per-trial: {[f'{a:.2f}' for a in all_test_accs]}")

    results = {
        'model': 'HRF_HRF_ablation_optionC', 'dataset': args.dataset,
        'use_best_config': args.use_best_config, 'n_trials': args.test_trials,
        'n_inp': n_inp, 'n_out': n_out, 'sequence_length': int(seq_length),
        'readout_mode': args.readout_mode, 'readout_C': float(args.readout_C),
        'dt': float(args.dt), 'rho': float(args.rho), 'inp_scaling': float(args.inp_scaling),
        'gamma_enc_min': float(args.gamma_enc_min), 'gamma_enc_max': float(args.gamma_enc_max),
        'epsilon_enc_min': float(args.epsilon_enc_min), 'epsilon_enc_max': float(args.epsilon_enc_max),
        'theta_enc': float(args.theta_enc),
        'gamma_res': str(res_cfg['gamma']), 'epsilon_res': str(res_cfg['epsilon']),
        'theta_rf': float(res_cfg['theta_rf']),
        'connectivity_enc2res': float(res_cfg['connectivity_enc2res']),
        'connectivity_res2res': float(res_cfg['connectivity_res2res']),
        'input_density': float(input_density),
        'n_enc2res_connections': int(model.n_enc2res_connections),
        'n_res2res_connections': int(model.n_res2res_connections),
        'n_input_connections_mean': float(np.mean(all_n_input_connections)),
        'train_acc_mean': float(np.mean(all_train_accs)), 'train_acc_std': float(np.std(all_train_accs)),
        'test_acc_mean': mean_test_acc, 'test_acc_std': std_test_acc,
        'test_accs_all': [float(a) for a in all_test_accs],
        'r_res_mean': float(np.mean(all_r_res)), 'r_enc_mean': float(np.mean(all_r_enc)),
        'energy_J_mean': float(np.mean(all_energies)), 'energy_J_std': float(np.std(all_energies)),
        'sops_mean': float(np.mean(all_sops)), 'sops_res_mean': float(np.mean(all_sops_res)),
        'sops_enc_mean': float(np.mean(all_sops_enc)),
        'n_hid': int(args.n_hid), 'base_seed': int(args.seed), 'args': vars(args),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    best_tag = '_bestcfg' if args.use_best_config else ''
    inp_str  = f"inp{input_density:.4f}" if input_density < 1.0 else "inpDense"
    fname = (f"results_hrf_hrf_{args.dataset}_nhid{args.n_hid}{best_tag}"
             f"_{inp_str}_{args.readout_mode}_trials{args.test_trials}_seed{args.seed}.json")
    fpath = os.path.join(args.results_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {fpath}")


if __name__ == '__main__':
    main()