"""
Unified training script for the LIF-reservoir ablation of s-RON.

Replaces HRF neurons in the reservoir with heterogeneous LIF neurons.
Encoder LIF layer is identical to s-RON -- only the reservoir (2nd population)
changes neuron model.

Supports: fordA | shd | sMNIST | dvs_gesture

Usage examples (manual params):
    python train_lif_ablation.py --dataset fordA        --use_test
    python train_lif_ablation.py --dataset shd          --use_test --n_hid 3000
    python train_lif_ablation.py --dataset sMNIST       --use_test
    python train_lif_ablation.py --dataset dvs_gesture  --use_test

Usage examples (best tuned configs for paper results):
    python train_lif_ablation.py --dataset fordA        --use_best_config --use_test --test_trials 3
    python train_lif_ablation.py --dataset shd --use_best_config --use_test --test_trials 3 --data_dir data/SHD
    python train_lif_ablation.py --dataset sMNIST       --use_best_config --use_test --test_trials 3
    python train_lif_ablation.py --dataset dvs_gesture  --use_best_config --use_test --test_trials 3 --data_dir data/DVSGesture

Results are saved to ablation_lif/ next to this script (override with
--results_dir).
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
from utils_aurora import estimate_snn_energy_sparse, spiking_LIF_reservoir

# Dataset loaders
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


# =============================================================================
# Best configs from hyperparameter search (one per dataset)
# These are loaded when --use_best_config is passed.
# All other args (test_trials, seed, use_test, results_dir) still apply.
# =============================================================================

BEST_CONFIGS = {
    'fordA': dict(
        # Rank 1: 69.17% test, 77.04% train, gap 7.9%
        n_hid              = 800,
        readout_mode       = 'final',
        readout_C          = 0.1,
        dt                 = 0.068,
        rho                = 0.875,
        inp_scaling        = 4.0365,
        theta_lif          = 0.1458,
        tau_m              = 36.68,
        tau_m_range        = 25.85,
        theta_res          = 0.15659,
        theta_res_range    = 0.01254,
        connectivity_lif2res = 0.2,
        connectivity_res2enc = 1.0,
        # dataset-specific
        input_density      = 1.0,
        num_steps          = 250,   # not used for fordA but kept for consistency
        max_time           = 1.4,
        spatial_factor     = 4,
    ),

    'shd': dict(
        # Rank 1: 75.00% test, 99.15% train, gap 24.2%
        n_hid              = 3000,
        readout_mode       = 'final',
        readout_C          = 0.01,
        dt                 = 0.084,
        rho                = 1.172,
        inp_scaling        = 0.1962,
        theta_lif          = 0.4188,
        tau_m              = 13.01,
        tau_m_range        = 2.38,
        theta_res          = 0.09598,
        theta_res_range    = 0.00375,
        connectivity_lif2res = 1.0, #0.2
        connectivity_res2enc = 1.0,
        # dataset-specific
        input_density      = 0.036,
        num_steps          = 250,
        max_time           = 1.4,
        spatial_factor     = 4,
    ),

    'sMNIST': dict(
        # Rank 1: 68.30% test, 69.10% train, gap 0.8%
        n_hid              = 800,
        readout_mode       = 'final',
        readout_C          = 0.01,
        dt                 = 0.034,
        rho                = 0.976,
        inp_scaling        = 0.8471,
        theta_lif          = 0.1507,
        tau_m              = 36.19,
        tau_m_range        = 3.60,
        theta_res          = 0.00254,
        theta_res_range    = 0.01180,
        connectivity_lif2res = 0.2,
        connectivity_res2enc = 1.0,
        # dataset-specific
        input_density      = 1.0,
        num_steps          = 250,
        max_time           = 1.4,
        spatial_factor     = 4,
    ),

    'dvs_gesture': dict(
        # Rank 1: 73.48% test, 99.81% train, gap 26.3%
        n_hid              = 3000,
        readout_mode       = 'final',
        readout_C          = 0.01,
        dt                 = 0.056,
        rho                = 0.866,
        inp_scaling        = 0.0310,
        theta_lif          = 0.5327,
        tau_m              = 58.59,
        tau_m_range        = 23.17,
        theta_res          = 0.02224,
        theta_res_range    = 0.00194,
        connectivity_lif2res = 1.0,
        connectivity_res2enc = 1.0,
        # dataset-specific
        input_density      = 0.0306,
        num_steps          = 200,
        max_time           = 1.4,
        spatial_factor     = 4,
    ),
}


# =============================================================================
# Seed / helpers
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_sparse_input_projection(model, input_density, n_inp, n_hid, device):
    """Identical sparse input projection used in SHD and DVS scripts."""
    if input_density >= 1.0:
        n_conn = n_inp * n_hid
        print(f"  Input projection: DENSE ({n_conn} connections)")
        return n_conn

    mask = (torch.rand(n_inp, n_hid, device=device) < input_density).float()
    for j in range(n_hid):
        if mask[:, j].sum() == 0:
            mask[torch.randint(0, n_inp, (1,)), j] = 1.0
    for i in range(n_inp):
        if mask[i, :].sum() == 0:
            mask[i, torch.randint(0, n_hid, (1,))] = 1.0

    scale = 1.0 / np.sqrt(input_density)
    model.x2h.data = model.x2h.data * mask * scale
    n_conn = int(mask.sum().item())
    print(f"  Input projection: SPARSE density={input_density:.3f}, "
          f"{n_conn}/{n_inp*n_hid} connections "
          f"({n_conn / (n_inp * n_hid) * 100:.1f}%), scale={scale:.2f}")
    print(f"  Avg inputs per neuron: {n_conn / n_hid:.1f}/{n_inp}")
    return n_conn


# =============================================================================
# Dataset loaders
# =============================================================================

def load_fordA(args, device):
    n_inp, n_out = 1, 2
    train_loader, _, test_loader = get_FordA_data(
        args.batch, 120, whole_train=True
    )
    sample_x, _ = next(iter(train_loader))
    seq_length   = sample_x.shape[1]
    print(f"FordA loaded: seq_length={seq_length}, n_inp={n_inp}, "
          f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, test_loader, n_inp, n_out, seq_length, 1.0


def load_shd(args, device):
    n_inp, n_out = 700, 20
    train_loader, _, test_loader = get_SHD_data(
        batch_train=args.batch, batch_test=256,
        data_dir=args.data_dir, num_steps=args.num_steps,
        max_time=args.max_time
    )
    sample_x, _ = next(iter(train_loader))
    seq_length   = sample_x.shape[1]
    print(f"SHD loaded: seq_length={seq_length}, n_inp={n_inp}, "
          f"num_steps={args.num_steps}, "
          f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, test_loader, n_inp, n_out, seq_length, args.input_density


def load_smnist(args, device):
    n_inp, n_out = 1, 10
    train_loader, _, test_loader = get_mnist_data(args.batch, 100)
    seq_length = 784
    print(f"sMNIST loaded: seq_length={seq_length}, n_inp={n_inp}, "
          f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    return train_loader, test_loader, n_inp, n_out, seq_length, 1.0


def load_dvs_gesture(args, device):
    assert TONIC_AVAILABLE, "pip install tonic to use dvs_gesture"

    sensor_size_orig = tonic.datasets.DVSGesture.sensor_size  # (128,128,2)
    H_orig = sensor_size_orig[1]
    W_orig = sensor_size_orig[0]
    C      = sensor_size_orig[2]
    sf     = args.spatial_factor
    H_ds   = H_orig // sf
    W_ds   = W_orig // sf
    n_inp  = C * H_ds * W_ds
    n_out  = 11

    frame_transform = tonic_transforms.ToFrame(
        sensor_size=sensor_size_orig, n_time_bins=args.num_steps
    )

    def collate_fn(batch):
        xs, ys = [], []
        for frames, label in batch:
            t = torch.tensor(frames, dtype=torch.float32)
            if sf > 1:
                T_ = t.size(0)
                t  = t.view(T_ * C, 1, H_orig, W_orig)
                t  = F.avg_pool2d(t, kernel_size=sf, stride=sf)
                t  = t.view(T_, C, H_ds, W_ds)
            t = t.reshape(t.size(0), -1)
            t = (t > 0).float()
            xs.append(t)
            ys.append(label)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    os.makedirs(args.data_dir, exist_ok=True)
    train_ds_raw = tonic.datasets.DVSGesture(
        save_to=args.data_dir, train=True,  transform=frame_transform)
    test_ds_raw  = tonic.datasets.DVSGesture(
        save_to=args.data_dir, train=False, transform=frame_transform)

    cache_tr = os.path.join(args.data_dir,
                            f'cache_train_T{args.num_steps}_sf{sf}')
    cache_te = os.path.join(args.data_dir,
                            f'cache_test_T{args.num_steps}_sf{sf}')
    train_ds = DiskCachedDataset(train_ds_raw, cache_path=cache_tr)
    test_ds  = DiskCachedDataset(test_ds_raw,  cache_path=cache_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True)

    sample_x, _ = next(iter(train_loader))
    seq_length   = sample_x.shape[1]
    print(f"DVS Gesture loaded: {len(train_ds)} train, {len(test_ds)} test")
    print(f"  Spatial: {H_orig}x{W_orig} -> {H_ds}x{W_ds} (factor={sf})")
    print(f"  n_inp={n_inp}, seq_length={seq_length}")
    return train_loader, test_loader, n_inp, n_out, seq_length, args.input_density


DATASET_LOADERS = {
    'fordA':       load_fordA,
    'shd':         load_shd,
    'sMNIST':      load_smnist,
    'dvs_gesture': load_dvs_gesture,
}


# =============================================================================
# Argument parser
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description='LIF reservoir ablation -- drop-in replacement for s-RON'
    )

    # Dataset
    p.add_argument('--dataset', type=str, required=True,
                   choices=['fordA', 'shd', 'sMNIST', 'dvs_gesture'])
    p.add_argument('--data_dir', type=str, default='data',
                   help='Root data directory (SHD / DVS Gesture)')

    # Best config flag -- overrides all model params below with tuned values
    p.add_argument('--use_best_config', action='store_true',
                   help='Load best tuned hyperparameters for this dataset. '
                        'Overrides all model/architecture args below.')

    # Architecture
    p.add_argument('--n_hid', type=int, default=800)
    p.add_argument('--batch', type=int, default=128)

    # Encoder LIF (fixed scalar, identical to s-RON)
    p.add_argument('--theta_lif',  type=float, default=0.05,
                   help='Encoder LIF firing threshold (fixed)')
    p.add_argument('--tau_filter', type=float, default=20.0)

    # Reservoir LIF -- heterogeneous (log-uniform sampling)
    p.add_argument('--tau_m',           type=float, default=20.0,
                   help='Reservoir LIF membrane time constant center (ms)')
    p.add_argument('--tau_m_range',     type=float, default=15.0,
                   help='Reservoir LIF tau_m range width')
    p.add_argument('--theta_res',       type=float, default=0.05,
                   help='Reservoir LIF firing threshold center')
    p.add_argument('--theta_res_range', type=float, default=0.04,
                   help='Reservoir LIF theta_res range width')

    # Reservoir dynamics
    p.add_argument('--dt',          type=float, default=0.051)
    p.add_argument('--rho',         type=float, default=0.75)
    p.add_argument('--inp_scaling', type=float, default=0.5)

    # Connectivity
    p.add_argument('--connectivity_lif2res', type=float, default=0.2,
                   help='Enc->Res sparsity (1.0 = dense)')
    p.add_argument('--connectivity_res2enc', type=float, default=1.0,
                   help='Res->Res recurrent sparsity (1.0 = dense)')

    # Dataset-specific
    p.add_argument('--input_density',  type=float, default=1.0,
                   help='Sparse input projection density (SHD / DVS Gesture)')
    p.add_argument('--num_steps',      type=int,   default=250,
                   help='Number of time bins (SHD / DVS Gesture)')
    p.add_argument('--max_time',       type=float, default=1.4,
                   help='Max time in seconds for SHD binning')
    p.add_argument('--spatial_factor', type=int,   default=4,
                   help='Spatial downsampling factor for DVS Gesture')

    # Readout
    p.add_argument('--readout_C',    type=float, default=0.1)
    p.add_argument('--readout_mode', type=str,   default='final',
                   choices=['final', 'mean', 'rms_std_final'])

    # Training
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--test_trials', type=int, default=3)
    p.add_argument('--use_test',    action='store_true')
    p.add_argument('--cpu',         action='store_true')

    # Results -- default: ablation_lif/ next to this script
    p.add_argument('--results_dir', type=str, default=None,
                   help='Output directory. Defaults to ablation_lif/ '
                        'next to this script.')

    return p


# =============================================================================
# Main
# =============================================================================

def main():
    args   = build_parser().parse_args()
    device = (torch.device('cuda')
              if torch.cuda.is_available() and not args.cpu
              else torch.device('cpu'))

    # Resolve results directory to be next to this script
    if args.results_dir is None:
        script_dir       = os.path.dirname(os.path.abspath(__file__))
        args.results_dir = os.path.join(script_dir, 'ablation_lif')

    # Apply best config if requested -- overrides all model args
    if args.use_best_config:
        cfg = BEST_CONFIGS[args.dataset]
        args.n_hid               = cfg['n_hid']
        args.readout_mode        = cfg['readout_mode']
        args.readout_C           = cfg['readout_C']
        args.dt                  = cfg['dt']
        args.rho                 = cfg['rho']
        args.inp_scaling         = cfg['inp_scaling']
        args.theta_lif           = cfg['theta_lif']
        args.tau_m               = cfg['tau_m']
        args.tau_m_range         = cfg['tau_m_range']
        args.theta_res           = cfg['theta_res']
        args.theta_res_range     = cfg['theta_res_range']
        args.connectivity_lif2res = cfg['connectivity_lif2res']
        args.connectivity_res2enc = cfg['connectivity_res2enc']
        args.input_density       = cfg['input_density']
        args.num_steps           = cfg['num_steps']
        args.max_time            = cfg['max_time']
        args.spatial_factor      = cfg['spatial_factor']
        print(f"[use_best_config] Loaded tuned params for {args.dataset}")

    print('=' * 70)
    print('LIF RESERVOIR ABLATION')
    print('=' * 70)
    print(f"Dataset:          {args.dataset}")
    print(f"Device:           {device}")
    print(f"Hidden units:     {args.n_hid}")
    print(f"Readout mode:     {args.readout_mode}   C={args.readout_C}")
    print(f"Trials:           {args.test_trials}   seed={args.seed}")
    print(f"dt:               {args.dt}")
    print(f"rho:              {args.rho}")
    print(f"inp_scaling:      {args.inp_scaling}")
    print(f"--- Encoder LIF (fixed) ---")
    print(f"theta_lif:        {args.theta_lif}")
    print(f"tau_filter:       {args.tau_filter}")
    print(f"--- Reservoir LIF (heterogeneous, log-uniform) ---")
    print(f"tau_m:            {args.tau_m} +/- {args.tau_m_range/2}")
    print(f"theta_res:        {args.theta_res} +/- {args.theta_res_range/2}")
    print(f"--- Connectivity ---")
    print(f"lif2res:          {args.connectivity_lif2res*100:.1f}%")
    print(f"res2res (h2h):    {args.connectivity_res2enc*100:.1f}%")
    print(f"input_density:    {args.input_density}")
    print('=' * 70)

    # Load dataset
    loader_fn = DATASET_LOADERS[args.dataset]
    (train_loader, test_loader,
     n_inp, n_out, seq_length, input_density) = loader_fn(args, device)

    needs_reshape = (args.dataset == 'sMNIST')

    # Per-trial accumulators
    all_test_accs, all_train_accs        = [], []
    all_energies                         = []
    all_sops, all_sops_res, all_sops_enc = [], [], []
    all_r_res, all_r_enc, all_r_total    = [], [], []
    all_n_input_connections              = []

    for trial in range(args.test_trials):
        print(f"\n{'='*70}\nTRIAL {trial+1}/{args.test_trials}\n{'='*70}")
        set_seed(args.seed + trial)

        print("\n=== Building LIF Reservoir ===")
        model = spiking_LIF_reservoir(
            n_inp=n_inp,
            n_hid=args.n_hid,
            dt=args.dt,
            tau_m=args.tau_m,
            tau_m_range=args.tau_m_range,
            theta_res=args.theta_res,
            theta_res_range=args.theta_res_range,
            rho=args.rho,
            input_scaling=args.inp_scaling,
            theta_lif=args.theta_lif,
            tau_filter=args.tau_filter,
            sparse_lif2res=(args.connectivity_lif2res < 1.0),
            connectivity_lif2res=args.connectivity_lif2res,
            sparse_res2enc=(args.connectivity_res2enc < 1.0),
            connectivity_res2enc=args.connectivity_res2enc,
            device=device,
            readout_mode=args.readout_mode,
        ).to(device)

        # Sparse input projection (SHD / DVS Gesture)
        if input_density < 1.0:
            n_input_conn = apply_sparse_input_projection(
                model, input_density, n_inp, args.n_hid, device
            )
        else:
            n_input_conn = n_inp * args.n_hid
        all_n_input_connections.append(n_input_conn)

        def _extract(loader, split_name):
            model.eval()
            feats, labels_all            = [], []
            r_tot_l, r_res_l, r_enc_l   = [], [], []
            with torch.no_grad():
                for x, y in tqdm(loader, ncols=80,
                                 desc=f"Extracting {split_name}"):
                    x = x.to(device)
                    if needs_reshape:
                        x = x.reshape(x.shape[0], 1, 784).permute(0, 2, 1)
                    features, r = model(x)
                    feats.append(features.cpu())
                    r_tot_l.append(r['r_total'])
                    r_res_l.append(r['r_hrf'])   # r_hrf = reservoir LIF rate
                    r_enc_l.append(r['r_lif'])   # r_lif = encoder LIF rate
                    labels_all.append(y)
            feats      = torch.cat(feats,      dim=0).numpy()
            labels_all = torch.cat(labels_all, dim=0).numpy()
            return (feats, labels_all,
                    torch.stack(r_tot_l).mean().item(),
                    torch.stack(r_res_l).mean().item(),
                    torch.stack(r_enc_l).mean().item())

        print("\n=== Extracting Reservoir Features ===")
        train_feats, train_labels, r_tot_tr, r_res_tr, r_enc_tr = \
            _extract(train_loader, 'train')
        print(f"Training features: {train_feats.shape}")
        print(f"r_res={r_res_tr:.4f}  r_enc={r_enc_tr:.4f}")

        if args.use_test:
            test_feats, test_labels, r_tot_te, r_res_te, r_enc_te = \
                _extract(test_loader, 'test')
            print(f"Test features: {test_feats.shape}")
        else:
            test_feats, test_labels = train_feats, train_labels

        scaler      = preprocessing.StandardScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats  = scaler.transform(test_feats)

        print("\n=== Training Logistic Regression Readout ===")
        # Solver matches original per-dataset scripts exactly
        if args.dataset in ('shd', 'dvs_gesture'):
            clf = LogisticRegression(
                max_iter=2000, verbose=0, n_jobs=-1,
                multi_class='multinomial', solver='lbfgs',
                C=args.readout_C
            ).fit(train_feats, train_labels)
        else:   # fordA, sMNIST -- sklearn defaults
            clf = LogisticRegression(
                max_iter=2000, verbose=0, n_jobs=-1,
                C=args.readout_C
            ).fit(train_feats, train_labels)

        train_acc = clf.score(train_feats, train_labels) * 100
        test_acc  = clf.score(test_feats,  test_labels)  * 100
        print(f"Train accuracy: {train_acc:.2f}%")
        print(f"Test accuracy:  {test_acc:.2f}%")

        # Energy (r_hrf API maps to reservoir LIF rate)
        snn_energy = estimate_snn_energy_sparse(
            r_hrf=r_res_tr, r_lif=r_enc_tr,
            n_hid=args.n_hid, T=seq_length,
            lif2hrf_connections=model.n_lif2res_connections,
            include_lif=True
        )
        print(f"Energy: {snn_energy['Energy_J']:.3e} J  |  "
              f"SOPs: {snn_energy['SOPs']:.3e}  "
              f"(Res: {snn_energy['HRF_SOPs']:.3e}, "
              f"Enc: {snn_energy['LIF_SOPs']:.3e})")

        all_test_accs.append(test_acc)
        all_train_accs.append(train_acc)
        all_energies.append(snn_energy['Energy_J'])
        all_sops.append(snn_energy['SOPs'])
        all_sops_res.append(snn_energy['HRF_SOPs'])
        all_sops_enc.append(snn_energy['LIF_SOPs'])
        all_r_res.append(r_res_tr)
        all_r_enc.append(r_enc_tr)
        all_r_total.append(r_tot_tr)

    # Final aggregation
    mean_test_acc  = float(np.mean(all_test_accs))
    std_test_acc   = float(np.std(all_test_accs))
    mean_train_acc = float(np.mean(all_train_accs))
    std_train_acc  = float(np.std(all_train_accs))
    mean_energy    = float(np.mean(all_energies))
    std_energy     = float(np.std(all_energies))
    mean_sops      = float(np.mean(all_sops))
    mean_sops_res  = float(np.mean(all_sops_res))
    mean_sops_enc  = float(np.mean(all_sops_enc))

    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Model:            LIF reservoir ablation")
    print(f"Dataset:          {args.dataset}")
    print(f"Best config used: {args.use_best_config}")
    print(f"Hidden units:     {args.n_hid},  Trials: {args.test_trials}")
    print(f"Readout mode:     {args.readout_mode}")
    print(f"Sequence length:  {seq_length}")
    if args.dataset in ('shd', 'dvs_gesture'):
        print(f"Num steps:        {args.num_steps}")
        print(f"Input density:    {input_density}")
    if args.dataset == 'dvs_gesture':
        print(f"Spatial factor:   {args.spatial_factor}")
    print(f"--- Encoder LIF (fixed) ---")
    print(f"theta_lif:        {args.theta_lif}")
    print(f"tau_filter:       {args.tau_filter}")
    print(f"--- Reservoir LIF (heterogeneous, log-uniform) ---")
    print(f"tau_m:            {args.tau_m} +/- {args.tau_m_range/2}  "
          f"[range {args.tau_m_range}]")
    print(f"theta_res:        {args.theta_res} +/- {args.theta_res_range/2}  "
          f"[range {args.theta_res_range}]")
    print(f"--- Connectivity ---")
    print(f"Enc->Res (lif2res): "
          f"{'DENSE' if args.connectivity_lif2res == 1.0 else 'SPARSE'} "
          f"({args.connectivity_lif2res*100:.1f}%),  "
          f"{model.n_lif2res_connections} connections")
    print(f"Res->Res (h2h):     "
          f"{'DENSE' if args.connectivity_res2enc == 1.0 else 'SPARSE'} "
          f"({args.connectivity_res2enc*100:.1f}%),  "
          f"{model.n_res2enc_connections} connections")
    print(f"--- Results ---")
    print(f"Train accuracy:   {mean_train_acc:.2f}% +/- {std_train_acc:.2f}%")
    print(f"Test accuracy:    {mean_test_acc:.2f}%  +/- {std_test_acc:.2f}%")
    print(f"Per-trial test:   {[f'{a:.2f}' for a in all_test_accs]}")
    print(f"r_res (mean):     {np.mean(all_r_res):.4f}")
    print(f"r_enc (mean):     {np.mean(all_r_enc):.4f}")
    print(f"Energy:           {mean_energy:.3e} +/- {std_energy:.3e} J")
    print(f"SOPs (total):     {mean_sops:.3e}  "
          f"(Res: {mean_sops_res:.3e}, Enc: {mean_sops_enc:.3e})")
    print(f"{'='*70}")

    results = {
        'model':              'LIF_reservoir_ablation',
        'dataset':            args.dataset,
        'use_best_config':    args.use_best_config,
        'args':               vars(args),
        'n_trials':           args.test_trials,
        'n_inp':              n_inp,
        'n_out':              n_out,
        'sequence_length':    int(seq_length),
        'readout_mode':       args.readout_mode,
        'readout_C':          float(args.readout_C),
        # Encoder LIF (fixed)
        'theta_lif':          float(args.theta_lif),
        'tau_filter':         float(args.tau_filter),
        # Reservoir LIF (heterogeneous)
        'tau_m':              float(args.tau_m),
        'tau_m_range':        float(args.tau_m_range),
        'theta_res':          float(args.theta_res),
        'theta_res_range':    float(args.theta_res_range),
        # Reservoir dynamics
        'dt':                 float(args.dt),
        'rho':                float(args.rho),
        'inp_scaling':        float(args.inp_scaling),
        # Connectivity
        'connectivity_lif2res':  float(args.connectivity_lif2res),
        'connectivity_res2enc':  float(args.connectivity_res2enc),
        'n_lif2res_connections': int(model.n_lif2res_connections),
        'n_res2enc_connections': int(model.n_res2enc_connections),
        'n_input_connections_mean': float(np.mean(all_n_input_connections)),
        'input_density':      float(input_density),
        # Dataset-specific
        'num_steps':          int(args.num_steps),
        'max_time':           float(args.max_time),
        'spatial_factor':     int(args.spatial_factor),
        # Accuracy
        'train_acc_mean':     mean_train_acc,
        'train_acc_std':      std_train_acc,
        'test_acc_mean':      mean_test_acc,
        'test_acc_std':       std_test_acc,
        'test_accs_all':      [float(a) for a in all_test_accs],
        # Firing rates
        'r_res_mean':         float(np.mean(all_r_res)),
        'r_enc_mean':         float(np.mean(all_r_enc)),
        'r_tot_mean':         float(np.mean(all_r_total)),
        # Energy
        'energy_J_mean':      mean_energy,
        'energy_J_std':       std_energy,
        'sops_mean':          mean_sops,
        'sops_res_mean':      mean_sops_res,
        'sops_enc_mean':      mean_sops_enc,
        # Meta
        'n_hid':              int(args.n_hid),
        'base_seed':          int(args.seed),
    }

    os.makedirs(args.results_dir, exist_ok=True)

    # Filename encodes key dimensions; tag best_config runs clearly
    best_tag = '_bestcfg' if args.use_best_config else ''
    inp_str  = (f"inp{input_density:.3f}"
                if input_density < 1.0 else "inpDense")
    l2r_str  = ("dense" if args.connectivity_lif2res == 1.0
                else f"l2r{args.connectivity_lif2res:.1f}")
    r2r_str  = ("dense" if args.connectivity_res2enc == 1.0
                else f"r2r{args.connectivity_res2enc:.1f}")
    fname = (
        f"results_lif_ablation_{args.dataset}"
        f"_nhid{args.n_hid}"
        f"{best_tag}"
        f"_{inp_str}_{l2r_str}_{r2r_str}"
        f"_{args.readout_mode}"
        f"_trials{args.test_trials}"
        f"_seed{args.seed}.json"
    )
    fpath = os.path.join(args.results_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {fpath}")


if __name__ == '__main__':
    main()
    