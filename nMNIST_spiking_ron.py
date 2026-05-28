"""
Spiking RON on the Neuromorphic MNIST (N-MNIST) dataset.

N-MNIST: 10-class digit classification using event-based camera spike trains.
- 34x34 pixels x 2 polarities = 2312 input channels (after flattening)
- Train: 60000, Test: 10000
- Events recorded by moving a DVS camera in front of standard MNIST digits

Data is binned into T time steps, yielding sparse binary tensors of shape (T, 2312).

Download (via Tonic):
    pip install tonic
    # Tonic downloads N-MNIST automatically on first run.
    # Default cache: ~/data/nmnist  (override with --data_dir)

Alternatively, manual download:
    https://www.garrickorchard.com/datasets/n-mnist
    Place the extracted Train/ and Test/ folders under --data_dir.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import argparse
import numpy as np
import random
import json
import os
import time

import tonic
import tonic.transforms as transforms


from utils_aurora import spiking_coESN_rescaled_II, estimate_snn_energy_sparse
from esn import spectral_norm_scaling


# -- Dataset -------------------------------------------------------------------

def get_NMNIST_data(batch_train, batch_test, data_dir='data/NMNIST',
                    num_steps=20, spatial_factor=1):
    sensor_size = tonic.datasets.NMNIST.sensor_size  # (34, 34, 2)
    H, W, C = sensor_size[1], sensor_size[0], sensor_size[2]

    H_ds = H // spatial_factor
    W_ds = W // spatial_factor
    n_inp = C * H_ds * W_ds

    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=num_steps)

    def collate_fn(batch):
        import torch.nn.functional as F
        xs, ys = [], []
        for frames, label in batch:
            t = torch.tensor(frames, dtype=torch.float32)  # (T, C, H, W)
            if spatial_factor > 1:
                T_ = t.size(0)
                t = t.view(T_ * C, 1, H, W)
                t = F.avg_pool2d(t, kernel_size=spatial_factor, stride=spatial_factor)
                t = t.view(T_, C, H_ds, W_ds)
            t = t.reshape(t.size(0), -1)   # (T, C*H_ds*W_ds)
            t = (t > 0).float()
            xs.append(t)
            ys.append(label)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    os.makedirs(data_dir, exist_ok=True)

    train_ds = tonic.datasets.NMNIST(save_to=data_dir, train=True,
                                     transform=frame_transform)
    test_ds  = tonic.datasets.NMNIST(save_to=data_dir, train=False,
                                     transform=frame_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_test,  shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    print(f"  N-MNIST: {len(train_ds)} train, {len(test_ds)} test samples")
    print(f"  Input shape per sample: ({num_steps}, {n_inp})")
    return train_loader, test_loader, n_inp


# -- Sparse input projection ---------------------------------------------------

def apply_sparse_input_projection(model, input_density, n_inp, n_hid, device):
    if input_density >= 1.0:
        n_input_connections = n_inp * n_hid
        print(f"  Input projection: DENSE ({n_input_connections} connections)")
        return n_input_connections

    mask = (torch.rand(n_inp, n_hid, device=device) < input_density).float()

    for j in range(n_hid):
        if mask[:, j].sum() == 0:
            mask[torch.randint(0, n_inp, (1,)), j] = 1.0
    for i in range(n_inp):
        if mask[i, :].sum() == 0:
            mask[i, torch.randint(0, n_hid, (1,))] = 1.0

    scale = 1.0 / np.sqrt(input_density)
    model.x2h.data = model.x2h.data * mask * scale

    n_input_connections = int(mask.sum().item())
    print(f"  Input projection: SPARSE density={input_density:.3f}, "
          f"{n_input_connections}/{n_inp * n_hid} connections "
          f"({n_input_connections / (n_inp * n_hid) * 100:.1f}%), scale={scale:.2f}")
    print(f"  Avg inputs per neuron: {n_input_connections / n_hid:.1f}/{n_inp}")
    return n_input_connections


# -- Feature extraction --------------------------------------------------------

def extract_features(loader, model, device):
    model.eval()
    feats, labels_all = [], []
    r_tot, r_hrf, r_lif = [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, ncols=80, desc="Extracting features"):
            x = x.to(device)
            features, r = model(x)
            feats.append(features.cpu())
            r_tot.append(r["r_total"])
            r_hrf.append(r["r_hrf"])
            r_lif.append(r["r_lif"])
            labels_all.append(y)

    if len(feats) == 0:
        return None, None, 0.0, 0.0, 0.0

    feats      = torch.cat(feats,      dim=0).numpy()
    labels_all = torch.cat(labels_all, dim=0).numpy()
    return (feats, labels_all,
            torch.stack(r_tot).mean().item(),
            torch.stack(r_hrf).mean().item(),
            torch.stack(r_lif).mean().item())


# -- Helpers -------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Spiking RON on N-MNIST Dataset')

    parser.add_argument('--n_hid', type=int, default=800)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--dt',            type=float, default=0.109)
    parser.add_argument('--gamma',         type=float, default=0.109)
    parser.add_argument('--epsilon',       type=float, default=0.0208)
    parser.add_argument('--gamma_range',   type=float, default=2.64)
    parser.add_argument('--epsilon_range', type=float, default=0.068)
    parser.add_argument('--inp_scaling', type=float, default=0.218)
    parser.add_argument('--rho',         type=float, default=1.21)
    parser.add_argument('--theta_lif',  type=float, default=0.189)
    parser.add_argument('--theta_rf',   type=float, default=0.045)
    parser.add_argument('--tau_filter', type=float, default=20.0)
    parser.add_argument('--connectivity_lif2hrf', type=float, default=1.0)
    parser.add_argument('--connectivity_hrf2lif', type=float, default=1.0)
    parser.add_argument('--input_density', type=float, default=0.063)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--spatial_factor', type=int, default=2)
    parser.add_argument('--cpu',         action='store_true')
    parser.add_argument('--use_test',    action='store_true')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--test_trials', type=int, default=5)
    parser.add_argument('--data_dir',    type=str, default='data/NMNIST')
    parser.add_argument('--readout_C', type=float, default=0.1)
    #parser.add_argument('--readout_mode', type=str, default='final',
                        #choices=['final', 'mean', 'rms_std_final'])
    parser.add_argument('--readout_mode', type=str, default='final',
                    choices=['final', 'mean', 'rms', 'std', 'rms_std_final', 'spikes_mean'])
    parser.add_argument('--results_dir', type=str, default='results_nmnist')

    args = parser.parse_args()

    print("=" * 70)
    print("SPIKING RON ON N-MNIST (NEUROMORPHIC MNIST) DATASET")
    print("=" * 70)
    print(args)
    print("=" * 70)

    device = (torch.device("cuda") if torch.cuda.is_available() and not args.cpu
              else torch.device("cpu"))
    print(f"Using device: {device}")

    n_out   = 10
    bs_test = 256

    gamma   = (args.gamma   - args.gamma_range   / 2., args.gamma   + args.gamma_range   / 2.)
    epsilon = (args.epsilon - args.epsilon_range / 2., args.epsilon + args.epsilon_range / 2.)
    use_sparse_lif2hrf = args.connectivity_lif2hrf < 1.0
    use_sparse_hrf2lif = args.connectivity_hrf2lif < 1.0

    print("\n=== Loading N-MNIST Dataset ===")
    train_loader, test_loader, n_inp = get_NMNIST_data(
        batch_train=args.batch, batch_test=bs_test,
        data_dir=args.data_dir, num_steps=args.num_steps,
        spatial_factor=args.spatial_factor
    )
    print(f"Loaded N-MNIST dataset")

    sample_x, sample_y = next(iter(train_loader))
    seq_length = sample_x.shape[1]
    print(f"   Sample shape: {sample_x.shape}  (batch, time, channels)")

    all_test_accs, all_train_accs, all_energies = [], [], []
    all_sops, all_sops_hrf, all_sops_lif = [], [], []
    all_r_hrf, all_r_lif, all_r_total = [], [], []
    all_n_input_connections = []

    for trial in range(args.test_trials):
        print(f"\n{'='*70}\nTRIAL {trial + 1}/{args.test_trials}\n{'='*70}")

        set_seed(args.seed + trial)

        print("\n=== Building Spiking RON ===")
        model = spiking_coESN_rescaled_II(
            n_inp=n_inp, n_hid=args.n_hid, dt=args.dt,
            gamma=gamma, epsilon=epsilon, rho=args.rho,
            input_scaling=args.inp_scaling,
            theta_lif=args.theta_lif, theta_rf=args.theta_rf,
            tau_filter=args.tau_filter,
            sparse_lif2hrf=use_sparse_lif2hrf,
            connectivity_lif2hrf=args.connectivity_lif2hrf,
            sparse_hrf2lif=use_sparse_hrf2lif,
            connectivity_hrf2lif=args.connectivity_hrf2lif,
            device=device,
            readout_mode=args.readout_mode,
        ).to(device)

        n_input_connections = apply_sparse_input_projection(
            model, args.input_density, n_inp, args.n_hid, device
        )
        all_n_input_connections.append(n_input_connections)
        print(f"Model created -- readout_mode='{args.readout_mode}'")

        # ── TIMED: feature extraction ──────────────────────────────────────
        print("\n=== Extracting Reservoir Features ===")
        t0 = time.time()
        train_feats, train_labels, r_tot_train, r_hrf_train, r_lif_train = extract_features(
            train_loader, model, device
        )
        print(f"  Train feature extraction: {time.time()-t0:.1f}s")
        print(f"  Training features: {train_feats.shape}")

        if args.use_test:
            t0 = time.time()
            test_feats, test_labels, r_tot_test, r_hrf_test, r_lif_test = extract_features(
                test_loader, model, device
            )
            print(f"  Test feature extraction:  {time.time()-t0:.1f}s")
            print(f"  Test features: {test_feats.shape}")
        else:
            test_feats, test_labels = train_feats, train_labels

        scaler      = preprocessing.StandardScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats  = scaler.transform(test_feats)

        # ── TIMED: logistic regression ─────────────────────────────────────
        print("\n=== Training Logistic Regression Readout ===")
        t0 = time.time()
        #clf = LogisticRegression(
            #max_iter=1000, verbose=0, n_jobs=1,
            #multi_class='multinomial', solver='lbfgs',
            #C=args.readout_C
        #).fit(train_feats, train_labels)

        clf = LogisticRegression(
            max_iter=1000, verbose=0, n_jobs=1,
            solver='liblinear',
            C=args.readout_C
            ).fit(train_feats, train_labels)
        print(f"  LR fit time: {time.time()-t0:.1f}s")

        train_acc = clf.score(train_feats, train_labels) * 100
        test_acc  = clf.score(test_feats,  test_labels)  * 100

        print(f"Train accuracy: {train_acc:.2f}%")
        print(f"Test accuracy:  {test_acc:.2f}%")
        print(f"r_hrf={r_hrf_train:.4f}, r_lif={r_lif_train:.4f}")

        snn_energy = estimate_snn_energy_sparse(
            r_hrf=r_hrf_train, r_lif=r_lif_train,
            n_hid=args.n_hid, T=seq_length,
            lif2hrf_connections=model.n_lif2hrf_connections,
            include_lif=True
        )
        print(f"Energy: {snn_energy['Energy_J']:.3e} J  |  "
              f"SOPs: {snn_energy['SOPs']:.3e}  "
              f"(HRF: {snn_energy['HRF_SOPs']:.3e}, LIF: {snn_energy['LIF_SOPs']:.3e})")

        all_test_accs.append(test_acc)
        all_train_accs.append(train_acc)
        all_energies.append(snn_energy['Energy_J'])
        all_sops.append(snn_energy['SOPs'])
        all_sops_hrf.append(snn_energy['HRF_SOPs'])
        all_sops_lif.append(snn_energy['LIF_SOPs'])
        all_r_hrf.append(r_hrf_train)
        all_r_lif.append(r_lif_train)
        all_r_total.append(r_tot_train)

    mean_test_acc  = np.mean(all_test_accs)
    std_test_acc   = np.std(all_test_accs)
    mean_train_acc = np.mean(all_train_accs)
    std_train_acc  = np.std(all_train_accs)
    mean_energy    = np.mean(all_energies)
    std_energy     = np.std(all_energies)
    mean_sops      = np.mean(all_sops)
    mean_sops_hrf  = np.mean(all_sops_hrf)
    mean_sops_lif  = np.mean(all_sops_lif)

    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset:        N-MNIST")
    print(f"Hidden units:   {args.n_hid}, Trials: {args.test_trials}")
    print(f"Readout mode:   {args.readout_mode}")
    print(f"Input channels: {n_inp}, Num steps: {args.num_steps}")
    print(f"Input density:  {args.input_density}")
    print(f"Train accuracy: {mean_train_acc:.2f}% +/- {std_train_acc:.2f}%")
    print(f"Test accuracy:  {mean_test_acc:.2f}%  +/- {std_test_acc:.2f}%")
    print(f"Per-trial test: {[f'{a:.2f}' for a in all_test_accs]}")
    print(f"Energy:         {mean_energy:.3e} +/- {std_energy:.3e} J")
    print(f"SOPs (total):   {mean_sops:.3e}  "
          f"(HRF: {mean_sops_hrf:.3e}, LIF: {mean_sops_lif:.3e})")
    print(f"{'='*70}")

    results = {
        'dataset': 'N-MNIST',
        'args': vars(args),
        'n_trials': args.test_trials,
        'n_inp': n_inp,
        'n_out': n_out,
        'num_steps': args.num_steps,
        'spatial_factor': args.spatial_factor,
        'readout_mode': args.readout_mode,
        'input_density': float(args.input_density),
        'readout_C': float(args.readout_C),
        'n_input_connections_mean': float(np.mean(all_n_input_connections)),
        'train_acc_mean': float(mean_train_acc),
        'train_acc_std':  float(std_train_acc),
        'test_acc_mean':  float(mean_test_acc),
        'test_acc_std':   float(std_test_acc),
        'test_accs_all':  [float(x) for x in all_test_accs],
        'r_hrf_mean':     float(np.mean(all_r_hrf)),
        'r_lif_mean':     float(np.mean(all_r_lif)),
        'r_tot_mean':     float(np.mean(all_r_total)),
        'energy_J_mean':  float(mean_energy),
        'energy_J_std':   float(std_energy),
        'sops_mean':      float(mean_sops),
        'sops_hrf_mean':  float(mean_sops_hrf),
        'sops_lif_mean':  float(mean_sops_lif),
        'n_lif2hrf_connections': int(model.n_lif2hrf_connections),
        'n_hrf2lif_connections': int(model.n_hrf2lif_connections),
        'connectivity_lif2hrf': float(args.connectivity_lif2hrf),
        'connectivity_hrf2lif': float(args.connectivity_hrf2lif),
        'n_hid':           int(args.n_hid),
        'base_seed':       int(args.seed),
        'sequence_length': int(seq_length),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    inp_str      = f"inp{args.input_density:.2f}" if args.input_density < 1.0 else "inpDense"
    conn_lif_str = "dense" if args.connectivity_lif2hrf == 1.0 else f"lif{args.connectivity_lif2hrf:.1f}"
    conn_hrf_str = "dense" if args.connectivity_hrf2lif == 1.0 else f"hrf{args.connectivity_hrf2lif:.1f}"
    results_filename = (
        f"results_nmnist_nhid{args.n_hid}_steps{args.num_steps}"
        f"_{inp_str}_{conn_lif_str}_{conn_hrf_str}"
        f"_{args.readout_mode}"
        f"_trials{args.test_trials}_seed{args.seed}.json"
    )
    results_path = os.path.join(args.results_dir, results_filename)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()