"""
Data loading utilities for:
  A) UCR Time Series Classification datasets: FordB, Wafer, Earthquakes
  B) Neuromorphic/Spiking datasets: SHD, N-MNIST, DVS Gesture

UCR datasets: https://www.timeseriesclassification.com/
Neuromorphic datasets:
  - SHD: download .h5.gz from https://zenkelab.org/datasets/
  - N-MNIST: use tonic (pip install tonic)
  - DVS Gesture: use tonic (pip install tonic)
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from pathlib import Path


# =============================================================================
# A) UCR DATASET UTILITIES
# =============================================================================

class UCR_dataset(data.Dataset):
    """Generic UCR dataset loader. Returns (time_series, label) pairs."""
    def __init__(self, mydata):
        self.mydata = mydata

    def __getitem__(self, idx):
        sample = self.mydata[idx]
        idx_inp, idx_targ = sample[0], sample[1]
        idx_inp = torch.Tensor(idx_inp)
        idx_targ = torch.Tensor([idx_targ])
        idx_inp = idx_inp.reshape(idx_inp.shape[0], 1)
        return idx_inp, idx_targ

    def __len__(self):
        return len(self.mydata)


def load_ucr_tsv(dataset_name, data_dir='data'):
    """Load a UCR dataset from TSV/TXT files."""
    base = Path(data_dir) / dataset_name
    for ext in ['.tsv', '.txt']:
        train_file = base / f"{dataset_name}_TRAIN{ext}"
        test_file = base / f"{dataset_name}_TEST{ext}"
        if train_file.exists() and test_file.exists():
            break
    if not train_file.exists():
        raise FileNotFoundError(
            f"Could not find {dataset_name}_TRAIN.tsv or .txt in {base}/. "
            f"Download from https://www.timeseriesclassification.com/ "
            f"and place in {base}/"
        )
    delim = '\t' if train_file.suffix == '.tsv' else None
    train_data = np.loadtxt(str(train_file), delimiter=delim)
    test_data = np.loadtxt(str(test_file), delimiter=delim)
    y_train, x_train = train_data[:, 0], train_data[:, 1:]
    y_test, x_test = test_data[:, 0], test_data[:, 1:]
    return x_train, y_train, x_test, y_test


def make_ucr_loaders(dataset_name, data_dir='data', batch_train=120, batch_test=120,
                     label_map=None, whole_train=True):
    """Create PyTorch DataLoaders for a UCR dataset."""
    x_train, y_train, x_test, y_test = load_ucr_tsv(dataset_name, data_dir)
    if label_map is not None:
        y_train = np.array([label_map[int(y)] for y in y_train])
        y_test = np.array([label_map[int(y)] for y in y_test])
    train_pairs = [(x_train[i], y_train[i]) for i in range(len(y_train))]
    test_pairs = [(x_test[i], y_test[i]) for i in range(len(y_test))]
    train_dataset = UCR_dataset(train_pairs)
    test_dataset = UCR_dataset(test_pairs)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_train, shuffle=True, drop_last=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_test, shuffle=False, drop_last=False)
    return train_loader, None, test_loader


def get_FordB_data(batch_train=120, batch_test=120, data_dir='data', whole_train=True):
    """FordB: Automotive engine noise (binary). Train:3636, Test:810, Length:500"""
    return make_ucr_loaders('FordB', data_dir=data_dir,
        batch_train=batch_train, batch_test=batch_test,
        label_map={-1: 0, 1: 1}, whole_train=whole_train)

def get_Wafer_data(batch_train=120, batch_test=120, data_dir='data', whole_train=True):
    """Wafer: Semiconductor wafer (binary, imbalanced). Train:1000, Test:6164, Length:152"""
    return make_ucr_loaders('Wafer', data_dir=data_dir,
        batch_train=batch_train, batch_test=batch_test,
        label_map={-1: 0, 1: 1}, whole_train=whole_train)

def get_Earthquakes_data(batch_train=120, batch_test=120, data_dir='data', whole_train=True):
    """Earthquakes: Seismology (binary, imbalanced). Train:322, Test:139, Length:512"""
    return make_ucr_loaders('Earthquakes', data_dir=data_dir,
        batch_train=batch_train, batch_test=batch_test,
        label_map=None, whole_train=whole_train)


# =============================================================================
# B) NEUROMORPHIC / SPIKING DATASET UTILITIES
# =============================================================================

# =============================================
# SHD (Spiking Heidelberg Digits)
# =============================================
# 20 classes (digits 0-9 in English + German), 700 input channels
# Train: 8332, Test: 2088. Max duration ~1.37s
#
# Download:
#   wget https://zenkelab.org/datasets/shd_train.h5.gz
#   wget https://zenkelab.org/datasets/shd_test.h5.gz
#   gunzip shd_train.h5.gz shd_test.h5.gz
#   mkdir -p data/SHD && mv shd_train.h5 shd_test.h5 data/SHD/

class SHD_dataset(data.Dataset):
    """
    Memory-efficient SHD dataset.
    Stores spike events in sparse form (bin_indices, unit_indices per sample).
    Converts to dense (num_steps, n_channels) on-the-fly in __getitem__.
    
    Memory: ~50 MB for sparse events vs ~6.5 GB for dense frames.
    """
    def __init__(self, h5_path, num_steps=250, n_channels=700, max_time=1.4):
        import h5py
        
        self.num_steps = num_steps
        self.n_channels = n_channels
        self.max_time = max_time
        
        # Store sparse events: list of (bin_indices, unit_indices) per sample
        self.events = []
        self.labels = []
        
        with h5py.File(h5_path, 'r') as f:
            spike_times = f['spikes']['times']
            spike_units = f['spikes']['units']
            labels = f['labels'][()]
            
            for i in range(len(labels)):
                times = spike_times[i][()]
                units = spike_units[i][()]
                
                # Pre-compute bin indices (cheap, small arrays)
                bins = np.floor(times / max_time * num_steps).astype(np.int32)
                bins = np.clip(bins, 0, num_steps - 1)
                unit_ids = units.astype(np.int32)
                
                self.events.append((bins, unit_ids))
                self.labels.append(int(labels[i]))
        
        print(f"    Loaded {len(self.labels)} samples (sparse), "
              f"~{sum(len(e[0]) for e in self.events) * 8 / 1e6:.1f} MB in memory")

    def __getitem__(self, idx):
        bins, units = self.events[idx]
        
        # Convert sparse → dense on-the-fly
        frame = np.zeros((self.num_steps, self.n_channels), dtype=np.float32)
        np.add.at(frame, (bins, units), 1.0)
        
        x = torch.FloatTensor(frame)            # (num_steps, n_channels)
        y = torch.LongTensor([self.labels[idx]]) # (1,)
        return x, y

    def __len__(self):
        return len(self.labels)


def get_SHD_data(batch_train=128, batch_test=256, data_dir='data/SHD',
                 num_steps=250, max_time=1.4, whole_train=True, force_reload=False):
    """
    SHD: Spiking Heidelberg Digits (20-class spoken digit classification).
    - 700 input channels (cochlear spike trains)
    - Train: 8332, Test: 2088
    - Binned into num_steps time bins over max_time seconds
    
    Returns:
        train_loader, valid_loader (None), test_loader
    """
    train_h5 = Path(data_dir) / 'shd_train.h5'
    test_h5 = Path(data_dir) / 'shd_test.h5'
    
    if not train_h5.exists():
        raise FileNotFoundError(
            f"Could not find {train_h5}. Download from:\n"
            f"  wget https://zenkelab.org/datasets/shd_train.h5.gz\n"
            f"  wget https://zenkelab.org/datasets/shd_test.h5.gz\n"
            f"  gunzip shd_train.h5.gz shd_test.h5.gz\n"
            f"  Place in {data_dir}/"
        )
    
    print(f"  Loading SHD train set (num_steps={num_steps})...")
    train_dataset = SHD_dataset(str(train_h5), num_steps, 700, max_time)
    print(f"  Loading SHD test set (num_steps={num_steps})...")
    test_dataset = SHD_dataset(str(test_h5), num_steps, 700, max_time)
    
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_train, shuffle=True,
        drop_last=False, num_workers=0
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_test, shuffle=False,
        drop_last=False, num_workers=0
    )
    
    return train_loader, None, test_loader


# =============================================
# N-MNIST (Neuromorphic MNIST)
# =============================================
# 10 classes, 34x34 DVS sensor, 2 polarities
# Train: 60000, Test: 10000. Duration: ~300ms
# Requires: pip install tonic

class SpikingDataset(data.Dataset):
    """Generic dense spiking dataset (for cached/preprocessed data)."""
    def __init__(self, frames_list, labels_list):
        self.frames = frames_list
        self.labels = labels_list
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.frames[idx])
        y = torch.LongTensor([self.labels[idx]])
        return x, y
    def __len__(self):
        return len(self.labels)


def preprocess_nmnist_with_tonic(save_dir='data/NMNIST', num_steps=300,
                                  sensor_size=(34, 34, 2), merge_polarities=False):
    """Use tonic to download N-MNIST and bin into dense frames."""
    try:
        import tonic
        import tonic.transforms as transforms
    except ImportError:
        raise ImportError("tonic required: pip install tonic")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    cache_name = f'nmnist_{"merged_" if merge_polarities else ""}steps{num_steps}'
    
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=num_steps),
    ])
    
    for split, train in [('train', True), ('test', False)]:
        cache_file = save_path / f'{cache_name}_{split}.pt'
        if cache_file.exists():
            print(f"  Cache exists: {cache_file}")
            continue
        print(f"  Preprocessing N-MNIST {split} (num_steps={num_steps})...")
        dataset = tonic.datasets.NMNIST(
            save_to=str(save_dir), train=train, transform=frame_transform
        )
        frames_list, labels_list = [], []
        for i in range(len(dataset)):
            if i % 5000 == 0:
                print(f"    {i}/{len(dataset)}")
            frame, label = dataset[i]
            frame = frame.reshape(num_steps, -1).astype(np.float32)
            if merge_polarities:
                frame_4d = frame.reshape(num_steps, sensor_size[2],
                                         sensor_size[0], sensor_size[1])
                frame = frame_4d.sum(axis=1).reshape(num_steps, -1)
            frames_list.append(frame)
            labels_list.append(int(label))
        torch.save({'frames': frames_list, 'labels': labels_list}, str(cache_file))
    print("  N-MNIST preprocessing complete!")


def get_NMNIST_data(batch_train=128, batch_test=256, data_dir='data/NMNIST',
                    num_steps=300, merge_polarities=False, whole_train=True):
    """N-MNIST: 10-class, 34x34x2 DVS. Train:60000, Test:10000."""
    save_path = Path(data_dir)
    cache_name = f'nmnist_{"merged_" if merge_polarities else ""}steps{num_steps}'
    cache_train = save_path / f'{cache_name}_train.pt'
    cache_test = save_path / f'{cache_name}_test.pt'
    if not cache_train.exists():
        preprocess_nmnist_with_tonic(save_dir=data_dir, num_steps=num_steps,
                                      merge_polarities=merge_polarities)
    print(f"  Loading N-MNIST from cache (num_steps={num_steps})...")
    train_data = torch.load(str(cache_train), map_location='cpu')
    test_data = torch.load(str(cache_test), map_location='cpu')
    train_dataset = SpikingDataset(train_data['frames'], train_data['labels'])
    test_dataset = SpikingDataset(test_data['frames'], test_data['labels'])
    train_loader = data.DataLoader(train_dataset, batch_size=batch_train, shuffle=True, drop_last=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_test, shuffle=False, drop_last=False)
    return train_loader, None, test_loader


# =============================================
# DVS Gesture
# =============================================
# 11 classes, 128x128 DVS → spatially downsampled
# Train: 1077, Test: 264. Requires: pip install tonic

def preprocess_dvs_gesture_with_tonic(save_dir='data/DVSGesture', num_steps=250,
                                       spatial_ds=16, merge_polarities=False):
    """Use tonic to download DVS Gesture and bin into dense frames."""
    try:
        import tonic
        import tonic.transforms as transforms
    except ImportError:
        raise ImportError("tonic required: pip install tonic")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    ds_factor = 128 // spatial_ds
    cache_name = f'dvsgesture_ds{spatial_ds}_{"merged_" if merge_polarities else ""}steps{num_steps}'
    
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.Downsample(spatial_factor=1.0 / ds_factor),
        transforms.ToFrame(sensor_size=(spatial_ds, spatial_ds, 2), n_time_bins=num_steps),
    ])
    
    for split, train in [('train', True), ('test', False)]:
        cache_file = save_path / f'{cache_name}_{split}.pt'
        if cache_file.exists():
            print(f"  Cache exists: {cache_file}")
            continue
        print(f"  Preprocessing DVS Gesture {split} (ds={spatial_ds}, steps={num_steps})...")
        dataset = tonic.datasets.DVSGesture(
            save_to=str(save_dir), train=train, transform=frame_transform
        )
        frames_list, labels_list = [], []
        for i in range(len(dataset)):
            if i % 100 == 0:
                print(f"    {i}/{len(dataset)}")
            frame, label = dataset[i]
            frame = frame.reshape(num_steps, -1).astype(np.float32)
            if merge_polarities:
                frame_4d = frame.reshape(num_steps, 2, spatial_ds, spatial_ds)
                frame = frame_4d.sum(axis=1).reshape(num_steps, -1)
            frames_list.append(frame)
            labels_list.append(int(label))
        torch.save({'frames': frames_list, 'labels': labels_list}, str(cache_file))
    print("  DVS Gesture preprocessing complete!")


def get_DVSGesture_data(batch_train=64, batch_test=64, data_dir='data/DVSGesture',
                        num_steps=250, spatial_ds=16, merge_polarities=False,
                        whole_train=True):
    """DVS Gesture: 11-class, 128x128→(ds×ds). Train:1077, Test:264."""
    save_path = Path(data_dir)
    cache_name = f'dvsgesture_ds{spatial_ds}_{"merged_" if merge_polarities else ""}steps{num_steps}'
    cache_train = save_path / f'{cache_name}_train.pt'
    cache_test = save_path / f'{cache_name}_test.pt'
    if not cache_train.exists():
        preprocess_dvs_gesture_with_tonic(save_dir=data_dir, num_steps=num_steps,
                                           spatial_ds=spatial_ds, merge_polarities=merge_polarities)
    print(f"  Loading DVS Gesture from cache (ds={spatial_ds}, steps={num_steps})...")
    train_data = torch.load(str(cache_train), map_location='cpu')
    test_data = torch.load(str(cache_test), map_location='cpu')
    train_dataset = SpikingDataset(train_data['frames'], train_data['labels'])
    test_dataset = SpikingDataset(test_data['frames'], test_data['labels'])
    train_loader = data.DataLoader(train_dataset, batch_size=batch_train, shuffle=True, drop_last=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_test, shuffle=False, drop_last=False)
    return train_loader, None, test_loader