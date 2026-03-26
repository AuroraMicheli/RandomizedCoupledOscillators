#!/bin/bash
#SBATCH --partition=prb,insy,general
#SBATCH --qos=long
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --mail-type=END
#SBATCH --gres=gpu

source ~/.bashrc
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
module load devtoolset/7
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/envs/pytorch

# Pre-warm DiskCachedDataset for all T values used in the search
# DVSGesture uses DiskCachedDataset so cache is built once per T value
# spatial_factor is applied in collate_fn so one cache covers all sf values
python - << 'PYWARM'
import tonic, tonic.transforms as T, os
from tonic import DiskCachedDataset

data_dir = 'data/DVSGesture'
os.makedirs(data_dir, exist_ok=True)
sensor_size = tonic.datasets.DVSGesture.sensor_size

for num_steps in [30, 50, 100, 200]:
    ft = T.ToFrame(sensor_size=sensor_size, n_time_bins=num_steps)
    for tag, train in [('train', True), ('test', False)]:
        ds_raw     = tonic.datasets.DVSGesture(save_to=data_dir, train=train, transform=ft)
        cache_path = os.path.join(data_dir, f'cache_{tag}_T{num_steps}_sf1')
        DiskCachedDataset(ds_raw, cache_path=cache_path)
        print(f"Warmed cache: {tag} T={num_steps}")

print("All caches warmed.")
PYWARM

srun python hyperparam_search_dvs-gesture_round2.py
