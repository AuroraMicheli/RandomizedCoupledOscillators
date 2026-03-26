#!/bin/bash
#SBATCH --partition=prb,insy,general
#SBATCH --qos=long
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24576
#SBATCH --mail-type=END
#SBATCH --gres=gpu

source ~/.bashrc
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
module load devtoolset/7
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/envs/pytorch

# Pre-warm tonic cache before the search starts
# This downloads/processes N-MNIST once so all subprocess calls hit disk cache
python - << 'PYWARM'
import tonic, tonic.transforms as T, os
sensor_size = tonic.datasets.NMNIST.sensor_size
ft = T.ToFrame(sensor_size=sensor_size, n_time_bins=50)
os.makedirs('data/NMNIST', exist_ok=True)
tonic.datasets.NMNIST(save_to='data/NMNIST', train=True,  transform=ft)
tonic.datasets.NMNIST(save_to='data/NMNIST', train=False, transform=ft)
print("Tonic cache warm.")
PYWARM

srun python hyperparam_search_nMNIST.py