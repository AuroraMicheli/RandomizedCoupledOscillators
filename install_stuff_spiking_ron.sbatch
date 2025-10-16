#!/bin/bash
#SBATCH --partition=prb,insy,general
#SBATCH --qos=medium
#SBATCH --job-name=install_spiking_ron_env
#SBATCH --output=conda_install.log
#SBATCH --error=conda_install.err
#SBATCH --time=30:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# -------------------------------
# 1. Load modules
# -------------------------------
module use /opt/insy/modulefiles
module load miniconda/3.9
module load cuda/12.4 cudnn/12-8.9.1.23
module load devtoolset/7

# -------------------------------
# 2. Activate Conda environment
# -------------------------------
source $HOME/.bashrc   # ensure conda is initialized
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/envs/spiking_ron

# -------------------------------
# 3. Define helper function with retry logic
# -------------------------------
run_conda_install () {
    local attempt=1
    local max_attempts=3
    local delay=120  # seconds between retries

    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt: $@"
        conda install -y $@ && return 0

        echo "⚠️  Conda installation failed on attempt $attempt. Retrying in $delay seconds..."
        sleep $delay
        attempt=$((attempt+1))
    done

    echo "❌ Installation failed after $max_attempts attempts: $@" >&2
    exit 1
}

# -------------------------------
# 4. Install packages (with retries)
# -------------------------------
echo "🚀 Starting installation process..."

# Core PyTorch + CUDA packages
run_conda_install pytorch=1.12.1 torchvision=0.13.1 torchtext=0.13.1 pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge

# Scientific stack
run_conda_install numpy=1.21.6 scipy=1.7.3 scikit-learn=1.0.2 matplotlib=3.5.3 tqdm=4.64.1 -c conda-forge

# Aeon (timeseries library)
run_conda_install aeon -c conda-forge

echo "✅ All packages installed successfully!"
