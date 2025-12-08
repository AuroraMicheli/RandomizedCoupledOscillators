#!/bin/bash
#SBATCH --partition=prb,insy,general
#SBATCH --qos=medium
#SBATCH --job-name=optuna_coesn
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=optuna_%A_%a.out
#SBATCH --error=optuna_%A_%a.err


module use /opt/insy/modulefiles
module load miniconda/3.9
module load cuda/12.4 cudnn/12-8.9.1.23
module load devtoolset/7

source $HOME/.bashrc
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/envs/pytorch

python experiment_optuna.py 

