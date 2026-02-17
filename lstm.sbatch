#!/bin/bash
#SBATCH --partition=prb,insy,general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32768
#SBATCH --mail-type=END
#SBATCH --gres=gpu

source ~/.bashrc
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
module load devtoolset/7
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/envs/pytorch

srun python sMNIST_task.py --lstm --use_test


            
