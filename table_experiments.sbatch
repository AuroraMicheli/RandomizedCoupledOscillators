#!/bin/bash
#SBATCH --partition=prb,insy,general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --output=%x_%j.log

source ~/.bashrc
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
module load devtoolset/7
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/envs/pytorch

echo "========================================================"
echo "Job:        $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node:       $SLURMD_NODENAME"
echo "Started:    $(date)"
echo "========================================================"

srun python sMNIST_spiking_ron.py \
    --use_test \
    --test_trials 3 \
    --n_hid 800 \
    --connectivity_lif2hrf 0.2 \
    --readout_mode mean


echo ""
echo "========================================================"
echo "Done: $(date)"
echo "========================================================"