#!/bin/bash
#SBATCH --partition=prb,insy,general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --output=results_npcifar10_%j.out
#SBATCH --error=results_npcifar10_%j.err

source ~/.bashrc
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
module load devtoolset/7
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/envs/pytorch

srun python npCIFAR10_spiking_ron.py --n_hid 800 --dt 0.055 --gamma 0.1152 \
 --gamma_range 1.42 --epsilon 0.1779 --epsilon_range 0.963 --rho 0.99 \
 --inp_scaling 0.0583 --theta_lif 0.1242 --theta_rf 0.1535 \
  --tau_filter 5.0 --connectivity_lif2hrf 0.2 --connectivity_hrf2lif 1.0 \
  --readout_C 0.01 --readout_mode rms_std_final --test_trials 1 \
  --seed 42 --use_test


            
