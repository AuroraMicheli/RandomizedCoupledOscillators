#!/bin/bash
#SBATCH --partition=prb,insy,general
#SBATCH --qos=medium
#SBATCH --time=8:00:00
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

# Configuration
RESULTS_DIR="results_sparse_connectivity"
mkdir -p $RESULTS_DIR

# Connectivity levels to test
CONNECTIVITIES=(1.0 0.8 0.6 0.4 0.2 0.1)

# Hidden units configurations
N_HIDS=(256 800)

# Seeds for multiple runs
SEEDS=(42 123 456)

echo "======================================"
echo "SPARSE CONNECTIVITY EXPERIMENTS"
echo "======================================"
echo "Results directory: $RESULTS_DIR"
echo "Connectivity levels: ${CONNECTIVITIES[@]}"
echo "Hidden units: ${N_HIDS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "======================================"
echo ""

# Loop over hidden units
for N_HID in "${N_HIDS[@]}"; do
    echo "=========================================="
    echo "TESTING N_HID = $N_HID"
    echo "=========================================="
    
    # Loop over connectivity levels
    for CONN in "${CONNECTIVITIES[@]}"; do
        echo ""
        echo "--- Connectivity: $CONN ---"
        
        # Determine if sparse
        if (( $(echo "$CONN < 1.0" | bc -l) )); then
            SPARSE_FLAG="--sparse_lif2hrf"
        else
            SPARSE_FLAG=""
        fi
        
        # Loop over seeds
        for SEED in "${SEEDS[@]}"; do
            echo "  Running seed $SEED..."
            
            srun python sparse_connectivity_lif_hrf_cluster.py \
                --n_hid $N_HID \
                --connectivity $CONN \
                $SPARSE_FLAG \
                --seed $SEED \
                --results_dir $RESULTS_DIR \
                --use_test
            
            if [ $? -eq 0 ]; then
                echo "  ✅ Completed: n_hid=$N_HID, conn=$CONN, seed=$SEED"
            else
                echo "  ❌ Failed: n_hid=$N_HID, conn=$CONN, seed=$SEED"
            fi
        done
    done
    echo ""
done

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "======================================"
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "To analyze results, run:"
echo "python analyze_results.py --results_dir $RESULTS_DIR"