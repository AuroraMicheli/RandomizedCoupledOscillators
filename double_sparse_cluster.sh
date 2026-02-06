#!/bin/bash
#SBATCH --partition=prb,insy,general
#SBATCH --qos=long
#SBATCH --time=13:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16384
#SBATCH --mail-type=END
#SBATCH --gres=gpu

source ~/.bashrc
module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
module load devtoolset/7
conda activate /tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/envs/pytorch

# Configuration
RESULTS_DIR="results_double_sparse"
mkdir -p $RESULTS_DIR

# Targeted connectivity configurations (LIF→HRF HRF→HRF)
# Format: "CONN_LIF CONN_HRF"
CONFIGS=(
    #"1.0 1.0"   # Dense-Dense baseline
    #"0.8 0.8"   # Symmetric 80%
    #"0.5 0.5"   # Symmetric 50%
    #"0.1 0.1"   # Ultra-sparse
    #"1.0 0.2"   # Dense LIF, sparse HRF
    #"0.2 1.0"   # Sparse LIF, dense HRF
    ""
    "1.0 1.0"
    "1.0 0.8"
    "1.0 0.5"
    "1.0 0.1"
    "0.8 1.0"
    "0.8 0.5"
    "0.8 0.5"
    "0.8 0.2"
    "0.8 0.1"
    "0.5 1.0"
    "0.5 0.8"
    "0.5 0.2"
    "0.5 0.1"
    "0.2 0.8"
    "0.2 0.5"
    "0.2 0.2"
    "0.2 0.1"
    "0.1 1.0"
    "0.1 0.8"
    "0.1 0.5"
    "0.1 0.2"

)

# Hidden units configurations
N_HIDS=(256 800)

# Seeds for multiple runs
SEEDS=(42 123 456)

echo "======================================"
echo "TARGETED DOUBLE SPARSE CONNECTIVITY EXPERIMENTS"
echo "======================================"
echo "Testing specific configurations:"
for CONFIG in "${CONFIGS[@]}"; do
    CONN_LIF=$(echo $CONFIG | awk '{print $1}')
    CONN_HRF=$(echo $CONFIG | awk '{print $2}')
    echo "  - LIF→HRF: ${CONN_LIF}, HRF→HRF: ${CONN_HRF}"
done
echo "======================================"
echo "Results directory: $RESULTS_DIR"
echo "Hidden units: ${N_HIDS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "======================================"
echo "Total experiments: $((${#CONFIGS[@]} * ${#N_HIDS[@]} * ${#SEEDS[@]}))"
echo "  = ${#CONFIGS[@]} configs × ${#N_HIDS[@]} n_hid × ${#SEEDS[@]} seeds"
echo "======================================"
echo ""

# Counter for progress
TOTAL_EXP=$((${#CONFIGS[@]} * ${#N_HIDS[@]} * ${#SEEDS[@]}))
CURRENT=0

# Loop over hidden units
for N_HID in "${N_HIDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "TESTING N_HID = $N_HID"
    echo "=========================================="
    
    # Loop over targeted configurations
    for CONFIG in "${CONFIGS[@]}"; do
        # Extract connectivity values
        CONN_LIF=$(echo $CONFIG | awk '{print $1}')
        CONN_HRF=$(echo $CONFIG | awk '{print $2}')
        
        echo ""
        echo "--- Configuration: LIF→HRF=$CONN_LIF | HRF→HRF=$CONN_HRF ---"
        
        # Determine sparse flags
        if (( $(echo "$CONN_LIF < 1.0" | bc -l) )); then
            SPARSE_LIF_FLAG="--sparse_lif2hrf"
        else
            SPARSE_LIF_FLAG=""
        fi
        
        if (( $(echo "$CONN_HRF < 1.0" | bc -l) )); then
            SPARSE_HRF_FLAG="--sparse_hrf2hrf"
        else
            SPARSE_HRF_FLAG=""
        fi
        
        # Loop over seeds
        for SEED in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            echo "  [${CURRENT}/${TOTAL_EXP}] Running seed $SEED..."
            
            srun python sparse_connectivity_lif_hrf_double_sparse.py \
                --n_hid $N_HID \
                --connectivity_lif2hrf $CONN_LIF \
                --connectivity_hrf2hrf $CONN_HRF \
                $SPARSE_LIF_FLAG \
                $SPARSE_HRF_FLAG \
                --seed $SEED \
                --results_dir $RESULTS_DIR \
                --use_test
            
            if [ $? -eq 0 ]; then
                echo "  ✅ Completed: n_hid=$N_HID, lif=$CONN_LIF, hrf=$CONN_HRF, seed=$SEED"
            else
                echo "  ❌ Failed: n_hid=$N_HID, lif=$CONN_LIF, hrf=$CONN_HRF, seed=$SEED"
            fi
        done
    done
    echo ""
done

echo ""
echo "======================================"
echo "ALL TARGETED EXPERIMENTS COMPLETED"
echo "======================================"
echo "Completed: $CURRENT / $TOTAL_EXP experiments"
echo "Results saved in: $RESULTS_DIR"
echo ""
echo "Configurations tested:"
for CONFIG in "${CONFIGS[@]}"; do
    CONN_LIF=$(echo $CONFIG | awk '{print $1}')
    CONN_HRF=$(echo $CONFIG | awk '{print $2}')
    LIF_PCT=$(echo "$CONN_LIF * 100" | bc)
    HRF_PCT=$(echo "$CONN_HRF * 100" | bc)
    echo "  - LIF→HRF: ${LIF_PCT}%, HRF→HRF: ${HRF_PCT}%"
done
echo ""
echo "To analyze results, run:"
echo "python analyze_double_sparse_results.py --results_dir $RESULTS_DIR"
echo "======================================"