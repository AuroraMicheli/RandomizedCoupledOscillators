#!/bin/bash

# ==== PATHS ====
CONTAINER=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/Containers/container.sif
FILE=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/RandomizedCoupledOscillators/spiking_ron_out_li.py
RESULTSFOLDER=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/RandomizedCoupledOscillators/result/shell

# ==== SETUP ====
export CUDA_VISIBLE_DEVICES=0

echo "Running the script $FILE in the container $CONTAINER"

# Ensure result folder exists
mkdir -p $RESULTSFOLDER

# Log file has the same base name as the Python file, with .log extension
LOGFILE=$RESULTSFOLDER/$(basename "$FILE" .py).log

# ==== RUN ====
nohup apptainer exec --nv $CONTAINER python -u $FILE \
    --n_hid 256 \
    --batch 256 \
    --dt 0.042 \
    --gamma 2.7 \
    --epsilon 4.7 \
    --gamma_range 2.7 \
    --epsilon_range 4.7 \
    --inp_scaling 1.0 \
    --rho 0.99 \
    --use_test \
    >> $LOGFILE 2>&1 &

# ==== LOG INFO ====
PID=$!
echo "Started job with PID: $PID"
echo $PID > "$RESULTSFOLDER/job.pid"
echo "Logging to: $LOGFILE"
echo "To monitor output live, run:"
echo ""
echo "  tail -f $LOGFILE"
echo ""
