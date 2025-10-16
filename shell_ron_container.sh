#!/bin/bash

#/home/acastanedagarc/Projects/Ideas_Exploration/UnbalanceUnsupervised/script.sh
CONTAINER=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/Containers/container.sif
FILE=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/RandomizedCoupledOscillators/sMNIST_task_try.py
RESULTSFOLDER=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/RandomizedCoupledOscillators/result/shell
MODELSFOLDER=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/dummy/models

export CUDA_VISIBLE_DEVICES=0

echo "Running the script $FILE in the container $CONTAINER" 

# Check if the folder exists; if not, create it
mkdir -p $RESULTSFOLDER

nohup apptainer exec --nv $CONTAINER python -u $FILE "$@" >> $RESULTSFOLDER/ron_sMNIST.log 2>&1 &
#nohup apptainer exec --nv $CONTAINER python -u $FILE "$@" &



PID=$!
echo "Started job with PID: $PID"sh shell_ron_c 
echo $PID > "$RESULTSFOLDER/ron_sMNIST.pid"