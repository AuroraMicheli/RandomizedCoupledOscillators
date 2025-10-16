#!/bin/bash
#SBATCH --job-name=dummyTest
#SBATCH --partition=prb,insy,general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/RandomizedCoupledOscillators/result/spiking_ron.out
#SBATCH --error=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/RandomizedCoupledOscillators/result/spiking_ron.err

# --- Paths ---
CONTAINER=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/Containers/container.sif
PROJ_DIR=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/RandomizedCoupledOscillators
DATA_DIR=/tudelft.net/staff-bulk/ewi/insy/VisionLab/amicheli/RandomizedCoupledOscillators/data

FILE="$PROJ_DIR/spiking_ron.py"

#mkdir -p "$PROJ_DIR/Results" "$PROJ_DIR/Checkpoints"

# --- Binds ---
BINDS=()
BINDS+=("--bind" "$PROJ_DIR:$PROJ_DIR")
BINDS+=("--bind" "$DATA_DIR:$DATA_DIR")

# --- Load modules ---
# apptainer exec --nv "$CONTAINER" python -m pip install --user --upgrade certifi >/dev/null 2>&1 || true
# export SSL_CERT_FILE="$(apptainer exec --nv "$CONTAINER" python -c 'import certifi; print(certifi.where())')"
# export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
# export CURL_CA_BUNDLE="$SSL_CERT_FILE"


# --- Run the dummy test ---
srun apptainer exec --nv --pwd "$PROJ_DIR" "${BINDS[@]}" "$CONTAINER" \
    python -u "$FILE"