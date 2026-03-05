#!/usr/bin/env bash
#
# reproduce.sh — Run all PLC simulation experiments.
#
# Total runtime: ~2 hours with GPU, ~6-8 hours CPU-only.
# All results are written to the results/ directory as JSON.
#
# Usage:
#   bash reproduce.sh           # Full reproduction
#   bash reproduce.sh --no-gpu  # CPU-only mode
#

set -euo pipefail

GPU_FLAG=""
if [[ "${1:-}" == "--no-gpu" ]]; then
    GPU_FLAG="--no-gpu"
    echo "Running in CPU-only mode."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p results

echo "========================================"
echo "  PLC Simulation: Full Reproduction"
echo "  $(date)"
echo "========================================"
echo

TOTAL_START=$SECONDS

# 1. Core experiments (Experiments 1-5)
echo "[1/9] Core experiments (main.py) ..."
START=$SECONDS
python main.py $GPU_FLAG
echo "       Done in $(( SECONDS - START ))s"
echo

# 2. Hardened statistics (Experiments 2 and 5 with bootstrap CIs)
echo "[2/9] Hardened statistics (run_hardened.py) ..."
START=$SECONDS
python run_hardened.py $GPU_FLAG
echo "       Done in $(( SECONDS - START ))s"
echo

# 3. Control experiments (1D chain, Haar random, planted partition)
echo "[3/9] Control experiments (run_controls.py) ..."
START=$SECONDS
python run_controls.py $GPU_FLAG
echo "       Done in $(( SECONDS - START ))s"
echo

# 4. Observable sweep (CMI, tripartite information, entanglement spectrum)
echo "[4/9] Observable sweep (run_observables.py) ..."
START=$SECONDS
python run_observables.py $GPU_FLAG
echo "       Done in $(( SECONDS - START ))s"
echo

# 5. Circularity-breaking tests
echo "[5/9] Circularity breaking (run_circularity.py) ..."
START=$SECONDS
python run_circularity.py $GPU_FLAG
echo "       Done in $(( SECONDS - START ))s"
echo

# 6. Symmetry-breaking control
echo "[6/9] Symmetry-breaking control (run_symmetry_breaking.py) ..."
START=$SECONDS
python run_symmetry_breaking.py $GPU_FLAG
echo "       Done in $(( SECONDS - START ))s"
echo

# 7. Null model battery
echo "[7/9] Null model battery (run_null_models.py) ..."
START=$SECONDS
python run_null_models.py $GPU_FLAG
echo "       Done in $(( SECONDS - START ))s"
echo

# 8. Distance robustness
echo "[8/9] Distance robustness (run_distance_robustness.py) ..."
START=$SECONDS
python run_distance_robustness.py $GPU_FLAG
echo "       Done in $(( SECONDS - START ))s"
echo

# 9. Scaling and large-N
echo "[9/9] Scaling study (run_scaling.py + run_large_n.py) ..."
START=$SECONDS
python run_scaling.py $GPU_FLAG
python run_large_n.py $GPU_FLAG
echo "       Done in $(( SECONDS - START ))s"
echo

TOTAL=$(( SECONDS - TOTAL_START ))
echo "========================================"
echo "  All experiments complete."
echo "  Total time: ${TOTAL}s ($(( TOTAL / 60 ))m $(( TOTAL % 60 ))s)"
echo "  Results in: results/"
echo "========================================"
