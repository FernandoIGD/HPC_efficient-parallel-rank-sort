#!/bin/bash
#SBATCH --job-name=ranking_sort
#SBATCH --output=logs/ranking_%j.out
#SBATCH --error=logs/ranking_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=256M
#SBATCH --time=00:30:00

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch --ntasks=<p> ranking_job.sh <repetitions>" >&2
    exit 2
fi

message_sizes=(3600 7200 10800 14400 18000)
repetitions=$1

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: repetitions must be a positive integer." >&2
    exit 2
fi

case "$SLURM_NTASKS" in
    1) grid_side=1 ;;
    4) grid_side=2 ;;
    9) grid_side=3 ;;
    16) grid_side=4 ;;
    25) grid_side=5 ;;
    36) grid_side=6 ;;
    49) grid_side=7 ;;
    64) grid_side=8 ;;
    *)
        echo "Error: process count must be a supported perfect square: 1, 4, 9, 16, 25, 36, 49, or 64." >&2
        exit 2
        ;;
esac

cd "$SLURM_SUBMIT_DIR"

module purge
module load gnu12/12.4.0 openmpi4/4.1.6

echo "job_id=$SLURM_JOB_ID"
echo "partition=$SLURM_JOB_PARTITION"
echo "nodes=$SLURM_JOB_NODELIST"
echo "processes=$SLURM_NTASKS"
echo "repetitions=$repetitions"
echo "source_sha256=$(sha256sum main.cpp | awk '{print $1}')"

for message_size in "${message_sizes[@]}"; do
    local_block_size=$((message_size / SLURM_NTASKS))
    effective_size=$((local_block_size * SLURM_NTASKS))
    aggregate_size=$((local_block_size * grid_side))

    if ((SLURM_NTASKS > 1 && aggregate_size >= 10000)); then
        echo "Error: this configuration exceeds the reference code's fixed receive buffer." >&2
        exit 2
    fi

    echo "case_start"
    echo "requested_size=$message_size"
    echo "effective_size=$effective_size"
    echo "local_block_size=$local_block_size"

    for ((run = 1; run <= repetitions; run++)); do
        echo "run=$run"
        mpirun -np "$SLURM_NTASKS" ./ranking_sort "$message_size"
    done

    echo "case_end"
done
