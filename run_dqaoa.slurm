#!/bin/bash
#SBATCH -A STF246
#SBATCH -J DQAOA-GPT
#SBATCH -p batch-gpu
#SBATCH -N 1
#SBATCH -t 23:59:00
#SBATCH -o slurm-DQAOA-%j.out

unset SLURM_EXPORT_ENV

cd $SLURM_SUBMIT_DIR
date

module load PrgEnv-cray/8.6.0
source /ccsopen/home/gvk/defiant/CudaQ/CudaQ/bin/activate

declare -A CKPT_MAP
CKPT_MAP[4]="nanoGPT/out-hubo_4nodes/ckpt_5000.pt"
CKPT_MAP[6]="nanoGPT/out-hubo_6nodes/ckpt_5000.pt"
CKPT_MAP[8]="nanoGPT/out-hubo_8nodes/ckpt_5000.pt"
CKPT_MAP[10]="nanoGPT/out-hubo_10nodes/ckpt_5000.pt"
CKPT_MAP[12]="nanoGPT/out-hubo_12nodes/ckpt_5000.pt"

mkdir -p results

for SIZE in 100; do
for SUBN in 4 6 8 10 12; do
for CTN in 0 1 2 3 4; do
for ITER in 30 60 90 120 150 180; do

    CKPT=${CKPT_MAP[$SUBN]}
    echo "=== hubo_size=$SIZE  ctn=$CTN  sub_n=$SUBN  iterations=$ITER ==="

    echo "--- Running GPT solver (ckpt=$CKPT) ---"
    srun -n 1 python step4_solve_large_hubo.py \
        --solver gpt \
        --load_txt --hubo_size $SIZE --ctn $CTN --hubo_dir ./HUBOs \
        --ckpt $CKPT \
        --sub_n $SUBN \
        --iterations $ITER \
        --save_solution results/solution_gpt_size${SIZE}_ctn${CTN}_subn${SUBN}_iter${ITER}.npy

    echo "--- Running QAOA solver ---"
    srun -n 1 python step4_solve_large_hubo.py \
        --solver qaoa \
        --load_txt --hubo_size $SIZE --ctn $CTN --hubo_dir ./HUBOs \
        --sub_n $SUBN \
        --iterations $ITER \
        --save_solution results/solution_qaoa_size${SIZE}_ctn${CTN}_subn${SUBN}_iter${ITER}.npy

done
done
done
done

echo "All done."
