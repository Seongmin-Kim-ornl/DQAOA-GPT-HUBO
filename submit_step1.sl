#!/bin/bash
#SBATCH -A STF246
#SBATCH -J dg_8nodes
#SBATCH -p batch-gpu
#SBATCH -N 1
#SBATCH -t 23:59:00
#SBATCH -o GPT-%j.out

cd $SLURM_SUBMIT_DIR
date

module load PrgEnv-cray/8.6.0
source /ccsopen/home/gvk/defiant/CudaQ/CudaQ/bin/activate

for GPU in {0..7}; do
CUDA_VISIBLE_DEVICES=$GPU python step1_generate_data.py \
      --from_formulated --hubo_size 100 \
      --n_nodes 8 --hubo_number 5000 \
      --rounding_digits 1 --quiet --no_sa \
      --approx_ratio 0.9 \
      --output_dir adapt_results_hubo_8nodes \
      --seed_hubo $((42 + GPU * 5000)) \
      --checkpoint_every 20 &
done

wait
echo "All done."