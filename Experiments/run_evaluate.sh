#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J run_evaluate
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 08:00
#BSUB -R "rusage[mem=20GB]"
#BSUB -u s204164@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o batch_output/run_evaluate_%J.out
#BSUB -e batch_output/run_evaluate_%J.err

nvidia-smi
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source .venv/bin/activate
cd src/Evaluation

python evaluate_all_experiments.py --device cuda:0