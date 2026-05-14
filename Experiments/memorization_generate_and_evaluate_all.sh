#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J generate_and_evaluate_all
#BSUB -n 12
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=20GB]"
##BSUB -u s204164@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o batch_output/generate_and_evaluate_all_%J.out
#BSUB -e batch_output/generate_and_evaluate_all_%J.err

nvidia-smi
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source .venv/bin/activate
cd src/Generation
python generate.py --generate_all_missing --saves_dir ../../Saves --device cuda:0 --Nsamples 100
cd ../Evaluation
python evaluate_all_experiments.py