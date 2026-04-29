#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J memorization_figure2_and_3
#BSUB -n 32
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -R "select[gpu32gb]"
#BSUB -W 24:00
#BSUB -R "rusage[mem=40GB]"
#BSUB -u s204164@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o batch_output/memorization_figure2_and_3_%J.out
#BSUB -e batch_output/memorization_figure2_and_3_%J.err

nvidia-smi
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source .venv/bin/activate
cd src/Training

python run_Sprites.py -n 512 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 -M Unet --n_steps 200000 --n_test_eval 1024 &
python run_Sprites.py -n 1024 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 -M Unet --n_steps 200000 --n_test_eval 1024 &
python run_Sprites.py -n 2048 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 -M Unet --n_steps 200000 --n_test_eval 1024 &
python run_Sprites.py -n 4096 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 -M Unet --n_steps 200000 --n_test_eval 1024 &
python run_Sprites.py -n 1024 -s 16 -LR 0.0001 -O Adam -W 8 -t -1 -M Unet --n_steps 200000 --n_test_eval 1024 &
python run_Sprites.py -n 1024 -s 16 -LR 0.0001 -O Adam -W 16 -t -1 -M Unet --n_steps 200000 --n_test_eval 1024 &
python run_Sprites.py -n 1024 -s 16 -LR 0.0001 -O Adam -W 64 -t -1 -M Unet --n_steps 200000 --n_test_eval 1024 &
python run_Sprites.py -n 1024 -s 16 -LR 0.0001 -O Adam -W 128 -t -1 -M Unet --n_steps 200000 --n_test_eval 1024 &
wait
