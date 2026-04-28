#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J figure2_experiments_celebA_ISIC
#BSUB -n 12
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -W 24:00
#BSUB -R "rusage[mem=20GB]"
##BSUB -u s204164@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o batch_output/figure2_experiments_celebA_ISIC_%J.out
#BSUB -e batch_output/figure2_experiments_celebA_ISIC_%J.err

nvidia-smi
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source .venv/bin/activate
cd src/Training

python run_Unet.py -n 512 -i 0 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 --n_steps 200000 &
python run_Unet.py -n 1024 -i 0 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 --n_steps 200000 &
python run_Unet.py -n 2048 -i 0 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 --n_steps 200000 &
python run_Unet.py -n 4096 -i 0 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 --n_steps 200000 &
python run_ISIC.py -n 512 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 --n_steps 200000 &
python run_ISIC.py -n 1024 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 --n_steps 200000 &
python run_ISIC.py -n 2048 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 --n_steps 200000 &
python run_ISIC.py -n 4096 -s 16 -LR 0.0001 -O Adam -W 32 -t -1 --n_steps 200000 &
wait
