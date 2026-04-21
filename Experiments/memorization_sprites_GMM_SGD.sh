#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J memorization_sprites
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the gpu resources:
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s204164@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o batch_output/memorization_sprites_GMM_SGD_%J.out
#BSUB -e batch_output/memorization_sprites_GMM_SGD_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source .venv/bin/activate
cd src/Training
python run_Sprites.py -n 4096 -s 16 --seed 1 -W 128 -O SGD_Momentum -B 512 -t -1 -M GMM -LR 0.0006 --n_steps 4000000