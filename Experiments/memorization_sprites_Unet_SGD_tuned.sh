#!/bin/sh
### General options
#BSUB -q gpuv100
#BSUB -J memorization_sprites_unet_sgd_tuned
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=20GB]"
#BSUB -u s204164@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o batch_output/memorization_sprites_Unet_SGD_tuned_%J.out
#BSUB -e batch_output/memorization_sprites_Unet_SGD_tuned_%J.err

nvidia-smi
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source .venv/bin/activate
cd src/Training

python run_Sprites.py \
  -n 1024 \
  -s 16 \
  -LR 0.003 \
  -O SGD_Momentum \
  -W 32 \
  -B 512 \
  -t -1 \
  -M Unet \
  --n_steps 200000 \
  --momentum 0.9 \
  --nesterov \
  --scheduler cosine \
  --warmup_steps 10000 \
  --min_lr_ratio 0.02 \
  --grad_clip 1.0 \
  --tag sgdcos_lr3e-3_warm10k
