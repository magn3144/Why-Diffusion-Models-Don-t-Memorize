#!/usr/bin/env bash
### General options
#BSUB -q gpuv100
#BSUB -J generate_all_samples
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "rusage[mem=20GB]"
#BSUB -u s204164@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o batch_output/generate_all_samples_%J.out
#BSUB -e batch_output/generate_all_samples_%J.err

nvidia-smi
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source .venv/bin/activate

set -euo pipefail

GEN_DIR="../src/Generation"

NSAMPLES="100"
DEVICE="cuda:0"

cd "${GEN_DIR}"

echo "Generating for CelebA32_1024_32_Adam_512_0.0001_index0"
python generate.py \
  -D CelebA \
  -s 32 \
  -n 1024 \
  -W 32 \
  -O Adam \
  -B 512 \
  -LR 0.0001 \
  -i 0 \
  -Ns "${NSAMPLES}" \
  --device "${DEVICE}"

echo "Generating for gmm_Sprites16_128_128_SGD_Momentum_128_0.0001_seed1"
python generate.py \
  -D Sprites \
  -s 16 \
  -n 128 \
  -W 128 \
  -O SGD_Momentum \
  -B 128 \
  -LR 0.0001 \
  -Ns "${NSAMPLES}" \
  --model_type gmm \
  --seed 1 \
  --device "${DEVICE}"

echo "Generating for gmm_Sprites16_4096_128_Adam_512_0.0006_seed1"
python generate.py \
  -D Sprites \
  -s 16 \
  -n 4096 \
  -W 128 \
  -O Adam \
  -B 512 \
  -LR 0.0006 \
  -Ns "${NSAMPLES}" \
  --model_type gmm \
  --seed 1 \
  --device "${DEVICE}"

echo "Generating for gmm_Sprites16_4096_128_SGD_Momentum_512_0.0006_seed1"
python generate.py \
  -D Sprites \
  -s 16 \
  -n 4096 \
  -W 128 \
  -O SGD_Momentum \
  -B 512 \
  -LR 0.0006 \
  -Ns "${NSAMPLES}" \
  --model_type gmm \
  --seed 1 \
  --device "${DEVICE}"

echo "Generating for gmm_Sprites16_4096_512_Adam_512_0.0006_seed1"
python generate.py \
  -D Sprites \
  -s 16 \
  -n 4096 \
  -W 512 \
  -O Adam \
  -B 512 \
  -LR 0.0006 \
  -Ns "${NSAMPLES}" \
  --model_type gmm \
  --seed 1 \
  --device "${DEVICE}"

echo "Generating for unet_Sprites16_1024_32_Adam_512_0.0001_seed1"
python generate.py \
  -D Sprites \
  -s 16 \
  -n 1024 \
  -W 32 \
  -O Adam \
  -B 512 \
  -LR 0.0001 \
  -Ns "${NSAMPLES}" \
  --model_type unet \
  --seed 1 \
  --device "${DEVICE}"

echo "Generating for unet_Sprites16_1024_32_SGD_Momentum_512_0.0001_seed1"
python generate.py \
  -D Sprites \
  -s 16 \
  -n 1024 \
  -W 32 \
  -O SGD_Momentum \
  -B 512 \
  -LR 0.0001 \
  -Ns "${NSAMPLES}" \
  --model_type unet \
  --seed 1 \
  --device "${DEVICE}"

echo "Generating for unet_Sprites16_1024_32_SGD_Momentum_512_0.0030_seed1_sgdcos_lr3e-3_warm10k"
python generate.py \
  -D Sprites \
  -s 16 \
  -n 1024 \
  -W 32 \
  -O SGD_Momentum \
  -B 512 \
  -LR 0.0030 \
  -Ns "${NSAMPLES}" \
  --model_type unet \
  --seed 1 \
  --tag sgdcos_lr3e-3_warm10k \
  --device "${DEVICE}"

echo "Skipping GMM8_4096_128_Adam_1_0.0006_B512_t-1 (no generator entry)"
