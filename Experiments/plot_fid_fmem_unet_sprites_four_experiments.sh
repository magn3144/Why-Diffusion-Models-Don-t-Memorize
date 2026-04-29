#!/bin/sh

set -eu

cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi

output_path="$PWD/Results/fid_fmem_unet_sprites_four_experiments.png"

python src/Evaluation/plot_fid_fmem.py \
    --experiment_dirs \
    Saves/unet_Sprites16_512_32_Adam_512_0.0001_seed1 \
    Saves/unet_Sprites16_1024_32_Adam_512_0.0001_seed1 \
    Saves/unet_Sprites16_2048_32_Adam_512_0.0001_seed1 \
    Saves/unet_Sprites16_4096_32_Adam_512_0.0001_seed1 \
    --labels n=512 n=1024 n=2048 n=4096 \
    --dataset_sizes 512 1024 2048 4096 \
    --title "Sprites UNet Adam: FID and f_mem vs tau" \
    --output "$output_path"