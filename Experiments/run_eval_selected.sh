#!/usr/bin/env bash
set -euo pipefail

# Update these before running.
DEVICE="cuda:0"
BATCH_SAMPLE_SIZE=100
GAP_THRESHOLD="0.3333333333333333"
FID_STAT_ID=1
ISIC_DATA_ROOT="../../Data/ISIC_2019_Training_Input/ISIC_2019_Training_Input"

EXP_BASE="../../Saves"

cd src/Evaluation

get_n_batches() {
	local exp_dir="$1"
	local max_idx=-1
	local file
	while IFS= read -r -d '' file; do
		local base="${file##*/}"
		local idx="${base#samples_a_}"
		if [[ "$idx" =~ ^[0-9]+$ ]] && (( idx > max_idx )); then
			max_idx=$idx
		fi
	done < <(find "${exp_dir}/Samples" -type f -name 'samples_a_*' -print0 2>/dev/null)

	if (( max_idx < 0 )); then
		echo 0
	else
		echo $((max_idx + 1))
	fi
}

run_sprites_eval() {
	local exp_name="$1"
	local nbase="$2"
	local n_batches
	n_batches=$(get_n_batches "${EXP_BASE}/${exp_name}")
	if (( n_batches <= 0 )); then
		echo "Skipping ${exp_name} (no generated samples)"
		return
	fi

	python compute_fmem.py -D Sprites -n 1024 -s 16 -LR 0.0001 -O Adam -W "${nbase}" -B 512 -t -1 \
		--model_type unet --seed 1 --experiment_dir "${exp_name}" -Ns "${n_batches}" \
		--batch_sample_size "${BATCH_SAMPLE_SIZE}" --gap_threshold "${GAP_THRESHOLD}" --device "${DEVICE}"

	python compute_FID.py -D Sprites -n 1024 -s 16 -LR 0.0001 -O Adam -W "${nbase}" -B 512 -t -1 \
		--model_type unet --seed 1 --experiment_dir "${exp_name}" -istat "${FID_STAT_ID}" \
		--N1 0 --N2 "${n_batches}" --batch_size_samples "${BATCH_SAMPLE_SIZE}" --rebuild_stats --device "${DEVICE}"
}

run_isic_eval() {
	local exp_name="$1"
	local nbase="$2"
	local n_batches
	n_batches=$(get_n_batches "${EXP_BASE}/${exp_name}")
	if (( n_batches <= 0 )); then
		echo "Skipping ${exp_name} (no generated samples)"
		return
	fi

	python compute_fmem.py -D ISIC -n 1024 -s 16 -LR 0.0001 -O Adam -W "${nbase}" -B 512 -t -1 \
		--seed 1 --experiment_dir "${exp_name}" -Ns "${n_batches}" --batch_sample_size "${BATCH_SAMPLE_SIZE}" \
		--gap_threshold "${GAP_THRESHOLD}" --device "${DEVICE}" --isic_data_root "${ISIC_DATA_ROOT}"

	python compute_FID.py -D ISIC -n 1024 -s 16 -LR 0.0001 -O Adam -W "${nbase}" -B 512 -t -1 \
		--seed 1 --experiment_dir "${exp_name}" -istat "${FID_STAT_ID}" --N1 0 --N2 "${n_batches}" \
		--batch_size_samples "${BATCH_SAMPLE_SIZE}" --rebuild_stats --device "${DEVICE}" \
		--isic_data_root "${ISIC_DATA_ROOT}"
}

# Sprites experiments (unet)
run_sprites_eval "unet_Sprites16_1024_8_Adam_512_0.0001_seed1" 8
run_sprites_eval "unet_Sprites16_1024_16_Adam_512_0.0001_seed1" 16
run_sprites_eval "unet_Sprites16_1024_64_Adam_512_0.0001_seed1" 64

# ISIC experiment
run_isic_eval "ISIC16_1024_32_Adam_512_0.0001_seed1" 32
