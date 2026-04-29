nvidia-smi
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
