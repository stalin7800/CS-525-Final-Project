#!/usr/bin/env bash
#SBATCH -A cs525
#SBATCH -p academic 
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH --gres=gpu:1
#SBATCH -C A30  
#SBATCH --mem 12g 
#SBATCH --job-name="Train pacman" 

source activate p4
python main.py --train_dqn --data_dir "pacmantest/" --model_name "noisy50.pth" --epsilon_decay_steps 500000 --prioritized_beta_increment 0.0001 --n_step 5 --episodes 50000 --max_buffer_size 100000 --buffer_start 50000