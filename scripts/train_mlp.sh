#!/bin/bash

#SBATCH --job-name=multi_gpu_training
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH -o /home/sc.uni-leipzig.de/sm589broe/logs/%x.out-%j
# #SBATCH --mail-type=END

module load PyTorch-Lightning/1.7.7-foss-2022a-CUDA-11.7.0

srun python src/runs/train_mlp.py
