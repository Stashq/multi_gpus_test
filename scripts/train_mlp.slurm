#!/bin/bash

#SBATCH --job-name=multi_gpu_training
#SBATCH --time=00:06:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=clara
#SBATCH --gres=gpu:v100:3
#SBATCH --mem=96G
#SBATCH --cpus-per-task=3
#SBATCH -o /home/sc.uni-leipzig.de/sm589broe/logs/%x.out-%j
# #SBATCH --mail-type=END

module load PyTorch-Lightning/1.7.7-foss-2022a-CUDA-11.7.0
export PYTHONPATH="/home/sc.uni-leipzig.de/zw82iqes/multi_gpus_test:$PYTHONPATH"

srun python src/runs/train_mlp.py
