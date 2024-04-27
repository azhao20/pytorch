#!/bin/bash
#SBATCH -c 16
#SBATCH -t 0-12:00
#SBATCH -p gpu_test
#SBATCH --mem=256000
#SBATCH --gres=gpu:1
#SBATCH -o timm.%j.out
#SBATCH -e timm.%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu

module load python/3.10.12-fasrc01
module load gcc/9.5.0-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load cmake

HOME=/n/holylabs/LABS/idreos_lab/Users/azhao

mamba deactivate
mamba activate $HOME/env

python3 $HOME/pytorch/benchmarks/dynamo/timm_models.py --performance --cold-start-latency --training --amp --backend inductor --device cuda