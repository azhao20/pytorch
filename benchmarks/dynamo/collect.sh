#!/bin/bash
#SBATCH -c 64
#SBATCH -t 0-12:00
#SBATCH -p gpu_test
#SBATCH --mem=256000
#SBATCH --gres=gpu:4
#SBATCH -o collect.%j.out
#SBATCH -e collect.%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=apzhao@college.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
module load gcc/9.5.0-fasrc01

export HOME="/n/holylabs/LABS/idreos_lab/Users/azhao"

source activate $HOME/env

export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cudnn/8.9.2.26_cuda12-fasrc01/lib:${LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# sbatch $HOME/pytorch/benchmarks/dynamo/huggingface.sh
# sbatch $HOME/pytorch/benchmarks/dynamo/timm.sh
# sbatch $HOME/pytorch/benchmarks/dynamo/torchbench.sh

python3 $HOME/pytorch/benchmarks/dynamo/torchbench.py --performance --cold-start-latency --training --amp --backend inductor --device cuda --repeat 1 --threads 56
echo "Finished torchbench-------------------"
python3 $HOME/pytorch/benchmarks/dynamo/timm_models.py --performance --cold-start-latency --training --amp --backend inductor --device cuda --repeat 1 --threads 56
echo "Finished timm-------------------"
python3 $HOME/pytorch/benchmarks/dynamo/huggingface.py --performance --cold-start-latency --training --amp --backend inductor --device cuda --repeat 1 --threads 56