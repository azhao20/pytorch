module load python/3.10.12-fasrc01

HOME=/n/holylabs/LABS/idreos_lab/Users/azhao

sbatch $HOME/pytorch/benchmarks/dynamo/huggingface.sh
sbatch $HOME/pytorch/benchmarks/dynamo/timm.sh
sbatch $HOME/pytorch/benchmarks/dynamo/torchbench.sh
