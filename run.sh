#!/bin/bash

#SBATCH --partition=long
#SBATCH -c 8                                                         
#SBATCH --mem=48G                                        
#SBATCH --time=96:00:00     
#SBATCH --gres=gpu:l40s:1                         
#SBATCH -o /home/mila/l/luke.rowe/test_repo/scenario-dreamer/gpudrive/slurm_logs/train_ppo.out

module load singularity
singularity exec -H $HOME:/home -B ~/scratch/:/scratch --nv ~/scratch/gpudrive_2025.sif bash -c "
cd test_repo/scenario-dreamer/gpudrive
source /scratch/conda-envs/gpudrive-scenario-dreamer/bin/activate
export PYTHONUNBUFFERED=1
python -u baselines/ppo/ppo_pufferlib.py
"