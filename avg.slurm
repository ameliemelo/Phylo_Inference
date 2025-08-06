#!/bin/bash
#SBATCH --job-name=check_avg     # Nom du jo
#SBATCH --output=/lustre/fswork/projects/rech/hvr/uhd88jk/Phylo_Inference/%x_%j.out      # Standard output (%j sera remplacé par l'ID du job)
#SBATCH --error=/lustre/fswork/projects/rech/hvr/uhd88jk/Phylo_Inference/%x_%j.err        # Standard error (%j sera remplacé par l'ID du job)
#SBATCH --time=02:00:00             # Limite de temps (HH:MM:SS)
#SBATCH -C a100
#SBATCH --nodes=1                   # Nombre de noeuds
#SBATCH --ntasks-per-node=1          # Une tâche par noeud (= nombre de GPU ici)
#SBATCH --cpus-per-task=10          # Nombre de cœurs CPU par tâche
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu_a100-dev              # Demande de ressources GPU
#SBATCH --hint=nomultithread
#SBATCH --account=hvr@a100 # Comptabilité pour V100

module purge

module load arch/a100 # pour les GPU A100
module load pytorch-gpu/py3/2.2.0  
set -x

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export TORCH_USE_CUDA_DSA=1
export CUBLAS_WORKSPACE_CONFIG=:16:8

srun python BiSSE/GNN_avg.py

