#!/bin/bash
#SBATCH --job-name=nanogpt_144M
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --gres=gpu:l40s:2        # number of allocated gpus per node
#SBATCH --cpus-per-task=6        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --partition=long
#SBATCH -o /network/scratch/m/mirceara/zsl_scratch/out/pretraining/nanogpt/%x/slurm.out
#SBATCH -e /network/scratch/m/mirceara/zsl_scratch/out/pretraining/nanogpt/%x/slurm.out.err

echo "------------------------------------------------------------------------"
echo "Job:     ${SLURM_JOBID}"
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "------------------------------------------------------------------------"

export ZSL_DIR="/home/mila/m/mirceara/1.workspace/zsl"
source $ZSL_DIR/.venv/bin/activate

export NANO_TRAIN_CFG="${ZSL_DIR}/zsl_pretraining/configs/ablation-nanogpt/nanogpt_144M.py"
export NANO_TRAIN_SCR="${ZSL_DIR}/zsl_pretraining/nanogpt/train.py"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
sig_handler() {
  echo "BATCH interrupted"
  wait
}
trap sig_handler SIGTERM

srun bash -c '\
export RANK=$SLURM_PROCID; \
export LOCAL_RANK=$(( RANK - SLURM_GPUS_ON_NODE * ( RANK / SLURM_GPUS_ON_NODE ) )); \
python $NANO_TRAIN_SCR $NANO_TRAIN_CFG
'
