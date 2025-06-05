#!/bin/bash
#SBATCH --job-name=olmo_14M-bsz_1M-seq_2048
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --gres=gpu:l40s:2        # number of allocated gpus per node
#SBATCH --cpus-per-task=6        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --partition=long
#SBATCH -o /network/scratch/m/mirceara/zsl_scratch/out/pretraining/olmo/%x/slurm.out
#SBATCH -e /network/scratch/m/mirceara/zsl_scratch/out/pretraining/olmo/%x/slurm.out.err
# Echo time and hostname into log

echo "------------------------------------------------------------------------"
echo "Job:     ${SLURM_JOBID}"
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "------------------------------------------------------------------------"
    
export ZSL_DIR="/home/mila/m/mirceara/1.workspace/zsl"
source $ZSL_DIR/.venv/bin/activate
source $ZSL_DIR/zsl_utils/olmo_env_setup.sh

export OLMO_TRAIN_CFG=$ZSL_DIR/pretraining/configs/ablation-bsz_seq/olmo_14M-bsz_1M-seq_2048.yaml
export OLMO_TRAIN_SCR="${ZSL_DIR}/pretraining/olmo/scripts/train.py"

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
sig_handler() {
  echo "BATCH interrupted"
  wait
}
trap sig_handler SIGTERM

srun  bash -c '\
export FS_LOCAL_RANK=$SLURM_PROCID; \
export RANK=$SLURM_PROCID; \
export LOCAL_RANK=$(( RANK - SLURM_GPUS_ON_NODE * ( RANK / SLURM_GPUS_ON_NODE ) )); \
python $OLMO_TRAIN_SCR $OLMO_TRAIN_CFG \
    --run_name=$SLURM_JOB_NAME \
    --wandb.name=$SLURM_JOB_NAME \
    --wandb.group=$SLURM_JOB_NAME \
    --wandb.project=zsl-tmp \
    --wandb.entity=amr-amr \
'