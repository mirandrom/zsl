#!/bin/bash
#SBATCH --job-name=1101-cosine-37m
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --gres=gpu:l40s:2        # number of allocated gpus per node
#SBATCH --cpus-per-task=6        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --partition=long
#SBATCH -o /network/scratch/m/mirceara/zsl-olmo/out/%x/slurm.out.j%j
#SBATCH -e /network/scratch/m/mirceara/zsl-olmo/out/%x/slurm.out.err.j%j

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
    
export ZSL_DIR="/home/mila/m/mirceara/1.workspace/zsl"
source $ZSL_DIR/.venv/bin/activate

export OLMO_DIR=$HOME/1.workspace/zsl/pretraining/olmo
export OLMO_RUN=$OLMO_DIR/scripts/train.py

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

export OLMO_ZSL_EXP="1101-cosine"
export OLMO_ZSL_RUN="37m"
export OLMO_CFG=$OLMO_DIR/zsl/exp/$OLMO_ZSL_EXP/$OLMO_ZSL_RUN.yaml

sig_handler() {
  echo "BATCH interrupted"
  wait # wait for all children, this is important!
  sbatch $OLMO_DIR/zsl/exp/$OLMO_ZSL_EXP/sbatch_$OLMO_ZSL_RUN.sh
}
trap sig_handler SIGTERM

# Execute Python script in each task (one per GPU)
srun  bash -c '\
export export FS_LOCAL_RANK=$SLURM_PROCID; \
export RANK=$SLURM_PROCID; \
export LOCAL_RANK=$(( RANK - SLURM_GPUS_ON_NODE * ( RANK / SLURM_GPUS_ON_NODE ) )); \
python $OLMO_RUN $OLMO_CFG \
    --run_name=$SLURM_JOB_NAME \
    --wandb.name=$SLURM_JOB_NAME.$SLURM_JOBID \
    --wandb.group=$SLURM_JOB_NAME \
    --wandb.project=zsl-tmp \
    --wandb.entity=amr-amr \
'