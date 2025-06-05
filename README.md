# zsl
Zero-sum learning (ACL 2025)

## Reproducibility
### Environment setup (uv)
```bash
git clone git@github.com:mirandrom/zsl.git
cd zsl
uv sync
source .venv/bin/activate
```
### Output directory setup
See `zsl_config.py` and change `ZSL_DIR_SCRATCH`

### Download pretraining data
See `pretraining/download_olmo_data`

### Run pretraining experiments
See folders in `pretraining/configs` and associated `sbatch` subfolders for launching training runs associated with each experiment in the paper. 

Note that we use SLURM and `srun` instead of `torchrun`; but you should be able to run the same code with `torchrun` if not on SLURM. 

Older `sbatch` scripts might also require minor edits to be compatible with the current codebase.