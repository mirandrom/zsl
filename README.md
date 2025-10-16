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

### Download checkpoints for analyses
Model weights and optimizer states can be found at:  
https://huggingface.co/mirandrom/zsl-checkpoints

You can [download](https://huggingface.co/docs/huggingface_hub/v0.31.1/guides/download) these with `huggingface_hub`: 

```python
from huggingface_hub import hf_hub_download

# model_sizes = ['14M','37M','78M','144M','285M','472M']
# steps = [int(2**i) for i in range(19)]
model_size = '14M' 
step = 8192
revision = f"OLMo-{model_size}-step{step}"

# Use this file structure to reproduce analyses
from zsl_config import ZSL_DIR_OUT_OLMO # change this in `zsl_config.py`
def get_run_dir(model_size, step):
    run = MODEL_SIZE_TO_RUN[model_size]
    return ZSL_DIR_OUT_OLMO / run / f"step{step}-unsharded"


# download checkpoint files
for filename in ['model.pt', 'optim.pt', 'train.pt', 'config.yaml']:
    hf_hub_download(
        repo_id="mirandrom/zsl-checkpoints", 
        filename=filename, 
        revision=revision,
        local_dir=get_run_dir(model_size, step)
    )
```

To download everything, you can run `reproducibility/download_checkpoints.ipynb`.

### Download eval data for analyses
We use the `val` split of `c4_en` from `allenai/paloma` (link).  
You will need to accept their T&C [here](https://huggingface.co/datasets/allenai/paloma) to get a token.  
Add your token to and run `reproducibility/download_c4_en_val.ipynb` to download & tokenize the data.  
The tokenized data will be cached in the right path for reproducing our analyses (`ZSL_DIR_DATA` in `zsl_config.py`).