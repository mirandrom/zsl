from zsl_config import ZSL_DIR_OUT_NANO, ZSL_DIR_DATA_NANO

wandb_log = True
wandb_project = 'zsl-tmp'
wandb_run_name='nanogpt_ademamix_144M'
log_interval = 1
ckp_interval = 1000
ckp_log2 = True
out_dir = ZSL_DIR_OUT_NANO / wandb_run_name

# model hparams
n_layer = 16
n_head = 16
n_embd = 1024
dropout = 0.0 
bias = False 

# data hparams
dir_data = ZSL_DIR_DATA_NANO
dataset = 'openwebtext'
max_iters = 1e5
block_size = 1024
bsz_global = 512
batch_size = 16
gradient_accumulation_steps = bsz_global // batch_size
assert batch_size * gradient_accumulation_steps == bsz_global, \
    f"{batch_size * gradient_accumulation_steps} != {bsz_global}"

# learning rate schedule
warmup_iters = 2000 
lr_decay_iters = max_iters
learning_rate = 6.8e-4 # max learning rate
min_lr = learning_rate # WSD schedule (no decay)

# optimizer hparams
optimizer_variant = 'ademamix'
ademamix_kwargs = {
    'betas': (0.9, 0.999, 0.9999),
    'alpha': 8,
    'beta3_warmup': 50_000,
    'alpha_warmup': 50_000,
}
weight_decay = 1e-1
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
