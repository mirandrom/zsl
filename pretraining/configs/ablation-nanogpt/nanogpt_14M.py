from zsl_config import ZSL_DIR_OUT_NANO, ZSL_DIR_DATA_NANO

wandb_log = True
wandb_project = 'zsl-tmp'
wandb_run_name='nanogpt_14M'
log_interval = 1
ckp_interval = 1000
ckp_log2 = True
out_dir = ZSL_DIR_OUT_NANO / wandb_run_name

# model hparams
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0 
bias = False 

# data hparams
dir_data = ZSL_DIR_DATA_NANO
dataset = 'openwebtext'
max_iters = 1e5
block_size = 1024
bsz_global = 512
batch_size = 64
gradient_accumulation_steps = bsz_global // batch_size
assert batch_size * gradient_accumulation_steps == bsz_global, \
    f"{batch_size * gradient_accumulation_steps} != {bsz_global}"

# learning rate schedule
warmup_iters = 2000 
lr_decay_iters = max_iters
learning_rate = 1.26e-3 # max learning rate
min_lr = learning_rate # WSD schedule (no decay)

# optimizer hparams
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
