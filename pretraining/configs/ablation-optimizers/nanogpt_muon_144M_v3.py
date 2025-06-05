from zsl_config import ZSL_DIR_OUT_NANO, ZSL_DIR_DATA_NANO

# key changes to be consistent with original (v2):
# set learning rate to 0.0036
# remove warmup
# remove weight decay
# remove grad clipping
# (v3)
# use rotary embeddings
# use float logits
# use default weight init
# fix lr setting

wandb_log = True
wandb_project = 'zsl-tmp'
wandb_run_name='nanogpt_muon_144M_v3'
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
rotary = True
float_logits = True
nanogpt_init = False

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
warmup_iters = 0 
lr_decay_iters = max_iters
learning_rate = 0.0036 # max learning rate
min_lr = learning_rate # WSD schedule (no decay)

# optimizer hparams
optimizer_variant = 'muon'
weight_decay = 0.0
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
