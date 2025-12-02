from datetime import datetime
import math
import gc
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist

from zsl_config import ZSL_DIR_ANALYSIS
from zsl_utils.grad_analysis.convert import convert_olmo_model
from zsl_utils.grad_analysis.reduce import reduce_metrics_olmo
from zsl_utils.olmo import (
    get_olmo_model_steps,
    get_olmo_device_bsz,
    load_olmo_model,
    load_olmo_optimizer,
)
from zsl_utils.load_data import get_olmo_train_batch, get_eval_dataloader


MODEL_CLASS = "olmo"
ANALYSIS_NAME = "LL_xsections_debug_ddp_fix" # fix = comment out dist.init_process_group
OUT_DIR = ZSL_DIR_ANALYSIS / f"{ANALYSIS_NAME}/{MODEL_CLASS}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    "1028-rmsnorm-14m",
    "1028-rmsnorm-37m",
    "1028-rmsnorm-78m",
    # "1028-rmsnorm-144m",
    # "1028-rmsnorm-285m",
    # "1028-rmsnorm-472m",
]
OVERWRITE = False
 
DEVICE_NAME = torch.cuda.get_device_name()
BASE_MICROBSZ = {
    'NVIDIA A100-SXM4-80GB': 96,
}
MICROBSZS = {
    "1028-rmsnorm-14m"  : int(7/8*BASE_MICROBSZ[DEVICE_NAME]),
    "1028-rmsnorm-37m"  : int(6/8*BASE_MICROBSZ[DEVICE_NAME]),
    "1028-rmsnorm-78m"  : int(5/8*BASE_MICROBSZ[DEVICE_NAME]),
    "1028-rmsnorm-144m" : int(4/8*BASE_MICROBSZ[DEVICE_NAME]),
    "1028-rmsnorm-285m" : int(3/8*BASE_MICROBSZ[DEVICE_NAME]),
    "1028-rmsnorm-472m" : int(2/8*BASE_MICROBSZ[DEVICE_NAME]),
}


def analysis_loop(run, step, train_bbatch, eval_bbatch, train_batch_step, eval_batch_step, microbsz, device, verbose: bool = False):
    # 1. load model and optimizer
    # 2. compute optimizer update on train batch
    # 3. measure loss landscapes along update for train batch
    # 4. do the same for eval batch(es).
    inf_ctx = torch.inference_mode()
    amp_ctx = torch.amp.autocast(device, dtype=torch.bfloat16)

    #---------------------------------------------------------------------------
    # skip batch dir if already done and no overwrite
    #---------------------------------------------------------------------------
    out_dir = OUT_DIR / f"{run}/train_batch={train_batch_step}"
    train_path = out_dir / "train" / f"step{step}.pt"
    eval_path = out_dir / f"eval_batch={eval_batch_step}" / f"step{step}.pt"

    if train_path.exists() and eval_path.exists:
        if not OVERWRITE:
            print(f"[{datetime.now()}][run={run}][step={step}] Skipping", end='\r')
            return
        else:
            print(f"[{datetime.now()}][run={run}][step={step}] Overwriting", end='\r')
            train_path.unlink()
            eval_path.unlink()

    train_path.parent.mkdir(exist_ok=True, parents=True)
    eval_path.parent.mkdir(exist_ok=True, parents=True)

    #---------------------------------------------------------------------------
    # train/eval batch loading
    #---------------------------------------------------------------------------
    num_microbatches = math.ceil(train_bbatch.shape[0] / microbsz)
    train_microbatch_dataloader = [
        train_bbatch[i * microbsz : (i + 1) * microbsz].to(device)
        for i in range(num_microbatches)
    ]
    eval_microbatch_dataloader = [
        eval_bbatch[i * microbsz : (i + 1) * microbsz].to(device)
        for i in range(num_microbatches)
    ]

    #---------------------------------------------------------------------------
    # 1. load model and optimizer
    #---------------------------------------------------------------------------'
    model = load_olmo_model(run, step, device=device)
    optimizer = load_olmo_optimizer(model, run, step, device=device)
    optimizer.zero_grad()

    # fix last optimizer step that had 0 lr
    if run.startswith('1028-rmsnorm') and step == 262144:
        _step = step - 1 
        for param_group in optimizer.param_groups:
            # because we are using WSD, the stable lr is initial_lr
            # but on the last step (262143), the lr was decayed to 0
            # which means the optimizer states were updated, but not the weights
            assert param_group['lr'] == 0
            param_group['lr'] = param_group["initial_lr"]
            
            # shorthands for optimizer update computations
            b1,b2 = param_group['betas']
            eps = param_group['eps']
            bc1 = 1 - b1**_step
            bc2_sqrt = math.sqrt(1 - b2**_step)
            wd = param_group['weight_decay']
            lr = param_group['lr']
            ss = lr / bc1

            # iterate over params and apply update
            with inf_ctx:
                for n,p in zip(param_group['param_names'], param_group['params']):
                    p.mul_(1-lr*wd)
                    m = optimizer.state[p]['exp_avg']
                    v = optimizer.state[p]['exp_avg_sq']
                    denom = (v.sqrt() / bc2_sqrt).add_(eps)
                    update = -ss * torch.div(m, denom)
                    p.add_(update)
                del denom, update

    #---------------------------------------------------------------------------
    # 1. compute optimizer step
    #---------------------------------------------------------------------------
    if verbose: print(f"[{datetime.now()}][run={run}][step={step}] Computing optimizer step", end='\r')
    init_W = {n: p.data.detach().cpu() for n, p in model.named_parameters()}
    delta_W = {}

    model.train()
    num_tokens = sum([x[:, :-1].numel() for x in train_microbatch_dataloader])
    for i, train_batch in enumerate(train_microbatch_dataloader):
        input_ids = train_batch[:, :-1]
        labels = train_batch[:, 1:].flatten()
        with amp_ctx:
            logits = model(input_ids).logits.flatten(0, 1)
            loss = F.cross_entropy(logits, labels, reduction="sum") / num_tokens
        loss.backward()
        del logits, loss
    optimizer.step()

    # compute and cache optimizer step
    del optimizer
    model.zero_grad()
    with inf_ctx:
        for n, p in model.named_parameters():
            delta_W[n] = (p - init_W[n].to(device)).cpu()
    delta_W_norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p) for p in delta_W.values()])
        ).item()

    #---------------------------------------------------------------------------
    # 2. Compute loss landscapes along update
    #---------------------------------------------------------------------------
    STEP_SIZES = [x/5 for x in range(1,10)] 
    STEP_SIZES = [-x for x in reversed(STEP_SIZES)] + [0] + STEP_SIZES
    model.eval()
    with inf_ctx:
        step_train_losses = []
        step_eval_losses = []
        for s in STEP_SIZES:
            if verbose: print(f"[{datetime.now()}][run={run}][step={step}][s={s}] Setting weights...", end='\r')
            for n,p in model.named_parameters():
                p.data.copy_(init_W[n].add(delta_W[n], alpha=s/delta_W_norm))

            if verbose: print(f"[{datetime.now()}][run={run}][step={step}][s={s}] Measuring losses...", end='\r')
            train_losses = []
            for i, train_batch in enumerate(train_microbatch_dataloader):
                input_ids = train_batch[:, :-1]
                bsz,seq = input_ids.shape
                labels = train_batch[:, 1:].flatten()
                with amp_ctx:
                    logits = model(input_ids).logits.flatten(0, 1)
                    train_losses += [F.cross_entropy(logits, labels, reduction="none").view(bsz,seq).cpu()] 
                del logits            
            step_train_losses += [torch.cat(train_losses)]
            
            eval_losses = []                    
            for i, eval_batch in enumerate(eval_microbatch_dataloader):
                input_ids = eval_batch[:, :-1]
                bsz,seq = input_ids.shape
                labels = eval_batch[:, 1:].flatten()
                with amp_ctx:
                    logits = model(input_ids).logits.flatten(0, 1)
                    eval_losses += [F.cross_entropy(logits, labels, reduction="none").view(bsz,seq).cpu()]      
                del logits
            step_eval_losses += [torch.cat(eval_losses)]
      
    train_metrics = {
        'losses': step_train_losses,
        'stepsizes': STEP_SIZES,
        'actual_stepsize': delta_W_norm
    }
    eval_metrics = {
        'losses': step_eval_losses,
        'stepsizes': STEP_SIZES,
        'actual_stepsize': delta_W_norm
    }
    torch.save(train_metrics, train_path)
    torch.save(eval_metrics, eval_path)

    del model, init_W, delta_W, delta_W_norm
    if verbose:
        print(f"\n[{datetime.now()}] done analysis loop")
        # print gpu memory usage
        print(f"[{datetime.now()}] GPU memory usage: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"[{datetime.now()}] GPU peak memory usage: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

if __name__ == "__main__":
    #---------------------------------------------------------------------------
    # setup DDP
    #---------------------------------------------------------------------------
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        # dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        ddp_rank = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #---------------------------------------------------------------------------
    # analysis
    #---------------------------------------------------------------------------
    train_batch_step = 300_000
    eval_batch_step = 300_001

    for run in RUNS:
        train_bbatch = get_olmo_train_batch(run, train_batch_step)
        eval_bbatch = get_olmo_train_batch(run, eval_batch_step)
        microbsz = MICROBSZS[run]
        steps = get_olmo_model_steps(run)
        for step in steps[ddp_rank::ddp_world_size]:
            print(f"[{ddp_rank}][{datetime.now()}][{run}][{step}] Starting", flush=True)
            analysis_loop(
                run=run,
                step=step,
                train_bbatch=train_bbatch,
                eval_bbatch=eval_bbatch,
                train_batch_step=train_batch_step,
                eval_batch_step=eval_batch_step,
                microbsz=microbsz,
                device=device,
                verbose=ddp_rank==0
            )
            print(f"[{ddp_rank}][{datetime.now()}][{run}][{step}] Finished", flush=True)
            torch.cuda.empty_cache()
            gc.collect()
            