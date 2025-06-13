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
DATASET = "c4_en_val"
ANALYSIS_NAME = "grad_intrf_token"
OUT_DIR = ZSL_DIR_ANALYSIS / f"{ANALYSIS_NAME}/{MODEL_CLASS}-{DATASET}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    "1028-rmsnorm-14m",
    "1028-rmsnorm-37m",
    "1028-rmsnorm-78m",
    "1028-rmsnorm-144m",
    "1028-rmsnorm-285m",
    "1028-rmsnorm-472m",
]
OVERWRITE = False
 
DEVICE_NAME = torch.cuda.get_device_name()
BASE_MICROBSZ = {
    'NVIDIA L40S': 96,
}
MICROBSZS = {
    "1028-rmsnorm-14m"  : BASE_MICROBSZ[DEVICE_NAME],
    "1028-rmsnorm-37m"  : BASE_MICROBSZ[DEVICE_NAME]//2,
    "1028-rmsnorm-78m"  : BASE_MICROBSZ[DEVICE_NAME]//4,
    "1028-rmsnorm-144m" : BASE_MICROBSZ[DEVICE_NAME]//8,
    "1028-rmsnorm-285m" : BASE_MICROBSZ[DEVICE_NAME]//16,
    "1028-rmsnorm-472m" : BASE_MICROBSZ[DEVICE_NAME]//32   
}

# pytorch_cuda_alloc_conf = [
#     'garbage_collection_threshold:0.95',
#     'expandable_segments:True'
# ]
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(pytorch_cuda_alloc_conf)


def analysis_loop(run, train_step, batch_seed, train_microbatch_dataloader, eval_microbatch_dataloader, device, verbose: bool = False):
    inf_ctx = torch.inference_mode()
    amp_ctx = torch.amp.autocast(device, dtype=torch.bfloat16)
    t = train_step
    bs = batch_seed
    #---------------------------------------------------------------------------
    # skip batch dir if already done and no overwrite
    #---------------------------------------------------------------------------
    out_dir = OUT_DIR / run
    out_dir.mkdir(exist_ok=True, parents=False)
    batch_dir = out_dir / f"train-batch={bs}"
    path_train_init = batch_dir / f"train/losses_init/step{t}.pt"
    path_train_post = batch_dir / f"train/losses_post/step{t}.pt"
    spath_eval_init = batch_dir / f"eval-{DATASET}/losses_init/step{t}.pt"
    path_eval_post = batch_dir / f"eval-{DATASET}/losses_post/step{t}.pt"
    path_train_grad = batch_dir / f"train/grad/step{t}.pt"
    path_eval_grad = batch_dir / f"eval-{DATASET}/grad/step{t}.pt"
    skip_batch_dir = all(
        [
            path_train_init.exists(),
            path_train_post.exists(),
            path_eval_post.exists(),
            spath_eval_init.exists(),
            path_train_grad.exists(),
            path_eval_grad.exists(),
            not OVERWRITE,
        ]
    )
    if skip_batch_dir:
        return
    #---------------------------------------------------------------------------
    # load model and optimizer
    #---------------------------------------------------------------------------
    model = load_olmo_model(run, t, device=device)
    model = model.to(torch.bfloat16)
    optimizer = load_olmo_optimizer(model, run, t, device=device)
    optimizer.zero_grad()
    for pg in optimizer.param_groups:
        if pg["lr"] == 0:
            assert run.startswith(
                "1028-rmsnorm"
            ), f"Learning rate hack might be wrong here."
            pg["lr"] = pg["initial_lr"]

    #---------------------------------------------------------------------------
    # 1. compute init eval losses (same across seeds, hence use of symlink)
    #---------------------------------------------------------------------------
    if verbose:
        print(f"[{datetime.now()}] init eval losses")
    model.eval()
    path_eval_init = out_dir / f".eval-{DATASET}/losses_init/step{t}.pt"
    if not path_eval_init.exists() or (path_eval_init.exists() and OVERWRITE):
        path_eval_init.parent.mkdir(parents=True, exist_ok=True)
        losses = []
        with inf_ctx, amp_ctx:
            for i, eval_batch in enumerate(eval_microbatch_dataloader):
                input_ids = eval_batch[:, :-1]
                labels = eval_batch[:, 1:].flatten()
                logits = model(input_ids).logits.flatten(0, 1)
                _losses = F.cross_entropy(logits, labels, reduction="none")
                losses.append(_losses.detach().cpu())
                del logits, _losses
        torch.save(torch.cat(losses, dim=0), path_eval_init)

        if spath_eval_init.exists() and OVERWRITE:
            spath_eval_init.unlink()
        if not spath_eval_init.exists():
            spath_eval_init.parent.mkdir(parents=True, exist_ok=True)
            spath_eval_init.symlink_to(path_eval_init)

    #---------------------------------------------------------------------------
    # 2. init train losses and optimizer step
    #---------------------------------------------------------------------------
    if verbose:
        print(f"[{datetime.now()}] init train losses")
    model.train()
    losses = []
    weights_delta = {n: p.data.cpu() for n, p in model.named_parameters()}
    num_tokens = sum([x[:, :-1].numel() for x in train_microbatch_dataloader])
    for i, train_microbatch in enumerate(train_microbatch_dataloader):
        input_ids = train_microbatch[:, :-1]
        labels = train_microbatch[:, 1:].flatten()
        with amp_ctx:
            logits = model(input_ids).logits.flatten(0, 1)
            _losses = F.cross_entropy(logits, labels, reduction="none")
            loss = _losses.sum() / num_tokens
        loss.backward()
        losses.append(_losses.detach().cpu())
        del logits, _losses, loss
    optimizer.step()
    for n, p in model.named_parameters():
        weights_delta[n] = p.data.cpu() - weights_delta[n]
    path_train_init.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(losses, dim=0), path_train_init)
    del optimizer

    #---------------------------------------------------------------------------
    # 3. post train losses
    #---------------------------------------------------------------------------
    if verbose:
        print(f"[{datetime.now()}] post train losses")
    model.eval()
    losses = []
    with inf_ctx, amp_ctx:
        for i, train_microbatch in enumerate(train_microbatch_dataloader):
            input_ids = train_microbatch[:, :-1]
            labels = train_microbatch[:, 1:].flatten()
            logits = model(input_ids).logits.flatten(0, 1)
            _losses = F.cross_entropy(logits, labels, reduction="none")
            losses.append(_losses.detach().cpu())
            del logits, _losses
    path_train_post.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(losses, dim=0), path_train_post)

    #---------------------------------------------------------------------------
    # 4. post eval losses
    #---------------------------------------------------------------------------
    if verbose:
        print(f"[{datetime.now()}] post eval losses")
    model.eval()
    losses = []
    with inf_ctx, amp_ctx:
        for i, eval_batch in enumerate(eval_microbatch_dataloader):
            input_ids = eval_batch[:, :-1]
            labels = eval_batch[:, 1:].flatten()
            logits = model(input_ids).logits.flatten(0, 1)
            _losses = F.cross_entropy(logits, labels, reduction="none")
            losses.append(_losses.detach().cpu())
            del logits, _losses
    path_eval_post.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(losses, dim=0), path_eval_post)

    #===========================================================================
    # 5a. gradient analysis (train)
    #===========================================================================
    path_train_grad.parent.mkdir(parents=True, exist_ok=True)
    if (path_train_grad.exists() and OVERWRITE) or not path_train_grad.exists():
        if verbose:
            print(f"[{datetime.now()}] gradient analysis (train)")
        model = load_olmo_model(run, t, device=device)
        model = model.to(torch.bfloat16)
        model = convert_olmo_model(model, weights_delta)
        overall_bsz = 0
        for i, train_microbatch in enumerate(train_microbatch_dataloader):
            input_ids = train_microbatch[:, :-1]
            labels = train_microbatch[:, 1:]
            bsz, seq_len = labels.shape
            overall_bsz += bsz
            # NOTE: compute logits and keep in memory to perform `seq_len` backward passes
            torch.cuda.empty_cache()
            gc.collect()
            with amp_ctx:
                logits = model(input_ids).logits
            for j in range(0, seq_len):
                # update curr_tok_idx across parameters for backward pass analysis
                for n, p in model.named_parameters():
                    p.curr_tok_idx = j
                # compute backward pass only for j'th tokens in a batch
                _logits = logits[:,j,:]
                _labels = labels[:,j]
                loss = F.cross_entropy(_logits, _labels, reduction="sum")
                loss.backward(retain_graph=True)
                del loss
                if verbose:
                    print(f"[{datetime.now()}]  batch.token {i}.{j}", end='\r')
            del logits

        metrics = reduce_metrics_olmo(model, overall_bsz, seq_len)
        torch.save(metrics, path_train_grad)
        del metrics, model


    #===========================================================================
    # 5b. gradient analysis (eval)
    #===========================================================================
    if verbose:
        print(f"\n[{datetime.now()}] gradient analysis (eval)")
    model = load_olmo_model(run, t, device=device)
    model = model.to(torch.bfloat16)
    model = convert_olmo_model(model, weights_delta)
    overall_bsz = 0
    for i, eval_batch in enumerate(eval_microbatch_dataloader):
        input_ids = eval_batch[:, :-1]
        labels = eval_batch[:, 1:]
        bsz, seq_len = labels.shape
        overall_bsz += bsz
        gc.collect()
        torch.cuda.empty_cache()
        # NOTE: compute logits and keep in memory to perform `seq_len` backward passes
        with amp_ctx:
            logits = model(input_ids).logits
        for j in range(0, seq_len):
            # update curr_tok_idx across parameters for backward pass analysis
            for n, p in model.named_parameters():
                p.curr_tok_idx = j
            # compute backward pass only for j'th tokens in a batch
            _logits = logits[:,j,:]
            _labels = labels[:,j]
            loss = F.cross_entropy(_logits, _labels, reduction="sum")
            loss.backward(retain_graph=True)
            del loss
            if verbose:
                print(f"[{datetime.now()}]  batch.token {i}.{j}", end='\r')
        del logits

    metrics = reduce_metrics_olmo(model, overall_bsz, seq_len)
    path_eval_grad.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metrics, path_eval_grad)
    del metrics, model
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
        dist.init_process_group(backend='nccl')
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
    # setup dataloaders
    #---------------------------------------------------------------------------
    num_seeds = 1  # i.e. number of train batches on which to replicate the analysis
    run = RUNS[0]
    steps = get_olmo_model_steps(run)
    train_batch_steps = [steps[-1] + i for i in range(num_seeds)]
    train_batches = {}
    for batch_step in train_batch_steps:
        out_path = OUT_DIR / f"tokens/train-batch={batch_step}.pt"
        if not out_path.exists() and ddp_rank == 0:
            batch = get_olmo_train_batch(run, batch_step)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(batch, out_path)
        if ddp:
            dist.barrier()
        batch = torch.load(out_path, weights_only=True)
        train_batches[batch_step] = batch

    bsz_eval = 128
    eval_dataloader = get_eval_dataloader(bsz=bsz_eval, device=device)
    eval_tokens = [b["input_ids"].cpu() for b in eval_dataloader]
    eval_tokens = torch.cat(eval_tokens, dim=0)
    if ddp_rank == 0:
        out_path = OUT_DIR / f"tokens/eval-{DATASET}.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(eval_tokens, out_path)

    #---------------------------------------------------------------------------
    # analysis
    #---------------------------------------------------------------------------
    for batch_seed, train_batch in train_batches.items():
        for run in RUNS:
            microbsz = MICROBSZS[run]
            num_eval_microbatches = math.ceil(eval_tokens.shape[0] / microbsz)
            num_train_microbatches = math.ceil(train_batch.shape[0] / microbsz)

            train_microbatch_dataloader = [
                train_batch[i * microbsz : (i + 1) * microbsz].to(device)
                for i in range(num_train_microbatches)
            ]
            eval_microbatch_dataloader = [
                eval_tokens[i * microbsz : (i + 1) * microbsz].to(device)
                for i in range(num_eval_microbatches)
            ]
            if ddp_rank == 0:
                print(f"[{ddp_rank}][{datetime.now()}][{batch_seed}][{run}] Train microbatches: {num_train_microbatches}")
                print(f"[{ddp_rank}][{datetime.now()}][{batch_seed}][{run}] Eval microbatches: {num_eval_microbatches}")

            for train_step in steps[ddp_rank::ddp_world_size]:
                print(f"[{ddp_rank}][{datetime.now()}][{batch_seed}][{run}][{train_step}]")
                analysis_loop(
                    run=run,
                    train_step=train_step,
                    batch_seed=batch_seed,
                    train_microbatch_dataloader=train_microbatch_dataloader,
                    eval_microbatch_dataloader=eval_microbatch_dataloader,
                    device=device,
                    verbose=ddp_rank==0
                )
            