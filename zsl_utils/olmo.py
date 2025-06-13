from olmo.config import TrainConfig
from olmo.checkpoint import load_state_dict
from olmo.model import OLMo
from olmo.optim import build_optimizer

# TODO: fix after rsync
from zsl_config import ZSL_DIR_OUT_OLMO

from typing import Optional

def get_olmo_device_bsz(run):
    yaml_path = ZSL_DIR_OUT_OLMO / run / 'config.yaml'
    cfg = TrainConfig.load(yaml_path)
    bsz = cfg.device_train_batch_size
    return bsz

def load_olmo_model(
    run: str, step: int, device="cuda", overrides: Optional[list] = None
):
    ckpt_dir = ZSL_DIR_OUT_OLMO / run / f"step{step}-unsharded"
    overrides = overrides or []
    overrides.append(f"model.init_device={device}")
    cfg = TrainConfig.load(
        ckpt_dir / "config.yaml", validate_paths=False, overrides=overrides
    )
    model = OLMo(cfg.model, init_params=False)
    state_dict = load_state_dict(ckpt_dir, "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    return model


def load_olmo_optimizer(
    model, run: str, step: int, device="cuda", overrides: Optional[list] = None
):
    ckpt_dir = ZSL_DIR_OUT_OLMO / run / f"step{step}-unsharded"
    overrides = overrides or []
    overrides.append(f"model.init_device={device}")
    cfg = TrainConfig.load(
        ckpt_dir / "config.yaml", validate_paths=False, overrides=overrides
    )
    optimizer = build_optimizer(cfg, model)
    state_dict = load_state_dict(ckpt_dir, "optim.pt", map_location=device)
    optimizer.load_state_dict(state_dict)
    return optimizer


def get_olmo_model_steps(run: str):
    ckpts_dir = ZSL_DIR_OUT_OLMO / run
    return sorted(
        [
            int(d.name.replace("step", "").replace("-unsharded", ""))
            for d in ckpts_dir.glob("step[1-9]*-unsharded")
        ]
    )
