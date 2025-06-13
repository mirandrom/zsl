import datasets as hfds

from torch.utils.data import DataLoader
from torch import LongTensor

from olmo.config import TrainConfig
from olmo.util import clean_opt
from olmo.torch_util import seed_all
from olmo.data import build_train_dataloader, IterableDataset


from zsl_config import ZSL_DIR_DATA
from zsl_config import ZSL_DIR_OUT_OLMO


def get_olmo_train_batch(run, step, *args) -> LongTensor:
    yaml_path = ZSL_DIR_OUT_OLMO / run / "config.yaml"
    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args])
    # Set `global_indices_file` to shared path
    cfg.data.global_indices_file = ZSL_DIR_OUT_OLMO / "train_data/global_indices.npy"
    # Load single batch instead of distributed
    cfg.device_train_batch_size = cfg.global_train_batch_size
    seed_all(cfg.seed)
    train_loader = build_train_dataloader(cfg)
    assert isinstance(train_loader.dataset, IterableDataset)

    global_train_examples_seen_this_epoch = cfg.global_train_batch_size * step
    train_loader.dataset.start_index = global_train_examples_seen_this_epoch
    batch = next(iter(train_loader))
    return batch["input_ids"]


def get_eval_dataloader(
    model_class: str = "olmo",
    dataset: str = "c4_en_val",
    bsz: int = 4,
    device: str = "cpu",
):
    tokenized_eval_data = ZSL_DIR_DATA / f"tokenized/{model_class}-{dataset}"
    assert tokenized_eval_data.exists()

    dataset = hfds.load_from_disk(tokenized_eval_data)
    dataset.set_format(type="torch", columns=["input_ids"])
    dataloader = DataLoader(
        dataset,
        batch_size=bsz,
        shuffle=False,
        pin_memory=device != "cpu",
        pin_memory_device=device,
    )
    return dataloader