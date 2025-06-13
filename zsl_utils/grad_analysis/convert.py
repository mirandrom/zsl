from functools import reduce

import torch

from .embed import GAEmbedding
from .linear import GALinear

from typing import Union


def convert_olmo_model(model, weights_delta):
    if model.config.weight_tying is True:
        model.config.weight_tying = False
        # create explicit linear output layer for tied weights
        # (work around the fact that olmo uses F.linear with wte weights)
        vsz = model.transformer.wte.num_embeddings
        hid = model.transformer.wte.embedding_dim
        model.transformer.ff_out = torch.nn.Linear(hid, vsz, bias=False)
        model.transformer.ff_out.weight.data = model.transformer.wte.weight.data.clone()
    model = convert_linear(model)
    model = convert_embed(model)
    # add update delta to parameters for backward computations
    if "transformer.ff_out.weight" not in weights_delta:
        weights_delta["transformer.ff_out.weight"] = weights_delta[
            "transformer.wte.weight"
        ]
    for n, p in model.named_parameters():
        p.delta = weights_delta[n].to(p.device)
    return model


def convert_embed(model):
    named_modules = [n for n, _ in model.named_modules() if n]
    for n in named_modules:
        m = get_module_by_name(model, n)
        if type(m) == torch.nn.Embedding:
            set_module_by_name(model, n, GAEmbedding.from_nnmodule(m))
    return model


def convert_linear(model):
    named_modules = [n for n, _ in model.named_modules() if n]
    for n in named_modules:
        m = get_module_by_name(model, n)
        if type(m) == torch.nn.Linear:
            set_module_by_name(model, n, GALinear.from_nnmodule(m))
    return model


def get_module_by_name(module: Union[torch.Tensor, torch.nn.Module], access_string: str):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def set_module_by_name(module, access_string, value):
    attrs = access_string.split(".")
    if len(attrs) == 1:
        submodule = module
    else:
        submodule = get_module_by_name(module, ".".join(attrs[:-1]))

    attr = attrs[-1]
    submodule.__setattr__(attr, value)
