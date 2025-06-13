import torch
import torch.nn.functional as F
from torch import autograd, nn, Tensor

from typing import Optional


class GAEmbedding(nn.Module):
    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = nn.Parameter(_weight)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return GAEmbeddingFunction.apply(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    @classmethod
    def from_nnmodule(cls, nnmodule: nn.Embedding):
        new_module = cls(
            num_embeddings=nnmodule.num_embeddings,
            embedding_dim=nnmodule.embedding_dim,
            padding_idx=nnmodule.padding_idx,
            max_norm=nnmodule.max_norm,
            norm_type=nnmodule.norm_type,
            scale_grad_by_freq=nnmodule.scale_grad_by_freq,
            sparse=nnmodule.sparse,
        )
        new_module.weight = nnmodule.weight
        new_module.weight.sum_i_gij = torch.zeros_like(
            new_module.weight, dtype=torch.float32
        )
        new_module.weight.sum_i_abs_gij = torch.zeros_like(
            new_module.weight, dtype=torch.float32
        )
        new_module.weight.sum_i_abs_pij = torch.zeros_like(
            new_module.weight, dtype=torch.float32
        )
        new_module.weight.sum_j_pij = []
        new_module.weight.sum_j_abs_pij = []
        new_module.weight.norm_l2sq_gi = []
        new_module.weight.norm_l1_gi = []
        new_module.weight.curr_tok_idx = 0

        return new_module.to(device=nnmodule.weight.device, dtype=nnmodule.weight.dtype)


class GAEmbeddingFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        ctx.save_for_backward(input, weight)
        # https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483/2
        ctx.padding_idx = padding_idx
        ctx.max_norm = max_norm
        ctx.norm_type = norm_type
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        return F.embedding(
            input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
        )

    @staticmethod
    def backward(ctx, output_grads):
        input_ids, W = ctx.saved_tensors
        output_grads = output_grads.to(dtype=torch.float32)

        # Because we only back prop for curr_tok_idx token across a batch,
        # output grads for tokens > i should be zero, so we truncate these.
        assert output_grads[:, W.curr_tok_idx + 1 :].sum() == 0
        output_grads_trunc = output_grads[:, : W.curr_tok_idx + 1, :]

        # If we are processing a new batch, then curr_tok_idx is 0 and we
        # instantiate nested list accumulators to recover the seq dimension
        if W.curr_tok_idx == 0:
            W.norm_l2sq_gi.append([])
            W.norm_l1_gi.append([])
            W.sum_j_pij.append([])
            W.sum_j_abs_pij.append([])

        # Reindex, truncate, then put on device to make use of `reindex_input_ids` cache.
        # reindexed_ids (bsz,seq) allows us to accumulate per-example grads to a smaller
        # (seq,dim) tensor instead of a much larger (vsz,dim) tensor.
        # reindex_maps (bsz,seq) allows us to map back to the original (vsz,dim) dimensions.
        # NOTE: reindex_maps is right-padded with 0 to allow vmap parallelism.
        #       Because 0 is a valid token id, this can lead to bugs if not careful.
        #       For our usecase, gradients are zero for these indices and have no effect.
        reindexed_ids, reindex_maps = reindex_input_ids(input_ids)
        reindexed_ids = reindexed_ids[:, : W.curr_tok_idx + 1].to(input_ids.device)
        reindex_maps = reindex_maps[:, : W.curr_tok_idx + 1].to(input_ids.device)

        # Accumulate per-example embedding gradients in reindexed tensor
        #       For example, if the first tokens in input_ids[i] are [42,42,9,100,9,...],
        #       then the first tokens in reindexed_ids[i] will be [0,0,1,2,1,...] with
        #       reindex_maps[i][0,1,2] = 42,9,100 and per_example_grads_reidx[i][0,1,2]
        #       corresponding to the gradients for embeddings of token ids 42,9,100.
        per_example_grads_reidx = torch.zeros_like(output_grads_trunc)
        reduce_embed_grad_reidx(
            output_grads_trunc, reindexed_ids, per_example_grads_reidx
        )

        # Compute and accumulate metrics
        # 1a) for gradients gij
        W.sum_i_gij.index_add_(
            0, reindex_maps.flatten(), per_example_grads_reidx.flatten(0, -2)
        )
        W.norm_l2sq_gi[-1] += [
            per_example_grads_reidx.square().sum(dim=(-1, -2)).detach().cpu()
        ]

        # 2a) for grad-delta products pij
        grad_delta_prod = per_example_grads_reidx.mul(W.delta[reindex_maps])
        W.sum_j_pij[-1] += [grad_delta_prod.sum(dim=(-1, -2)).detach().cpu()]

        # 2b) for absolute grad-delta products
        grad_delta_prod.abs_()
        W.sum_i_abs_pij.index_add_(
            0, reindex_maps.flatten(), grad_delta_prod.flatten(0, -2)
        )
        W.sum_j_abs_pij[-1] += [grad_delta_prod.sum(dim=(-1, -2)).detach().cpu()]
        del grad_delta_prod

        # 1b) for absolute gradients gij
        per_example_grads_reidx.abs_()
        W.sum_i_abs_gij.index_add_(
            0, reindex_maps.flatten(), per_example_grads_reidx.flatten(0, -2)
        )
        W.norm_l1_gi[-1] += [per_example_grads_reidx.sum(dim=(-1, -2)).detach().cpu()]
        del per_example_grads_reidx

        W.curr_tok_idx += 1
        return None, None, None, None, None, None, None


poor_person_cache = {}  # good enough work around for lru_cache issues and tensors
def reindex_input_ids(input_ids):
    k = str(input_ids)
    if k in poor_person_cache:
        return poor_person_cache[k]

    # Note: cast to int32 for speedup https://github.com/pytorch/pytorch/issues/42109
    #       also put on cpu to avoid memory leak with cache
    bsz, seq = input_ids.shape
    reindex_maps = torch.zeros(bsz, seq, dtype=torch.int32, device="cpu")
    reindexed_ids = torch.zeros(bsz, seq, dtype=torch.int32, device="cpu")
    for i, xx in enumerate(input_ids):
        reindex_map = dict()
        for j, x in enumerate(xx.tolist()):
            r = reindex_map.get(x, None)
            if r is None:
                r = len(reindex_map)
                reindex_map[x] = r
            reindexed_ids[i][j] = r
            reindex_maps[i][r] = x
    poor_person_cache[k] = reindexed_ids, reindex_maps
    return poor_person_cache[k]


def _reduce_embed_grad_reidx(_grad_output, _reindexed_ids, _per_example_grads_reidx):
    _per_example_grads_reidx.index_add_(0, _reindexed_ids, _grad_output)
    return


reduce_embed_grad_reidx = torch.vmap(_reduce_embed_grad_reidx, out_dims=None)
