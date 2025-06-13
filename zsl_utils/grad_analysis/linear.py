import math
import gc

import torch
import torch.nn.functional as F
from torch import autograd, nn, Tensor


class GALinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return GALinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )

    @classmethod
    def from_nnmodule(cls, nnmodule: nn.Linear):
        new_linear = cls(nnmodule.in_features, nnmodule.out_features, nnmodule.bias)
        new_linear.weight = nnmodule.weight
        new_linear.bias = nnmodule.bias
        new_linear.weight.sum_i_gij = torch.zeros_like(
            new_linear.weight, dtype=torch.float32
        )
        new_linear.weight.sum_i_abs_gij = torch.zeros_like(
            new_linear.weight, dtype=torch.float32
        )
        new_linear.weight.sum_i_abs_pij = torch.zeros_like(
            new_linear.weight, dtype=torch.float32
        )
        new_linear.weight.sum_j_pij = []
        new_linear.weight.sum_j_abs_pij = []
        new_linear.weight.norm_l2sq_gi = []
        new_linear.weight.norm_l1_gi = []
        new_linear.weight.curr_tok_idx = 0

        if new_linear.bias is not None:
            raise NotImplementedError("Bias not supported")
        return new_linear.to(device=nnmodule.weight.device, dtype=nnmodule.weight.dtype)


class GALinearFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, output_grads):
        # output_grads: (bsz, seq, dim)
        # input_activations: (bsz, seq, dim)
        # grad_weight = outer_product(output_grads, input_activations)
        input, W, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if bias is not None and ctx.needs_input_grad[2]:
            raise NotImplementedError("Bias not supported")

        if ctx.needs_input_grad[0]:
            grad_input = torch.einsum("B...p,pd->B...d", output_grads, W)

        if not ctx.needs_input_grad[1]:
            return grad_input, None, None

        # Because we only back prop for curr_tok_idx token across a batch,
        # output grads for tokens > i should be zero, so we truncate these.
        torch.Tensor
        assert output_grads[:, W.curr_tok_idx + 1 : ].sum() == 0
        output_grads_trunc = output_grads[:, : W.curr_tok_idx + 1, :]
        input_trunc = input[:, : W.curr_tok_idx + 1, :]
        output_grads_trunc = output_grads_trunc.to(dtype=torch.float32)
        input_trunc = input_trunc.to(dtype=torch.float32)

        # If we are processing a new batch, then curr_tok_idx is 0 and we
        # instantiate nested list accumulators to recover the seq dimension
        if W.curr_tok_idx == 0:
            W.norm_l2sq_gi.append([])
            W.norm_l1_gi.append([])
            W.sum_j_pij.append([])
            W.sum_j_abs_pij.append([])


        # Compute per-example grads for each sequence in a batch
        grads = torch.einsum("Bsd,Bsp->Bpd", input_trunc, output_grads_trunc)

        # Compute and accumulate metrics. Use tmp tensor for reduced memory.
        tmp_tensor = torch.zeros_like(grads)

        W.sum_i_gij.add_(grads.sum(dim=0))
        W.norm_l2sq_gi[-1] += [torch.square(grads, out=tmp_tensor).sum(dim=(-1, -2)).detach().cpu()]
        
        abs_grads = torch.abs(grads, out=tmp_tensor)
        W.sum_i_abs_gij.add_(abs_grads.sum(dim=0))
        W.norm_l1_gi[-1] += [abs_grads.sum(dim=(-1, -2)).detach().cpu()]

        grad_delta_prod = torch.mul(grads, W.delta.unsqueeze(0), out=tmp_tensor)
        W.sum_j_pij[-1] += [grad_delta_prod.sum(dim=(-1, -2)).detach().cpu()]

        abs_grad_delta_prod = grad_delta_prod.abs_()
        W.sum_i_abs_pij.add_(abs_grad_delta_prod.sum(dim=0))
        W.sum_j_abs_pij[-1] += [abs_grad_delta_prod.sum(dim=(-1, -2)).detach().cpu()]
        del grad_delta_prod, abs_grad_delta_prod, abs_grads, tmp_tensor, grads
        # torch.cuda.empty_cache()
        # gc.collect() 

        W.curr_tok_idx += 1
        return grad_input, None, None
