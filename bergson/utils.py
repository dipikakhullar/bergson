from typing import Any, Type, TypeVar, cast
import copy

import torch
from torch import nn
from transformers import PreTrainedModel

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore[return-value]


def get_layer_list(model: PreTrainedModel) -> nn.ModuleList:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        mod
        for mod in model.base_model.modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


def patch_fsdp_int8_model(model: PreTrainedModel, dtype: torch.dtype):
    """Patches bitsandbytes internal code and the model to make `int8` precision work with FSDP.
    It does this by making views of important parameters in the floating point parameter dtype (`dtype`).
    List of patches:
    - transmute `weight.data` and `state.CB` to `dtype` so it is not int8 and FSDP can handle it
    - custom forward pass for Linear8bitLt: use `CB` from `weight.data` so it is sharded; move `SCB` to cuda (it is not automatically dispatched)
    - patch `.to` on Linear8bitLt: by default it will attempt to convert dtypes
    - bnb.matmul: view the dtype-transmuted `CB` (code block) array as int8 and call the original matmul kernel

    Args:
        model (PreTrainedModel): model to patch
        dtype (torch.dtype): dtype to convert parameters to

    Raises:
        ImportError: if bitsandbytes is not installed
    """
    from importlib.util import find_spec
    if find_spec("bitsandbytes") is None:
        raise ImportError("bitsandbytes is not installed, but load")
    from bitsandbytes.nn.modules import Linear8bitLt
    import bitsandbytes as bnb

    def forward(self, x):
        self.state.CB = self.weight.data
        self.state.SCB = self.state.SCB.cuda()
        return bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

    def to_dtype(x):
        return x.view(dtype).clone()

    for m in model.modules():
        if isinstance(m, Linear8bitLt):
            m.init_8bit_state()
            m.weight.data = to_dtype(m.weight.data)
            m.state.CB = to_dtype(m.state.CB)
            m.weight.__class__.to = torch.nn.Parameter.to
            m.__class__.forward = forward
    base_matmul = bnb.matmul
    def new_matmul(
        A,
        B,
        out = None,
        state = None,
        threshold = 0.0,
        bias = None,
    ):
        state = copy.copy(state)
        state.CB = state.CB.view(torch.int8)
        return base_matmul(A, B, out, state, threshold, bias)
    bnb.matmul = new_matmul


def post_patch_fsdp_int8_model(model: PreTrainedModel):
    """Moves `SCB` to GPU after FSDPing an int8 model."""
    from importlib.util import find_spec
    if find_spec("bitsandbytes") is None:
        raise ImportError("bitsandbytes is not installed, but load")
    from bitsandbytes.nn.modules import Linear8bitLt
    
    for m in model.modules():
        if isinstance(m, Linear8bitLt):
            m.weight.SCB = m.state.SCB.cuda()
            m.state.SCB = m.state.SCB.cuda()
