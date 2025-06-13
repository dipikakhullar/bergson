from typing import Any, Type, TypeVar, cast
import copy

import torch

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore[return-value]


def hide_int8_model(model, dtype):
    from bitsandbytes.nn.modules import Linear8bitLt
    import bitsandbytes as bnb

    def forward(self, x):
        self.state.CB = self.weight.data
        self.state.SCB = self.state.SCB.cuda()
        return bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

    for m in model.modules():
        if isinstance(m, Linear8bitLt):
            m.init_8bit_state()
            def to_dtype(x):
                return x.view(dtype).clone()
            m.weight.data = to_dtype(m.weight.data)
            m.state.CB = to_dtype(m.state.CB)
            # ????
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
