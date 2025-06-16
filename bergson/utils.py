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


from transformers.modeling_attn_mask_utils import AttentionMaskConverter, is_torchdynamo_compiling
from typing import Optional

def _ignore_causal_mask_sdpa(
    attention_mask: Optional[torch.Tensor],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
    is_training: bool = False,
) -> bool:
    """
    Detects whether the optional user-specified attention_mask & the automatically created causal mask can be
    ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

    In case no token is masked in the `attention_mask` argument, if `query_length == 1` or
    `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
    passed).
    """
    is_tracing = torch.jit.is_tracing() or isinstance(inputs_embeds, torch.fx.Proxy) or is_torchdynamo_compiling()

    ignore_causal_mask = False

    if attention_mask is None:
        # TODO: When tracing with TorchDynamo with fullgraph=True, the model is recompiled depending on the input
        # shape, thus SDPA's `is_causal` argument is rightfully updated
        # (see https://gist.github.com/fxmarty/1313f39037fc1c112508989628c57363). However, when using
        # `torch.export` or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is
        # hard-coded. If a user exports a model with q_len > 1, the exported model will hard-code `is_causal=True`
        # which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108).
        # Thus, we only set `ignore_causal_mask = True` if the model is set to training.
        #
        # Besides, jit.trace can not handle the `q_len > 1` condition for `is_causal`
        # ("TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Tensor").
        try:
            _, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
            key_value_length = query_length + past_key_values_length

            if (
                (is_training or not is_tracing)
                and (query_length == 1 or key_value_length == query_length)
                and (sliding_window is None or key_value_length < sliding_window)
            ):
                ignore_causal_mask = True
        except AttributeError:
            pass
    elif sliding_window is None or key_value_length < sliding_window:
        if len(attention_mask.shape) == 4:
            return False
        elif not is_tracing and torch.all(attention_mask == 1):
            if query_length == 1 or key_value_length == query_length:
                # For query_length == 1, causal attention and bi-directional attention are the same.
                ignore_causal_mask = True

            # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore
            # the attention mask, as SDPA causal mask generation may be wrong. We will set `is_causal=False` in
            # SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
            # Reference: https://github.com/pytorch/pytorch/issues/108108
            # TODO: maybe revisit this with https://github.com/pytorch/pytorch/pull/114823 in PyTorch 2.3.

    return ignore_causal_mask


AttentionMaskConverter._ignore_causal_mask_sdpa = _ignore_causal_mask_sdpa
