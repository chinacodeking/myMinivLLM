import pytest
import torch

from myvllm.models.llama import LlamaDecoderLayer
from myvllm.models.qwen3 import Qwen3DecoderLayer
from myvllm.utils.context import reset_context, set_context


class IdentityNorm(torch.nn.Module):
    def forward(
        self,
        x,
        residual=None,
    ):
        if residual is None:
            return x

        return x, residual


class RecordingAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_positions = None

    def forward(
        self,
        x,
        positions,
    ):
        self.seen_positions = positions
        return x


def make_lightweight_decoder_layer(layer_type):
    layer = layer_type.__new__(layer_type)
    torch.nn.Module.__init__(layer)

    layer.input_layernorm = IdentityNorm()
    layer.self_attn = RecordingAttention()
    layer.post_attention_layernorm = IdentityNorm()
    layer.mlp = torch.nn.Identity()

    return layer


@pytest.mark.parametrize(
    "layer_type",
    [
        pytest.param(
            Qwen3DecoderLayer,
            id="qwen3",
        ),
        pytest.param(
            LlamaDecoderLayer,
            id="llama",
        ),
    ],
)
def test_decoder_layer_uses_explicit_context_positions(
    layer_type,
):
    layer = make_lightweight_decoder_layer(layer_type)

    expected_positions = torch.tensor(
        [4, 5],
        dtype=torch.long,
    )

    set_context(
        is_prefill=True,
        cu_seqlens_q=torch.tensor(
            [0, 2],
            dtype=torch.int32,
        ),
        positions=expected_positions,
    )

    try:
        layer(
            torch.zeros(
                (2, 4),
                dtype=torch.float32,
            )
        )
    finally:
        reset_context()

    seen_positions = layer.self_attn.seen_positions

    assert seen_positions.tolist() == [4, 5]
    assert seen_positions is expected_positions