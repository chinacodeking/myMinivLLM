from myvllm.utils.context import (
    get_context,
    reset_context,
    set_context,
)


def test_context_preserves_explicit_positions():
    positions = object()

    reset_context()

    try:
        set_context(
            is_prefill=True,
            positions=positions,
        )

        context = get_context()

        assert context.is_prefill
        assert context.positions is positions
    finally:
        reset_context()