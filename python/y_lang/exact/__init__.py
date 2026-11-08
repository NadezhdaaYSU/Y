"""Reproducible inference for Qwen2-architecture language models.

    from y_lang.exact import load, generate

    model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
    tokens = generate(model, tokenizer, "The capital of France is", max_new_tokens=32)

A prompt decodes to the same tokens whatever else is in its batch, and to the
same tokens on a CPU and on a GPU. Every reduction whose order can change with
the batch shape or the device is carried out in integers, and integer
addition is associative; the float operations that remain (the quantisers'
divides, RoPE's sine and cosine, SiLU) are written so that no device rounds
them differently. The measurements, their controls and what is not claimed
are in the repository's `docs/bit_identical_decode.md`; in short, the claim is
"deterministic", not "exact": the model is quantised (int8 weights, and
quantised activations and K/V cache), so its tokens are not the bf16 model's.

Submodules, which the repository's `tools/` harnesses import by their old
names:
- `model`: `ExactLinear` (int8 x int8 with an exact int32 sum), the exact
  norm, SiLU and rotary embedding, and `convert`. Importing it sets two
  process-wide Inductor flags (`division_rounding`, `emulate_precision_casts`)
  without which `torch.compile` rounds unlike eager PyTorch.
- `kv`: the K/V quantisers and cache.
- `attention`: exact attention for Qwen2 (integer scores, the integer `exp2`,
  an exact p.V), and the demo's `build`, `decode` and sanity gate.
- `greedy`: fixed-length greedy decoding without a host sync per token.
- `w8a8_gemv`: the Triton int8 kernels (GPU only).
- `large_model`: a builder that never holds a float32 copy of a bf16
  checkpoint, for models too large for that.

On a GPU the linear layers' int8 products run on Triton's kernel by default,
on cuBLASLt with `Y_EXACT_GEMM=torch`, or on Y's own kernel, whose schedule and
value have a machine-checked proof, with `Y_EXACT_GEMM=y`. `Y_EXACT_V_INT8=1`
stores V as int8 and runs attention's p.V on Y's `exact_pv` kernel. Every
choice gives the same bits. These are read when `model` and `kv` are first
imported.
"""

__all__ = ["load", "generate", "SUPPORTED_MODEL_TYPES"]

SUPPORTED_MODEL_TYPES = ("qwen2",)


def load(model_id, device="cpu", *, low_memory=False):
    """`(model, tokenizer)` for a Qwen2-architecture checkpoint, converted to
    the exact form on `device`.

    `low_memory=True` builds layer by layer from a bf16 checkpoint without a
    float32 copy of the whole model (`large_model.build_exact`), for models
    such as Qwen2.5-7B; it refuses a checkpoint not stored in bf16.

    Another architecture is refused rather than converted: the exact
    attention replaces Qwen2's, and nothing else has been checked.

    The conversion replaces `transformers`' Qwen2 eager attention for the
    whole process. Stock models loaded with `attn_implementation="sdpa"`
    are not affected.
    """
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id)
    if config.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"{model_id} is a {config.model_type!r} model; y_lang.exact supports "
            f"{', '.join(SUPPORTED_MODEL_TYPES)} only")
    if low_memory:
        from .large_model import build_exact
        model = build_exact(model_id, device)
    else:
        from .attention import build_exact_model
        model = build_exact_model(model_id, device)
    return model, AutoTokenizer.from_pretrained(model_id)


def generate(model, tokenizer, prompt, max_new_tokens=64):
    """Greedy decoding of `prompt` alone: the new token ids, as a list."""
    from .attention import decode

    device = next(model.parameters()).device
    return decode(model, tokenizer, prompt, 1, max_new_tokens, device)
