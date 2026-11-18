"""Greedy decoding with each decode step replayed from one CUDA graph.

    from y_lang.exact.graphed import GraphedGreedy
    run = GraphedGreedy(model, batch, prompt_len, max_new_tokens)
    new_ids = run(input_ids, attention_mask)     # what `greedy.greedy` returns

A compiled decode step of the exact path launches about 600 kernels. On an
RTX 4070 Ti SUPER its GPU work took 2.6 ms and the step 4.4 ms: the GPU
waited on the host that launched them. A CUDA graph launches the whole step
at once. It needs a step whose shapes never change, and Hugging Face's
caches grow by a row each step. So the decode step here is built from the
model's own modules, with:

  * a cache of fixed buffers, `[batch, kv heads, capacity, row]` per layer,
    filled by the prefill and then written one row a step at a position
    held on the device (`index_copy_`);
  * an attention mask built on the device from that position: a row sees
    its own prompt (not its left padding) and every token decoded so far,
    and nothing in the spare rows;
  * positions, the next token and the step counter advanced on the device,
    inside the graph.

The step is compiled once (`torch.compile`, `dynamic=False`), warmed up and
recorded once per (batch, prompt length, new tokens), and replayed. The
prefill runs as usual (`model(...)`, the caller's compiled forward if it
has one) into a fresh cache, whose rows are copied into the buffers.

Same tokens: every kernel the step launches is one the uncompiled loop
launches, on the same values. Attention now spans the buffer's spare rows
too, and the mask blocks them. The exact path's attention gives a blocked
key no weight, exactly (its batch invariance with left padding rests on
that). `python/y_lang/tests/test_exact_graphed.py` checks the tokens against
`greedy.greedy`, bit for bit, with and without left padding.

Refused: a CPU model or one split across devices, a model with
sliding-window layers, an attention
implementation other than "eager" (the exact path's) or "sdpa", positions
past the model's table, and inputs of another shape than the graph's.
"""

import contextlib

import torch

from . import kv as _kv


def _has_exact_layers(model):
    from .model import ExactLinear
    return any(isinstance(m, ExactLinear) for m in model.modules())


class _StaticCache:
    """What a decoder layer's attention needs from a cache: `update`, which
    writes the new rows at position `t` (a device int64 [1]) and returns the
    whole buffers. Rows past t are blocked by the step's mask."""

    def __init__(self, keys, values, t):
        self.keys, self.values, self.t = keys, values, t

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if key_states.shape[-2] != 1:
            raise RuntimeError("the graphed cache takes one new row a step")
        self.keys[layer_idx].index_copy_(2, self.t, key_states)
        self.values[layer_idx].index_copy_(2, self.t, value_states)
        return self.keys[layer_idx], self.values[layer_idx]


class GraphedGreedy:
    """Greedy decoding of `max_new_tokens` tokens for inputs of one shape,
    `(batch, prompt_len)`, each decode step one CUDA graph replay.

    `attention` is installed as transformers' Qwen2 eager attention while the
    step is traced; by default the exact path's (`attention.exact_attention`)
    when the model has `ExactLinear` layers, else none. `inductor_patch`
    (a dict for `torch._inductor.config.patch`) is applied while the step
    compiles, so a baseline can compile with PyTorch's defaults.
    """

    def __init__(self, model, batch, prompt_len, max_new_tokens, attention="auto",
                 inductor_patch=None, capacity_multiple=32):
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1, not {max_new_tokens}")
        dev = next(model.parameters()).device
        if dev.type != "cuda":
            raise ValueError("GraphedGreedy needs a CUDA model (it records CUDA graphs)")
        for t in (*model.parameters(), *model.buffers()):
            if t.device != dev:
                raise ValueError(f"GraphedGreedy: the model has tensors on {t.device} and {dev}; "
                                 f"a graph cannot hold a copy to or from the host")
        inner = getattr(model, "model", None)
        for name in ("layers", "embed_tokens", "rotary_emb", "norm"):
            if inner is None or not hasattr(inner, name):
                raise ValueError(f"GraphedGreedy: the model has no `model.{name}`; it runs "
                                 f"decoder-only models laid out as transformers' Qwen2")
        if getattr(inner, "has_sliding_layers", False):
            raise ValueError("GraphedGreedy: sliding-window layers are not supported")
        cfg = model.config.get_text_config(decoder=True)
        impl = getattr(cfg, "_attn_implementation", None)
        if impl not in ("eager", "sdpa"):
            raise ValueError(f"GraphedGreedy: attention implementation {impl!r}; it builds "
                             f"masks for 'eager' and 'sdpa' only")
        self.model, self.cfg, self.dev = model, cfg, dev
        self.batch, self.prompt_len, self.new = batch, prompt_len, max_new_tokens
        self.cap = -(-(prompt_len + max_new_tokens) // capacity_multiple) * capacity_multiple
        limit = getattr(cfg, "max_position_embeddings", None)
        if limit is not None and prompt_len + max_new_tokens > limit:
            raise ValueError(f"GraphedGreedy: {prompt_len} + {max_new_tokens} positions are "
                             f"past the model's {limit}")
        self.sdpa = impl == "sdpa"
        if attention == "auto":
            attention = None
            if _has_exact_layers(model):
                from .attention import exact_attention
                attention = exact_attention
        self.attention, self.inductor_patch = attention, inductor_patch

        self.ar = torch.arange(self.cap, device=dev)
        self.base = torch.ones((batch, self.cap), dtype=torch.bool, device=dev)
        self.t = torch.zeros(1, dtype=torch.int64, device=dev)
        self.step_i = torch.zeros((), dtype=torch.int64, device=dev)
        self.out = torch.zeros((batch, max_new_tokens), dtype=torch.int64, device=dev)
        self.token = torch.zeros(batch, dtype=torch.int64, device=dev)
        self.pos = torch.zeros((batch, 1), dtype=torch.int64, device=dev)
        self.keys = self.values = None
        self.graph = None
        self._step_compiled = torch.compile(self._step, dynamic=False)

    # The model's layers read the attention function from a module global
    # when they run (see tools/exact_throughput.py `with_attention`).
    @contextlib.contextmanager
    def _installed(self):
        import transformers.models.qwen2.modeling_qwen2 as qwen2
        prev = qwen2.eager_attention_forward
        if self.attention is not None:
            qwen2.eager_attention_forward = self.attention
        try:
            yield
        finally:
            qwen2.eager_attention_forward = prev

    def _prefill(self, input_ids, attention_mask):
        cache = _kv.new_cache(self.cfg)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         position_ids=position_ids, past_key_values=cache, use_cache=True,
                         logits_to_keep=1, return_dict=True)
        first = out.logits[:, -1].to(dtype=torch.float32).argmax(-1)
        rows = []
        for layer in cache.layers:
            k, v = layer._rows() if hasattr(layer, "_rows") else (layer.keys, layer.values)
            if k.dim() != 4 or k.shape[0] != self.batch or k.shape[2] != self.prompt_len:
                raise RuntimeError(f"GraphedGreedy: a prefill cache layer holds {tuple(k.shape)}, "
                                   f"not [batch {self.batch}, heads, {self.prompt_len}, row]")
            rows.append((k, v))
        return first, position_ids[:, -1:] + 1, rows

    def _load(self, first, next_pos, rows, attention_mask):
        if self.keys is None:
            self.keys = [k.new_zeros((*k.shape[:2], self.cap, k.shape[-1])) for k, _ in rows]
            self.values = [v.new_zeros((*v.shape[:2], self.cap, v.shape[-1])) for _, v in rows]
            self.cache = _StaticCache(self.keys, self.values, self.t)
        for kb, vb, (k, v) in zip(self.keys, self.values, rows):
            if k.shape != kb[:, :, :self.prompt_len].shape or k.dtype != kb.dtype:
                raise RuntimeError("GraphedGreedy: the prefill's cache rows changed shape or type")
            # The spare rows are masked, so what they hold does not reach a token
            # (a mutation that drops these two lines passes the tests). Zeroed
            # anyway, so a stale row from the last call is never read as data.
            kb.zero_()
            vb.zero_()
            kb[:, :, :self.prompt_len].copy_(k)
            vb[:, :, :self.prompt_len].copy_(v)
        self.base.fill_(True)
        self.base[:, :self.prompt_len].copy_(attention_mask.bool())
        self.token.copy_(first)
        self.pos.copy_(next_pos)
        self.t.fill_(self.prompt_len)
        self.step_i.zero_()
        self.out.zero_()
        self.out[:, 0] = first

    def _step(self, token, pos, t):
        inner = self.model.model
        h = inner.embed_tokens(token[:, None])
        pe = inner.rotary_emb(h, pos)
        allowed = (self.ar <= t) & self.base                       # [batch, capacity]
        if self.sdpa:
            mask = allowed.view(self.batch, 1, 1, self.cap)
        else:
            mask = torch.where(allowed, 0.0, float("-inf")).to(h.dtype)
            mask = mask.view(self.batch, 1, 1, self.cap)
        for layer in inner.layers:
            h = layer(h, attention_mask=mask, position_ids=pos, past_key_values=self.cache,
                      use_cache=True, position_embeddings=pe)
        h = inner.norm(h)
        return self.model.lm_head(h)[:, -1].to(dtype=torch.float32).argmax(-1)

    def _advance(self):
        nxt = self._step_compiled(self.token, self.pos, self.t)
        self.token.copy_(nxt)
        self.pos.add_(1)
        self.t.add_(1)
        self.step_i.add_(1)
        self.out.index_copy_(1, self.step_i.view(1), nxt.view(self.batch, 1))

    def _record(self):
        import torch._inductor.config as inductor_config
        patch = (inductor_config.patch(self.inductor_patch) if self.inductor_patch
                 else contextlib.nullcontext())
        with self._installed(), patch, torch.no_grad():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):              # compile, then settle the allocator
                    self._advance()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._advance()
        self.graph = graph

    def __call__(self, input_ids, attention_mask):
        if tuple(input_ids.shape) != (self.batch, self.prompt_len) or \
                tuple(attention_mask.shape) != (self.batch, self.prompt_len):
            raise ValueError(f"GraphedGreedy was recorded for inputs {(self.batch, self.prompt_len)}, "
                             f"not {tuple(input_ids.shape)}")
        with torch.no_grad():
            with self._installed():
                first, next_pos, rows = self._prefill(input_ids, attention_mask)
            self._load(first, next_pos, rows, attention_mask)
            if self.new > 1 and self.graph is None:
                self._record()
                # Warm-up and recording moved the state: put it back.
                self._load(first, next_pos, rows, attention_mask)
            for _ in range(self.new - 1):
                self.graph.replay()
        return self.out.clone()
