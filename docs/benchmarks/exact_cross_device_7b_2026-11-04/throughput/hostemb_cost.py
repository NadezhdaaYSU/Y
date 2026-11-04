"""What does keeping stock's token embedding on the host cost? 0.5B, where both
fit, compiled, interleaved, batch 32 x 64. 0.5B ties lm_head to the embedding,
so the host copy gets its own CPU weights (as 7B's untied table is)."""
import sys
sys.path.insert(0, sys.argv[1])
import torch
from transformers import AutoTokenizer
import exact_throughput as T
import exact_large_model as L

torch.backends.cuda.matmul.allow_tf32 = False
tok = AutoTokenizer.from_pretrained(T.MODEL)
a = T.build("stock", "cuda", True)
b = T.build("stock", "cuda", True)
emb = b.model.embed_tokens
he = L.HostEmbedding(emb.num_embeddings, emb.embedding_dim,
                     padding_idx=emb.padding_idx,
                     _weight=emb.weight.detach().cpu().clone(), _freeze=True,
                     device="cpu", dtype=emb.weight.dtype)
he.target = torch.device("cuda")
b.model.embed_tokens = he
L._pin_device(b, "cuda")
L._pin_device(b.model, "cuda")
assert he.weight.device.type == "cpu"
models = {"stock": a, "stock_hostemb": b}
assert T.gate(models, tok, "cuda")
res = T.run_interleaved(models, tok, 32, 64, "cuda", 5)
for k, (best, med) in res.items():
    print(f"{k:<16}{best:8.3f}{64 / best:10.1f} tok/s/seq  spread {100 * (med - best) / best:.1f}%")
print(f"host embedding costs {res['stock_hostemb'][0] / res['stock'][0]:.3f}x")
