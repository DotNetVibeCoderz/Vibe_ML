"""Fixtures for the GPT-2 tests. Run from this folder: python make_gpt2.py

gpt2-tiny/            a random GPT2LMHeadModel (2 layers, 32 wide, 4 heads, vocab 99, 32 positions),
                      saved with the "transformer." prefix
gpt2-reference.json   torch float64 logits for a sequence, and a greedy continuation
"""
import json, os, shutil
import torch
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(0)
config = GPT2Config(vocab_size=99, n_positions=32, n_embd=32, n_layer=2, n_head=4, bos_token_id=98, eos_token_id=98)
model = GPT2LMHeadModel(config).eval()

# Widen the default initialisation so every parameter matters, norms included.
with torch.no_grad():
    for name, p in model.named_parameters():
        p.normal_(0, 0.3 if p.dim() > 1 else 0.2)
        if name.endswith("ln_1.weight") or name.endswith("ln_2.weight") or name.endswith("ln_f.weight"):
            p.add_(1.0)

if os.path.exists("gpt2-tiny"):
    shutil.rmtree("gpt2-tiny")
model.save_pretrained("gpt2-tiny")

model = model.double()
ids = [5, 17, 42, 3, 77, 60, 11]
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0]
    greedy = model.generate(torch.tensor([ids[:3]]), max_new_tokens=12, do_sample=False,
                            pad_token_id=98, eos_token_id=None)[0].tolist()
    penalised = model.generate(torch.tensor([ids[:3]]), max_new_tokens=12, do_sample=False,
                               repetition_penalty=1.3, pad_token_id=98, eos_token_id=None)[0].tolist()

json.dump({"ids": ids, "logits": logits.flatten().tolist(), "greedy": greedy, "penalised": penalised},
          open("gpt2-reference.json", "w"))
from safetensors import safe_open
with safe_open("gpt2-tiny/model.safetensors", "pt") as f:
    print(sorted(k for k in f.keys() if ".h.1." not in k))
print("greedy", greedy)
