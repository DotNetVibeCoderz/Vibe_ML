"""Fixtures for the streaming encoder tests. Run from this folder: python make_bert.py

bert-small/           a random BertModel (2 layers, 16 wide, 2 heads) with its own 25-word vocabulary
bert-small-sharded/   the same weights split into several safetensors shards with an index
bert-reference.json   torch float64 last_hidden_state for a text
"""
import json, os, shutil
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

words = "the a cat dog sat on mat ran fast slow happy sad big small red blue house tree river sky".split()
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
for folder in ("bert-small", "bert-small-sharded"):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    open(f"{folder}/vocab.txt", "w").write("\n".join(vocab) + "\n")

tokenizer = BertTokenizerFast("bert-small/vocab.txt", do_lower_case=True)
torch.manual_seed(0)
config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=32, max_position_embeddings=32)
model = BertModel(config, add_pooling_layer=False).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        p.normal_(0, 0.3 if p.dim() > 1 else 0.2)
        if "LayerNorm.weight" in name:
            p.add_(1.0)

model.save_pretrained("bert-small")
tokenizer.save_pretrained("bert-small")
model.save_pretrained("bert-small-sharded", max_shard_size="8KB")
tokenizer.save_pretrained("bert-small-sharded")

text = "the big red house by the river"
with torch.no_grad():
    hidden = model.double()(**tokenizer(text, return_tensors="pt")).last_hidden_state[0]
json.dump({"text": text, "hidden": hidden.flatten().tolist()}, open("bert-reference.json", "w"))
print(sorted(os.listdir("bert-small-sharded")))
