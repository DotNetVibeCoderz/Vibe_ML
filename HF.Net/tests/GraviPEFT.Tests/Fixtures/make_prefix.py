"""Fixtures for the prefix-tuning tests. Run from this folder: python make_prefix.py

bert-small/        a random BertForSequenceClassification (2 layers, 16 wide, 2 heads) with its own
                   40-word vocabulary, so the pooler and classifier are real
prefix-adapter/    a PEFT 0.21 PREFIX_TUNING adapter for it (3 virtual tokens), saved by PEFT
prefix-reference.json   torch float64 logits of the adapted model for two texts, and the base
                   model's, which must differ
"""
import json, os, shutil
import torch
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

words = "the a cat dog sat on mat ran fast slow happy sad big small red blue house tree river sky".split()
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
for folder in ("bert-small", "prefix-adapter"):
    if os.path.exists(folder):
        shutil.rmtree(folder)
os.makedirs("bert-small")
open("bert-small/vocab.txt", "w").write("\n".join(vocab) + "\n")
tokenizer = BertTokenizerFast("bert-small/vocab.txt", do_lower_case=True)

torch.manual_seed(0)
config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=32, max_position_embeddings=32, num_labels=2)
model = BertForSequenceClassification(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        p.normal_(0, 0.3 if p.dim() > 1 else 0.2)
        if "LayerNorm.weight" in name:
            p.add_(1.0)
model.save_pretrained("bert-small")
tokenizer.save_pretrained("bert-small")

peft_model = get_peft_model(model, PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=3))
with torch.no_grad():
    peft_model.prompt_encoder["default"].embedding.weight.normal_(0, 1.0)
peft_model.save_pretrained("prefix-adapter")

peft_model = peft_model.double().eval()
base = BertForSequenceClassification.from_pretrained("bert-small").double().eval()
texts = ["the cat sat on the mat", "a big red house by the river"]
reference = {"texts": texts, "adapted": [], "base": []}
with torch.no_grad():
    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        reference["adapted"].append(peft_model(**enc).logits[0].tolist())
        reference["base"].append(base(**enc).logits[0].tolist())
json.dump(reference, open("prefix-reference.json", "w"), indent=1)
print(reference)
print(sorted(os.listdir("prefix-adapter")), sorted(os.listdir("bert-small")))
