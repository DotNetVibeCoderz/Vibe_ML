"""Fixtures for the ESM tests. Run from this folder:

    python make_esm.py           # the tiny random checkpoints and esm-reference.json
    python make_esm.py --real    # also esm2-t6-reference.json, from facebook/esm2_t6_8M_UR50D (downloads 30 MB)

esm-rotary/     a random EsmForMaskedLM in ESM-2's layout: rotary positions, token dropout, contact head
esm-absolute/   the ESM-1b layout: learned positions (offset by pad + 1), a norm before the first layer,
                no token dropout
esm-reference.json   transformers' float64 outputs for both, and EsmTokenizer's ids for awkward inputs

transformers computes the rotary angles, and the rotation itself, in float32 even inside a float64 model.
HF.Net does both in double, so the reference patches EsmRotaryEmbedding.forward and apply_rotary_pos_emb to
stay in double; without the patch expect differences around 3e-7, which are the reference's rounding, not a
bug. Token dropout's observed mask ratio is float32 in the reference too, and HF.Net reproduces that one.
"""
import json, os, shutil, sys
import torch
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer
from transformers.models.esm import modeling_esm

here = os.path.dirname(os.path.abspath(__file__))

# ESM-2's own vocabulary, in id order.
VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
         "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]


def double_rotary(self, x, position_ids, layer_type=None):
    dim = self.inv_freq.shape[0] * 2
    base = getattr(self.config, "rope_theta", None) or 10000.0
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    freqs = position_ids[:, :, None].double() * inv[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_in_dtype(q, k, cos, sin, unsqueeze_dim=1):
    # transformers' version casts q and k to float32 before rotating them, even in a float64 model.
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + modeling_esm.rotate_half(q) * sin, k * cos + modeling_esm.rotate_half(k) * sin


modeling_esm.EsmRotaryEmbedding.forward = double_rotary
modeling_esm.apply_rotary_pos_emb = rotate_in_dtype


def tokenizer_in(folder):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "vocab.txt"), "w", newline="\n") as f:
        f.write("\n".join(VOCAB))
    tok = EsmTokenizer(os.path.join(folder, "vocab.txt"))
    tok.save_pretrained(folder)
    return tok


def random_model(folder, **overrides):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    tok = tokenizer_in(folder)
    settings = dict(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=32, max_position_embeddings=40, mask_token_id=32, pad_token_id=1,
                    layer_norm_eps=1e-5)
    settings.update(overrides)
    model = EsmForMaskedLM(EsmConfig(**settings, attn_implementation="eager")).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.normal_(0, 0.3 if p.dim() > 1 else 0.2)
            if "LayerNorm.weight" in name or "layer_norm" in name and name.endswith("weight"):
                p.add_(1.0)
    model.save_pretrained(folder)
    return tok, model.double()


def hidden(model, tok, text):
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        return model.esm(**enc).last_hidden_state[0]


def logits_at_mask(model, tok, text):
    enc = tok(text, return_tensors="pt")
    position = enc["input_ids"][0].tolist().index(tok.mask_token_id)
    with torch.no_grad():
        return position, model(**enc).logits[0, position]


def tiny():
    torch.manual_seed(0)
    tok, rotary = random_model(os.path.join(here, "esm-rotary"), position_embedding_type="rotary",
                               token_dropout=True, emb_layer_norm_before=False)
    torch.manual_seed(1)
    _, absolute = random_model(os.path.join(here, "esm-absolute"), position_embedding_type="absolute",
                               token_dropout=False, emb_layer_norm_before=True)

    sequence = "MKTAYIAKQRQISFVKSHFSRQ"
    masked = "MKTAY<mask>AKQRQISF"
    reference = {"sequence": sequence, "masked": masked}

    for name, model in (("rotary", rotary), ("absolute", absolute)):
        position, logits = logits_at_mask(model, tok, masked)
        reference[name] = {
            "hidden": hidden(model, tok, sequence).flatten().tolist(),
            "mask_position": position,
            "mask_logits": logits.tolist(),
        }

    # Padding: a short sequence padded to the length of a longer one. Token dropout divides by the
    # number of real tokens, and the pad rows are zeroed before the first layer.
    padded = tok(["MKTAY<mask>AK", sequence], padding=True, return_tensors="pt")
    with torch.no_grad():
        states = rotary.esm(**padded).last_hidden_state[0]
    reference["padded"] = {
        "ids": padded["input_ids"][0].tolist(),
        "mask": padded["attention_mask"][0].tolist(),
        "hidden": states.flatten().tolist(),
    }

    contact_input = "MKTAYIAKQRQISFVK"
    enc = tok(contact_input, return_tensors="pt")
    with torch.no_grad():
        contacts = rotary.predict_contacts(enc["input_ids"], enc["attention_mask"])[0]
    reference["contacts"] = {"sequence": contact_input, "values": contacts.flatten().tolist(),
                             "size": contacts.shape[0]}

    texts = ["MKTAYIAKQR", "M K T A", "MKT<mask>AY", "mkta", "MKJJXB", "<cls>MK<eos>", "", "MK  <mask> J A<pad>"]
    reference["tokenizer"] = [
        {"text": t, "ids": tok(t)["input_ids"], "plain": tok(t, add_special_tokens=False)["input_ids"],
         "decoded": tok.decode(tok(t)["input_ids"]),
         "decoded_skip": tok.decode(tok(t)["input_ids"], skip_special_tokens=True)}
        for t in texts]
    reference["pair"] = tok("MKT", "AYI")["input_ids"]

    with open(os.path.join(here, "esm-reference.json"), "w") as f:
        json.dump(reference, f)
    print("tiny fixtures written")


def real():
    repo = "facebook/esm2_t6_8M_UR50D"
    tok = EsmTokenizer.from_pretrained(repo)
    model = EsmForMaskedLM.from_pretrained(repo, attn_implementation="eager").eval().double()
    ubiquitin = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

    states = hidden(model, tok, ubiquitin)
    pooled = states[1:-1].mean(0)

    # Masked marginal for K11P: mask residue 11 (token 11, after <cls>) and compare log P(P) with log P(K).
    ids = tok(ubiquitin, return_tensors="pt")["input_ids"]
    ids[0, 11] = tok.mask_token_id
    with torch.no_grad():
        log_p = torch.log_softmax(model(input_ids=ids).logits[0, 11], -1)
    k, p = tok.convert_tokens_to_ids("K"), tok.convert_tokens_to_ids("P")

    position, logits = logits_at_mask(model, tok, "<mask>" + ubiquitin[1:])
    probabilities = torch.softmax(logits, -1)
    top = torch.topk(probabilities, 5)

    enc = tok(ubiquitin, return_tensors="pt")
    with torch.no_grad():
        contacts = model.predict_contacts(enc["input_ids"], enc["attention_mask"])[0]

    reference = {
        "repo": repo,
        "sequence": ubiquitin,
        "pooled": pooled.tolist(),
        "hidden_every_97th": states.flatten()[::97].tolist(),
        "k11p": {"log_p_k": log_p[k].item(), "log_p_p": log_p[p].item(), "score": (log_p[p] - log_p[k]).item()},
        "first_residue_top5": [{"token": tok.convert_ids_to_tokens(int(i)), "score": s.item()}
                               for s, i in zip(top.values, top.indices)],
        "contacts_every_31st": contacts.flatten()[::31].tolist(),
        "contacts_size": contacts.shape[0],
    }
    with open(os.path.join(here, "esm2-t6-reference.json"), "w") as f:
        json.dump(reference, f)
    print("pooled[0] =", pooled[0].item(), " pooled[4] =", pooled[4].item(), " K11P =", reference["k11p"]["score"])


tiny()
if "--real" in sys.argv:
    real()
