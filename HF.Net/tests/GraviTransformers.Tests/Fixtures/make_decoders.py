"""Fixtures for the Llama-family and GPT-NeoX tests. Run from this folder: python make_decoders.py

decoders/<name>/          a tiny random checkpoint for each case below
decoders-reference.json   torch float64 logits for a 12-token sequence, and a greedy continuation

transformers computes three things in float32 even inside a float64 model: RMSNorm, the rotary
angles, and the eager softmax. HF.Net computes all three in double, so the reference patches them
to double too - the comparison is then of the same arithmetic, to 1e-12, rather than of float32
rounding. Against unpatched transformers the difference is a few units in the seventh digit.
"""
import json, math, os, shutil, zlib
import torch
from transformers import (GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig, LlamaForCausalLM, MistralConfig,
                          MistralForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM)
from transformers.models.gpt_neox import modeling_gpt_neox
from transformers.models.llama import modeling_llama
from transformers.models.mistral import modeling_mistral
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3


def rms_forward(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


def inverse_frequency(config, dim):
    params = getattr(config, "rope_parameters", None) or {}
    theta = params.get("rope_theta", getattr(config, "rope_theta", None) or getattr(config, "rotary_emb_base", 10000))
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    kind = params.get("rope_type", "default")
    if kind == "linear":
        inv = inv / params["factor"]
    elif kind == "llama3":
        factor, low, high = params["factor"], params["low_freq_factor"], params["high_freq_factor"]
        original = params["original_max_position_embeddings"]
        low_wavelength, high_wavelength = original / low, original / high
        wavelength = 2 * math.pi / inv
        scaled = torch.where(wavelength > low_wavelength, inv / factor, inv)
        smooth = (original / wavelength - low) / (high - low)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        medium = ~(wavelength < high_wavelength) & ~(wavelength > low_wavelength)
        inv = torch.where(medium, smoothed, scaled)
    return inv


def rotary_forward(self, x, position_ids):
    config = self.config
    head = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial = (getattr(config, "rope_parameters", None) or {}).get("partial_rotary_factor",
                                                                    getattr(config, "rotary_pct", 1.0))
    dim = int(head * partial)
    inv = inverse_frequency(config, dim)
    freqs = position_ids[:, :, None].double() * inv[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


for module in (modeling_llama, modeling_mistral, modeling_qwen2, modeling_qwen3):
    for name in dir(module):
        if name.endswith("RMSNorm"):
            getattr(module, name).forward = rms_forward
        if name.endswith("RotaryEmbedding"):
            getattr(module, name).forward = rotary_forward
for name in dir(modeling_gpt_neox):
    if name.endswith("RotaryEmbedding"):
        getattr(modeling_gpt_neox, name).forward = rotary_forward

base = dict(vocab_size=99, hidden_size=32, intermediate_size=40, num_hidden_layers=2, num_attention_heads=4,
            max_position_embeddings=64, bos_token_id=1, eos_token_id=98)
cases = {
    "llama": (LlamaConfig, LlamaForCausalLM, dict(num_key_value_heads=2)),
    "llama-linear": (LlamaConfig, LlamaForCausalLM, dict(num_key_value_heads=4,
                     rope_parameters={"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0})),
    "llama3": (LlamaConfig, LlamaForCausalLM, dict(num_key_value_heads=2, tie_word_embeddings=True,
               rope_parameters={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                "original_max_position_embeddings": 16, "rope_theta": 500000.0})),
    "mistral": (MistralConfig, MistralForCausalLM, dict(num_key_value_heads=1, sliding_window=4)),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM, dict(num_key_value_heads=2, tie_word_embeddings=True)),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM, dict(num_key_value_heads=2, head_dim=16)),
    "neox": (GPTNeoXConfig, GPTNeoXForCausalLM, dict(rotary_pct=0.25, use_parallel_residual=True)),
    "neox-sequential": (GPTNeoXConfig, GPTNeoXForCausalLM, dict(rotary_pct=0.5, use_parallel_residual=False)),
}

ids = [1, 5, 17, 42, 3, 77, 60, 11, 23, 90, 8, 31]
reference = {"ids": ids}
if os.path.exists("decoders"):
    shutil.rmtree("decoders")

for name, (config_type, model_type, extra) in cases.items():
    torch.manual_seed(zlib.crc32(name.encode()))  # str hash is randomised per process
    config = config_type(**{**base, **extra})
    config._attn_implementation = "sdpa"
    model = model_type(config).eval()
    with torch.no_grad():
        for parameter_name, p in model.named_parameters():
            p.normal_(0, 0.3 if p.dim() > 1 else 0.2)
            if "norm" in parameter_name and parameter_name.endswith("weight"):
                p.add_(1.0)
    model.save_pretrained(f"decoders/{name}")

    model = model.double()
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
        greedy = model.generate(torch.tensor([ids[:4]]), max_new_tokens=10, do_sample=False,
                                pad_token_id=0, eos_token_id=None)[0].tolist()
    reference[name] = {"logits": logits.flatten().tolist(), "greedy": greedy}
    print(name, "greedy", greedy[4:])

json.dump(reference, open("decoders-reference.json", "w"))
