# Modelfile

A Modelfile names a base model and layers prompt, sampling and load settings on top, in the spirit
of a Dockerfile. Creating one produces a named variant that shares the base model's weights, so a
variant costs nothing on disk.

```bash
localgen create my-assistant -f Modelfile
localgen run my-assistant
localgen show my-assistant --modelfile
```

## Format

One instruction per line. Values spanning several lines are wrapped in triple quotes. Lines
starting with `#` are comments.

```
FROM qwen2.5-7b-instruct:q4_k_m

PARAMETER temperature 0.3
PARAMETER num_ctx 8192
PARAMETER stop "<|im_end|>"

SYSTEM """
You are a senior .NET engineer.
Answer with code first, explanation second.
"""
```

## Instructions

### `FROM` — required

The base model: an installed model id, a path, or a reference LocalGen can pull.

```
FROM qwen2.5-7b-instruct:q4_k_m
FROM ./models/custom.gguf
FROM huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF
```

### `SYSTEM`

The system prompt prepended to every conversation.

```
SYSTEM You are concise.

SYSTEM """
Multiple lines
are fine.
"""
```

### `PARAMETER`

Sampling and load settings. Sampling values can be overridden per request; load values apply when
the model is loaded.

**Sampling**

| Parameter | Meaning |
| --- | --- |
| `temperature` | Randomness. 0 is near-deterministic; 0.7 is a common default. |
| `top_p` | Nucleus sampling threshold. |
| `top_k` | Consider only the k most likely tokens. |
| `min_p` | Minimum probability relative to the most likely token. |
| `repeat_penalty` | Penalty on tokens already in context. 1.0 disables it. |
| `repeat_last_n` | How far back the repeat penalty looks. |
| `presence_penalty`, `frequency_penalty` | OpenAI-style penalties. |
| `num_predict` (`max_tokens`) | Cap on generated tokens. |
| `seed` | Fixed seed for reproducible sampling. |
| `stop` | A stop string. Repeat the instruction for several. |

**Load-time**

| Parameter | Meaning |
| --- | --- |
| `num_ctx` (`context_length`) | Context window in tokens. |
| `num_gpu` (`gpu_layers`) | Layers offloaded to the GPU. |
| `num_thread` (`threads`) | CPU threads. |
| `num_batch` (`batch_size`) | Prompt ingestion batch size. |
| `device` | `auto`, `cpu`, `cuda`, `vulkan`, `metal`, `directml`, `npu`. |
| `tensor_split` | Per-GPU weight split, e.g. `0.6, 0.4`. Relative weights on any scale — `60, 40` is the same split. Empty divides the model in proportion to free VRAM. |
| `split_mode` | `auto`, `layer`, `row` or `none`. How the model is divided between GPUs. |
| `main_gpu` | Index of the GPU holding the KV cache and unsplit tensors. |
| `use_mmap`, `use_mlock` | Memory-map the weights; lock them in RAM. |

Unrecognised keys are kept rather than rejected, so a file written for a newer version still loads.

### `TEMPLATE`

Overrides the chat template. LocalGen uses the template stored in the GGUF file by default, and
falls back to ChatML when there is none — so this is rarely needed.

```
TEMPLATE """{{ .System }}
User: {{ .Prompt }}
Assistant: """
```

### `MESSAGE`

Few-shot examples seeded into every new conversation.

```
MESSAGE user "What is the capital of Indonesia?"
MESSAGE assistant "Jakarta."
```

### `EMBEDDING`

Marks the model as an embedding model rather than a chat model.

```
EMBEDDING true
```

### `ADAPTER`

A LoRA adapter applied over the base weights. Repeat for several, applied in order.

```
ADAPTER ./adapters/domain-tuning.gguf
```

### `LICENSE`

Free text, carried with the model.

## Examples

### A focused code assistant

```
FROM qwen2.5-coder-7b-instruct:q4_k_m

PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER num_ctx 16384
PARAMETER num_predict 2048

SYSTEM """
You are a .NET code assistant.

- Give working code before prose.
- Prefer the standard library over dependencies.
- Say when you are unsure rather than guessing an API.
"""
```

### Structured extraction

```
FROM qwen2.5-7b-instruct:q4_k_m

# Near-deterministic, with a fixed seed, so the same input gives the same output.
PARAMETER temperature 0.1
PARAMETER seed 42
PARAMETER num_predict 512

SYSTEM """
Extract invoice fields and reply with JSON only, matching:
{"invoice_number": string, "date": "YYYY-MM-DD", "total": number, "currency": string}

Use null for anything the document does not state.
"""
```

Pair this with `"response_format": {"type": "json_object"}` on the request. On the LlamaSharp
backend that constrains decoding with a grammar, so the output cannot be anything but JSON.

### An Indonesian-language assistant

```
FROM qwen2.5-7b-instruct:q4_k_m

PARAMETER temperature 0.6
PARAMETER num_ctx 8192

SYSTEM """
Anda adalah asisten yang membantu dan ramah.
Selalu jawab dalam Bahasa Indonesia yang baik dan benar.
Jika pertanyaan diajukan dalam bahasa lain, tetap jawab dalam Bahasa Indonesia.
"""

MESSAGE user "Halo, siapa kamu?"
MESSAGE assistant "Halo! Saya asisten AI yang berjalan secara lokal di komputer Anda."
```

### Long-context summarisation on a GPU

```
FROM qwen2.5-14b-instruct:q4_k_m

PARAMETER num_ctx 32768
PARAMETER num_gpu 40
PARAMETER device cuda
PARAMETER temperature 0.4

SYSTEM """
Summarise long documents. Preserve figures, dates and named entities exactly.
Structure the summary under headings that follow the document's own.
"""
```

## Errors

Parse errors name the line:

```
Modelfile error on line 4: 'temperature' expects a number but got 'high'
```

Common causes: a missing `FROM`, an unterminated `"""` block, or a misspelled instruction.
