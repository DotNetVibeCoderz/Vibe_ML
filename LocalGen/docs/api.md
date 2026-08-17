# HTTP API

LocalGen serves the OpenAI REST API. Endpoint shapes, field names and the SSE framing follow
OpenAI's, so existing clients switch by changing only their base URL.

Default address: `http://127.0.0.1:11434`, with the OpenAI surface under `/v1`.

> Every endpoint below can be exercised from the **API test** screen in Admin Control, which ships
> a runnable example payload for each one and shows the raw response.
>
> [![API test](images/admin-apitest.png)](images/admin-apitest.png)

## Authentication

None by default — LocalGen binds to loopback. When `LocalGen:Server:ApiKey` is set, every request
must present it:

```
Authorization: Bearer <key>
```

`X-API-Key: <key>` is also accepted. `/` and `/api/health` stay open so probes work without a key.

---

## OpenAI-compatible endpoints

### `POST /v1/chat/completions`

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct:q4_k_m",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "Explain quantization."}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

Supported request fields: `model`, `messages`, `temperature`, `top_p`, `top_k`, `max_tokens`,
`max_completion_tokens`, `presence_penalty`, `frequency_penalty`, `seed`, `stop`,
`response_format`, `tools`, `stream`, `stream_options`.

`n` is accepted and ignored — LocalGen returns one choice.

**Streaming** — set `"stream": true` to receive server-sent events. Each frame is a
`chat.completion.chunk`; the stream ends with `data: [DONE]`. Add
`"stream_options": {"include_usage": true}` to get token counts on the final frame.

**Multimodal content** — the `content` field accepts either a string or an array of typed parts.
Image parts as `data:` URIs are decoded; remote URLs are passed through as text, since fetching
them would break offline mode.

**JSON mode** — `"response_format": {"type": "json_object"}` constrains decoding with a JSON
grammar on the LlamaSharp backend. Other backends fall back to instructing the model.

**Tools** — pass OpenAI-shaped `tools`. Calls come back as `tool_calls` on the message. LocalGen's
backends have no native function-call channel, so tools are described in the prompt and calls are
parsed out of the token stream; the wire contract is unchanged.

The *form* of that prompt follows the model. LocalGen reads the chat template baked into the GGUF
file and speaks the convention that family was fine-tuned on, because a model follows its own
convention more reliably than one it was merely instructed in:

| Family | Recognised by | A call looks like |
| --- | --- | --- |
| Qwen, Hermes, most others | `<tool_call>` in the template, or nothing more specific | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |
| Llama 3.1 / 3.2 | `<\|start_header_id\|>` | a bare `{"name": …, "parameters": {…}}`, ended by `<\|eot_id\|>` |
| Mistral, Nemo | `[TOOL_CALLS]` | `[TOOL_CALLS][{"name": …, "arguments": {…}}]` |

None of this reaches the wire — the request and the `tool_calls` you get back are OpenAI's shape
either way. A model whose family is not recognised is asked for the first form, which is what most
instruction-tuned models understand from instructions alone.

### `POST /v1/completions`

The legacy text completion endpoint. The prompt is wrapped in a single user turn, which is the
only sensible mapping onto instruction-tuned models.

### `POST /v1/embeddings`

```bash
curl http://127.0.0.1:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text", "input": ["first", "second"]}'
```

`input` may be a string or an array. Requires the LlamaSharp backend — ONNX Runtime GenAI has no
embedding path.

### `GET /v1/models`, `GET /v1/models/{id}`

Standard OpenAI shape, plus a `localgen` block that OpenAI clients ignore:

```json
{
  "id": "qwen2.5-7b-instruct:q4_k_m",
  "object": "model",
  "owned_by": "bartowski",
  "localgen": {
    "format": "Gguf", "quantization": "Q4_K_M", "size_bytes": 4683073024,
    "parameter_count_b": 7.6, "context_length": 32768,
    "capabilities": ["Chat", "TextGeneration", "ToolCalling"],
    "engines": ["LlamaSharp"]
  }
}
```

---

## Management endpoints

LocalGen's own API, outside the OpenAI surface. The desktop app, the CLI and the web UI are all
built on these.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness probe. Never requires a key. |
| `GET` | `/api/status` | Service state, uptime, resident models |
| `GET` | `/api/engines` | Installed backends, availability, recommendation |
| `GET` | `/api/metrics` | Inference and resource telemetry |
| `GET` | `/api/usage` | Per-key quota consumption and cache statistics |
| `GET` | `/api/logs?limit&level` | Recent log entries |
| `DELETE` | `/api/logs` | Clear the log buffer |
| `GET` | `/api/models` | Installed models with full metadata |
| `POST` | `/api/models/pull` | Download a model, streaming progress |
| `DELETE` | `/api/models/{id}` | Delete a model |
| `POST` | `/api/models/{id}/load` | Load into memory |
| `POST` | `/api/models/{id}/unload` | Unload |
| `GET` | `/api/models/{id}/modelfile` | The model's Modelfile |
| `GET` | `/api/catalog/search?q=&limit=` | Search remote catalogues |
| `GET` | `/api/catalog/entry?reference=` | One entry with its quantizations |

### Pulling a model

Downloads run for minutes, so progress streams as server-sent events:

```bash
curl -N -X POST http://127.0.0.1:11434/api/models/pull \
  -H "Content-Type: application/json" \
  -d '{"reference": "huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF"}'
```

```
data: {"model_id":"qwen2.5-7b-instruct:q4_k_m","file_name":"…Q4_K_M.gguf",
       "bytes_downloaded":52428800,"total_bytes":4683073024,"bytes_per_second":11534336}
…
data: {"status":"success","model":{…}}
data: [DONE]
```

Naming a repository without a file lets LocalGen choose the quantization, preferring `Q4_K_M`.
Name a file explicitly to override that.

---

## Attachments

Conversation attachments need an address: the OpenAI wire format carries images as `image_url`
parts, and a transcript renders one by URL. Uploading gives a local file that address, so nothing
has to be inlined as base64 or fetched from the internet.

### `POST /api/files`

`multipart/form-data` with a single file part.

```bash
curl -X POST http://127.0.0.1:11434/api/files -F "file=@diagram.png"
```

```json
{
  "id": "a3b5185112d19a141ddeae34.png",
  "file_name": "diagram.png",
  "media_type": "image/png",
  "url": "http://127.0.0.1:11434/api/files/a3b5185112d19a141ddeae34.png",
  "size_bytes": 84213
}
```

Files are content-addressed by hash, so uploading the same bytes twice stores them once and
returns the same id. Uploads are capped at 64 MB — attachments are context, not a file transfer
service. A client that sends the generic `application/octet-stream` content type has the media
type inferred from the extension instead, since that generic value says nothing.

### `GET /api/files/{id}`

Serves the file with its media type, and supports range requests so a client can seek within an
attached video or audio file.

### Using an attachment

Pass the returned URL as an `image_url` part:

```json
{
  "model": "qwen2.5-7b-instruct:q4_k_m",
  "messages": [{
    "role": "user",
    "content": [
      { "type": "text", "text": "What does this diagram show?" },
      { "type": "image_url",
        "image_url": { "url": "http://127.0.0.1:11434/api/files/a3b5185112d19a141ddeae34.png" } }
    ]
  }]
}
```

LocalGen loads the bytes for URLs it serves itself before the request reaches a backend. Remote
URLs are deliberately **not** fetched — doing so would make inference depend on the network, which
offline mode exists to prevent. A text-only model is told an image was attached that it cannot
read, rather than being handed a silent placeholder.

Documents are not uploaded as model input: extract their text and send it in the message. The
Admin Control Playground does both — it uploads the file for the link and sends the extracted text
for the model.

---

## Vision

A model can read images when it ships a multimodal projector — the `mmproj` file that encodes
pixels into the same embedding space as the text. `localgen pull` fetches it alongside the weights
and the store marks such a model `Vision`; nothing else is needed to turn the feature on.

```bash
localgen pull huggingface:ggml-org/SmolVLM-256M-Instruct-GGUF
```

Send the image as a base64 data URL, or as an `image_url` pointing at a file this server stores:

```json
{
  "model": "smolvlm-256m-instruct:q8_0",
  "messages": [{
    "role": "user",
    "content": [
      { "type": "text", "text": "What shape and colour is in this image?" },
      { "type": "image_url", "image_url": { "url": "data:image/png;base64,iVBORw0KGgo…" } }
    ]
  }]
}
```

Several images in one message are allowed, and they are read in the order they appear — the nth
image lands where the nth image part sat in the text, so a picture can be referred to by its
position in the sentence around it.

Two things worth knowing:

- **A model without a projector is told, not silently failed.** The message says an image was
  attached that this model cannot read, and generation continues on the text.
- **Prompt token counts exclude the image.** The tokens a picture expands into are added inside
  llama.cpp where LocalGen cannot observe them, so `usage.prompt_tokens` on a vision request
  counts the text only.

---

## Errors

Errors use OpenAI's shape:

```json
{ "error": { "message": "Model 'x' is not installed…", "type": "invalid_request_error",
             "code": "model_not_found" } }
```

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request_error` | Malformed request |
| 401 | `invalid_api_key` | Missing or wrong key |
| 403 | `offline_mode` | The operation needs the network |
| 403 | `tool_forbidden` | A tool call fell outside its allowed paths |
| 403 | `model_forbidden` | The key may not use the model it asked for |
| 429 | `rate_limit_exceeded` | A key exceeded a quota; `Retry-After` says how long to wait |
| 404 | `model_not_found` | Not installed |
| 503 | `engine_unavailable` | No backend can serve the model |

When a stream has already begun, the headers are gone — the error arrives as one final SSE frame
carrying the same error object, followed by `data: [DONE]`.
