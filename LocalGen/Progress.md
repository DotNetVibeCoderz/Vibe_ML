# LocalGen — Development Progress

*Gravicode Studios · dipimpin oleh Kang Fadhil*

A record of what has been built and verified. "Verified" means exercised at runtime, not merely
compiled — the distinction matters most for the inference path, where a clean build proves very
little.

---

## Status

| Area | State | Verified how |
| --- | --- | --- |
| Core contracts and Modelfile | Complete | 47 unit tests |
| Markdown rendering | Complete | 12 headless Avalonia tests, incl. the table → grid path |
| Attachments | Complete | Upload and serve exercised over HTTP; media types verified |
| API test panel | Complete | Sent a live chat completion from the panel: 200 OK in 2,995 ms |
| GGUF metadata reader | Complete | Read architecture, context length and chat template from a real 105 MB model |
| Model store and Hugging Face downloader | Complete | Pulled `SmolLM2-135M-Instruct-Q4_K_M` end to end; quantization auto-selected |
| LlamaSharp backend | Complete | Real generation at ~104 tok/s on CPU |
| LlamaSharp on GPU | Complete | CUDA build on an RTX 4060: 163 tokens/s against 21 on CPU |
| ONNX Runtime backend | Complete | Probes available; native library loads |
| Foundry Local backend | Complete | Compiles and probes; not exercised against an installed Foundry service |
| OpenAI-compatible API | Complete | `/v1/chat/completions` non-streaming and SSE, `/v1/models`, `/api/*` all returned correct payloads |
| Semantic Kernel layer | Complete | Builds; agent loop and plugins wired into the Playground |
| Skills and MCP | Complete | Store, gallery and kernel registration implemented |
| RAG | Complete | Chunking under test; vector backends implemented |
| .NET SDK | Complete | Used by the CLI, the web UI and the Foundry backend |
| CLI | Complete | `list`, `run` and `pull` exercised against real models |
| Avalonia Admin Control | Complete | All six screens rendered and inspected |
| Blazor web UI | Complete | Served and rendered |
| Documentation | Complete | Bilingual README, nine guides |
| Samples | Complete | Five samples; console and embedded-Avalonia exercised against real models |
| Project templates | Complete | All three packed, installed, generated and built |
| VS Code extension | Complete | Compiles under strict TypeScript; its client tested against a live server |

---

## What was verified at runtime

These are the checks that were actually run, with their results.

### Model download

`POST /api/models/pull` with `huggingface:bartowski/SmolLM2-135M-Instruct-GGUF`:

- The downloader listed the repository, skipped sharded files and chose `Q4_K_M` — the best
  size/quality trade-off among the available quantizations.
- The GGUF reader extracted `llama` architecture, 30 layers, an 8,192-token context and the
  model's own chat template, all from the file header without loading the weights.

### Inference

`POST /v1/chat/completions`, non-streaming, returned the expected OpenAI shape:

```json
{
  "id": "chatcmpl-…", "object": "chat.completion",
  "choices": [{ "index": 0, "message": { "role": "assistant", "content": "Hello, hello, hello" },
                "finish_reason": "stop" }],
  "usage": { "prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22 },
  "system_fingerprint": "localgen-0.1.0"
}
```

Streaming produced correct SSE framing: the role on the first delta, content deltas after it, and
a `data: [DONE]` terminator.

### Engines

`GET /api/engines` reported both installed backends as available, with LlamaSharp's real
llama.cpp build banner (`AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 …`) and a recommendation
derived from what probed successfully.

### CLI

`localgen list` and `localgen run` worked in embedded mode with no server running, reporting
**104 tokens/s** and **608 ms to first token** on CPU.

### Monitoring

The Admin Control dashboard read live hardware through `nvidia-smi`: **NVIDIA GeForce RTX 4060**,
41 °C, 891 MB of 8.00 GB VRAM in use.

### Templates

All three were packed, installed with `dotnet new install`, generated and built. The `--Model`
parameter reached both the generated code and its `appsettings.json`, and the generated console
application answered a prompt against a running server.

### VS Code extension

Compiles under strict TypeScript with no errors. Its HTTP client was exercised against a live
server outside VS Code — `ping`, `status`, `listModels` and a streamed chat all worked.

### OpenAI compatibility

Checked against `qwen2.5-1.5b-instruct:q4_k_m` over HTTP:

- **Function calling** — `finish_reason: "tool_calls"`, empty content, the call in `tool_calls`
- **Structured output** — `response_format: json_object` produced JSON that parsed cleanly
- **Streaming tool calls** — tool-call frame, then a closing frame carrying `finish_reason` once
- **`stream_options.include_usage`** — usage on the final chunk
- **Attachments** — upload returned `text/markdown` and `image/png` correctly; download round-tripped

### GPU

Built with `-p:LlamaBackend=cuda12` and run on an RTX 4060 (8 GB, driver 610.88). The engine
reported `Auto · Cpu · Cuda` with the banner `CUDA0 | CUDA : ARCHS = …,890`, which matches the
card's compute capability. Loading a 940 MB model raised VRAM from 881 MiB to 1914 MiB, so the
weights really were resident on the card.

Same model, same prompt, 128 tokens, three runs each:

| | CPU | CUDA |
| --- | ---: | ---: |
| Throughput | 21.2 tokens/s | 163.2 tokens/s |
| Time to first token | 339 ms | 54 ms |

### Quantization

`localgen quantize` and the Admin Control's Quantize button both convert in-process through
llama.cpp. A 258 MB F16 SmolLM2 became 107 MB at Q5_K_M and 89 MB at Q3_K_M, each in about two
seconds, and the Q5_K_M result generated at **97.8 tokens/s** — so the output is a working model,
not just a smaller file.

### Split models

Verified against `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF`, which publishes five of its
quantizations as two-part sets. LocalGen now lists each set once with the size of the whole
(Q8_0 shows 74.98 GB across both parts) where it previously hid them entirely. Downloading the
parts is covered by tests against a stubbed Hub rather than by pulling the real thing: split
models start around 50 GB because that is the file-size limit that forces the split. **Loading**
a split model has therefore not been run end to end.

### Embedded runtime

`samples/AvaloniaEmbedded` loaded a model inside its own process — no server involved — and
reported **80.8 tokens/s** with 704 ms to first token, reading the same model directory the CLI
uses.

---

## Decisions worth recording

Choices that were not obvious, and why they went the way they did.

**The server is a library as well as an executable.** Admin Control hosts it in-process, so
starting the service from the desktop exposes the models the Playground has already loaded rather
than loading a second copy into memory.

**Tool calling is a prompt protocol, not a native API.** Neither llama.cpp nor ONNX Runtime has a
structured function-call channel. Tools are described in the system prompt and calls are parsed
back out of the token stream, with a filter that holds back text which might still turn into a
tag — so a partial `<tool` never reaches the transcript.

**The agent loop is LocalGen's, not Semantic Kernel's auto-invocation.** Owning it is what lets
the Playground show each call, its arguments and its result as they happen, which is most of the
value of a local playground.

**Chroma is spoken directly over HTTP.** It has no current `Microsoft.Extensions.VectorData`
connector and the legacy one is alpha; its REST surface is small enough that talking to it
directly beats taking a pre-release dependency. The other backends go through the abstraction.

**Vector dimensions are supplied at runtime.** The embedding model decides them, so the collection
is described with a `VectorStoreCollectionDefinition` built after asking the model for one vector,
rather than from a compile-time attribute.

**The CPU backend is the build default.** The CUDA package is hundreds of megabytes; GPU support
is opt-in with `-p:LlamaBackend=cuda12` so a first clone stays quick.

---

## Fixes made during development

**Chunk overlap is capped at half the chunk size.** A test with an overlap larger than the chunk
showed the cursor advancing one character at a time, turning a 5,000-character document into
thousands of near-duplicate chunks. Overlap beyond half a chunk is not useful anyway.

**llama.cpp's native logging is redirected into the .NET logging pipeline.** Left alone it writes
its build banner and graph allocations straight to stderr, which corrupted CLI output.

**Two dependency advisories were closed** by pinning `AngleSharp` to 1.7.1 and the native SQLite
library forward within its patched line.

**Fluent's accent colour was overridden.** Checkboxes, sliders and selection were painting Windows
blue — a colour belonging to no part of the palette.

**Function calling was silently broken.** Qwen2.5 emits `<tool_call>` and the JSON, then stops —
the end-of-sequence token arrives where the closing tag should be. The stream filter released the
payload as prose, so the user saw raw JSON and the call was lost. Now an unterminated block is
parsed on its own, with brace tracking so trailing prose does not spoil it, and only genuinely
truncated JSON falls through to text. Found by sending one request; it compiled cleanly either way.

**`finish_reason` was sent twice** on a streaming tool call — once on the tool-call chunk and once
on the closing chunk. The OpenAI contract sends it once.

**Uploaded media types were wrong.** A client that sends the generic `application/octet-stream`
had that taken as the answer, so an uploaded PNG would never have rendered as an image. The
generic value is now treated as "unknown" and the extension decides.

**Chat bubbles were clipped** at the settings panel: a fixed 760 px maximum exceeded the column
when both side panels were open.

**GPU detection read the wrong API.** Accelerators were inferred from llama.cpp's system-info
banner, which in current builds lists only CPU feature flags — so a machine with working CUDA
reported CPU alone. Devices now come from the registered ggml devices, which is what actually
answers the question.

**A CUDA build silently ran on the CPU**, for two separate reasons found while testing on an
RTX 4060. The CUDA package ships no `ggml-cpu.dll` even though its `ggml.dll` imports from it, so
the libraries failed to load; and LLamaSharp judged the machine's CUDA 13 toolkit incompatible
with its CUDA 12 backend, despite the driver running those binaries fine. The build now copies the
CPU module into the CUDA folder and the loader is pointed at the CUDA library outright. Both
failures were invisible — the only symptom was that inference was slow.

**Every quantization variant reported zero bytes.** The Hugging Face model endpoint omits file
sizes unless asked for them with `blobs=true`, so the gallery was showing "0 B" beside every
choice — for a screen whose whole job is helping you weigh size against quality. Only reading a
real response revealed it; the field was present and parsed, just always absent.

**A model quantized from the Admin Control did not appear until restart.** The store rescans the
models directory only when its cache is cold, so refreshing the list after writing the file found
nothing new. The result is now registered explicitly. The CLI never had the problem — each
invocation is a fresh process.

**Split models registered as several models.** The directory scan treated every shard as its own
model, so a three-part download would list three entries, two of which cannot be loaded. Deleting
one also left the other parts behind as orphaned gigabytes.

**The VS Code extension read the wrong casing.** Running its client against a live server showed
model size and context arriving empty: the management API serialises `snake_case`, matching the
OpenAI surface it sits beside, and the client was reading `camelCase`. Only running it surfaced
this — it compiled cleanly either way.

---

## Next

Phase 5 in [PLAN.md](PLAN.md): quantization and sharded GGUF are done; vision models, multi-GPU
tensor parallelism, native tool-calling templates, batched inference and Agent Framework
orchestration remain.

Two Phase 4 items remain, both needing infrastructure rather than code: publishing the packages
to nuget.org and the extension to the marketplace, and hosting the community skill gallery index.
