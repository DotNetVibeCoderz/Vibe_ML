# LocalGen — Development Progress

*Gravicode Studios · dipimpin oleh Kang Fadhil*

A record of what has been built and verified. "Verified" means exercised at runtime, not merely
compiled — the distinction matters most for the inference path, where a clean build proves very
little.

---

## Status

| Area | State | Verified how |
| --- | --- | --- |
| Core contracts and Modelfile | Complete | 134 unit tests |
| Multi-GPU splitting | Configurable; one-GPU behaviour verified | Real GPU enumerated as `CUDA0`; a two-GPU split refused on it with the right message; 15 unit tests over the resolver |
| Batched inference | Complete | 4 concurrent requests: 24.3s serialised → 8.6s batched |
| Multi-agent orchestration | Complete | All three patterns run against a real model |
| Vision models | Complete | SmolVLM-256M read two images it had never seen and described each correctly |
| Native tool calling | Complete | Dialects checked against five real published chat templates |
| Telemetry export | Complete | Prometheus scrape served live counters over HTTP |
| Multi-tenant keys and quotas | Complete | Rate, token and model limits each refused a real request |
| Response caching | Complete | Identical prompt returned in 112 ms against 1,120 ms cold |
| Helm chart | Templates verified | `helm lint` and `helm template` clean; not applied to a cluster |
| Markdown rendering | Complete | 12 headless Avalonia tests, incl. the table → grid path |
| Windows installer | Complete | MSI built, extracted, and the CLI inside it generated tokens |
| Linux packages | Complete | `.deb` installed on a bare `debian:12` and generated at 176.9 tokens/s |
| macOS package | Written, unbuilt | No macOS machine here; scripts and templates validated only |
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
| RAG ingestion | Complete | Word, Excel, PowerPoint, EPUB, RTF, CSV, HTML converted and ingested; 22 tests over real files |
| RAG search | Complete | Questions answered from a .docx, .pptx and the second sheet of an .xlsx, over SQLite and Chroma |
| .NET SDK | Complete | Used by the CLI, the web UI and the Foundry backend |
| CLI | Complete | `list`, `run` and `pull` exercised against real models |
| Avalonia Admin Control | Complete | All six screens rendered and inspected |
| Blazor web UI | Complete | Served and rendered |
| Documentation | Complete | Bilingual README, nine guides |
| Samples | Complete | Six samples; console, embedded-Avalonia and the agent team run against real models |
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

### Vision

`ggml-org/SmolVLM-256M-Instruct-GGUF` pulled end to end — the downloader fetched the `mmproj`
alongside the weights and stored it under the canonical name, and the store marked the model
`Vision` without being told to.

Two images were drawn specifically for the test, so the model could not have seen either before
and no caption exists anywhere to recite. Same prompt, `temperature: 0`, both over
`/v1/chat/completions` as base64 `image_url` parts:

| Image sent | What came back |
| --- | --- |
| A red circle on white | "The shape is circle. The colour is red." |
| A blue square on white | "The shape is a square. The colour is blue." |

The answer tracks the pixels, which is the only thing that distinguishes real vision from a model
guessing from the prompt. Two further checks mattered as much:

- **The same model still answers text-only prompts** — "Say the word hello" returned "Hello."
- **A second image request against the still-loaded model was correct**, which is what proves the
  per-request context works. The stateful executor multimodal input requires would otherwise have
  decoded the second request on top of the first conversation's KV cache.

### Native tool calling

Detection was checked against the chat templates Hugging Face actually publishes, not fixtures:

| Model | Template says | Detected |
| --- | --- | --- |
| Qwen2.5-7B-Instruct | `<tool_call>` | hermes |
| Hermes-3-Llama-3.1-8B | ChatML + `<tool_call>` | hermes |
| Mistral-Nemo-Instruct-2407 | `[TOOL_CALLS]` | mistral |
| Llama-3.2-3B-Instruct | `<\|start_header_id\|>`, ipython | llama3 |
| Meta-Llama-3.1-8B-Instruct | `<\|start_header_id\|>`, no tools branch | llama3 |

Reading the real templates corrected two assumptions that fixtures would have preserved:

- **Llama 3 marks a tool call with nothing at all.** `<|python_tag|>` appears in none of its
  templates — it belongs to the built-in search and interpreter tools. A custom function call is a
  bare JSON object naming its arguments `parameters`, ended by `<|eot_id|>`, which llama.cpp
  consumes as end-of-sequence. That needed a markerless path through the stream filter, which
  decides from the first non-whitespace character whether the reply is a call or prose.
- **Hermes 3 is a Llama 3 model that speaks ChatML.** Detecting the architecture before the
  convention would have picked exactly the wrong dialect for it, so an explicit `<tool_call>` in
  the template now wins over the family token.

Function calling against `qwen2.5-1.5b-instruct:q4_k_m` still returns `finish_reason: tool_calls`
with the call in `tool_calls` and empty content, unchanged by the rework.

### Batched inference

Four concurrent requests against `qwen2.5-1.5b-instruct:q4_k_m`, each generating 128 tokens, the
model already warm so loading is excluded:

| | Serialised | Batched (4 sequences) |
| --- | ---: | ---: |
| Wall clock | 24.29 s | **8.55 s** |
| Aggregate throughput | 20.7 tok/s | **59.3 tok/s** |

The serialised run's completions land at 6.3 s, 12.5 s, 18.6 s and 24.3 s — one after another,
which is the queue doing exactly what it was designed to do. Batched, all four finish together at
8.55 s. Each answer stayed on its own subject, which is what rules out the sequences bleeding into
one another in the shared KV cache.

Also checked, because a batching bug shows up in the edges rather than the throughput number:

- **Admission control** — six requests against a four-sequence limit: four finished at ~8 s and
  the two queued behind them at ~14.5 s.
- **Streaming** — SSE framing unchanged, `finish_reason` once, usage on the final chunk.
- **Tool calling** — still returns `finish_reason: tool_calls` with the call in `tool_calls`.
- **Abandonment** — a client hung up mid-stream and the pump kept serving; the next request
  answered normally. A dropped reader has to retire its conversation, or one abandoned stream
  would hold a KV slot until the model unloaded.

### Multi-agent orchestration

Three patterns, all run end to end against `qwen2.5-1.5b-instruct:q4_k_m` through
`samples/MultiAgentTeam`:

- **Sequential** — Researcher → Critic → Writer, 20.0 s, each doing its own job and the Writer
  producing one finished paragraph from the chain.
- **Concurrent** — all three answered simultaneously (7.8 s, 11.3 s, 21.8 s) and a reducer
  consolidated them. This is the pattern that depends on batched inference; without it the three
  members simply queue.
- **Handoff** — the coordinator called `ask_researcher`, the specialist answered in 2.6 s, and the
  coordinator wrote the final answer from what came back.

The handoff result came with a caveat worth recording, because it is a property of small models
rather than a defect. With all three specialists offered, the 1.5B coordinator did not delegate at
all — it answered from its own knowledge, since it is perfectly capable of producing *an* answer
and that is the path of least resistance. Making the instruction directive ("you must not answer
this yourself; call at least one ask_* function first") moved it from ignoring the team to naming
a specialist in prose without actually calling it.

Before concluding that was the model, the plumbing was checked separately: the delegation plugin
advertises `Team-ask_researcher`, `Team-ask_critic` and `Team-ask_writer` with the right parameter
schema through the same `FunctionChoiceBehavior` the chat service uses, and a `FunctionCallContent`
against one invokes it. With a single specialist — the smallest possible routing decision — the
same model delegated correctly on the first attempt. So the machinery is complete and the limit is
how many colleagues a 1.5B model can choose between.

### Telemetry, quotas and caching

Run with three keys configured, Prometheus on and the cache enabled:

- **Authentication** — no key, a wrong key: 401 in the OpenAI error shape. Both the
  `Authorization: Bearer` and `X-API-Key` forms accepted.
- **Rate limit** — a key capped at 3/minute served three requests then returned 429 with
  `Retry-After: 60`, while another key was unaffected.
- **Token quota** — a key capped at 10 tokens/day was refused after one 25-token generation, with
  `Retry-After: 83617` — 23.2 hours, which is the rolling day emptying an hour at a time.
- **Model allow-list** — a key restricted to another model got 403 `model_forbidden`.
- **Cache** — the same prompt at `temperature: 0` returned in **112 ms** against **1,120 ms** cold,
  byte-identical including usage. Two sampled requests at `temperature: 0.8` produced different
  answers and added no cache entry, which is the default policy holding.
- **Prometheus** — `/metrics` served `localgen_cache_hits_responses_total`,
  `localgen_tenant_tokens_today` and the ASP.NET Core instrumentation, unauthenticated by default
  so a cluster scraper needs no credentials.

### Document ingestion

Office, ebook and tabular formats are converted to markdown by
[ElBruno.MarkItDotNet](https://github.com/elbruno/ElBruno.MarkItDotNet), locally and with no
network, so ingestion still works in offline mode.

Which formats go through the converter was decided by looking at the output rather than by taking
everything on offer. The test is whether the conversion recovers structure retrieval can use:

| Format | Path | Why |
| --- | --- | --- |
| `.docx`, `.xlsx`, `.pptx`, `.epub`, `.rtf` | converted | Nothing in LocalGen could read them at all |
| `.html` | converted | Markdown keeps the headings and tables that the previous text-only extraction threw away |
| `.csv`, `.tsv` | converted | A markdown table is retrievable in a way a comma-separated line is not |
| `.pdf` | kept LocalGen's own | The converter separates pages with a horizontal rule and no number; LocalGen writes `[page N]`, which is what a citation from a long manual needs |
| `.json`, `.xml`, `.yaml` | kept as text | Conversion wraps them in a fenced code block: the same text, re-printed, none of it easier to find |
| source code | kept as text | Same |
| images | kept as text | The converter emits a placeholder with no OCR, which is no more use than the one LocalGen already wrote |

Converted text is chunked on its headings rather than on blank lines, so a passage retrieved from
the middle of a report arrives labelled with the section it came from. That is what
`ExtractedDocument.IsMarkdown` is for — the kind stays `docx`, because that is what a citation
should say, while the chunker is told the text is markdown.

**Verified end to end**, against a folder holding a real `.docx`, `.xlsx`, `.pptx`, `.rtf`, `.csv`,
`.html`, `.json`, `.xml` and `.yaml`. Embeddings from `nomic-embed-text-v1.5:q4_k_m`, answers from
`qwen2.5-1.5b-instruct:q4_k_m`, retrieval over SQLite — the default backend — and over Chroma,
which returned the same passages and the same answers:

| Question | Answer | Where the fact lives |
| --- | --- | --- |
| Which region led growth, and what revenue did it report? | "Asia Pacific led growth, and it reported a revenue of 4.2M." | Prose and a table in the **.docx** |
| What is on the roadmap? | "Shipping installers and verifying multi-GPU functionality." | Slide text in the **.pptx** |
| Are the figures audited? | "The figures are unaudited." | The **second worksheet** of the .xlsx |

The third question is the one worth keeping: that sentence exists only on a sheet the converter
had to reach past the first one to find, and it came back through the retrieval path rather than
from a converter's return value.

22 tests cover the extractor itself, against files authored at run time — a real Word document
built with the OpenXML SDK, a real workbook from ClosedXML, a real EPUB assembled as an OPC
package — rather than committed binaries nobody can review in a diff.

### Multi-GPU splitting

This machine has one GPU, so what could be verified here is everything except the thing the
feature is for. Recorded precisely, because the gap matters.

**What the hardware answered.** Built with `-p:LlamaBackend=cuda12` on the RTX 4060. The engine
now enumerates the accelerators ggml registered rather than only naming the device families, and
`GET /api/engines` returned `"accelerators": ["CUDA0"]` — one entry, from the real device list.
`localgen engines` prints the same as a `GPUs` row, and appends `(splittable)` only above one.

That count is the number that decides everything else, and it is not the number of cards in the
machine: a CPU-only build sees none, and `CUDA_VISIBLE_DEVICES` or a driver that bound one card
narrows it. Configuring a split against the card count rather than against this one is how a
model ends up on fewer GPUs than intended.

Configured against that single GPU, with a real model loading each time:

| Configuration | What happened |
| --- | --- |
| `TensorSplit: [0.6, 0.4]` | `warn: tensor_split is set but only one GPU is visible, so the whole model goes on it` — then generated normally |
| `MainGpu: 1` | `warn: MainGpu is set to 1 but only 1 accelerator(s) are visible; using device 0` — then generated normally |

Both loaded and answered. That is the intended behaviour rather than leniency: a wrong split
should cost you the split, not the model.

**What tests answered instead of hardware.** `TensorSplitPlan` resolves a configured split against
a device count, so the two-, three- and four-GPU cases are ordinary unit tests — 15 of them, over
normalisation, a split longer than the device list, one shorter than it, zero weights, negative
and NaN weights, and the scale-invariance that makes `60, 40` and `0.6, 0.4` the same split. Seven
more drive the Engine screen through a stub backend reporting two accelerators, which is the only
way to see the panel appear, name both devices, and resolve `0.6, 0.4` to `CUDA0 60% · CUDA1 40%`
before anything is loaded.

**What the chart answered.** `helm template` with `gpu.count=2` and a two-entry split renders the
right environment variables. A three-entry split against `gpu.count=2`, and any split with
`gpu.enabled=false`, now fail the render with a message naming both numbers — the mistake is
cheaper to catch there than in a pod's logs.

**What is left.** Whether a model split across two cards produces correct tokens at a useful
speed. That needs two cards. The line that applies the resolved split to `ModelParams` is trivial;
it is called unverified because llama.cpp accepts a wrong split silently, so a successful load
would not be evidence of anything.

### Installers

**Windows — verified end to end.** `packaging/windows/build.ps1` publishes the CLI, the server and
the Admin Control self-contained into one folder — 559 files, 275 MB — and WiX packs them into an
85 MB MSI. Extracted with `msiexec /a`, the payload landed as expected: `localgen.exe`,
`localgen-server.exe` and `LocalGen.Desktop.exe` in `Program Files\LocalGen`.

The check that mattered was running what came out of the installer rather than what went into it:

```
localgen.exe --version                     → 0.1.0
localgen.exe run smollm2-135m… --stats     → 14 prompt + 9 completion tokens · 100.3 tokens/s
```

That is inference from an installed copy on a machine where nothing resolves through the SDK, so
the self-contained payload really is self-contained.

**Linux — verified end to end.** `packaging/linux/build.sh` under `mcr.microsoft.com/dotnet/sdk:10.0`
produced a 78 MB `.deb` and a 118 MB tarball from a 307 MB payload. The package was then installed
into a bare `debian:12` container — a machine with no .NET, no SDK and none of this repository:

```
apt-get install ./localgen_0.1.0_amd64.deb
which localgen                             → /usr/bin/localgen
localgen --version                         → 0.1.0
getent passwd localgen                     → localgen:x:100:101::/var/lib/localgen:/usr/sbin/nologin
localgen run smollm2-135m… --stats         → 11 prompt + 54 completion tokens · 176.9 tokens/s
```

The symlink, the system user, the systemd unit and the declared dependencies all came out right,
and the last line is the one that matters: real inference from an installed package on a clean
machine.

Two things were found by reading the package back rather than by building it:

- **An empty `Conflicts:` field would have truncated the control file.** The CUDA package needs
  `Conflicts`/`Replaces`/`Provides` against `localgen` because both install the same paths; the CPU
  package needs none. Substituting an empty variable left a blank line, and dpkg reads a blank
  line as the end of the stanza — `Homepage` and the whole `Description` would have been dropped
  silently. Blank lines are now stripped before the file is written.
- **The CUDA package was named `localgen_0.1.0_amd64-cuda12.deb`.** Debian tooling expects
  `<package>_<version>_<arch>.deb`, and the package is `localgen-cuda12`, so the name is now
  `localgen-cuda12_0.1.0_amd64.deb`.

`--skip-publish` came out of that: repackaging to check a control file took seconds, publishing
three self-contained applications to reach it took minutes.

**macOS — written, not built.** No macOS machine was available. The script, `Info.plist`,
entitlements and `distribution.xml` are validated as far as they can be from here (shell syntax,
XML well-formedness) and no further. The release workflow builds it on `macos-14` and `macos-13`,
which is where it will first actually run.

**Signing is wired but unexercised.** Each script signs when credentials are present and warns
when they are not. Nothing here can produce a Windows code-signing certificate or an Apple
Developer ID, so those paths have been written and reviewed rather than run.

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

**Semantic Kernel is pinned to the version its vector-store connectors were released with.** The
newest Semantic Kernel and the newest connectors cannot both be had: 1.79 requires an abstractions
package that removed a type the connectors call, so every vector search threw while ingestion went
on succeeding — the worst shape for a bug, since a corpus indexes cleanly and only fails when
someone asks it something. Holding Semantic Kernel at 1.74 makes the set agree on abstractions
10.1.0.

The alternative was to wait for connectors built against the current abstractions, which would
have meant shipping a RAG feature whose search worked on one backend out of five. Nothing LocalGen
uses came from 1.75–1.79: the downgrade compiled with no source change, and all three
orchestration patterns, the kernel plugins and the Playground still run. The four packages move
together or not at all — that constraint is written into `Directory.Packages.props` beside them,
because the failure it prevents does not show up at build time.

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

**RAG had never been run end to end, and two things were waiting there.** Both were found by
ingesting a folder rather than by any test, and neither had anything to do with document formats:

- **The SQLite index could not be created at all.** The collection asked for
  `DistanceFunction.CosineSimilarity`, which the sqlite-vec connector rejects outright — it
  computes cosine *distance*. Every ingest failed at the first file. The distance function is now
  chosen per backend and the scores converted back to similarity, so `SearchHit.Score` still means
  what its documentation says on every provider.
- **Search threw on every `VectorData` connector** — SQLite, Qdrant, Azure AI Search and the
  in-memory store alike, while ingestion succeeded. They call `VectorSearchOptions.OldFilter`,
  removed in abstractions 10.5.0, while Semantic Kernel 1.79 requires 10.5.2 or newer. Fixed by
  holding Semantic Kernel at 1.74, the release the connectors shipped alongside, so the whole set
  agrees on abstractions 10.1.0. See the decision recorded below.

**A sample swallowed the errors that would have explained it.** `DocumentQA` registered a logging
level but no logging provider, so ingestion warnings went nowhere and a run that indexed nothing
printed "Indexed 0 file(s)" with no reason. That is what made the SQLite failure above look like
an empty folder.

**An empty Debian `Conflicts:` field silently truncated the control file.** A blank line ends the
stanza as far as dpkg is concerned, so the fields after it — including the entire description —
would have gone missing from the CPU package while the build reported success. Found by reading
`dpkg-deb -I` output rather than by the package building.

**The VS Code extension read the wrong casing.** Running its client against a live server showed
model size and context arriving empty: the management API serialises `snake_case`, matching the
OpenAI surface it sits beside, and the client was reading `camelCase`. Only running it surfaced
this — it compiled cleanly either way.

---

## Next

**Phase 5.** Complete as far as this machine can take it. Multi-GPU is configurable end to end and
its one-GPU behaviour is verified; splitting a model across two cards needs two cards. Anyone with
a two-GPU host can settle it in an afternoon: set `TensorSplit`, load a 30B model that fits in
neither card alone, and check `nvidia-smi` shows weights on both.

Two things are worth doing next that the work above exposed rather than completed. Batching does
not yet apply to vision models, and the daily token quota undercharges image requests because the
image's tokens are counted inside llama.cpp. Neither blocks anything; both are recorded under
known limitations.

**Phase 6.** The installers are built, and the Windows one has been run out of its own package.
What is left is not code: a Windows code-signing certificate and an Apple Developer Program
membership, without which macOS will not open the package at all. The Kubernetes operator stays
deferred with a rationale recorded in [PLAN.md](PLAN.md).

**Phase 4.** Still two infrastructure items: publishing the packages to nuget.org and the
extension to the marketplace, and hosting the community skill gallery index.

**RAG.** Complete and verified end to end on the default backend. The one thing to watch is the
Semantic Kernel pin: raising it without the connectors reintroduces the search failure, which is
silent until someone runs a query.
