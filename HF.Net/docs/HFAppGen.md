# HFAppGen

**An IDE whose assistant builds HF.Net applications from a prompt.**

The assistant is **Jack, the Code Bender**. It writes the files, runs the build, and fixes what the
compiler says — rather than printing code into a chat window for you to paste.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```bash
dotnet run --project tools/HFAppGen
dotnet run --project tools/HFAppGen -- --selftest      # one headless round trip
```

## The window

![HFAppGen with a project open](screenshots/hfappgen-main.png)

The banded rule under the toolbar is the **offset rail** — eight spans, one per HF.Net library, each
as wide as that library's real share of the source. GraviHub is nearly a third of it, and the rail
says so.

## Design

The identity is built on HF.Net's characteristic artifact: the **safetensors header**, a contiguous
buffer carved into named, unequal byte spans. A gutter, a file tree and a build log are the same
shape of thing.

Two accents, each with a meaning that never varies:

| | Used for |
|---|---|
| **Amber** `#FFB454` | Identity and action |
| **Cyan** `#57C7E3` | Measured quantities only — offsets, shapes, timings, token counts |

So the caret readout, numeric literals in the editor and a latency figure in the logs are all the
same colour, and that colour means one thing. Status colours and library colours are separate
palettes; in the predecessor app they were one, which meant green had to mean both *success* and one
particular library, and neither reading survived the other.

There is no decorative display face. In an IDE the monospace **is** the display face: shapes,
offsets and timings are the content with the most personality, so Cascadia Mono carries it and Inter
handles the chrome.

## Configuration

![Settings](screenshots/hfappgen-settings.png)

Everything lives in `app.config` and is editable from **Tools → Settings**. The UI writes changes
back to that file, so hand edits and UI edits are equivalent.

```xml
<add key="llm.provider" value="AzureOpenAI" />   <!-- OpenAI | AzureOpenAI | Anthropic | Google | Ollama -->
<add key="llm.model" value="gpt-5-mini" />
<add key="llm.apiKey" value="" />
<add key="llm.endpoint" value="" />
<add key="llm.temperature" value="0.3" />
<add key="llm.systemPrompt" value="..." />
```

Keys are blank in source control. Environment variables always win:

| Variable | Overrides |
|---|---|
| `HFAPPGEN_APIKEY` | `llm.apiKey` |
| `HFAPPGEN_ENDPOINT` | `llm.endpoint` |
| `TAVILY_API_KEY` | `tools.tavilyApiKey` |
| `HF_TOKEN` | used by generated apps to reach the Hub |

## Providers

Semantic Kernel supplies OpenAI, Azure OpenAI, Google and Ollama. **Anthropic has no official
connector**, so Claude is served by a hand-written `IChatCompletionService` against the Messages API
in `Services/AnthropicChatCompletionService.cs`.

`gpt-5`, `o1`, `o3` and `o4` reject `max_tokens` and require `max_completion_tokens`, and reject a
non-default temperature. `AssistantService.UsesCompletionTokenLimit` routes around that.

## What the assistant can do

| Function | Purpose |
|---|---|
| `HFNetReference(library)` | The real API surface of an HF.Net library |
| `HFNetExample(task)` | A complete working program for a common task |
| `HFNetProjectReferences(libraries, directory)` | The exact `<ItemGroup>` a generated `.csproj` needs |
| Project and file functions | Create a project, write, read and list files |
| `BuildProject` | Run the build and return what it said |
| `SearchInternet` | Tavily |
| `ScrapeWebPage`, `MathCalculation`, date and time | The usual utilities |

### Why the reference tool matters

HF.Net is newer than any training corpus, so a model asked to write against it **invents method
names that compile in its head and not in the project**. The reference is not an optimisation here;
it is the only source of truth available.

`HFNetProjectReferences` exists because of a failure caught in testing. Asked to build a project,
the assistant guessed at a NuGet package called `Gravicode.HFNet.GraviTransformers`, hit NU1101 —
HF.Net is not published yet — and then **wrote a fake shim implementing the API so the build would
go green**. A project that compiles against an invented type is worse than one that does not
compile, because it looks finished. The tool now hands over the resolved local paths, and the system
prompt forbids shims outright.

## Templates

![New project dialog](screenshots/hfappgen-new-project.png)

New Project offers **Blank** or one of twelve templates:

| | |
|---|---|
| Sentiment analysis | Classify text with a fine-tuned model |
| Semantic search | Embed a corpus once, rank it against a query |
| Masked language model | Ask an encoder to fill in a blank |
| Tokenizer laboratory | Compare WordPiece, byte-level BPE and Unigram |
| Hub explorer | Search, inspect a repository, read its tensors |
| Dataset pipeline | Load, inspect, split and batch |
| ONNX inference | Run an export through ONNX Runtime and measure it |
| LoRA adapters | Attach, merge and save adapters |
| Text to image | Stable Diffusion through ONNX |
| Notebook: text analysis | .NET Interactive, for data scientists |
| Notebook: performance study | Managed inference against ONNX Runtime |

Templates live in code rather than as loose files so one cannot go missing from an installed copy,
and so the project name substitutes properly rather than by find-and-replace over a directory.

Library references cannot be a fixed relative path — a project created in Documents is nowhere near
the repository — so the path is computed from the new project's location to wherever the libraries
actually are.

## Self-test

The LLM path is the one part that cannot be covered by a unit test: it needs a real endpoint, a real
key and a real model.

```bash
dotnet run --project tools/HFAppGen -- --selftest
dotnet run --project tools/HFAppGen -- --selftest "Create a project in C:\tmp\Demo that ..."
```

```
provider : AzureOpenAI
model    : gpt-5-mini
endpoint : https://.../
key      : set (84 chars)

> In one sentence: what does GraviHub do in HF.Net? Call HFNetReference first.

GraviHub is the HF.Net Hugging Face Hub client and file readers, providing model and
dataset search, download/upload and repo info, cache management, and SafeTensors/PyTorch
checkpoint readers.
```

Exit codes: `0` success, `2` not configured, `3` kernel build failed, `4` request failed, `5` empty
reply.

## Keyboard

| | |
|---|---|
| `Ctrl+Enter` | Send |
| `Ctrl+Shift+N` / `Ctrl+O` / `Ctrl+S` | New project / Open / Save |
| `F6` / `F5` | Build / Run |
| `Ctrl+G` / `Ctrl+K` | Go to line / Format |
| `Ctrl+B` / `Ctrl+L` | Toggle chat / logs |
| `Ctrl+,` | Settings |

## Notes for maintainers

**Avalonia bindings must not cast in the path.**
`{Binding $parent[Window].((vm:Shell)DataContext).X}` compiles and then throws at startup. Use
`{Binding $parent[Window].DataContext.X}`.

**AvaloniaEdit's bundled `.xshd` definitions are written for a white page.**
`Services/SyntaxTheme.cs` remaps their named colours onto the `Code*` tokens. It mutates
`HighlightingManager.Instance`, which is process-wide, so it must be re-run on
`ActualThemeVariantChanged`.

**The AvaloniaEdit package is `Avalonia.AvaloniaEdit` but the assembly is `AvaloniaEdit`**, so the
style include is `avares://AvaloniaEdit/Themes/Fluent/AvaloniaEdit.xaml`.

## See also

[Getting started](getting-started.md) · [GraviTransformers](GraviTransformers.md)
