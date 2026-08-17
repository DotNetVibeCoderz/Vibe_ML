# RAG

Ingest documents, search them by meaning, and give the model passages it can cite. Embeddings are
produced by a local model, so retrieval works offline.

## Setup

RAG needs an embedding model in addition to your chat model:

```bash
localgen pull huggingface:nomic-ai/nomic-embed-text-v1.5-GGUF
```

```json
{
  "LocalGen": {
    "Rag": {
      "Provider": "sqlite",
      "EmbeddingModel": "nomic-embed-text-v1.5:q4_k_m",
      "ChunkSize": 1000,
      "ChunkOverlap": 200,
      "TopK": 5,
      "MinRelevance": 0.5
    }
  }
}
```

Embeddings require the LlamaSharp backend — ONNX Runtime GenAI has no embedding path.

## Vector backends

| Provider | `ConnectionString` | When to use it |
| --- | --- | --- |
| `sqlite` | Optional; defaults to `rag.localgen.db` in the data directory | The default. No server to run. |
| `inmemory` | — | Tests, and experiments you do not want to persist |
| `qdrant` | `http://localhost:6334` | Large collections, filtered search |
| `chroma` | `http://localhost:8000` | An existing Chroma deployment |
| `azureaisearch` | `Endpoint=https://…;ApiKey=…` | Azure-hosted, hybrid search |

SQLite is the default because it needs nothing installed: a laptop gets working RAG out of the box.

Chroma is spoken directly over its HTTP API rather than through
`Microsoft.Extensions.VectorData` — it has no current connector, and its REST surface is small
enough that talking to it directly beats a pre-release dependency. Nothing above the storage
interface knows the difference.

## Supported documents

| Kind | Extensions | How it is read |
| --- | --- | --- |
| Word | `.docx` | Converted to markdown — headings, lists and tables kept |
| Excel | `.xlsx` | One markdown table per worksheet, each under its sheet name |
| PowerPoint | `.pptx` | One section per slide |
| EPUB | `.epub` | Converted to markdown, chapter by chapter |
| Rich text | `.rtf` | Converted to markdown |
| PDF | `.pdf` | Text per page, with `[page N]` markers kept for citation |
| Markdown | `.md`, `.markdown` | Read as-is |
| HTML | `.html`, `.htm` | Converted to markdown; scripts and styles dropped |
| Tabular | `.csv`, `.tsv` | Converted to a markdown table |
| Plain text | `.txt`, `.log`, `.json`, `.xml`, `.yaml`, `.toml`, `.ini` | Read as-is |
| Source code | `.cs`, `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, … | Read as-is |
| Images | `.png`, `.jpg`, `.svg`, … | Recorded by name, not OCR'd |

The office and ebook formats go through
[ElBruno.MarkItDotNet](https://github.com/elbruno/ElBruno.MarkItDotNet), which converts them to
markdown locally — no service, no network, so ingestion still works in offline mode.

**Why some formats are converted and others are not.** The test is whether the conversion recovers
structure that retrieval can use, not whether a converter exists for the format:

- A `.docx` becomes markdown because its headings then survive into the chunker, and every
  retrieved passage arrives labelled with the section it came from.
- A `.pdf` keeps LocalGen's own reader because the page number is what makes a citation from a
  300-page manual worth anything, and a general-purpose converter marks page breaks with a
  horizontal rule instead.
- `.json`, `.xml` and `.yaml` are read as-is. Converting them would wrap the file in a fenced code
  block: the same text, re-printed, none of it easier to find.
- Source code is read as-is for the same reason.

Images are deliberately not OCR'd: shipping an OCR engine would be a large dependency for
something most users would not reach for. An image is recorded so a search can surface "there is a
diagram called X here", and a vision-capable model can read it when asked.

One property of the conversion is worth knowing about: characters that would otherwise be markdown
syntax come back escaped, so `Q4_K_M` is stored as `Q4\_K\_M`. Retrieval is unaffected — an
embedding is unbothered by a backslash — but a passage quoted back to a model carries them.

## Chunking

Chunking decides whether retrieval returns something useful. A fixed character count alone cuts
sentences in half and strands the answer across two chunks, so LocalGen picks the largest natural
boundary that fits — paragraph, then sentence, then word — and carries an overlap forward so a
fact spanning a boundary still appears whole somewhere.

Markdown is chunked by heading, and each chunk carries its heading, so a retrieved passage still
says what it is about. That path covers everything that arrives as markdown, which now includes
converted Word, Excel, PowerPoint, EPUB, RTF and HTML — the structure the conversion recovers is
the structure the chunker splits on.

Overlap is capped at half the chunk size. Beyond that each chunk barely advances past the last,
turning a long document into thousands of near-duplicates.

| Setting | Guidance |
| --- | --- |
| `ChunkSize` | 500–1500 characters. Smaller is more precise; larger keeps more context together. |
| `ChunkOverlap` | 10–20% of the chunk size. |
| `TopK` | 3–10. More passages cost context. |
| `MinRelevance` | 0.5 is a reasonable floor. Raise it if irrelevant passages get through. |

## From the Playground

With the Skills function enabled, the model can reach the index itself:

| Function | Purpose |
| --- | --- |
| `search_documents(query, limit, collection)` | Retrieve relevant passages with sources |
| `ingest_file(path, collection)` | Add one document |
| `ingest_directory(path, recursive, collection)` | Add a folder |
| `forget_source(source)` | Remove a document |

> Ingest the docs folder, then tell me what LocalGen does about offline mode.

Retrieval is offered as a tool rather than stuffed into every prompt, so the model only pays for
context when the question calls for it — and can search again with a better query if the first
attempt misses.

## From code

```csharp
using LocalGen.Rag;

var rag = services.GetRequiredService<RagService>();

await rag.IngestFileAsync("handbook.pdf", collection: "hr");
await rag.IngestDirectoryAsync("./docs", collection: "product", recursive: true);

var hits = await rag.SearchAsync("what is the leave policy", top: 5, collection: "hr");

foreach (var hit in hits)
{
    Console.WriteLine($"{hit.Record.Source} · {hit.Score:P0}");
    Console.WriteLine(hit.Record.Text);
}

// Formats the passages as a context block with sources, ready to prepend to a prompt.
var context = RagService.FormatContext(hits);
```

Register it with:

```csharp
services.AddLocalGenRuntime(configuration);
services.AddLocalGenRag();
```

## Collections

Collections keep unrelated corpora apart, so an HR question does not retrieve product
documentation. Pass a collection name to ingestion and to search; omitting it uses `default`.

## Re-ingesting

Chunk ids are derived from the source and chunk index, so re-ingesting a document replaces its
chunks rather than duplicating them. Editing a file and adding it again leaves no stale passages
behind.

## Notes

- **Vector dimensions come from the model.** The index is created after asking the embedding model
  for one vector, since the dimension is not known until then. Changing the embedding model means
  rebuilding the index — clear it and re-ingest.
- **Ingestion embeds every chunk**, so a large corpus takes a while on CPU. It is a one-off cost.
- **A file that fails to read is skipped** during a bulk ingest rather than abandoning the run.
  The reason is logged, so a run that indexes fewer files than expected has an explanation — make
  sure your host has a logging provider registered, or the warning has nowhere to go.
- **Semantic Kernel is pinned to the 1.74 line so the vector-store connectors work.** The
  connectors call an abstraction that newer Semantic Kernel releases require a version of
  `Microsoft.Extensions.VectorData.Abstractions` that no longer has, and the symptom is that every
  search throws while ingestion happily succeeds. Raise `Microsoft.SemanticKernel`,
  `.Abstractions`, `.Core` and `Microsoft.Extensions.VectorData.Abstractions` together, or not at
  all. See [PLAN.md](../PLAN.md#known-limitations).
