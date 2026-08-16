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

| Kind | Extensions |
| --- | --- |
| PDF | `.pdf` — text extracted per page, with page markers kept for citation |
| Markdown | `.md`, `.markdown` — chunked by heading |
| HTML | `.html`, `.htm` — scripts and styles stripped |
| Plain text | `.txt`, `.log`, `.csv`, `.json`, `.xml`, `.yaml`, … |
| Source code | `.cs`, `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, … |
| Images | `.png`, `.jpg`, … — indexed by name, not OCR'd |

Images are deliberately not OCR'd: shipping an OCR engine would be a large dependency for
something most users would not reach for. An image is recorded so a search can surface "there is a
diagram called X here", and a vision-capable model can read it when asked.

## Chunking

Chunking decides whether retrieval returns something useful. A fixed character count alone cuts
sentences in half and strands the answer across two chunks, so LocalGen picks the largest natural
boundary that fits — paragraph, then sentence, then word — and carries an overlap forward so a
fact spanning a boundary still appears whole somewhere.

Markdown is chunked by heading, and each chunk carries its heading, so a retrieved passage still
says what it is about.

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
