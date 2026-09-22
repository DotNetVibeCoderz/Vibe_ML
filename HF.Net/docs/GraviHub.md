# GraviHub

**The Hugging Face Hub from .NET, and readers for the formats it serves.**

Mirrors `huggingface_hub`. Everything else in HF.Net depends on it.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;
```

## One-line use

```csharp
var directory = Hub.DownloadModel("bert-base-uncased");
var info      = Hub.ModelInfo("bert-base-uncased");
var path      = Hub.DownloadFile("bert-base-uncased", "config.json");

foreach (var hit in Hub.SearchModels("sentiment", limit: 5, task: "text-classification"))
    Console.WriteLine($"{hit.Id} - {hit.Downloads:N0} downloads");
```

`Hub` is a facade over a process-wide `HubClient` and it **blocks**. That is fine in a console app
or a notebook and wrong in a UI or a server - use `HubClient` there.

## HubClient

One instance holds one `HttpClient` and is safe to share across threads. **Create it once for the
lifetime of the process.** Creating one per download exhausts sockets under load, which surfaces as
an intermittent `SocketException` long after the code that caused it.

```csharp
using var client = new HubClient(new HubOptions
{
    Token      = null,                  // falls back to HF_TOKEN
    CacheRoot  = null,                  // falls back to HF_HUB_CACHE, then HF_HOME
    Timeout    = TimeSpan.FromMinutes(30),
    MaxRetries = 3,
});

var progress = new Progress<TransferProgress>(p => Console.WriteLine(p));

await client.SnapshotAsync("bert-base-uncased", RepoKind.Model, "main",
    allowPatterns: ["*.safetensors", "*.json"],
    ignorePatterns: null,
    progress);
```

### Why patterns matter

A typical model repository carries the same weights two or three times over: safetensors beside a
PyTorch pickle beside an ONNX export, plus TensorFlow and Flax variants on the older ones.
Downloading a snapshot without patterns routinely transfers three times what is needed - it is often
the difference between 400 MB and 1.6 GB.

`Hub.DownloadModel(id, weightsOnly: true)` - the default - applies a sensible allow-list for you.

## Repository metadata

```csharp
var info = Hub.ModelInfo("bert-base-uncased");

info.Id; info.Sha; info.LastModified; info.Downloads; info.Likes;
info.Tags; info.PipelineTag; info.Private; info.Gated;
info.HasSafeTensors;
info.FilesWithExtension(".safetensors");
info.File("config.json");
```

File listings come from the tree API rather than the `siblings` array, because `siblings` reports
names only - a caller trying to pick the smaller of two weight files, or to skip a 10 GB shard, has
nothing to decide on. `RepoFile` therefore carries `Size`, `Sha` and `IsLfs`.

## The cache

```csharp
Hub.Cache.Root;
Hub.Cache.Contains("bert-base-uncased", RepoKind.Model, "main", "config.json");
Hub.Cache.SizeInBytes();
Hub.Cache.Evict("bert-base-uncased", RepoKind.Model);
```

A cached file is **revalidated, not re-fetched**: one HEAD request compares its ETag, which for an
LFS object is the content hash. The steady-state cost of `DownloadFile` on a 400 MB checkpoint is
therefore one round trip.

A partial download goes to a `.part` file and is moved into place only once complete, so an
interrupted transfer can never be mistaken for a finished one. A `.part` left from a previous run is
resumed with a range request when the server allows it.

The layout is a plain readable directory tree rather than the blob-and-symlink arrangement the
Python client uses. Symlinks need elevation or Developer Mode on Windows, and a cache a user cannot
inspect with a file browser is one they cannot clear when a download goes wrong. The cost is that
two revisions sharing an identical file store it twice, which is the right trade here.

## safetensors

The format the Hub serves. A little-endian `u64` header length, a JSON header mapping each name to
its dtype, shape and byte range, then one contiguous data block.

```csharp
// Inspect without reading any tensor data.
foreach (var tensor in SafeTensors.Inspect(path))
    Console.WriteLine(tensor);          // bert.encoder.layer.0... F32 [768x768]

// Memory-mapped; pull out only what you need.
using var reader = SafeTensors.Open(path);
var embeddings = reader.Read("bert.embeddings.word_embeddings.weight");

// Sharded checkpoints.
var shards = SafeTensors.ReadShardIndex("model.safetensors.index.json");

// Writing.
SafeTensors.Write(path, tensors, SafeTensorDType.F32, metadata);
SafeTensors.Write(path, tensors, name => name.Contains("position") ? SafeTensorDType.F32
                                                                   : SafeTensorDType.BF16);
```

Supported dtypes on read: `Bool U8 I8 F8E4M3 F8E5M2 U16 I16 F16 BF16 U32 I32 F32 U64 I64 F64`.
Writing is limited to `F16 BF16 F32 F64`, because `NdArray` holds `double`.

**Offsets in the header are relative to the start of the data block, not to the start of the file.**
Reading them as absolute produces tensors shifted by the header length and full of plausible-looking
numbers.

The file is memory-mapped rather than read into a byte array. A checkpoint is routinely larger than
RAM, and the common case - pulling twenty of four hundred tensors - should not pay for the rest.
`ReadAll` is the eager convenience and is expensive: a 7B checkpoint in F16 is 14 GB on disk and
56 GB once widened to `double`.

### float8

`F8_E4M3` is decoded as `float8_e4m3fn`: the `fn` means *finite*, so the all-ones exponent holds
normal values and only the all-ones mantissa beside it is NaN. Decoding it as IEEE yields infinities
where the format stores 448, which then poisons every downstream sum. `F8_E5M2` is IEEE-shaped and
does have real infinities.

### Writing bfloat16

Narrowing rounds to nearest even rather than truncating. Truncation is the obvious implementation
and it biases every weight towards zero - small per value, systematic across a checkpoint, and
exactly the kind of error that survives a round-trip test written with a loose tolerance.

## PyTorch checkpoints

Most of the Hub still ships `pytorch_model.bin`, so a loader that understands only safetensors
cannot open a large share of models - including many small and widely-used ones.

```csharp
using var checkpoint = PyTorchCheckpoint.Open("pytorch_model.bin");

foreach (var tensor in checkpoint.Tensors) Console.WriteLine(tensor);
var weights = checkpoint.Read("bert.embeddings.word_embeddings.weight");
```

A `.bin` is a ZIP holding a pickled state dict beside raw tensor storages. Pickle is a stack machine
whose `REDUCE` opcode calls an arbitrary named callable - which is precisely why safetensors exists.
**Nothing in the file is executed here:** `GLOBAL` resolves against a fixed allow-list of the torch
tensor constructors and nothing else, and anything outside it throws.

Where both formats are published, prefer safetensors anyway: it needs no interpretation at all.

The pre-1.6 bare-pickle format is detected and refused with a message saying so, rather than failing
somewhere deep in the opcode loop.

## Uploading

```csharp
Hub.Upload("my-user/my-model", "adapter_config.json", "adapter_config.json");
```

**Capped at 10 MB.** The commit API takes file content inline as base64, so the whole file is held
in memory twice and travels in the request body. Anything the Hub would store in LFS - which is to
say any real weight file - needs the LFS batch protocol, which GraviHub does not implement. The cap
is enforced rather than letting a 5 GB upload fail slowly and unhelpfully.

## Errors

`HubException` carries the HTTP status and the repository, and the message names the likely cause:

| Status | Meaning |
|---|---|
| 401 / 403 | Private or gated. Set `HF_TOKEN`, and accept the licence on the Hub if gated. |
| 404 | Check the id, the revision, and whether it is a dataset rather than a model. |
| 429 | Rate limited. An authenticated client gets a much higher limit. |

Retries use exponential backoff with a cap and apply only to 429, 408 and 5xx. A 4xx will not become
true by being asked again.

## See also

[GraviTransformers](GraviTransformers.md) · [GraviDatasets](GraviDatasets.md) · [GraviOptimum](GraviOptimum.md)
