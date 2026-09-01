// Retrieval-augmented question answering, end to end.
//
// Ingests a folder, then answers questions from it with citations. Everything runs locally: the
// embedding model, the vector index and the chat model.
//
//   dotnet run --project samples/DocumentQA -- ./docs

using LocalGen.Core.Configuration;
using LocalGen.Core.Inference;
using LocalGen.Engines.LlamaSharp;
using LocalGen.Rag;
using LocalGen.Runtime;
using LocalGen.Runtime.Sessions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

var folder = args.FirstOrDefault() ?? "./docs";

if (!Directory.Exists(folder))
{
    Console.Error.WriteLine($"No directory at '{folder}'.");
    Console.Error.WriteLine("Usage: dotnet run --project samples/DocumentQA -- <folder>");
    return 1;
}

var configuration = new ConfigurationBuilder()
    .AddInMemoryCollection(new Dictionary<string, string?>
    {
        // SQLite needs no server, which is what makes this sample runnable as-is.
        ["LocalGen:Rag:Provider"] = "sqlite",
        ["LocalGen:Rag:EmbeddingModel"] = Environment.GetEnvironmentVariable("LOCALGEN_EMBED_MODEL")
                                          ?? "nomic-embed-text-v1.5:q4_k_m",
        ["LocalGen:Rag:ChunkSize"] = "800",
        ["LocalGen:Rag:ChunkOverlap"] = "150",
        ["LocalGen:Rag:TopK"] = "4"
    })
    .AddEnvironmentVariables("LOCALGEN_")
    .Build();

var services = new ServiceCollection();

// A console provider, not just a level: ingestion skips a file it cannot read and logs why, and
// without somewhere for that warning to go the sample reports "Indexed 0 file(s)" and no reason.
services.AddLogging(logging =>
{
    logging.AddSimpleConsole(console => console.SingleLine = true);
    logging.SetMinimumLevel(LogLevel.Warning);
});
services.AddLocalGenRuntime(configuration);
services.AddLocalGenRag();
services.AddLlamaSharpEngine();

await using var provider = services.BuildServiceProvider();

var rag = provider.GetRequiredService<RagService>();
var sessions = provider.GetRequiredService<ModelSessionManager>();
var options = provider.GetRequiredService<Microsoft.Extensions.Options.IOptions<LocalGenOptions>>().Value;

var chatModel = Environment.GetEnvironmentVariable("LOCALGEN_CHAT_MODEL")
                ?? "qwen2.5-7b-instruct:q4_k_m";

Console.WriteLine($"Embedding model: {options.Rag.EmbeddingModel}");
Console.WriteLine($"Chat model:      {chatModel}");
Console.WriteLine();

// ─────────────────────────  Ingest  ─────────────────────────

Console.WriteLine($"Indexing {folder}…");

try
{
    var results = await rag.IngestDirectoryAsync(folder, recursive: true);

    var chunks = results.Sum(r => r.ChunkCount);
    Console.WriteLine($"Indexed {results.Count} file(s) as {chunks} chunk(s).");
    Console.WriteLine();
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Ingestion failed: {ex.Message}");
    Console.Error.WriteLine();
    Console.Error.WriteLine("Both models need to be installed:");
    Console.Error.WriteLine("  localgen pull huggingface:nomic-ai/nomic-embed-text-v1.5-GGUF");
    Console.Error.WriteLine("  localgen pull huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF");
    return 1;
}

// ─────────────────────────  Ask  ─────────────────────────

Console.WriteLine("Ask a question about those documents, or /bye to exit.");
Console.WriteLine();

while (true)
{
    Console.Write("> ");

    var question = Console.ReadLine()?.Trim();

    if (question is null or "/bye" or "/exit")
    {
        break;
    }

    if (question.Length == 0)
    {
        continue;
    }

    var hits = await rag.SearchAsync(question);

    if (hits.Count == 0)
    {
        Console.WriteLine("Nothing relevant was found in those documents.");
        Console.WriteLine();
        continue;
    }

    Console.WriteLine($"[{hits.Count} passage(s): {string.Join(", ", hits.Select(h => h.Record.Source).Distinct())}]");

    // FormatContext renders the passages with numbered sources so the model can cite them.
    var context = RagService.FormatContext(hits);

    using var lease = await sessions.AcquireAsync(chatModel);

    var request = new ChatRequest
    {
        Model = chatModel,
        Options = new GenerationOptions { Temperature = 0.3f, MaxTokens = 800 },
        Messages =
        [
            ChatMessage.System(
                "Answer only from the retrieved context. Cite sources by name. " +
                "If the context does not contain the answer, say so rather than guessing."),
            ChatMessage.User($"{context}\n\nQuestion: {question}")
        ]
    };

    await foreach (var chunk in lease.Session.StreamAsync(request))
    {
        Console.Write(chunk.Delta);
    }

    Console.WriteLine();
    Console.WriteLine();
}

return 0;
