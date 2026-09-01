# .NET SDK

`LocalGen.Sdk` is a typed client for the LocalGen API. Because LocalGen speaks the OpenAI wire
format, the same client works against OpenAI or any compatible gateway — useful when an
application needs to fall back to a hosted model.

```bash
dotnet add package LocalGen.Sdk
```

## Getting started

```csharp
using LocalGen.Sdk;

using var client = new LocalGenClient(new LocalGenClientOptions
{
    Endpoint = "http://127.0.0.1:11434",
    DefaultModel = "qwen2.5-7b-instruct:q4_k_m"
});

var answer = await client.ChatAsync("Explain quantization in one paragraph.");
Console.WriteLine(answer);
```

## Streaming

```csharp
await foreach (var token in client.StreamTextAsync(
    prompt: "Write a haiku about local inference.",
    systemPrompt: "You are a poet."))
{
    Console.Write(token);
}
```

For full control over the request, stream the raw chunks:

```csharp
using LocalGen.Core.Protocol;
using System.Text.Json;

var request = new ChatCompletionRequest
{
    Model = "qwen2.5-7b-instruct:q4_k_m",
    Temperature = 0.3f,
    MaxTokens = 800,
    StreamOptions = new StreamOptions { IncludeUsage = true },
    Messages =
    [
        new OpenAiMessage { Role = "system", Content = JsonSerializer.SerializeToElement("Be brief.") },
        new OpenAiMessage { Role = "user", Content = JsonSerializer.SerializeToElement("Why is Q4_K_M popular?") }
    ]
};

await foreach (var chunk in client.StreamChatAsync(request))
{
    Console.Write(chunk.Choices.FirstOrDefault()?.Delta?.Content);

    if (chunk.Usage is { } usage)
    {
        Console.WriteLine($"\n{usage.CompletionTokens} tokens generated");
    }
}
```

## Embeddings

```csharp
float[] vector = await client.EmbedAsync("text to embed", model: "nomic-embed-text");

var batch = await client.EmbedAsync(["first", "second", "third"], "nomic-embed-text");
foreach (var item in batch.Data)
{
    Console.WriteLine($"[{item.Index}] {item.Embedding.Length} dimensions");
}
```

## Dependency injection

```csharp
builder.Services.AddLocalGenClient(options =>
{
    options.Endpoint = builder.Configuration["LocalGen:Endpoint"]!;
    options.ApiKey = builder.Configuration["LocalGen:ApiKey"];
    options.DefaultModel = "qwen2.5-7b-instruct:q4_k_m";
});
```

The underlying `HttpClient` comes from `IHttpClientFactory`, so socket handling and DNS refresh
follow the usual .NET conventions.

## Microsoft.Extensions.AI

`LocalGenChatClient` implements `IChatClient`, the abstraction Semantic Kernel, the Agent
Framework and most .NET AI middleware build on:

```csharp
builder.Services.AddLocalGenChatClient(
    modelId: "qwen2.5-7b-instruct:q4_k_m",
    options => options.Endpoint = "http://127.0.0.1:11434");
```

Which means the function-invoking and caching decorators work unchanged:

```csharp
using Microsoft.Extensions.AI;

IChatClient client = new LocalGenChatClient("http://127.0.0.1:11434", "qwen2.5-7b-instruct:q4_k_m")
    .AsBuilder()
    .UseFunctionInvocation()
    .Build();

var response = await client.GetResponseAsync(
    "What is 17.5% of 84,320?",
    new ChatOptions { Tools = [AIFunctionFactory.Create(Calculate)] });

static double Calculate(double part, double whole) => part / 100 * whole;
```

## Semantic Kernel

```csharp
using LocalGen.Sdk;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel;

var builder = Kernel.CreateBuilder();
builder.Services.AddSingleton<IChatClient>(
    new LocalGenChatClient("http://127.0.0.1:11434", "qwen2.5-7b-instruct:q4_k_m"));

var kernel = builder.Build();

var summary = await kernel.InvokePromptAsync(
    "Summarise in three bullets: {{$input}}",
    new KernelArguments { ["input"] = document });
```

## Management operations

LocalGen-specific calls live on `Admin`, so it is obvious which parts of the client are portable
to another provider.

```csharp
var status = await client.Admin.GetStatusAsync();
Console.WriteLine($"{status.Status} · {status.LoadedModels.Count} model(s) resident");

foreach (var model in await client.Admin.ListModelsAsync())
{
    Console.WriteLine($"{model.Id}  {model.Quantization.Name}  {model.SizeBytes / 1024 / 1024} MB");
}

// Downloads stream progress because they run for minutes
await foreach (var update in client.Admin.PullModelAsync("huggingface:owner/repo"))
{
    if (update.IsComplete)
    {
        Console.WriteLine(update.Status);
        break;
    }

    Console.Write($"\r{update.Percentage:N1}%");
}

await client.Admin.LoadModelAsync("qwen2.5-7b-instruct:q4_k_m");   // preload
var engines = await client.Admin.GetEnginesAsync();
Console.WriteLine(engines.Rationale);
```

## Errors and cancellation

Failures surface as `LocalGenClientException`, carrying the server's message, HTTP status and
error code rather than a bare status line.

```csharp
try
{
    await client.ChatAsync("Hello", model: "not-installed");
}
catch (LocalGenClientException ex) when (ex.ErrorCode == "model_not_found")
{
    Console.Error.WriteLine(ex.Message);
}
```

Every method takes a `CancellationToken`. Cancelling a stream stops generation on the server.

```csharp
using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));

await foreach (var token in client.StreamTextAsync("…", cancellationToken: timeout.Token))
{
    Console.Write(token);
}
```

The default per-request timeout is ten minutes, because a cold model load on CPU can take a while
before the first token appears.

## Checking availability

```csharp
if (!await client.PingAsync())
{
    Console.Error.WriteLine("LocalGen is not running. Start it with: localgen serve");
    return 1;
}
```

`PingAsync` never throws — it returns false when the server is unreachable.
