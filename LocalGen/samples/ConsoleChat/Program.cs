// Streaming chat against a local model — the smallest useful LocalGen client.
//
//   dotnet run --project samples/ConsoleChat [model-id]

using System.Text.Json;
using LocalGen.Core.Protocol;
using LocalGen.Sdk;

var endpoint = Environment.GetEnvironmentVariable("LOCALGEN_HOST") ?? "http://127.0.0.1:11434";

using var client = new LocalGenClient(new LocalGenClientOptions
{
    Endpoint = endpoint,
    ApiKey = Environment.GetEnvironmentVariable("LOCALGEN_API_KEY")
});

if (!await client.PingAsync())
{
    Console.Error.WriteLine($"No LocalGen server at {endpoint}.");
    Console.Error.WriteLine("Start one with: localgen serve");
    return 1;
}

// Take the model from the command line, or fall back to whatever is installed.
var models = await client.ListModelsAsync();

if (models.Count == 0)
{
    Console.Error.WriteLine("No models installed. Try:");
    Console.Error.WriteLine("  localgen pull huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF");
    return 1;
}

var model = args.FirstOrDefault() ?? models[0].Id;

Console.WriteLine($"Model:  {model}");
Console.WriteLine("Type a message, or /bye to exit.");
Console.WriteLine();

// The conversation is replayed on every turn — the server keeps no session state, which is what
// makes it safe to point several clients at one instance.
var history = new List<OpenAiMessage>
{
    Message("system", "You are a helpful assistant running locally on the user's machine.")
};

using var cancellation = new CancellationTokenSource();

Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    cancellation.Cancel();
};

while (!cancellation.IsCancellationRequested)
{
    Console.Write("> ");

    var input = Console.ReadLine()?.Trim();

    if (input is null or "/bye" or "/exit")
    {
        break;
    }

    if (input.Length == 0)
    {
        continue;
    }

    history.Add(Message("user", input));

    var reply = new System.Text.StringBuilder();

    try
    {
        var request = new ChatCompletionRequest
        {
            Model = model,
            Temperature = 0.7f,
            Messages = history
        };

        await foreach (var chunk in client.StreamChatAsync(request, cancellation.Token))
        {
            var delta = chunk.Choices.FirstOrDefault()?.Delta?.Content;

            if (!string.IsNullOrEmpty(delta))
            {
                reply.Append(delta);
                Console.Write(delta);
            }
        }

        Console.WriteLine();
        Console.WriteLine();

        history.Add(Message("assistant", reply.ToString()));
    }
    catch (OperationCanceledException)
    {
        Console.WriteLine("\n[stopped]");
        break;
    }
    catch (LocalGenClientException ex)
    {
        Console.Error.WriteLine($"\nError: {ex.Message}");

        // Drop the turn that failed so the next request is not sent with a dangling user message.
        history.RemoveAt(history.Count - 1);
    }
}

return 0;

// The content field is polymorphic on the wire — a string or an array of typed parts — so it is
// serialised as a JsonElement rather than typed as string.
static OpenAiMessage Message(string role, string text) => new()
{
    Role = role,
    Content = JsonSerializer.SerializeToElement(text)
};
