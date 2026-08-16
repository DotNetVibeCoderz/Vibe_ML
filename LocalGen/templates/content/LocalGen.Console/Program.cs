// A console client for a local model, generated from the LocalGen template.
//
//   dotnet run              chat interactively
//   dotnet run -- "prompt"  one-shot

using System.Text.Json;
using LocalGen.Core.Protocol;
using LocalGen.Sdk;
using Microsoft.Extensions.Configuration;

var configuration = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables("LOCALGEN_")
    .Build();

using var client = new LocalGenClient(new LocalGenClientOptions
{
    Endpoint = configuration["LocalGen:Endpoint"] ?? "LOCALGEN_ENDPOINT_PLACEHOLDER",
    ApiKey = configuration["LocalGen:ApiKey"],
    DefaultModel = configuration["LocalGen:Model"] ?? "LOCALGEN_MODEL_PLACEHOLDER"
});

if (!await client.PingAsync())
{
    Console.Error.WriteLine("LocalGen is not reachable. Start it with: localgen serve");
    return 1;
}

var systemPrompt = configuration["LocalGen:SystemPrompt"]
                   ?? "You are a helpful assistant running locally on the user's machine.";

// One-shot mode when a prompt is passed on the command line.
if (args.Length > 0)
{
    await StreamAsync([Message("system", systemPrompt), Message("user", string.Join(' ', args))]);
    return 0;
}

Console.WriteLine("Type a message, or /bye to exit.");
Console.WriteLine();

// The server keeps no session state, so the conversation is replayed on every turn.
var history = new List<OpenAiMessage> { Message("system", systemPrompt) };

while (true)
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

    var reply = await StreamAsync(history);

    if (reply is null)
    {
        // The turn failed; drop it so the next request has no dangling user message.
        history.RemoveAt(history.Count - 1);
        continue;
    }

    history.Add(Message("assistant", reply));
}

return 0;

async Task<string?> StreamAsync(IReadOnlyList<OpenAiMessage> messages)
{
    var reply = new System.Text.StringBuilder();

    try
    {
        var request = new ChatCompletionRequest
        {
            Model = client.Options.DefaultModel ?? string.Empty,
            Temperature = 0.7f,
            Messages = [.. messages]
        };

        await foreach (var chunk in client.StreamChatAsync(request))
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

        return reply.ToString();
    }
    catch (LocalGenClientException ex)
    {
        Console.Error.WriteLine($"Error: {ex.Message}");
        return null;
    }
}

// The wire format's content field is polymorphic — a string or an array of typed parts — so it
// is carried as a JsonElement rather than typed as string.
static OpenAiMessage Message(string role, string text) => new()
{
    Role = role,
    Content = JsonSerializer.SerializeToElement(text)
};
