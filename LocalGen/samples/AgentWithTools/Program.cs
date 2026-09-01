// Function calling against a local model.
//
// LocalGen implements IChatClient, so Microsoft.Extensions.AI's function-invocation middleware
// works unchanged — the same code would run against a hosted provider.
//
//   dotnet run --project samples/AgentWithTools [model-id]

using System.ComponentModel;
using LocalGen.Sdk;
using Microsoft.Extensions.AI;

var endpoint = Environment.GetEnvironmentVariable("LOCALGEN_HOST") ?? "http://127.0.0.1:11434";

using var probe = new LocalGenClient(endpoint);

if (!await probe.PingAsync())
{
    Console.Error.WriteLine($"No LocalGen server at {endpoint}. Start one with: localgen serve");
    return 1;
}

var models = await probe.ListModelsAsync();

if (models.Count == 0)
{
    Console.Error.WriteLine("No models installed. Try: localgen pull huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF");
    return 1;
}

var model = args.FirstOrDefault() ?? models[0].Id;

// UseFunctionInvocation adds the loop: when the model asks for a tool, the middleware runs it,
// feeds the result back and continues, without the caller writing that loop.
IChatClient client = new LocalGenChatClient(endpoint, model)
    .AsBuilder()
    .UseFunctionInvocation()
    .Build();

var tools = new List<AITool>
{
    AIFunctionFactory.Create(GetCurrentTime),
    AIFunctionFactory.Create(Calculate),
    AIFunctionFactory.Create(GetStockLevel)
};

var options = new ChatOptions { Tools = tools, Temperature = 0.3f };

Console.WriteLine($"Model: {model}");
Console.WriteLine("Tools: current time, arithmetic, warehouse stock lookup");
Console.WriteLine();

string[] questions =
[
    "What time is it right now?",
    "We have 340 units in the Jakarta warehouse. If we ship 17.5% of them, how many remain?",
    "How much stock is there in Surabaya, and is that more than Bandung?"
];

foreach (var question in questions)
{
    Console.WriteLine($"> {question}");

    var response = await client.GetResponseAsync(question, options);

    Console.WriteLine(response.Text);
    Console.WriteLine();
}

return 0;

// ─────────────────────────  The tools  ─────────────────────────
//
// The [Description] attributes are not documentation for humans — they are what the model reads
// to decide whether a tool applies, so they are written for that reader.

[Description("Returns the current local date and time.")]
static string GetCurrentTime() =>
    DateTimeOffset.Now.ToString("dddd, d MMMM yyyy 'at' HH:mm");

[Description("Performs exact arithmetic. Use this instead of calculating in your head.")]
static double Calculate(
    [Description("The left operand")] double left,
    [Description("One of: add, subtract, multiply, divide, percent_of")] string operation,
    [Description("The right operand")] double right) => operation.ToLowerInvariant() switch
{
    "add" => left + right,
    "subtract" => left - right,
    "multiply" => left * right,
    "divide" when right != 0 => left / right,
    "divide" => throw new ArgumentException("Cannot divide by zero."),
    "percent_of" => left / 100 * right,
    _ => throw new ArgumentException($"Unknown operation '{operation}'.")
};

[Description("Looks up how many units are in stock at a warehouse.")]
static string GetStockLevel(
    [Description("Warehouse city, for example Jakarta or Surabaya")] string warehouse)
{
    // Stands in for whatever system a real deployment would query.
    var stock = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
    {
        ["Jakarta"] = 340,
        ["Surabaya"] = 128,
        ["Bandung"] = 76
    };

    return stock.TryGetValue(warehouse, out var units)
        ? $"{warehouse}: {units} units"
        : $"No warehouse named '{warehouse}'. Known warehouses: {string.Join(", ", stock.Keys)}.";
}
