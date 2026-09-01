using LocalGen.Cli.Commands;
using Spectre.Console;
using Spectre.Console.Cli;

var app = new CommandApp();

app.Configure(config =>
{
    config.SetApplicationName("localgen");
    config.SetApplicationVersion(
        typeof(Program).Assembly.GetName().Version?.ToString(3) ?? "0.1.0");

    config.AddCommand<ListCommand>("list")
          .WithAlias("ls")
          .WithDescription("List the models installed locally.")
          .WithExample("list");

    config.AddCommand<PullCommand>("pull")
          .WithDescription("Download a model.")
          .WithExample("pull", "huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF")
          .WithExample("pull", "huggingface:bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf");

    config.AddCommand<RunCommand>("run")
          .WithDescription("Chat with a model. Omit the prompt for an interactive session.")
          .WithExample("run", "qwen2.5-7b-instruct:q4_k_m")
          .WithExample("run", "qwen2.5-7b-instruct:q4_k_m", "\"Explain quantization in one paragraph\"");

    config.AddCommand<RemoveCommand>("rm")
          .WithDescription("Delete an installed model.")
          .WithExample("rm", "qwen2.5-7b-instruct:q4_k_m");

    config.AddCommand<ShowCommand>("show")
          .WithDescription("Show a model's details and Modelfile.")
          .WithExample("show", "qwen2.5-7b-instruct:q4_k_m");

    config.AddCommand<CreateCommand>("create")
          .WithDescription("Create a model variant from a Modelfile.")
          .WithExample("create", "my-assistant", "-f", "Modelfile");

    config.AddCommand<QuantizeCommand>("quantize")
          .WithDescription("Convert a model to a smaller quantization.")
          .WithExample("quantize", "smollm2-135m-instruct:f16", "Q4_K_M")
          .WithExample("quantize", "./models/model-F16.gguf", "Q5_K_M", "-t", "8");

    config.AddCommand<BenchmarkCommand>("benchmark")
          .WithAlias("bench")
          .WithDescription("Measure throughput and time to first token.")
          .WithExample("benchmark", "qwen2.5-7b-instruct:q4_k_m", "-n", "5");

    config.AddCommand<ServeCommand>("serve")
          .WithDescription("Start the OpenAI-compatible web service.")
          .WithExample("serve")
          .WithExample("serve", "--port", "8080", "--preload", "qwen2.5-7b-instruct:q4_k_m");

    config.AddCommand<StatusCommand>("status")
          .WithAlias("ps")
          .WithDescription("Show service status and loaded models.");

    config.AddCommand<EnginesCommand>("engines")
          .WithDescription("List inference backends and what this machine supports.");

    // Unhandled exceptions become a readable message; --verbose brings back the stack trace.
    config.SetExceptionHandler((exception, _) =>
    {
        if (exception is CommandParseException or CommandRuntimeException)
        {
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(exception.Message)}[/]");
            AnsiConsole.MarkupLine("[dim]Run 'localgen --help' for usage.[/]");
            return 1;
        }

        AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(exception.Message)}");

        if (Environment.GetCommandLineArgs().Any(static a => a is "-v" or "--verbose"))
        {
            AnsiConsole.WriteException(exception, ExceptionFormats.ShortenPaths);
        }

        return 1;
    });
});

return await app.RunAsync(args);
