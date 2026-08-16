using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using LocalGen.Core;
using LocalGen.Core.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;

namespace LocalGen.Kernel.Plugins;

/// <summary>
/// Runs code the model writes, in Python, JavaScript, C#, Bash or PowerShell.
/// </summary>
/// <remarks>
/// This is the most powerful — and most dangerous — built-in function: it executes arbitrary code
/// with the privileges of the LocalGen process. Three things keep it usable rather than reckless:
/// the working directory is confined to the workspace, every run is killed at
/// <see cref="ToolOptions.CodeExecutionTimeout"/>, and installing packages requires
/// <see cref="ToolOptions.AllowDependencyInstall"/> to be turned on deliberately. Anyone exposing
/// LocalGen beyond loopback should turn this function off.
/// </remarks>
public sealed class CodeExecutionPlugin
{
    private const string ToolName = "CodeExecution";
    private const int MaxOutputLength = 16_000;

    private readonly PathGuard _guard;
    private readonly LocalGenOptions _options;
    private readonly ILogger<CodeExecutionPlugin> _logger;

    public CodeExecutionPlugin(
        PathGuard guard,
        LocalGenOptions options,
        ILogger<CodeExecutionPlugin> logger)
    {
        _guard = guard;
        _options = options;
        _logger = logger;
    }

    [KernelFunction("execute_code")]
    [Description(
        "Runs a snippet of code and returns its output. " +
        "Supported languages: python, javascript (node), csharp, bash, powershell. " +
        "Code runs in the LocalGen workspace directory.")]
    public async Task<string> ExecuteAsync(
        [Description("Language: python, javascript, csharp, bash or powershell")] string language,
        [Description("The source code to run")] string code,
        CancellationToken cancellationToken = default)
    {
        if (!_options.Tools.CodeExecution)
        {
            return "Error: code execution is disabled in this LocalGen configuration.";
        }

        var runtime = CodeRuntime.Resolve(language);
        if (runtime is null)
        {
            return $"Error: '{language}' is not supported. Use python, javascript, csharp, bash or powershell.";
        }

        // C# has no single-file interpreter in the SDK, so it gets its own project-based path.
        if (runtime.Value.Kind == RuntimeKind.CSharp)
        {
            return await ExecuteCSharpAsync(code, cancellationToken).ConfigureAwait(false);
        }

        var scriptPath = Path.Combine(
            _guard.Workspace,
            $".localgen-run-{Guid.NewGuid():N}{runtime.Value.Extension}");

        try
        {
            await File.WriteAllTextAsync(scriptPath, code, cancellationToken).ConfigureAwait(false);

            var result = await RunAsync(
                runtime.Value.Executable,
                [.. runtime.Value.Arguments, scriptPath],
                _guard.Workspace,
                cancellationToken).ConfigureAwait(false);

            return Format(result);
        }
        catch (Win32Exception)
        {
            return $"Error: '{runtime.Value.Executable}' is not installed or not on PATH. " +
                   $"Install it to run {language} code.";
        }
        finally
        {
            TryDelete(scriptPath);
        }
    }

    [KernelFunction("run_command")]
    [Description(
        "Runs a shell command in the workspace and returns its output. " +
        "Uses PowerShell on Windows and Bash elsewhere.")]
    public async Task<string> RunCommandAsync(
        [Description("The command line to run")] string command,
        CancellationToken cancellationToken = default)
    {
        if (!_options.Tools.CodeExecution)
        {
            return "Error: code execution is disabled in this LocalGen configuration.";
        }

        try
        {
            var (executable, arguments) = OperatingSystem.IsWindows()
                ? ("powershell", new[] { "-NoProfile", "-NonInteractive", "-Command", command })
                : ("/bin/bash", ["-c", command]);

            var result = await RunAsync(executable, arguments, _guard.Workspace, cancellationToken)
                .ConfigureAwait(false);

            return Format(result);
        }
        catch (Win32Exception ex)
        {
            return $"Error: could not start a shell — {ex.Message}";
        }
    }

    [KernelFunction("install_dependency")]
    [Description(
        "Installs a package or SDK needed to run code. " +
        "Managers: pip, npm, dotnet, winget, apt. Disabled unless the operator has allowed it.")]
    public async Task<string> InstallDependencyAsync(
        [Description("Package manager: pip, npm, dotnet, winget or apt")] string manager,
        [Description("Package name, optionally with a version")] string package,
        CancellationToken cancellationToken = default)
    {
        if (!_options.Tools.CodeExecution)
        {
            return "Error: code execution is disabled in this LocalGen configuration.";
        }

        // Installing packages changes the host machine, so it stays behind its own switch even
        // when code execution itself is allowed.
        if (!_options.Tools.AllowDependencyInstall)
        {
            return "Error: installing dependencies is disabled. " +
                   "Set LocalGen:Tools:AllowDependencyInstall to true to allow it.";
        }

        // The package name reaches a process launcher, so only safe characters are permitted.
        if (!IsSafePackageName(package))
        {
            return $"Error: '{package}' contains characters that are not allowed in a package name.";
        }

        var (executable, arguments) = manager.ToLowerInvariant() switch
        {
            "pip" or "pip3" => ("python", (string[])["-m", "pip", "install", package]),
            "npm" => ("npm", ["install", package]),
            "dotnet" => ("dotnet", ["add", "package", package]),
            "winget" => ("winget", ["install", "--accept-package-agreements", "--accept-source-agreements", "-e", "--id", package]),
            "apt" or "apt-get" => ("sudo", ["apt-get", "install", "-y", package]),
            _ => (string.Empty, [])
        };

        if (string.IsNullOrEmpty(executable))
        {
            return $"Error: unknown package manager '{manager}'. Use pip, npm, dotnet, winget or apt.";
        }

        _logger.LogInformation("Installing {Package} with {Manager}", package, manager);

        try
        {
            // Installs pull from the network and can be slow; they get a longer budget than code.
            var result = await RunAsync(
                executable,
                arguments,
                _guard.Workspace,
                cancellationToken,
                TimeSpan.FromMinutes(10)).ConfigureAwait(false);

            return Format(result);
        }
        catch (Win32Exception)
        {
            return $"Error: '{executable}' is not installed or not on PATH.";
        }
    }

    [KernelFunction("check_runtime")]
    [Description("Reports which language runtimes are installed and their versions.")]
    public async Task<string> CheckRuntimesAsync(CancellationToken cancellationToken = default)
    {
        var probes = new (string Label, string Executable, string[] Arguments)[]
        {
            ("Python", "python", ["--version"]),
            ("Node.js", "node", ["--version"]),
            (".NET SDK", "dotnet", ["--version"]),
            ("PowerShell", "powershell", ["-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"]),
            ("Bash", "bash", ["--version"])
        };

        var builder = new StringBuilder("Installed runtimes:\n");

        foreach (var (label, executable, arguments) in probes)
        {
            try
            {
                var result = await RunAsync(
                    executable, arguments, _guard.Workspace, cancellationToken, TimeSpan.FromSeconds(10))
                    .ConfigureAwait(false);

                var version = result.Output.Split('\n').FirstOrDefault()?.Trim();

                builder.Append("  ").Append(label).Append(": ")
                       .AppendLine(result.ExitCode == 0 && !string.IsNullOrEmpty(version)
                           ? version
                           : "not available");
            }
            catch (Exception ex) when (ex is Win32Exception or TimeoutException)
            {
                builder.Append("  ").Append(label).AppendLine(": not installed");
            }
        }

        return builder.ToString();
    }

    /// <summary>
    /// Compiles and runs a C# snippet. The SDK has no single-file runner, so a throwaway console
    /// project is generated around the code.
    /// </summary>
    private async Task<string> ExecuteCSharpAsync(string code, CancellationToken cancellationToken)
    {
        var projectDirectory = Path.Combine(_guard.Workspace, $".localgen-csharp-{Guid.NewGuid():N}");

        try
        {
            Directory.CreateDirectory(projectDirectory);

            await File.WriteAllTextAsync(
                Path.Combine(projectDirectory, "Program.csproj"),
                """
                <Project Sdk="Microsoft.NET.Sdk">
                  <PropertyGroup>
                    <OutputType>Exe</OutputType>
                    <TargetFramework>net10.0</TargetFramework>
                    <Nullable>disable</Nullable>
                    <ImplicitUsings>enable</ImplicitUsings>
                    <ManagePackageVersionsCentrally>false</ManagePackageVersionsCentrally>
                  </PropertyGroup>
                </Project>
                """,
                cancellationToken).ConfigureAwait(false);

            await File.WriteAllTextAsync(
                Path.Combine(projectDirectory, "Program.cs"), code, cancellationToken).ConfigureAwait(false);

            var result = await RunAsync(
                "dotnet",
                ["run", "--project", projectDirectory, "--verbosity", "quiet", "--nologo"],
                projectDirectory,
                cancellationToken,
                // Compilation plus restore needs more headroom than an interpreted script.
                TimeSpan.FromMinutes(3)).ConfigureAwait(false);

            return Format(result);
        }
        catch (Win32Exception)
        {
            return "Error: the .NET SDK is not installed or not on PATH.";
        }
        finally
        {
            TryDeleteDirectory(projectDirectory);
        }
    }

    private async Task<ProcessResult> RunAsync(
        string executable,
        string[] arguments,
        string workingDirectory,
        CancellationToken cancellationToken,
        TimeSpan? timeout = null)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = executable,
            WorkingDirectory = workingDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            // stdin is closed so an interactive prompt fails fast instead of hanging until timeout.
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        using var process = Process.Start(startInfo)
            ?? throw new LocalGenException($"Could not start '{executable}'.");

        process.StandardInput.Close();

        var stdoutTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
        var stderrTask = process.StandardError.ReadToEndAsync(cancellationToken);

        using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutSource.CancelAfter(timeout ?? _options.Tools.CodeExecutionTimeout);

        var timedOut = false;

        try
        {
            await process.WaitForExitAsync(timeoutSource.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            timedOut = true;

            try
            {
                // Child processes are killed too; a build or install spawns several.
                process.Kill(entireProcessTree: true);
            }
            catch (InvalidOperationException)
            {
                // The process exited between the timeout firing and the kill.
            }
        }

        var stdout = await SafeRead(stdoutTask).ConfigureAwait(false);
        var stderr = await SafeRead(stderrTask).ConfigureAwait(false);

        return new ProcessResult
        {
            ExitCode = timedOut ? -1 : process.ExitCode,
            Output = stdout,
            Error = stderr,
            TimedOut = timedOut
        };

        static async Task<string> SafeRead(Task<string> task)
        {
            try
            {
                return await task.ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                return string.Empty;
            }
        }
    }

    private string Format(ProcessResult result)
    {
        var builder = new StringBuilder();

        if (result.TimedOut)
        {
            builder.Append("[timed out after ")
                   .Append(_options.Tools.CodeExecutionTimeout.TotalSeconds)
                   .AppendLine("s and was terminated]");
        }

        if (!string.IsNullOrWhiteSpace(result.Output))
        {
            builder.AppendLine(result.Output.TrimEnd());
        }

        if (!string.IsNullOrWhiteSpace(result.Error))
        {
            builder.AppendLine("[stderr]").AppendLine(result.Error.TrimEnd());
        }

        if (result.ExitCode != 0 && !result.TimedOut)
        {
            builder.Append("[exit code ").Append(result.ExitCode).AppendLine("]");
        }

        var text = builder.ToString();

        if (text.Length == 0)
        {
            return "(the program produced no output)";
        }

        return text.Length <= MaxOutputLength
            ? text
            : text[..MaxOutputLength] + $"\n… [truncated, {text.Length:N0} characters total]";
    }

    /// <summary>Rejects shell metacharacters so a package name cannot smuggle in a command.</summary>
    private static bool IsSafePackageName(string package) =>
        !string.IsNullOrWhiteSpace(package) &&
        package.Length <= 200 &&
        package.All(static c =>
            char.IsAsciiLetterOrDigit(c) || c is '.' or '-' or '_' or '@' or '/' or '=' or '<' or '>' or '~' or '+');

    private void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException ex)
        {
            _logger.LogDebug(ex, "Could not delete temporary script {Path}", path);
        }
    }

    private void TryDeleteDirectory(string path)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
            }
        }
        catch (IOException ex)
        {
            _logger.LogDebug(ex, "Could not delete temporary project {Path}", path);
        }
    }

    private sealed record ProcessResult
    {
        public int ExitCode { get; init; }

        public string Output { get; init; } = string.Empty;

        public string Error { get; init; } = string.Empty;

        public bool TimedOut { get; init; }
    }
}

internal enum RuntimeKind
{
    Interpreted,
    CSharp
}

/// <summary>How each supported language is launched.</summary>
internal readonly record struct CodeRuntime(
    RuntimeKind Kind,
    string Executable,
    string[] Arguments,
    string Extension)
{
    public static CodeRuntime? Resolve(string language) => language.ToLowerInvariant() switch
    {
        "python" or "py" or "python3" =>
            new CodeRuntime(RuntimeKind.Interpreted, "python", [], ".py"),

        "javascript" or "js" or "node" or "nodejs" =>
            new CodeRuntime(RuntimeKind.Interpreted, "node", [], ".js"),

        "typescript" or "ts" =>
            new CodeRuntime(RuntimeKind.Interpreted, "npx", ["tsx"], ".ts"),

        "bash" or "sh" or "shell" =>
            new CodeRuntime(RuntimeKind.Interpreted, OperatingSystem.IsWindows() ? "bash" : "/bin/bash", [], ".sh"),

        "powershell" or "pwsh" or "ps1" =>
            new CodeRuntime(
                RuntimeKind.Interpreted,
                OperatingSystem.IsWindows() ? "powershell" : "pwsh",
                ["-NoProfile", "-NonInteractive", "-File"],
                ".ps1"),

        "csharp" or "cs" or "c#" or "dotnet" =>
            new CodeRuntime(RuntimeKind.CSharp, "dotnet", [], ".cs"),

        _ => null
    };
}
