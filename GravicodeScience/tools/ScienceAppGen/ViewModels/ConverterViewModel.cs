using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using ScienceAppGen.Models;
using ScienceAppGen.Services;
using ScienceAppGen.Services.Conversion;

namespace ScienceAppGen.ViewModels;

/// <summary>One Python module on the converter page.</summary>
public sealed partial class ConvertedModuleItem : ObservableObject
{
    /// <summary>Creates the item for a planned module.</summary>
    public ConvertedModuleItem(PlannedModule module)
    {
        Module = module;
        _status = ModuleStatus.Pending;
    }

    /// <summary>The plan entry.</summary>
    public PlannedModule Module { get; }

    /// <summary>Python path.</summary>
    public string PythonPath => Module.PythonPath;

    /// <summary>C# path.</summary>
    public string CSharpPath => Module.CSharpPath;

    /// <summary>"→ Program.cs · entry point" and the like.</summary>
    public string Target => "→ " + Module.CSharpPath + (Module.IsEntryPoint ? "  ·  entry point" : "");

    /// <summary>The Python source.</summary>
    public string PythonSource => Module.Analysis.File.Source;

    /// <summary>Lines of Python.</summary>
    public int LineCount => Module.Analysis.File.LineCount;

    /// <summary>The latest C#, or null before conversion.</summary>
    [ObservableProperty] private string? _cSharp;

    /// <summary>Where the module is in the pipeline.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(StatusText), nameof(StatusColour), nameof(IsBusy))]
    private ModuleStatus _status;

    /// <summary>Error count from the last build, shown next to the status.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(StatusText))]
    private int _errorCount;

    /// <summary>Short status label.</summary>
    public string StatusText => Status switch
    {
        ModuleStatus.Pending => "waiting",
        ModuleStatus.Converting => "converting",
        ModuleStatus.Converted => "converted",
        ModuleStatus.HasErrors => ErrorCount == 1 ? "1 error" : $"{ErrorCount} errors",
        ModuleStatus.Repairing => "repairing",
        ModuleStatus.Compiles => "compiles",
        ModuleStatus.Failed => "needs you",
        _ => "",
    };

    /// <summary>Status colour from the spectrum: red failed, orange errors, yellow written, green compiles, cyan and violet in flight.</summary>
    public string StatusColour => Status switch
    {
        ModuleStatus.Converting => ConverterViewModel.Info,
        ModuleStatus.Converted => ConverterViewModel.Build,
        ModuleStatus.HasErrors => ConverterViewModel.Warning,
        ModuleStatus.Repairing => ConverterViewModel.Assistant,
        ModuleStatus.Compiles => ConverterViewModel.Success,
        ModuleStatus.Failed => ConverterViewModel.Error,
        _ => ConverterViewModel.Muted,
    };

    /// <summary>True while the model is working on this module.</summary>
    public bool IsBusy => Status is ModuleStatus.Converting or ModuleStatus.Repairing;
}

/// <summary>How one Python package maps, as a chip.</summary>
/// <param name="Package">Python package.</param>
/// <param name="Target">What it becomes.</param>
/// <param name="Colour">Hex colour of the target library.</param>
/// <param name="Note">Caveat, shown as a tooltip.</param>
/// <param name="IsUnsupported">True when nothing covers it.</param>
public sealed record PackageChip(string Package, string Target, string Colour, string? Note, bool IsUnsupported);

/// <summary>A line in the activity feed.</summary>
/// <param name="Time">When.</param>
/// <param name="Message">What happened.</param>
/// <param name="Colour">Hex colour for the bullet.</param>
/// <param name="IsProgramOutput">True for lines the converted program printed.</param>
public sealed record ActivityLine(string Time, string Message, string Colour, bool IsProgramOutput);

/// <summary>
/// The Python to C# page: choose a file or folder, see the plan, convert, watch it build.
/// </summary>
/// <remarks>
/// Scanning and planning happen as soon as a source is chosen and cost nothing, so the user sees
/// the module names, the package mapping and anything unsupported <i>before</i> spending a single
/// model request.
/// </remarks>
public sealed partial class ConverterViewModel : ObservableObject
{
    internal const string Error = "#E0555A", Warning = "#E08A3C", Build = "#D9C04A", Success = "#56B87F",
        Info = "#3FA9C9", Assistant = "#7B7BD6", Muted = "#7D8A9C";

    private readonly AppSettings _settings;
    private readonly Func<string?> _selectedModel;
    private readonly LogService _logs;
    private CancellationTokenSource? _cancel;
    private ConversionPlan? _plan;

    /// <summary>Creates the page.</summary>
    /// <param name="settings">Provider and key.</param>
    /// <param name="selectedModel">The model picked in the chat panel, which wins over the settings default.</param>
    /// <param name="logs">The shared log.</param>
    public ConverterViewModel(AppSettings settings, Func<string?> selectedModel, LogService logs)
    {
        _settings = settings;
        _selectedModel = selectedModel;
        _logs = logs;
    }

    /// <summary>Raised with the output folder when the user asks to open the result in the workspace.</summary>
    public event Action<string>? OpenProjectRequested;

    /// <summary>Modules in conversion order.</summary>
    public ObservableCollection<ConvertedModuleItem> Modules { get; } = [];

    /// <summary>Package mapping chips.</summary>
    public ObservableCollection<PackageChip> Packages { get; } = [];

    /// <summary>The activity feed.</summary>
    public ObservableCollection<ActivityLine> Activity { get; } = [];

    /// <summary>TODO notes the model left in the result.</summary>
    public ObservableCollection<string> Todos { get; } = [];

    /// <summary>The chosen file or folder.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasSource), nameof(CanConvert), nameof(SourceName))]
    private string _sourcePath = "";

    /// <summary>The C# project name.</summary>
    [ObservableProperty] private string _projectName = "";

    /// <summary>Where the C# project goes.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanConvert))]
    private string _outputDirectory = "";

    /// <summary>Build after converting and send compiler errors back to the model.</summary>
    [ObservableProperty] private bool _buildAndRepair = true;

    /// <summary>How many repair rounds.</summary>
    [ObservableProperty] private decimal _repairRounds = 3;

    /// <summary>Run the program once it builds.</summary>
    [ObservableProperty] private bool _runAfterBuild = true;

    /// <summary>True while a conversion runs.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanConvert), nameof(IsIdle))]
    private bool _isRunning;

    /// <summary>One line under the Convert button.</summary>
    [ObservableProperty] private string _statusText = "Choose a Python file or folder to start.";

    /// <summary>"12 modules · 3 data files" and the like.</summary>
    [ObservableProperty] private string _sourceSummary = "";

    /// <summary>Notes about skipped files.</summary>
    [ObservableProperty] private string _skippedSummary = "";

    /// <summary>The module shown in the two editors.</summary>
    [ObservableProperty] private ConvertedModuleItem? _selectedModule;

    /// <summary>The result line after a conversion.</summary>
    [ObservableProperty] private string _resultSummary = "";

    /// <summary>True once a conversion has written a project that can be opened.</summary>
    [ObservableProperty] private bool _hasResult;

    /// <summary>True when the last conversion built cleanly.</summary>
    [ObservableProperty] private bool _resultSucceeded;

    /// <summary>True when a source has been scanned.</summary>
    public bool HasSource => _plan is not null;

    /// <summary>The file or folder name, shown above the full path.</summary>
    public string SourceName => Path.GetFileName(SourcePath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));

    /// <summary>True when Convert may be pressed.</summary>
    public bool CanConvert => _plan is not null && !IsRunning && OutputDirectory.Trim().Length > 0;

    /// <summary>Inverse of <see cref="IsRunning"/>, for bindings.</summary>
    public bool IsIdle => !IsRunning;

    /// <summary>"AzureOpenAI · gpt-5-mini".</summary>
    public string ModelLabel => $"{_settings.Provider} · {CurrentModel}";

    private string CurrentModel => _selectedModel() is { Length: > 0 } picked ? picked : _settings.Model;

    /// <summary>Scans a file or folder and plans the conversion. Makes no model calls.</summary>
    public void LoadSource(string path)
    {
        try
        {
            var project = PythonProjectScanner.Scan(path);
            _plan = ConversionPlanner.Plan(project);

            SourcePath = path;
            ProjectName = _plan.ProjectName;
            OutputDirectory = ConvertCommand.DefaultOutput(project, _plan);

            Modules.Clear();
            foreach (var module in _plan.Modules) Modules.Add(new ConvertedModuleItem(module));
            SelectedModule = Modules.FirstOrDefault(m => m.Module.IsEntryPoint) ?? Modules.FirstOrDefault();

            Packages.Clear();
            foreach (var package in _plan.Packages)
            {
                var target = package.Target;
                Packages.Add(new PackageChip(
                    package.Package,
                    target is null ? "no counterpart" : target.PackageId is null ? ".NET" : target.Id,
                    target is null ? Error : ColourOf(target),
                    package.Note,
                    package.IsUnsupported));
            }

            var lines = project.Files.Sum(f => f.LineCount);
            SourceSummary = $"{project.Files.Count} module{(project.Files.Count == 1 ? "" : "s")} · {lines:N0} lines"
                            + (project.DataFiles.Count > 0 ? $" · {project.DataFiles.Count} data file{(project.DataFiles.Count == 1 ? "" : "s")}" : "");
            SkippedSummary = project.Skipped.Count == 0 ? "" : "Skipped: " + string.Join("; ", project.Skipped);

            Activity.Clear();
            Todos.Clear();
            HasResult = false;
            ResultSummary = "";
            var unsupported = _plan.Unsupported.Count();
            StatusText = unsupported > 0
                ? $"{unsupported} package(s) have no counterpart; the model will port those calls by hand."
                : "Ready. Nothing is sent to the model until you press Convert.";

            OnPropertyChanged(nameof(HasSource));
            OnPropertyChanged(nameof(CanConvert));
            OnPropertyChanged(nameof(ModelLabel));
        }
        catch (Exception ex) when (ex is InvalidOperationException or FileNotFoundException or IOException or UnauthorizedAccessException)
        {
            StatusText = ex.Message;
            _logs.Warning("convert", ex.Message);
        }
    }

    /// <summary>Runs the conversion.</summary>
    public async Task ConvertAsync()
    {
        if (_plan is null || IsRunning) return;

        var settings = SettingsForRun();
        if (!settings.IsConfigured)
        {
            StatusText = settings.MissingConfiguration ?? "The model is not configured. Open Tools > Settings.";
            return;
        }

        // Re-plan with the chosen name so namespaces and the csproj follow it.
        if (!string.Equals(ProjectName, _plan.ProjectName, StringComparison.Ordinal))
        {
            _plan = ConversionPlanner.Plan(_plan.Project, ProjectName);
            ProjectName = _plan.ProjectName;
            var selected = SelectedModule?.PythonPath;
            Modules.Clear();
            foreach (var module in _plan.Modules) Modules.Add(new ConvertedModuleItem(module));
            SelectedModule = Modules.FirstOrDefault(m => m.PythonPath == selected) ?? Modules.FirstOrDefault();
        }

        foreach (var module in Modules) { module.Status = ModuleStatus.Pending; module.CSharp = null; module.ErrorCount = 0; }
        Activity.Clear();
        Todos.Clear();
        HasResult = false;
        ResultSummary = "";
        IsRunning = true;
        StatusText = $"Converting with {settings.Provider} · {settings.Model}...";
        _cancel = new CancellationTokenSource();

        // Created on the UI thread, so every report is posted back to it.
        var progress = new Progress<ConversionEvent>(OnEvent);

        try
        {
            using var llm = new SemanticKernelLlmClient(settings);
            var options = new ConversionOptions
            {
                OutputDirectory = OutputDirectory.Trim(),
                ProjectName = _plan.ProjectName,
                Build = BuildAndRepair,
                MaxRepairRounds = (int)RepairRounds,
                RunAfterBuild = BuildAndRepair && RunAfterBuild,
            };

            var plan = _plan;
            var result = await Task.Run(() => new PythonConverter(llm).ConvertAsync(plan, options, progress, _cancel.Token));

            HasResult = true;
            ResultSucceeded = result.Succeeded && (result.ProgramExitCode ?? 0) == 0;
            foreach (var todo in result.Todos) Todos.Add(todo);
            ResultSummary = (result.Succeeded ? "Builds" : $"{result.RemainingErrors.Count} error(s) left")
                            + $" · {result.LlmCalls} request{(result.LlmCalls == 1 ? "" : "s")}"
                            + $" · {result.RepairRounds} repair round{(result.RepairRounds == 1 ? "" : "s")}"
                            + $" · {result.Duration.TotalSeconds:F0} s"
                            + (result.ProgramExitCode is { } code ? $" · program exit {code}" : "");
            StatusText = result.Succeeded
                ? "Done. Open the project to keep working on it, or read CONVERSION.md in the folder."
                : "The project is written but does not build yet. Open it and finish the last errors, or convert again.";
            _logs.Success("convert", $"{plan.ProjectName}: {ResultSummary}");
        }
        catch (OperationCanceledException)
        {
            StatusText = "Cancelled. Files converted so far are in the output folder.";
        }
        catch (Exception ex)
        {
            StatusText = ex.Message;
            AddActivity(ex.Message, Error);
            _logs.Error("convert", ex.Message);
        }
        finally
        {
            IsRunning = false;
            _cancel?.Dispose();
            _cancel = null;
        }
    }

    /// <summary>Stops a running conversion.</summary>
    public void Cancel() => _cancel?.Cancel();

    /// <summary>Asks the shell to open the converted project.</summary>
    public void OpenResult()
    {
        if (HasResult) OpenProjectRequested?.Invoke(OutputDirectory.Trim());
    }

    /// <summary>Refreshes the model label when the chat's picker or the settings change.</summary>
    public void RefreshModel() => OnPropertyChanged(nameof(ModelLabel));

    private AppSettings SettingsForRun() => new()
    {
        Provider = _settings.Provider,
        Model = CurrentModel,
        ApiKey = _settings.ApiKey,
        Endpoint = _settings.Endpoint,
        Temperature = _settings.Temperature,
        MaxTokens = _settings.MaxTokens,
        TimeoutSeconds = _settings.TimeoutSeconds,
    };

    private void OnEvent(ConversionEvent e)
    {
        switch (e.Kind)
        {
            case ConversionEventKind.Build:
                return;   // the parsed errors are reported; raw MSBuild output stays in the transcript

            case ConversionEventKind.Module:
                var item = Modules.FirstOrDefault(m => m.PythonPath == e.ModulePath);
                if (item is not null)
                {
                    if (e.Status is { } status)
                    {
                        if (status == ModuleStatus.HasErrors && int.TryParse(e.Message.Split(':').Last().Trim().Split(' ')[0], out var count))
                            item.ErrorCount = count;
                        item.Status = status;
                    }
                    if (e.CSharp is not null) item.CSharp = e.CSharp;
                }
                AddActivity(e.Message, item?.StatusColour ?? Muted);
                return;

            case ConversionEventKind.ProgramOutput:
                Activity.Add(new ActivityLine("", e.Message, Muted, true));
                return;

            default:
                AddActivity(e.Message, e.Kind switch
                {
                    ConversionEventKind.Error => Error,
                    ConversionEventKind.Warning => Warning,
                    _ => Info,
                });
                if (e.Kind == ConversionEventKind.Info) StatusText = e.Message;
                return;
        }
    }

    private void AddActivity(string message, string colour)
        => Activity.Add(new ActivityLine(DateTime.Now.ToString("HH:mm:ss"), message, colour, false));

    private static string ColourOf(TargetLibrary target) => target.Id switch
    {
        "GraviNum" => Error,
        "GraviFrame" => Warning,
        "GraviLearn" => Build,
        "GraviText" => Success,
        "GraviGraph" => Info,
        "GraviProb" => Assistant,
        "SciNet" => "#5CC4E2",
        "BCL" => Muted,
        _ => "#C4CEDA",   // Visual.Net: a neutral light, so charts read as one family
    };
}
