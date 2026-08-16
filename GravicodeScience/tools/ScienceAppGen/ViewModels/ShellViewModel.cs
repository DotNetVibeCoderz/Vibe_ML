using CommunityToolkit.Mvvm.ComponentModel;
using ScienceAppGen.Models;
using ScienceAppGen.Services;

namespace ScienceAppGen.ViewModels;

/// <summary>
/// The application shell: owns the services and the panel view models.
/// </summary>
public sealed partial class ShellViewModel : ObservableObject
{
    private readonly ConfigurationService _configuration;

    /// <summary>Creates the shell and its panels.</summary>
    public ShellViewModel(ConfigurationService configuration, LogService logs, AppSettings settings)
    {
        _configuration = configuration;
        Settings = settings;
        Logs = logs;

        Projects = new ProjectService(logs);
        Assistant = new AssistantService(Projects, logs);

        Explorer = new ExplorerViewModel(Projects);
        Editor = new EditorViewModel(settings);
        Chat = new ChatViewModel(Assistant, Projects, logs, settings);

        Explorer.FileOpenRequested += path => Editor.OpenFile(path);
        Projects.ProjectChanged += _ =>
        {
            Explorer.Refresh();
            Chat.NoteProjectChanged();
            OnPropertyChanged(nameof(Title));
            OnPropertyChanged(nameof(ProjectLabel));
        };

        _chatVisible = settings.ChatPanelVisible;
        _chatWidth = settings.ChatPanelWidth;
        _explorerWidth = settings.ExplorerWidth;
        _logsHeight = settings.LogsHeight;

        logs.Info("app", $"ScienceAppGen ready. {AssistantService.FullName} is standing by.");

        if (!settings.IsConfigured)
            logs.Warning("app", settings.MissingConfiguration ?? "The assistant is not configured.");

        // A project that has since been moved or deleted is not an error worth reporting on
        // launch; the app simply starts empty, as it does on a first run.
        if (settings.LastProject.Length > 0 && Directory.Exists(settings.LastProject))
        {
            try { Projects.OpenProject(settings.LastProject); }
            catch (Exception ex) { logs.Warning("app", $"Could not reopen the last project: {ex.Message}"); }
        }
    }

    /// <summary>Current settings.</summary>
    public AppSettings Settings { get; }

    /// <summary>The shared log sink.</summary>
    public LogService Logs { get; }

    /// <summary>File and build operations.</summary>
    public ProjectService Projects { get; }

    /// <summary>The LLM-backed assistant.</summary>
    public AssistantService Assistant { get; }

    /// <summary>Left panel.</summary>
    public ExplorerViewModel Explorer { get; }

    /// <summary>Centre panel.</summary>
    public EditorViewModel Editor { get; }

    /// <summary>Right panel.</summary>
    public ChatViewModel Chat { get; }

    /// <summary>Window title, including the open project.</summary>
    public string Title => Projects.HasProject
        ? $"{Projects.ProjectName} - ScienceAppGen"
        : "ScienceAppGen";

    /// <summary>What the status bar shows for the project slot.</summary>
    public string ProjectLabel => Projects.ProjectName ?? "No project";

    [ObservableProperty] private bool _chatVisible;
    [ObservableProperty] private double _chatWidth;
    [ObservableProperty] private double _explorerWidth;
    [ObservableProperty] private double _logsHeight;
    [ObservableProperty] private bool _logsVisible = true;
    [ObservableProperty] private string _statusMessage = "Ready";
    [ObservableProperty] private bool _isBusy;

    /// <summary>Writes panel sizes and the open project back to app.config.</summary>
    public void PersistLayout()
    {
        Settings.ChatPanelVisible = ChatVisible;
        Settings.ChatPanelWidth = ChatWidth;
        Settings.ExplorerWidth = ExplorerWidth;
        Settings.LogsHeight = LogsHeight;
        Settings.LastProject = Projects.ProjectRoot ?? "";

        try { _configuration.Save(Settings); }
        catch (Exception ex) { Logs.Warning("app", $"Could not save settings: {ex.Message}"); }
    }
}
