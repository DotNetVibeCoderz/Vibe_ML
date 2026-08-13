using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Avalonia.VisualTree;
using ScienceAppGen.Services;
using ScienceAppGen.ViewModels;

namespace ScienceAppGen.Views;

/// <summary>The application shell window.</summary>
public partial class MainWindow : Window
{
    /// <summary>Creates the window and registers the keyboard shortcuts.</summary>
    public MainWindow()
    {
        AvaloniaXamlLoader.Load(this);
        AddShortcuts();
    }

    private ShellViewModel? Shell => DataContext as ShellViewModel;

    private void AddShortcuts()
    {
        void Bind(Key key, KeyModifiers modifiers, Action action)
            => KeyBindings.Add(new KeyBinding
            {
                Gesture = new KeyGesture(key, modifiers),
                Command = new RelayCommand(action),
            });

        Bind(Key.N, KeyModifiers.Control | KeyModifiers.Shift, () => _ = NewProjectAsync());
        Bind(Key.O, KeyModifiers.Control, () => _ = OpenProjectAsync());
        Bind(Key.S, KeyModifiers.Control, Save);
        Bind(Key.S, KeyModifiers.Control | KeyModifiers.Shift, SaveAll);
        Bind(Key.G, KeyModifiers.Control, () => _ = GoToLineAsync());
        Bind(Key.K, KeyModifiers.Control, Format);
        Bind(Key.B, KeyModifiers.Control, ToggleChat);
        Bind(Key.L, KeyModifiers.Control, ToggleLogs);
        Bind(Key.F6, KeyModifiers.None, () => _ = BuildAsync());
        Bind(Key.F5, KeyModifiers.None, () => _ = RunAsync());
        Bind(Key.OemComma, KeyModifiers.Control, () => _ = SettingsAsync());
    }

    // ---------------------------------------------------------------- project

    private void OnNewProject(object? sender, RoutedEventArgs e) => _ = NewProjectAsync();

    private async Task NewProjectAsync()
    {
        if (Shell is null) return;

        var dialog = new NewProjectDialog(Shell.Settings.ProjectsRoot);
        var result = await dialog.ShowDialog<NewProjectResult?>(this);
        if (result is null) return;

        try
        {
            var template = TemplateService.Find(result.TemplateId) ?? TemplateService.All[0];
            var target = Path.Combine(result.Location, result.Name);

            TemplateService.Create(template, target, result.Name);
            Shell.Projects.OpenProject(target);
            Shell.Logs.Success("project", $"Created {result.Name} from '{template.Name}'.");
            Shell.StatusMessage = $"Created {result.Name}";

            OpenFirstSourceFile(target);
        }
        catch (Exception ex)
        {
            Shell.Logs.Error("project", ex.Message);
        }
    }

    private void OnOpenProject(object? sender, RoutedEventArgs e) => _ = OpenProjectAsync();

    private async Task OpenProjectAsync()
    {
        if (Shell is null) return;

        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Open project folder",
            AllowMultiple = false,
        });

        if (folders.Count == 0) return;

        try
        {
            Shell.Projects.OpenProject(folders[0].Path.LocalPath);
            Shell.StatusMessage = $"Opened {Shell.Projects.ProjectName}";
            OpenFirstSourceFile(folders[0].Path.LocalPath);
        }
        catch (Exception ex)
        {
            Shell.Logs.Error("project", ex.Message);
        }
    }

    private void OpenFirstSourceFile(string projectRoot)
    {
        // Landing on Program.cs is almost always what the user wants next.
        var candidate = Directory.EnumerateFiles(projectRoot, "Program.cs", SearchOption.AllDirectories)
            .FirstOrDefault(p => !ProjectService.IsIgnored(p));

        candidate ??= Directory.EnumerateFiles(projectRoot, "*.cs", SearchOption.AllDirectories)
            .FirstOrDefault(p => !ProjectService.IsIgnored(p));

        candidate ??= Directory.EnumerateFiles(projectRoot, "*.ipynb", SearchOption.AllDirectories)
            .FirstOrDefault(p => !ProjectService.IsIgnored(p));

        if (candidate is not null) Shell?.Editor.OpenFile(candidate);
    }

    private async void OnOpenFile(object? sender, RoutedEventArgs e)
    {
        if (Shell is null) return;

        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Open file",
            AllowMultiple = true,
        });

        foreach (var file in files) Shell.Editor.OpenFile(file.Path.LocalPath);
    }

    private void OnCloseProject(object? sender, RoutedEventArgs e)
    {
        if (Shell is null) return;

        Shell.Editor.SaveAll();
        Shell.Editor.CloseAll();
        Shell.Projects.CloseProject();
        Shell.StatusMessage = "Project closed";
    }

    private void OnExit(object? sender, RoutedEventArgs e)
    {
        Shell?.PersistLayout();
        Close();
    }

    // ---------------------------------------------------------------- editing

    private void OnSave(object? sender, RoutedEventArgs e) => Save();

    private void Save()
    {
        if (Shell is null) return;

        Shell.StatusMessage = Shell.Editor.SaveActive()
            ? $"Saved {Shell.Editor.Active?.Name}"
            : "Nothing to save";
    }

    private void OnSaveAll(object? sender, RoutedEventArgs e) => SaveAll();

    private void SaveAll()
    {
        if (Shell is null) return;

        var count = Shell.Editor.SaveAll();
        Shell.StatusMessage = count == 0 ? "Nothing to save" : $"Saved {count} file(s)";
    }

    private void OnFormat(object? sender, RoutedEventArgs e) => Format();

    private void Format()
    {
        if (Shell is null) return;

        Shell.StatusMessage = Shell.Editor.FormatActive();

        // FormatActive rewrites the document's Text; the editor control holds its own copy,
        // so it has to be told to re-read it.
        FindEditorView()?.ReloadFromModel();
    }

    private void OnGoToLine(object? sender, RoutedEventArgs e) => _ = GoToLineAsync();

    private async Task GoToLineAsync()
    {
        if (Shell?.Editor.Active is null) return;

        var dialog = new GoToLineDialog();
        var line = await dialog.ShowDialog<int?>(this);
        if (line is null) return;

        FindEditorView()?.GoToLine(line.Value);
        Shell.StatusMessage = $"Jumped to line {line}";
    }

    private EditorView? FindEditorView() => this.GetVisualDescendants()
        .OfType<EditorView>()
        .FirstOrDefault();

    // ---------------------------------------------------------------- build

    private void OnBuild(object? sender, RoutedEventArgs e) => _ = BuildAsync();

    private async Task BuildAsync()
    {
        if (Shell is null) return;
        if (!Shell.Projects.HasProject) { Shell.StatusMessage = "No project open"; return; }

        Shell.Editor.SaveAll();
        Shell.IsBusy = true;
        Shell.StatusMessage = "Building…";

        var result = await Shell.Projects.BuildAsync();

        Shell.IsBusy = false;
        Shell.StatusMessage = result.Succeeded ? "Build succeeded" : "Build failed";
    }

    private void OnRun(object? sender, RoutedEventArgs e) => _ = RunAsync();

    private async Task RunAsync()
    {
        if (Shell is null) return;
        if (!Shell.Projects.HasProject) { Shell.StatusMessage = "No project open"; return; }

        Shell.Editor.SaveAll();
        Shell.IsBusy = true;
        Shell.StatusMessage = "Building…";

        var build = await Shell.Projects.BuildAsync();
        if (!build.Succeeded)
        {
            Shell.IsBusy = false;
            Shell.StatusMessage = "Build failed";
            return;
        }

        Shell.StatusMessage = "Running…";
        var run = await Shell.Projects.RunAsync();

        Shell.IsBusy = false;
        Shell.StatusMessage = run.Succeeded ? "Run finished" : "Run failed";
    }

    private async void OnDeploy(object? sender, RoutedEventArgs e)
    {
        if (Shell is null) return;
        if (!Shell.Projects.HasProject) { Shell.StatusMessage = "No project open"; return; }

        Shell.Editor.SaveAll();
        Shell.IsBusy = true;
        Shell.StatusMessage = "Publishing…";

        var result = await Shell.Projects.PublishAsync();

        Shell.IsBusy = false;
        Shell.StatusMessage = result.Succeeded
            ? $"Published to {Path.Combine(Shell.Projects.ProjectRoot!, "publish")}"
            : "Publish failed";
    }

    // ---------------------------------------------------------------- view

    private void OnToggleLineNumbers(object? sender, RoutedEventArgs e)
    {
        if (Shell is null) return;

        Shell.Editor.ShowLineNumbers = !Shell.Editor.ShowLineNumbers;
        Shell.Settings.ShowLineNumbers = Shell.Editor.ShowLineNumbers;
        Shell.StatusMessage = Shell.Editor.ShowLineNumbers ? "Line numbers on" : "Line numbers off";
    }

    private void OnToggleChat(object? sender, RoutedEventArgs e) => ToggleChat();

    private void ToggleChat()
    {
        if (Shell is null) return;
        Shell.ChatVisible = !Shell.ChatVisible;
    }

    private void OnToggleLogs(object? sender, RoutedEventArgs e) => ToggleLogs();

    private void ToggleLogs()
    {
        if (Shell is null) return;
        Shell.LogsVisible = !Shell.LogsVisible;
    }

    private void OnClearLogs(object? sender, RoutedEventArgs e) => Shell?.Logs.Clear();

    // ---------------------------------------------------------------- settings

    private void OnSettings(object? sender, RoutedEventArgs e) => _ = SettingsAsync();

    private async Task SettingsAsync()
    {
        if (Shell is null) return;

        var dialog = new SettingsDialog(Shell.Settings);
        var saved = await dialog.ShowDialog<bool>(this);
        if (!saved) return;

        App.Configuration.Save(Shell.Settings);
        Shell.Assistant.Configure(Shell.Settings);
        Shell.Editor.ShowLineNumbers = Shell.Settings.ShowLineNumbers;
        Shell.StatusMessage = "Settings saved";
        Shell.Logs.Info("app", "Settings updated.");
    }
}

/// <summary>A minimal command so key bindings can call plain methods.</summary>
internal sealed class RelayCommand(Action action) : System.Windows.Input.ICommand
{
    /// <inheritdoc />
    public event EventHandler? CanExecuteChanged;

    /// <inheritdoc />
    public bool CanExecute(object? parameter) => true;

    /// <inheritdoc />
    public void Execute(object? parameter) => action();

    /// <summary>Never raised; the commands here are always available.</summary>
    public void RaiseCanExecuteChanged() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}
