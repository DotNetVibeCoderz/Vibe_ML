using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using HFAppGen.Services;

namespace HFAppGen.Views;

/// <summary>What the New Project dialog returns.</summary>
/// <param name="Name">Project name.</param>
/// <param name="Location">Parent directory.</param>
/// <param name="TemplateId">Chosen template.</param>
public sealed record NewProjectResult(string Name, string Location, string TemplateId);

/// <summary>Collects a name, a location and a template.</summary>
public partial class NewProjectDialog : Window
{
    /// <summary>Designer constructor.</summary>
    public NewProjectDialog() : this("") { }

    /// <summary>Creates the dialog rooted at a default location.</summary>
    public NewProjectDialog(string defaultLocation)
    {
        AvaloniaXamlLoader.Load(this);

        var location = this.FindControl<TextBox>("LocationBox");
        if (location is not null)
        {
            location.Text = string.IsNullOrWhiteSpace(defaultLocation)
                ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "HFAppGen")
                : defaultLocation;
        }

        var list = this.FindControl<ListBox>("TemplateList");
        if (list is not null)
        {
            list.ItemsSource = TemplateService.All;
            list.SelectedIndex = 0;
            list.SelectionChanged += (_, _) => UpdateHint();
        }

        UpdateHint();
    }

    private void UpdateHint()
    {
        var list = this.FindControl<ListBox>("TemplateList");
        var hint = this.FindControl<TextBlock>("Hint");
        if (hint is null) return;

        hint.Text = list?.SelectedItem is ProjectTemplate template
            ? $"{template.Files.Count} file(s) will be created."
            : "Pick a template to continue.";
    }

    private async void OnBrowse(object? sender, RoutedEventArgs e)
    {
        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Choose a location",
            AllowMultiple = false,
        });

        if (folders.Count == 0) return;

        var location = this.FindControl<TextBox>("LocationBox");
        if (location is not null) location.Text = folders[0].Path.LocalPath;
    }

    private void OnCreate(object? sender, RoutedEventArgs e)
    {
        var name = this.FindControl<TextBox>("NameBox")?.Text?.Trim();
        var location = this.FindControl<TextBox>("LocationBox")?.Text?.Trim();
        var template = this.FindControl<ListBox>("TemplateList")?.SelectedItem as ProjectTemplate;
        var hint = this.FindControl<TextBlock>("Hint");

        if (string.IsNullOrWhiteSpace(name))
        {
            if (hint is not null) hint.Text = "Enter a project name.";
            return;
        }

        if (string.IsNullOrWhiteSpace(location))
        {
            if (hint is not null) hint.Text = "Choose a location.";
            return;
        }

        if (template is null)
        {
            if (hint is not null) hint.Text = "Pick a template.";
            return;
        }

        var target = Path.Combine(location, name);
        if (Directory.Exists(target) && Directory.EnumerateFileSystemEntries(target).Any())
        {
            if (hint is not null) hint.Text = $"'{name}' already exists here and is not empty.";
            return;
        }

        Close(new NewProjectResult(name, location, template.Id));
    }

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(null);
}
