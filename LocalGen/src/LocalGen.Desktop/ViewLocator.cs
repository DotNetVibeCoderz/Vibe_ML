using Avalonia.Controls;
using Avalonia.Controls.Templates;
using LocalGen.Desktop.ViewModels;

namespace LocalGen.Desktop;

/// <summary>
/// Resolves a view for a view model by name: <c>…ViewModels.ModelsViewModel</c> maps to
/// <c>…Views.ModelsView</c>. Keeps navigation a matter of setting a view model rather than
/// wiring every screen into the shell by hand.
/// </summary>
public sealed class ViewLocator : IDataTemplate
{
    public Control Build(object? data)
    {
        if (data is null)
        {
            return new TextBlock { Text = "No content." };
        }

        var name = data.GetType().FullName!
            .Replace("ViewModels", "Views", StringComparison.Ordinal)
            .Replace("ViewModel", "View", StringComparison.Ordinal);

        var type = Type.GetType(name);

        if (type is null)
        {
            // Surfaced in the UI rather than thrown: a missing view should not take the shell down.
            return new TextBlock
            {
                Text = $"View not found: {name}",
                Margin = new Avalonia.Thickness(20)
            };
        }

        return (Control)Activator.CreateInstance(type)!;
    }

    public bool Match(object? data) => data is ViewModelBase;
}
