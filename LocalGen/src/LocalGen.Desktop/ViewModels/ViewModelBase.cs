using CommunityToolkit.Mvvm.ComponentModel;

namespace LocalGen.Desktop.ViewModels;

/// <summary>Base for every view model, and the type <see cref="ViewLocator"/> matches on.</summary>
public abstract partial class ViewModelBase : ObservableObject
{
    /// <summary>Set while a long operation is in flight, so views can disable their controls.</summary>
    [ObservableProperty]
    private bool _isBusy;

    /// <summary>User-facing error for the current screen. Empty when there is nothing wrong.</summary>
    [ObservableProperty]
    private string _errorMessage = string.Empty;

    /// <summary>
    /// Called the first time a screen is shown. Loading here rather than in the constructor keeps
    /// startup fast: a screen the user never opens never queries anything.
    /// </summary>
    public virtual Task InitializeAsync() => Task.CompletedTask;

    /// <summary>
    /// Runs an operation with the busy flag set and any failure reported on the screen instead of
    /// thrown — an unhandled exception on a UI thread would take the window down.
    /// </summary>
    protected async Task RunAsync(Func<Task> operation)
    {
        if (IsBusy)
        {
            return;
        }

        IsBusy = true;
        ErrorMessage = string.Empty;

        try
        {
            await operation().ConfigureAwait(true);
        }
        catch (OperationCanceledException)
        {
            // Cancellation is a user action, not a failure.
        }
        catch (Exception ex)
        {
            ErrorMessage = ex.Message;
        }
        finally
        {
            IsBusy = false;
        }
    }
}
