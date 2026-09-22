using System.Globalization;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using HFAppGen.Models;

namespace HFAppGen.Views;

/// <summary>
/// Edits <see cref="AppSettings"/>. Everything here maps one-for-one onto <c>app.config</c>.
/// </summary>
public partial class SettingsDialog : Window
{
    private readonly AppSettings _settings;

    /// <summary>Designer constructor.</summary>
    public SettingsDialog() : this(new AppSettings()) { }

    /// <summary>Creates the dialog bound to a settings object.</summary>
    public SettingsDialog(AppSettings settings)
    {
        _settings = settings;
        AvaloniaXamlLoader.Load(this);

        var providers = this.FindControl<ComboBox>("ProviderBox");
        if (providers is not null)
        {
            providers.ItemsSource = Enum.GetValues<LlmProvider>();
            providers.SelectedItem = settings.Provider;
        }

        Set("ModelBox", settings.Model);
        Set("KeyBox", settings.ApiKey);
        Set("EndpointBox", settings.Endpoint);
        Set("MaxTokensBox", settings.MaxTokens.ToString(CultureInfo.InvariantCulture));
        Set("ModelsBox", string.Join(", ", settings.AvailableModels));
        Set("PromptBox", settings.SystemPrompt);
        Set("TavilyBox", settings.TavilyApiKey);
        Set("WebTimeoutBox", settings.WebTimeoutSeconds.ToString(CultureInfo.InvariantCulture));
        Set("FontSizeBox", settings.EditorFontSize.ToString(CultureInfo.InvariantCulture));
        Set("TabSizeBox", settings.TabSize.ToString(CultureInfo.InvariantCulture));

        var lineNumbers = this.FindControl<CheckBox>("LineNumbersBox");
        if (lineNumbers is not null) lineNumbers.IsChecked = settings.ShowLineNumbers;

        var slider = this.FindControl<Slider>("TemperatureSlider");
        if (slider is not null)
        {
            slider.Value = settings.Temperature;
            slider.PropertyChanged += (_, e) =>
            {
                if (e.Property.Name == nameof(Slider.Value)) UpdateTemperatureLabel();
            };
        }
        UpdateTemperatureLabel();

        var path = this.FindControl<TextBlock>("ConfigPathLabel");
        if (path is not null) path.Text = App.Configuration.ConfigPath;

        UpdateProviderHint();
    }

    private void Set(string name, string? value)
    {
        var box = this.FindControl<TextBox>(name);
        if (box is not null) box.Text = value ?? "";
    }

    private string Get(string name) => this.FindControl<TextBox>(name)?.Text?.Trim() ?? "";

    private void UpdateTemperatureLabel()
    {
        var slider = this.FindControl<Slider>("TemperatureSlider");
        var label = this.FindControl<TextBlock>("TemperatureLabel");
        if (slider is not null && label is not null)
            label.Text = slider.Value.ToString("0.0", CultureInfo.InvariantCulture);
    }

    private void OnProviderChanged(object? sender, SelectionChangedEventArgs e) => UpdateProviderHint();

    /// <summary>Tells the user what the selected provider actually needs.</summary>
    private void UpdateProviderHint()
    {
        var hint = this.FindControl<TextBlock>("ProviderHint");
        if (hint is null) return;

        var provider = this.FindControl<ComboBox>("ProviderBox")?.SelectedItem as LlmProvider?;
        hint.Text = provider switch
        {
            LlmProvider.AzureOpenAI =>
                "Azure OpenAI: model is the deployment name, and the endpoint is required, " +
                "e.g. https://your-resource.openai.azure.com/",
            LlmProvider.Ollama =>
                "Ollama: no key needed. Endpoint is usually http://localhost:11434 and the model " +
                "must already be pulled.",
            LlmProvider.Anthropic =>
                "Anthropic: served by a built-in chat service, because Semantic Kernel has no " +
                "official Claude connector. Leave the endpoint blank for the default host.",
            LlmProvider.Google =>
                "Google: uses the Gemini API key. Leave the endpoint blank.",
            _ => "OpenAI: leave the endpoint blank unless you are using a compatible gateway.",
        };
    }

    private void OnSave(object? sender, RoutedEventArgs e)
    {
        if (this.FindControl<ComboBox>("ProviderBox")?.SelectedItem is LlmProvider provider)
            _settings.Provider = provider;

        _settings.Model = Get("ModelBox");
        _settings.ApiKey = Get("KeyBox");
        _settings.Endpoint = Get("EndpointBox");
        _settings.SystemPrompt = this.FindControl<TextBox>("PromptBox")?.Text ?? "";
        _settings.TavilyApiKey = Get("TavilyBox");

        _settings.AvailableModels = Get("ModelsBox")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList();

        if (this.FindControl<Slider>("TemperatureSlider") is { } slider)
            _settings.Temperature = Math.Round(slider.Value, 2);

        if (int.TryParse(Get("MaxTokensBox"), out var maxTokens) && maxTokens > 0)
            _settings.MaxTokens = maxTokens;

        if (int.TryParse(Get("WebTimeoutBox"), out var timeout) && timeout > 0)
            _settings.WebTimeoutSeconds = timeout;

        if (double.TryParse(Get("FontSizeBox"), NumberStyles.Float, CultureInfo.InvariantCulture, out var fontSize)
            && fontSize is >= 8 and <= 32)
            _settings.EditorFontSize = fontSize;

        if (int.TryParse(Get("TabSizeBox"), out var tabSize) && tabSize is >= 1 and <= 8)
            _settings.TabSize = tabSize;

        _settings.ShowLineNumbers = this.FindControl<CheckBox>("LineNumbersBox")?.IsChecked ?? true;

        Close(true);
    }

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(false);
}
