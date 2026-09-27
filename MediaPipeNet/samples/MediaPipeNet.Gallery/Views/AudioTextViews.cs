using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using MediaPipeNet.Gallery.Controls;
using MediaPipeNet.Gallery.Services;
using MediaPipeNet.Tasks.Audio;
using MediaPipeNet.Tasks.Text;

namespace MediaPipeNet.Gallery.Views;

/// <summary>Audio classification (YAMNet) and voice activity detection on bundled or user-picked clips.</summary>
public sealed class AudioView : IGalleryPage
{
    private static readonly string[] s_sources = ["speech_16000_hz_mono.wav", "tone:440", "noise", "tone+speech"];
    private readonly ComboBox _source = new() { Width = 260, SelectedIndex = 0 };
    private readonly Canvas _wave = new() { Height = 150, ClipToBounds = true };
    private readonly StackPanel _windows = new() { Spacing = 6 };
    private readonly StackPanel _segments = new() { Spacing = 4 };
    private readonly TextBlock _status = new TextBlock().With("muted");
    private readonly CodeView _code = new();
    private AudioData? _audio;
    private VoiceActivityResult? _vad;
    private AudioClassifier? _classifier;
    private VoiceActivityDetector? _detector;
    private string _name = "";

    public AudioView() => View = Build();

    public Control View { get; }

    private Control Build()
    {
        _source.ItemsSource = new[]
        {
            Loc.Pick("Speech (sample recording)", "Ucapan (rekaman contoh)"),
            Loc.Pick("440 Hz sine tone (synthetic)", "Nada sinus 440 Hz (sintetis)"),
            Loc.Pick("White noise (synthetic)", "Derau putih (sintetis)"),
            Loc.Pick("Tone, then speech", "Nada, lalu ucapan"),
        };
        _source.SelectionChanged += async (_, _) => await RunAsync(null);
        var open = Ui.Button(Loc.Pick("Open WAV…", "Buka WAV…"), "quiet", async (_, _) => await OpenAsync());
        var controls = Ui.Stack(Orientation.Horizontal, 12, _source, open, _status);
        foreach (var c in controls.Children) c.VerticalAlignment = VerticalAlignment.Center;
        controls.Margin = new Thickness(0, 18, 0, 14);

        var waveBorder = new Border { Child = _wave, CornerRadius = new CornerRadius(10), Padding = new Thickness(0), ClipToBounds = true };
        waveBorder.Bind(Border.BackgroundProperty, waveBorder.GetResourceObservable("StageBrush"));
        _wave.SizeChanged += (_, _) => DrawWave();

        var left = Ui.Stack(Orientation.Vertical, 12,
            waveBorder,
            Ui.Panel(Ui.Stack(Orientation.Vertical, 8, Ui.Text(Loc.Pick("AUDIO EVENTS PER 0.975 s WINDOW", "EVENT AUDIO PER JENDELA 0,975 s"), "eyebrow"), _windows)));
        var right = Ui.Stack(Orientation.Vertical, 12,
            Ui.Panel(Ui.Stack(Orientation.Vertical, 8, Ui.Text(Loc.Pick("VOICE ACTIVITY", "AKTIVITAS SUARA"), "eyebrow"), _segments)),
            Ui.Panel(Ui.Stack(Orientation.Vertical, 8, Ui.Text(Loc.T("task.code"), "eyebrow"), new ScrollViewer { Content = _code, HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto }), "panel code"));
        var grid = new Grid { ColumnDefinitions = new ColumnDefinitions("3*,2*") };
        grid.Children.Add(left);
        right.Margin = new Thickness(16, 0, 0, 0);
        Grid.SetColumn(right, 1);
        grid.Children.Add(right);

        var page = new StackPanel { Margin = new Thickness(32, 26, 32, 24) };
        page.Children.Add(Ui.Header("YAMNET · 521 AUDIOSET CLASSES · 16 kHz",
            Loc.Pick("Audio classification", "Klasifikasi audio"),
            Loc.Pick("YAMNet names the sounds in every 0.975-second window — speech, music, animals, alarms, engines — and the voice activity detector turns its speech classes into speech segments. Any WAV works; it is resampled to 16 kHz.",
                     "YAMNet menamai suara di setiap jendela 0,975 detik — ucapan, musik, hewan, alarm, mesin — dan detektor aktivitas suara mengubah kelas ucapannya menjadi segmen bicara. WAV apa pun bisa; otomatis di-resample ke 16 kHz.")));
        page.Children.Add(controls);
        page.Children.Add(grid);
        _code.SetCode("""
            using MediaPipeNet.Tasks.Audio;

            var audio = AudioData.LoadWav("speech.wav");        // any rate, any channel count

            using var classifier = AudioClassifier.Create(new AudioClassifierOptions
            {
                Classifier = new() { MaxResults = 3 },
            });
            foreach (var window in classifier.Classify(audio))
                Console.WriteLine($"{window.TimestampMs} ms: {window.TopCategory}");

            using var vad = VoiceActivityDetector.Create();
            foreach (var s in vad.Detect(audio).Segments)
                Console.WriteLine($"speech {s.StartMs}-{s.EndMs} ms");
            """);
        _ = RunAsync(null);
        return new ScrollViewer { Content = page };
    }

    private async Task OpenAsync()
    {
        var top = TopLevel.GetTopLevel(View);
        if (top is null) return;
        var files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = Loc.Pick("Open WAV…", "Buka WAV…"),
            FileTypeFilter = [new FilePickerFileType("WAV") { Patterns = ["*.wav"] }],
        });
        if (files.Count > 0 && files[0].TryGetLocalPath() is { } path) await RunAsync(path);
    }

    private static string SamplePath(string name) => System.IO.Path.Combine(AppContext.BaseDirectory, "audio", name);

    private AudioData Synthesize(int index)
    {
        const int rate = 16000;
        var rng = new Random(7);
        float[] Tone(int seconds) => Enumerable.Range(0, rate * seconds).Select(i => 0.4f * MathF.Sin(2 * MathF.PI * 440 * i / rate)).ToArray();
        return index switch
        {
            1 => new AudioData(Tone(3), rate),
            2 => new AudioData(Enumerable.Range(0, rate * 3).Select(_ => (float)(rng.NextDouble() * 0.6 - 0.3)).ToArray(), rate),
            _ => new AudioData([.. Tone(2), .. AudioData.LoadWav(SamplePath(s_sources[0])).Samples], rate),
        };
    }

    private async Task RunAsync(string? path)
    {
        _status.Text = Loc.T("task.running");
        int index = _source.SelectedIndex;
        try
        {
            var (audio, name, windows, vad) = await Task.Run(() =>
            {
                var a = path is not null ? AudioData.LoadWav(path) : index == 0 ? AudioData.LoadWav(SamplePath(s_sources[0])) : Synthesize(index);
                var b = TaskEngine.BaseOptions;
                _classifier ??= AudioClassifier.Create(new() { BaseOptions = b, Classifier = new() { MaxResults = 3 } });
                _detector ??= VoiceActivityDetector.Create(new() { BaseOptions = b });
                return (a, path is null ? _source.SelectedItem?.ToString() ?? "" : System.IO.Path.GetFileName(path), _classifier.Classify(a), _detector.Detect(a));
            });
            _audio = audio;
            _vad = vad;
            _name = name;
            _windows.Children.Clear();
            foreach (var w in windows)
            {
                var top = w.Categories;
                var line = Ui.Stack(Orientation.Vertical, 2, Ui.Text($"{w.TimestampMs / 1000.0:0.00} s", "mono"));
                foreach (var c in top) line.Children.Add(Ui.Meter(new ResultRow(c.CategoryName ?? c.Index.ToString(CultureInfo.InvariantCulture), c.Score.ToString("P0", CultureInfo.CurrentCulture), c.Score)));
                _windows.Children.Add(line);
            }
            _segments.Children.Clear();
            _segments.Children.Add(Ui.Meter(new ResultRow(Loc.Pick("Speech windows", "Jendela berisi ucapan"), vad.SpeechRatio.ToString("P0", CultureInfo.CurrentCulture), vad.SpeechRatio)));
            if (vad.Segments.Count == 0) _segments.Children.Add(Ui.Text(Loc.Pick("No speech detected.", "Tidak ada ucapan terdeteksi."), "muted"));
            foreach (var s in vad.Segments)
                _segments.Children.Add(Ui.Text($"{s.StartMs / 1000.0:0.00} s – {s.EndMs / 1000.0:0.00} s · p {s.MeanProbability:P0}", "mono"));
            _status.Text = $"{name} · {audio.Duration.TotalSeconds:0.00} s · {audio.SampleRate} Hz";
            DrawWave();
        }
        catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or MediaPipeException)
        {
            _status.Text = Loc.T("task.error") + e.Message;
        }
    }

    private void DrawWave()
    {
        _wave.Children.Clear();
        if (_audio is not { Samples.Length: > 0 } audio) return;
        double w = _wave.Bounds.Width, h = _wave.Bounds.Height;
        if (w <= 0) return;
        double durationMs = audio.Duration.TotalMilliseconds;
        foreach (var s in _vad?.Segments ?? [])
        {
            var band = new Rectangle { Width = Math.Max(1, (s.EndMs - s.StartMs) / durationMs * w), Height = h, Fill = new SolidColorBrush(Color.Parse("#3335E0B5")) };
            Canvas.SetLeft(band, s.StartMs / durationMs * w);
            _wave.Children.Add(band);
        }
        int columns = (int)w;
        var points = new List<Point>(columns * 2);
        var samples = audio.Samples;
        float peak = Math.Max(1e-3f, samples.Max(MathF.Abs));
        for (int x = 0; x < columns; x++)
        {
            int from = (int)((long)x * samples.Length / columns), to = (int)((long)(x + 1) * samples.Length / columns);
            float min = 0, max = 0;
            for (int i = from; i < to; i++)
            {
                min = MathF.Min(min, samples[i]);
                max = MathF.Max(max, samples[i]);
            }
            points.Add(new Point(x, h / 2 - max / peak * h * 0.45));
            points.Add(new Point(x, h / 2 - min / peak * h * 0.45));
        }
        _wave.Children.Add(new Polyline { Points = points, Stroke = new SolidColorBrush(Color.Parse("#6D8BFF")), StrokeThickness = 1 });
        var label = new TextBlock { Text = _name, Foreground = new SolidColorBrush(Color.Parse("#98A2B3")), FontSize = 11, Margin = new Thickness(10, 6) };
        _wave.Children.Add(label);
    }

    public async Task PrepareForScreenshotAsync()
    {
        _source.SelectedIndex = 3;
        await RunAsync(null);
    }

    public void Deactivate()
    {
        _classifier?.Dispose();
        _detector?.Dispose();
        _classifier = null;
        _detector = null;
    }
}

/// <summary>Sentiment, language identification and sentence similarity.</summary>
public sealed class TextView : IGalleryPage
{
    private static readonly string[] s_presets =
    [
        "It's beautiful outside.",
        "The movie was a complete waste of time, boring and far too long.",
        "Selamat pagi, apa kabar hari ini?",
        "Il fait très beau aujourd'hui.",
        "Das ist ein wunderbares Buch.",
    ];

    private readonly TextBox _input = new() { AcceptsReturn = true, TextWrapping = TextWrapping.Wrap, Height = 90, Text = s_presets[0] };
    private readonly TextBox _compare = new() { AcceptsReturn = true, TextWrapping = TextWrapping.Wrap, Height = 60, Text = "What a lovely sunny day." };
    private readonly StackPanel _sentiment = new() { Spacing = 4 };
    private readonly StackPanel _language = new() { Spacing = 4 };
    private readonly StackPanel _similarity = new() { Spacing = 4 };
    private readonly TextBlock _status = new TextBlock().With("muted");
    private readonly CodeView _code = new();
    private TextClassifier? _bert, _fast;
    private LanguageDetector? _languageDetector;
    private TextEmbedder? _embedder;

    public TextView() => View = Build();

    public Control View { get; }

    private Control Build()
    {
        var presets = new WrapPanel();
        foreach (var p in s_presets)
        {
            var b = Ui.Button(p.Length > 34 ? p[..32] + "…" : p, "quiet chip", async (_, _) =>
            {
                _input.Text = p;
                await RunAsync();
            });
            b.Margin = new Thickness(0, 0, 8, 8);
            presets.Children.Add(b);
        }
        var run = Ui.Button(Loc.Pick("Analyze", "Analisis"), "primary", async (_, _) => await RunAsync());
        var inputPanel = Ui.Panel(Ui.Stack(Orientation.Vertical, 8,
            Ui.Text(Loc.Pick("TEXT", "TEKS"), "eyebrow"), _input, presets,
            Ui.Text(Loc.Pick("COMPARE WITH (SENTENCE SIMILARITY)", "BANDINGKAN DENGAN (KEMIRIPAN KALIMAT)"), "eyebrow"), _compare,
            Ui.Stack(Orientation.Horizontal, 12, run, _status)));

        var results = Ui.Stack(Orientation.Vertical, 12,
            Ui.Panel(Ui.Stack(Orientation.Vertical, 8, Ui.Text(Loc.Pick("SENTIMENT", "SENTIMEN"), "eyebrow"), _sentiment)),
            Ui.Panel(Ui.Stack(Orientation.Vertical, 8, Ui.Text(Loc.Pick("LANGUAGE", "BAHASA"), "eyebrow"), _language)),
            Ui.Panel(Ui.Stack(Orientation.Vertical, 8, Ui.Text(Loc.Pick("SIMILARITY (MOBILEBERT EMBEDDINGS)", "KEMIRIPAN (EMBEDDING MOBILEBERT)"), "eyebrow"), _similarity)));

        var grid = new Grid { ColumnDefinitions = new ColumnDefinitions("3*,2*") };
        var left = Ui.Stack(Orientation.Vertical, 12, inputPanel,
            Ui.Panel(Ui.Stack(Orientation.Vertical, 8, Ui.Text(Loc.T("task.code"), "eyebrow"), new ScrollViewer { Content = _code, HorizontalScrollBarVisibility = Avalonia.Controls.Primitives.ScrollBarVisibility.Auto }), "panel code"));
        grid.Children.Add(left);
        results.Margin = new Thickness(16, 0, 0, 0);
        Grid.SetColumn(results, 1);
        grid.Children.Add(results);

        var page = new StackPanel { Margin = new Thickness(32, 26, 32, 24) };
        page.Children.Add(Ui.Header("MOBILEBERT · AVERAGE WORD · 110 LANGUAGES",
            Loc.Pick("Text understanding", "Pemahaman teks"),
            Loc.Pick("Sentiment from MobileBERT and a tiny average-word model, language identification for 110 languages, and sentence embeddings for semantic similarity — with MediaPipe's exact tokenizers.",
                     "Sentimen dari MobileBERT dan model average-word yang mungil, identifikasi 110 bahasa, dan embedding kalimat untuk kemiripan semantik — dengan tokenizer yang persis sama dengan MediaPipe.")));
        page.Children.Add(new Border { Height = 18 });
        page.Children.Add(grid);
        _code.SetCode("""
            using MediaPipeNet.Tasks.Text;

            using var classifier = TextClassifier.Create();                 // MobileBERT, SST-2
            Console.WriteLine(classifier.Classify("It's beautiful outside.").TopCategory);

            using var detector = LanguageDetector.Create();
            Console.WriteLine(detector.Detect("Selamat pagi, apa kabar?").TopLanguage);   // id

            using var embedder = TextEmbedder.Create(new TextEmbedderOptions { L2Normalize = true });
            double similarity = TextEmbedder.CosineSimilarity(
                embedder.Embed("It's beautiful outside.").Embedding,
                embedder.Embed("What a lovely sunny day.").Embedding);
            """);
        _ = RunAsync();
        return new ScrollViewer { Content = page };
    }

    private async Task RunAsync()
    {
        string text = _input.Text ?? "", other = _compare.Text ?? "";
        if (string.IsNullOrWhiteSpace(text)) return;
        _status.Text = Loc.T("task.running");
        try
        {
            var r = await Task.Run(() =>
            {
                var b = TaskEngine.BaseOptions;
                var bert = _bert ??= TextClassifier.Create(new() { BaseOptions = b });
                var fast = _fast ??= TextClassifier.Create(new() { BaseOptions = b, Model = TextClassifierModel.AverageWord });
                var language = _languageDetector ??= LanguageDetector.Create(new() { BaseOptions = b, Classifier = new() { MaxResults = 4 } });
                var embedder = _embedder ??= TextEmbedder.Create(new() { BaseOptions = b, L2Normalize = true });
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                var sentiment = bert.Classify(text);
                var quick = fast.Classify(text);
                var languages = language.Detect(text);
                double? similarity = string.IsNullOrWhiteSpace(other) ? null
                    : TextEmbedder.CosineSimilarity(embedder.Embed(text).Embedding, embedder.Embed(other).Embedding);
                return (sentiment, quick, languages, similarity, System.Diagnostics.Stopwatch.GetElapsedTime(t0));
            });
            _sentiment.Children.Clear();
            foreach (var c in r.sentiment.Categories)
                _sentiment.Children.Add(Ui.Meter(new ResultRow("MobileBERT · " + c.CategoryName, c.Score.ToString("P1", CultureInfo.CurrentCulture), c.Score)));
            var positive = r.quick.Categories.FirstOrDefault(c => c.CategoryName == "1");
            if (positive is not null)
                _sentiment.Children.Add(Ui.Meter(new ResultRow(Loc.Pick("Average word · positive", "Average word · positif"), positive.Score.ToString("P1", CultureInfo.CurrentCulture), positive.Score)));
            _language.Children.Clear();
            foreach (var p in r.languages.Predictions)
                _language.Children.Add(Ui.Meter(new ResultRow(p.LanguageCode, p.Probability.ToString("P1", CultureInfo.CurrentCulture), p.Probability)));
            _similarity.Children.Clear();
            if (r.similarity is { } s)
                _similarity.Children.Add(Ui.Meter(new ResultRow(Loc.Pick("Cosine similarity", "Kemiripan kosinus"), s.ToString("0.000", CultureInfo.InvariantCulture), Math.Max(0, s))));
            _status.Text = $"{r.Item5.TotalMilliseconds:0} ms";
        }
        catch (MediaPipeException e)
        {
            _status.Text = Loc.T("task.error") + e.Message;
        }
    }

    public async Task PrepareForScreenshotAsync()
    {
        _input.Text = s_presets[2];
        _compare.Text = "Good morning, how are you today?";
        await RunAsync();
    }

    public void Deactivate()
    {
        _bert?.Dispose();
        _fast?.Dispose();
        _languageDetector?.Dispose();
        _embedder?.Dispose();
        _bert = _fast = null;
        _languageDetector = null;
        _embedder = null;
    }
}
