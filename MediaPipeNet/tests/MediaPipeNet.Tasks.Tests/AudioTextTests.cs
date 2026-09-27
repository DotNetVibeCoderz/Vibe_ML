using MediaPipeNet.Tasks.Audio;
using MediaPipeNet.Tasks.Text;
using MediaPipeNet.Tasks.Text.Tokenizers;
using MediaPipeNet.Tests;

namespace MediaPipeNet.Tasks.Tests;

/// <summary>Audio and text tasks, cross-validated against the official MediaPipe Python package where possible.</summary>
public class AudioTextTests
{
    private static readonly string[] s_sentences =
    [
        "It's beautiful outside.",
        "The movie was a complete waste of time, boring and far too long.",
        "Selamat pagi, apa kabar hari ini?",
        "Il fait très beau aujourd'hui.",
        "Das ist ein wunderbares Buch.",
    ];

    [Fact]
    public void Audio_classifier_matches_mediapipe()
    {
        using var classifier = AudioClassifier.Create(new() { BaseOptions = Fixtures.Base, Classifier = new() { MaxResults = 3 } });
        var results = classifier.Classify(AudioData.LoadWav(TestPaths.Audio("speech_16000_hz_mono.wav")));
        var expected = Golden.V2("audio_classifier").GetProperty("speech_16000_hz_mono.wav");
        results.Should().HaveCount(expected.GetArrayLength());
        for (int i = 0; i < results.Count; i++)
        {
            var e = expected[i];
            results[i].TimestampMs.Should().Be(e.GetProperty("timestamp_ms").GetInt64());
            var top = e.GetProperty("categories")[0];
            // YAMNet ships int8-quantized; the ONNX float model differs from TFLite by up to ~0.07.
            if (top.GetProperty("score").GetSingle() > 0.5f)
            {
                results[i].TopCategory!.CategoryName.Should().Be(top.GetProperty("name").GetString());
                results[i].TopCategory!.Score.Should().BeApproximately(top.GetProperty("score").GetSingle(), 0.08f);
            }
        }
    }

    [Fact]
    public void Audio_stream_mode_matches_clip_mode()
    {
        var audio = AudioData.LoadWav(TestPaths.Audio("speech_16000_hz_mono.wav"));
        using var clips = AudioClassifier.Create(new() { BaseOptions = Fixtures.Base });
        var expected = clips.Classify(audio);
        var streamed = new List<AudioClassificationResult>();
        using var stream = AudioClassifier.Create(new() { BaseOptions = Fixtures.Base, RunningMode = AudioRunningMode.AudioStream, ResultCallback = streamed.Add });
        // Feed 100 ms chunks.
        for (int start = 0, ts = 0; start < audio.Samples.Length; start += 1600, ts += 100)
            stream.ClassifyStream(new AudioData(audio.Samples[start..Math.Min(start + 1600, audio.Samples.Length)], 16000), ts);
        streamed.Should().HaveCount(expected.Count - 1); // the partial last window is only classified in clip mode
        for (int i = 0; i < streamed.Count; i++)
        {
            streamed[i].TimestampMs.Should().Be(expected[i].TimestampMs);
            streamed[i].TopCategory!.Score.Should().BeApproximately(expected[i].TopCategory!.Score, 1e-5f);
        }
    }

    [Fact]
    public void Resampling_preserves_the_classification()
    {
        var audio = AudioData.LoadWav(TestPaths.Audio("speech_16000_hz_mono.wav"));
        var at44k = audio.Resample(44_100);
        at44k.SampleRate.Should().Be(44_100);
        at44k.Duration.TotalSeconds.Should().BeApproximately(audio.Duration.TotalSeconds, 0.01);
        using var classifier = AudioClassifier.Create(new() { BaseOptions = Fixtures.Base });
        classifier.Classify(at44k)[1].TopCategory!.CategoryName.Should().Be("Speech");
    }

    [Fact]
    public void Voice_activity_detector_finds_the_speech()
    {
        using var vad = VoiceActivityDetector.Create(new() { BaseOptions = Fixtures.Base });
        var audio = AudioData.LoadWav(TestPaths.Audio("speech_16000_hz_mono.wav"));
        var result = vad.Detect(audio);
        result.Frames.Should().NotBeEmpty();
        result.Segments.Should().ContainSingle();
        var speech = result.Segments[0];
        speech.StartMs.Should().Be(0);
        speech.EndMs.Should().BeInRange(3500, 4400); // the clip ends with ~0.4 s of ticks and silence
        var silence = vad.Detect(new AudioData(new float[32000], 16000));
        silence.Segments.Should().BeEmpty();
        silence.SpeechRatio.Should().Be(0);
    }

    [Fact]
    public void Wav_loader_reads_pcm_and_float()
    {
        var audio = AudioData.LoadWav(TestPaths.Audio("speech_16000_hz_mono.wav"));
        audio.SampleRate.Should().Be(16000);
        audio.Samples.Should().HaveCount(68360);
        audio.Samples.Max(MathF.Abs).Should().BeInRange(0.01f, 1f);
        // A stereo float WAV is down-mixed to mono.
        var stereo = AudioData.LoadWav(MakeFloatWav([0.5f, -0.5f, 0.25f, 0.25f], 8000, 2));
        stereo.Samples.Should().Equal(0f, 0.25f);
        stereo.SampleRate.Should().Be(8000);
    }

    [Theory]
    [InlineData(TextClassifierModel.Bert, "bert_classifier.tflite")]
    [InlineData(TextClassifierModel.AverageWord, "average_word_classifier.tflite")]
    public void Text_classifier_matches_mediapipe(TextClassifierModel model, string goldenKey)
    {
        using var classifier = TextClassifier.Create(new() { BaseOptions = Fixtures.Base, Model = model });
        var expected = Golden.V2("text_classifier").GetProperty(goldenKey);
        foreach (var sentence in s_sentences[..2])
        {
            var result = classifier.Classify(sentence);
            var e = expected.GetProperty(sentence);
            result.Categories.Should().HaveCount(e.GetArrayLength());
            for (int i = 0; i < result.Categories.Count; i++)
            {
                result.Categories[i].CategoryName.Should().Be(e[i].GetProperty("name").GetString());
                result.Categories[i].Score.Should().BeApproximately(e[i].GetProperty("score").GetSingle(), 0.01f, sentence);
            }
        }
    }

    [Fact]
    public void Text_embedder_matches_mediapipe()
    {
        var expected = Golden.V2("text_embedder");
        using var embedder = TextEmbedder.Create(new() { BaseOptions = Fixtures.Base, L2Normalize = true });
        var a = embedder.Embed("I love sunny days.").Embedding;
        var b = embedder.Embed("Sunny weather makes me happy.").Embedding;
        var c = embedder.Embed("The server crashed at midnight.").Embedding;
        a.Dimension.Should().Be(expected.GetProperty("dim").GetInt32());
        // The TFLite model runs int8 weights with dynamically quantized activations; the ONNX model computes
        // the same network in float, so similarities differ by ~0.02.
        TextEmbedder.CosineSimilarity(a, b).Should().BeApproximately(expected.GetProperty("similarity_related").GetDouble(), 0.03);
        TextEmbedder.CosineSimilarity(a, c).Should().BeApproximately(expected.GetProperty("similarity_unrelated").GetDouble(), 0.03);
        TextEmbedder.CosineSimilarity(a, b).Should().BeGreaterThan(TextEmbedder.CosineSimilarity(a, c));
    }

    [Fact]
    public void Language_detector_matches_mediapipe()
    {
        using var detector = LanguageDetector.Create(new() { BaseOptions = Fixtures.Base });
        var expected = Golden.V2("language_detector");
        foreach (var sentence in s_sentences)
        {
            var result = detector.Detect(sentence);
            var e = expected.GetProperty(sentence);
            result.Predictions.Should().HaveCount(e.GetArrayLength());
            for (int i = 0; i < result.Predictions.Count; i++)
            {
                result.Predictions[i].LanguageCode.Should().Be(e[i].GetProperty("language").GetString(), sentence);
                result.Predictions[i].Probability.Should().BeApproximately(e[i].GetProperty("probability").GetSingle(), 1e-3f, sentence);
            }
        }
    }

    [Fact]
    public void Bert_tokenizer_follows_wordpiece_rules()
    {
        var vocab = new[] { "[PAD]", "[UNK]", "[CLS]", "[SEP]", "un", "##aff", "##able", "it", "'", "s", "!", "你" };
        var tokenizer = new BertTokenizer(vocab);
        tokenizer.Tokenize("Unaffable IT's!你").Should().Equal("un", "##aff", "##able", "it", "'", "s", "!", "你");
        tokenizer.Tokenize("unknownword").Should().Equal("[UNK]");
        var (ids, mask, segments) = tokenizer.Encode("unaffable", 8);
        ids.Should().Equal(2, 4, 5, 6, 3, 0, 0, 0);
        mask.Should().Equal(1, 1, 1, 1, 1, 0, 0, 0);
        segments.Should().OnlyContain(v => v == 0);
    }

    [Fact]
    public void Regex_tokenizer_splits_like_re2()
    {
        var tokenizer = new RegexTokenizer(@"[^\w\']+", ["<PAD> 0", "<START> 1", "<UNKNOWN> 2", "it's 3", "fine 4"]);
        // RE2's \w is ASCII-only, so "é" is a delimiter; no lower-casing.
        tokenizer.Tokenize("It's fine, it's FINE! é").Should().Equal("It's", "fine", "it's", "FINE");
        tokenizer.Encode("it's fine, really", 6).Should().Equal(1, 3, 4, 2, 0, 0);
    }

    [Fact]
    public void Ngram_hasher_wraps_and_hashes_characters()
    {
        NGramHasher.MurmurHash64(""u8, 0).Should().Be(0UL);
        var hasher = new NGramHasher();
        var (ids, tokens) = hasher.Hash("Hi!");
        tokens.Should().Be(5); // ^ h i (space for "!") $
        ids.Should().HaveCount(4 * 5).And.OnlyContain(i => i >= 1 && i <= 13000);
        hasher.Hash("HI!").Ids.Should().Equal(ids); // lower-cased first
    }

    private static byte[] MakeFloatWav(float[] samples, int rate, int channels)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write("RIFF"u8);
        w.Write(36 + samples.Length * 4);
        w.Write("WAVE"u8);
        w.Write("fmt "u8);
        w.Write(16);
        w.Write((short)3);
        w.Write((short)channels);
        w.Write(rate);
        w.Write(rate * channels * 4);
        w.Write((short)(channels * 4));
        w.Write((short)32);
        w.Write("data"u8);
        w.Write(samples.Length * 4);
        foreach (var s in samples) w.Write(s);
        return ms.ToArray();
    }
}
