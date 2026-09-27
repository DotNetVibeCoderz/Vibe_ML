using MediaPipeNet.Tasks.Audio;
using MediaPipeNet.Tasks.Text;
using MediaPipeNet.Tasks.Vision;
using MediaPipeNet.Tests;

namespace MediaPipeNet.Tasks.Tests;

/// <summary>
/// Custom models (ModelPath + Labels / companion files). The catalog models stand in for user models, copied
/// under a new name with their own label files, exactly as <c>convert_models.py --custom</c> would write them.
/// </summary>
public sealed class CustomModelTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mpnet-custom").FullName;

    private string Copy(string modelFile, string newName, IEnumerable<string>? labels = null, string? vocabFrom = null)
    {
        var path = Path.Combine(_dir, newName + ".onnx");
        File.Copy(Path.Combine(TestPaths.Models, modelFile), path);
        if (labels is not null) File.WriteAllLines(Path.Combine(_dir, newName + ".labels.txt"), labels);
        if (vocabFrom is not null) File.Copy(Path.Combine(TestPaths.Models, vocabFrom), Path.Combine(_dir, newName + ".vocab.txt"));
        return path;
    }

    [Fact]
    public void Image_classifier_uses_the_companion_label_file()
    {
        var labels = File.ReadAllLines(Path.Combine(TestPaths.Models, "efficientnet_lite0.labels.txt")).Select(l => "custom:" + l).ToArray();
        var model = Copy("efficientnet_lite0.onnx", "my_food_classifier", labels);
        using var classifier = ImageClassifier.Create(new() { BaseOptions = Fixtures.Base, ModelPath = model, MaxResults = 1 });
        using var image = Fixtures.Load("burger.jpg");
        classifier.Classify(image).Categories[0].CategoryName.Should().Be("custom:cheeseburger");
    }

    [Fact]
    public void Object_detector_uses_explicit_labels_and_the_model_input_size()
    {
        var model = Copy("efficientdet_lite0.onnx", "my_detector");
        var labels = Enumerable.Range(0, 90).Select(i => $"class{i}").ToArray();
        using var detector = ObjectDetector.Create(new() { BaseOptions = Fixtures.Base, ModelPath = model, Labels = labels });
        using var image = Fixtures.Load("cats_and_dogs.jpg");
        var result = detector.Detect(image);
        result.Detections.Should().NotBeEmpty().And.OnlyContain(d => d.TopCategory.CategoryName!.StartsWith("class", StringComparison.Ordinal));
        result.Detections.Select(d => d.TopCategory.CategoryName).Should().Contain(["class16", "class17"]); // COCO cat / dog ids
    }

    [Fact]
    public void Audio_classifier_accepts_a_custom_model_and_labels()
    {
        var labels = Enumerable.Range(0, 521).Select(i => i == 0 ? "voice" : $"sound{i}").ToArray();
        var model = Copy("yamnet.onnx", "my_audio", labels);
        using var classifier = AudioClassifier.Create(new() { BaseOptions = Fixtures.Base, ModelPath = model });
        classifier.Classify(AudioData.LoadWav(TestPaths.Audio("speech_16000_hz_mono.wav")))[1].TopCategory!.CategoryName.Should().Be("voice");
    }

    [Fact]
    public void Text_classifier_reads_labels_and_vocabulary_next_to_the_model()
    {
        var model = Copy("average_word_classifier.onnx", "my_sentiment", ["bad", "good"], vocabFrom: "average_word_classifier.vocab.txt");
        using var classifier = TextClassifier.Create(new() { BaseOptions = Fixtures.Base, Model = TextClassifierModel.AverageWord, ModelPath = model });
        classifier.Classify("It's beautiful outside.").TopCategory!.CategoryName.Should().Be("good");
    }

    [Fact]
    public void Gesture_recognizer_uses_a_custom_classifier()
    {
        string[] labels = ["nothing", "fist", "palm", "point_up", "thumbs_down", "thumbs_up", "peace", "love"];
        var model = Copy("canned_gesture_classifier.onnx", "my_gestures_custom_gesture_classifier", labels);
        using var recognizer = GestureRecognizer.Create(new() { BaseOptions = Fixtures.Base, ClassifierModelPath = model });
        using var image = Fixtures.Load("victory.jpg");
        recognizer.Recognize(image).Hands[0].TopGesture.CategoryName.Should().Be("peace");
    }

    [Fact]
    public void A_missing_custom_model_is_reported()
    {
        var act = () => ImageClassifier.Create(new() { BaseOptions = Fixtures.Base, ModelPath = Path.Combine(_dir, "nope.onnx") });
        act.Should().Throw<ModelNotFoundException>();
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }
}
