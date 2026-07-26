using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.ML.Execution.Executors;
using BlazorML.ML.Training;

namespace BlazorML.Tests;

/// <summary>
/// The vision and NLP modules are in the catalogue in every build, but their backbones are only
/// compiled in when the solution is built with <c>-p:EnableVisionNlp=true</c>. What matters in a
/// standard build is that they are discoverable and that trying to run one explains itself
/// rather than failing obscurely.
/// </summary>
public class VisionNlpTests
{
    private readonly AlgorithmExecutor _algorithms = new();
    private readonly TrainingExecutor _training = new();

    private static TabularData Labelled(string valueColumn)
    {
        var table = TabularData.WithColumns(valueColumn, "kelas");
        table.AddRow("baris satu", "a");
        table.AddRow("baris dua", "b");

        return table;
    }

    [Theory]
    [InlineData("algo.vision.imageClassification", MlTask.ImageClassification)]
    [InlineData("algo.nlp.textClassification", MlTask.TextClassification)]
    public void The_modules_are_in_the_catalogue_whatever_the_build(string moduleId, MlTask task)
    {
        var module = ModuleCatalog.Find(moduleId);

        Assert.NotNull(module);
        Assert.Equal(task, module!.Task);
        Assert.Equal(ModuleCategory.Algorithm, module.Category);
        Assert.Contains(ModuleCatalog.TrainersFor(task), m => m.Id == moduleId);
    }

    [Fact]
    public void They_are_findable_by_the_words_a_user_would_search_for()
    {
        Assert.Contains(ModuleCatalog.Search("gambar"), m => m.Id == "algo.vision.imageClassification");
        Assert.Contains(ModuleCatalog.Search("vision"), m => m.Id == "algo.vision.imageClassification");
        Assert.Contains(ModuleCatalog.Search("teks"), m => m.Id == "algo.nlp.textClassification");
        Assert.Contains(ModuleCatalog.Search("nlp"), m => m.Id == "algo.nlp.textClassification");
    }

    [Fact]
    public void AutoML_offers_the_vision_and_nlp_tasks()
    {
        // The specification asks for AutoML across classification, regression, vision and NLP.
        var task = ModuleCatalog.Find("train.autoML")!.Parameters.First(p => p.Name == "task");

        Assert.Contains(task.Choices, c => c.Value == "ImageClassification");
        Assert.Contains(task.Choices, c => c.Value == "TextClassification");
    }

    [Theory]
    [InlineData("algo.vision.imageClassification")]
    [InlineData("algo.nlp.textClassification")]
    public void They_take_the_special_training_path_not_the_features_vector(string moduleId)
    {
        // Neither concatenates its inputs into a Features vector; both need their own wiring.
        Assert.True(PipelineBuilder.NeedsSpecialPath(moduleId));
    }

    [Theory]
    [InlineData("algo.vision.imageClassification", "imagePathColumn")]
    [InlineData("algo.nlp.textClassification", "textColumn")]
    public void The_input_column_is_required_because_nothing_can_be_guessed_from_a_default(
        string moduleId, string parameterName)
    {
        var parameter = ModuleCatalog.Find(moduleId)!.Parameters.First(p => p.Name == parameterName);

        Assert.True(parameter.Required);
    }

    [Theory]
    [InlineData("algo.vision.imageClassification", "imagePathColumn")]
    [InlineData("algo.nlp.textClassification", "textColumn")]
    public void Training_without_a_label_says_so_before_anything_expensive_happens(
        string moduleId, string inputParameter)
    {
        var spec = _algorithms.Output<TrainerSpec>(
            TestContext.For(moduleId, [(inputParameter, "nilai")]));

        var context = TestContext.For("train.model", [("labelColumn", null)], spec, Labelled("nilai"));

        var error = Assert.Throws<InvalidOperationException>(() =>
            _training.ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains("label column", error.Message);
    }

    /// <summary>
    /// In a standard build the backbones are not compiled in. Running one has to name the module,
    /// say how to enable it, and be honest about what that costs — not fail with a missing-type
    /// exception the user cannot act on.
    /// </summary>
    [Theory]
    [InlineData("algo.vision.imageClassification", "imagePathColumn", "Image Classification")]
    [InlineData("algo.nlp.textClassification", "textColumn", "Text Classification")]
    public void Without_the_pack_the_failure_explains_how_to_enable_it(
        string moduleId, string inputParameter, string displayName)
    {
        if (VisionNlpProbe.IsBuiltIn)
        {
            // Built with the pack, so there is no refusal to check for.
            return;
        }

        var spec = _algorithms.Output<TrainerSpec>(
            TestContext.For(moduleId, [(inputParameter, "nilai")]));

        var context = TestContext.For("train.model", [("labelColumn", "kelas")], spec, Labelled("nilai"));

        var error = Assert.Throws<InvalidOperationException>(() =>
            _training.ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains(displayName, error.Message);
        Assert.Contains("EnableVisionNlp=true", error.Message);
        Assert.Contains("1.2 GB", error.Message);
    }

    [Fact]
    public void AutoML_asks_which_column_holds_the_images_rather_than_guessing()
    {
        var input = Labelled("berkas");

        var context = TestContext.For("train.autoML",
            [("task", "ImageClassification"), ("labelColumn", "kelas"), ("inputColumn", null)], input);

        var error = Assert.Throws<InvalidOperationException>(() =>
            new TrainingExecutor().ExecuteAsync(context).GetAwaiter().GetResult());

        Assert.Contains("image paths", error.Message);
    }
}

/// <summary>
/// Mirrors the compile-time switch so a test can tell which build it is running in without
/// reaching into internals.
/// </summary>
internal static class VisionNlpProbe
{
    public static bool IsBuiltIn =>
#if VISION_NLP
        true;
#else
        false;
#endif
}
