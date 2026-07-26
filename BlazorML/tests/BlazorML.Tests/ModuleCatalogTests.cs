using BlazorML.Core.Domain;
using BlazorML.Core.Modules;

namespace BlazorML.Tests;

/// <summary>
/// The catalog drives the palette, the canvas, the inspector and the run engine from one
/// declaration. These are the invariants that make that safe — a malformed entry would surface
/// as a broken screen rather than a compile error, so it is checked here instead.
/// </summary>
public class ModuleCatalogTests
{
    public static TheoryData<string> AllModuleIds()
    {
        var data = new TheoryData<string>();

        foreach (var module in ModuleCatalog.All)
        {
            data.Add(module.Id);
        }

        return data;
    }

    [Fact]
    public void Every_module_id_is_unique()
    {
        var duplicates = ModuleCatalog.All
            .GroupBy(m => m.Id, StringComparer.OrdinalIgnoreCase)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToList();

        Assert.Empty(duplicates);
    }

    [Theory]
    [MemberData(nameof(AllModuleIds))]
    public void Module_is_well_formed(string moduleId)
    {
        var module = ModuleCatalog.Find(moduleId)!;

        Assert.False(string.IsNullOrWhiteSpace(module.Name), $"{moduleId} has no name.");
        Assert.False(string.IsNullOrWhiteSpace(module.Description), $"{moduleId} has no description.");

        // The description is shown as the palette subtitle and the inspector blurb, so it has to
        // be a sentence rather than a restatement of the name.
        Assert.EndsWith(".", module.Description);
        Assert.NotEqual(module.Name, module.Description);

        var parameterNames = module.Parameters.Select(p => p.Name).ToList();
        Assert.Equal(parameterNames.Count, parameterNames.Distinct(StringComparer.Ordinal).Count());

        foreach (var parameter in module.Parameters)
        {
            Assert.False(string.IsNullOrWhiteSpace(parameter.Label),
                $"{moduleId}.{parameter.Name} has no label.");

            if (parameter.Kind == ParameterKind.Choice)
            {
                Assert.NotEmpty(parameter.Choices);

                // A choice must default to one of its own options, or the select renders blank.
                Assert.Contains(parameter.Choices, c => c.Value == parameter.Default);
            }

            // A conditional field must point at a parameter that exists on the same module.
            if (parameter.VisibleWhen is not null)
            {
                Assert.Contains(parameter.VisibleWhen, parameterNames);
            }
        }
    }

    [Theory]
    [MemberData(nameof(AllModuleIds))]
    public void Default_parameters_cover_every_declared_parameter(string moduleId)
    {
        var module = ModuleCatalog.Find(moduleId)!;
        var defaults = module.BuildDefaultParameters();

        Assert.Equal(module.Parameters.Count, defaults.Count);
        Assert.All(module.Parameters, p => Assert.True(defaults.ContainsKey(p.Name)));
    }

    /// <summary>
    /// A required parameter with a default contradicts itself — the default satisfies the
    /// requirement, so the field would never actually be reported as missing.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllModuleIds))]
    public void A_required_parameter_carries_no_default(string moduleId)
    {
        var module = ModuleCatalog.Find(moduleId)!;

        foreach (var parameter in module.Parameters.Where(p => p.Required))
        {
            Assert.True(string.IsNullOrEmpty(parameter.Default),
                $"{moduleId}.{parameter.Name} is required but defaults to '{parameter.Default}'.");
        }
    }

    /// <summary>
    /// Requiredness used to be inferred from "has no default", which marked genuinely optional
    /// fields — a positive-class override, a rename target — as blocking. Validation that
    /// reports things the user does not have to fill in is validation people learn to ignore.
    /// </summary>
    [Theory]
    [InlineData("score.evaluate", "positiveClass")]
    [InlineData("tf.editMetadata", "newName")]
    [InlineData("tf.splitData", "stratifyColumn")]
    [InlineData("tf.cleanMissing", "columns")]
    [InlineData("data.web", "headers")]
    [InlineData("out.registerModel", "description")]
    [InlineData("train.model", "featureColumns")]
    public void Optional_fields_are_not_marked_required(string moduleId, string parameterName)
    {
        var parameter = ModuleCatalog.Find(moduleId)!.Parameters.First(p => p.Name == parameterName);

        Assert.False(parameter.Required);
    }

    [Theory]
    [InlineData("data.dataset", "datasetId")]
    [InlineData("train.model", "labelColumn")]
    [InlineData("tf.join", "leftKey")]
    [InlineData("out.saveDataset", "name")]
    public void Genuinely_required_fields_are_marked_required(string moduleId, string parameterName)
    {
        var parameter = ModuleCatalog.Find(moduleId)!.Parameters.First(p => p.Name == parameterName);

        Assert.True(parameter.Required);
    }

    [Fact]
    public void Every_algorithm_declares_the_task_it_solves()
    {
        var untyped = ModuleCatalog.ByCategory(ModuleCategory.Algorithm)
            .Where(m => m.Task == MlTask.None)
            .Select(m => m.Id);

        Assert.Empty(untyped);
    }

    [Fact]
    public void Every_algorithm_emits_exactly_one_untrained_model()
    {
        foreach (var module in ModuleCatalog.ByCategory(ModuleCategory.Algorithm))
        {
            Assert.Single(module.Outputs);
            Assert.Equal(PortType.UntrainedModel, module.Outputs[0].Type);
            Assert.Empty(module.Inputs);
        }
    }

    [Fact]
    public void Every_category_maps_to_a_pen_and_a_label()
    {
        foreach (var category in Enum.GetValues<ModuleCategory>())
        {
            Assert.False(string.IsNullOrWhiteSpace(ModuleCatalog.CategoryPen(category)));
            Assert.False(string.IsNullOrWhiteSpace(ModuleCatalog.CategoryLabel(category)));
        }
    }

    [Fact]
    public void Every_supervised_task_family_offers_more_than_one_algorithm()
    {
        // The spec asks for a choice of algorithm per task; a family of one is not a choice.
        foreach (var task in (MlTask[])
                 [MlTask.BinaryClassification, MlTask.MulticlassClassification, MlTask.Regression])
        {
            Assert.True(ModuleCatalog.TrainersFor(task).Count() > 1,
                $"{task} offers fewer than two algorithms.");
        }
    }

    [Fact]
    public void Search_finds_modules_by_name_description_and_indonesian_keyword()
    {
        Assert.Contains(ModuleCatalog.Search("Split Data"), m => m.Id == "tf.splitData");
        Assert.Contains(ModuleCatalog.Search("regresi"), m => m.Task == MlTask.Regression);
        Assert.Contains(ModuleCatalog.Search("rekomendasi"), m => m.Task == MlTask.Recommendation);
        Assert.Contains(ModuleCatalog.Search("anomali"), m => m.Task == MlTask.AnomalyDetection);
    }

    [Fact]
    public void Search_with_no_term_returns_the_whole_catalog()
    {
        Assert.Equal(ModuleCatalog.All.Count, ModuleCatalog.Search(null).Count());
        Assert.Equal(ModuleCatalog.All.Count, ModuleCatalog.Search("   ").Count());
    }

    [Fact]
    public void Find_is_case_insensitive_and_null_safe()
    {
        Assert.NotNull(ModuleCatalog.Find("TF.SPLITDATA"));
        Assert.Null(ModuleCatalog.Find("tidak.ada"));
        Assert.Null(ModuleCatalog.Find(null));
    }

    [Fact]
    public void Catalog_covers_every_capability_area_the_specification_names()
    {
        Assert.NotEmpty(ModuleCatalog.ByCategory(ModuleCategory.DataInput));
        Assert.NotEmpty(ModuleCatalog.ByCategory(ModuleCategory.DataTransform));
        Assert.NotEmpty(ModuleCatalog.ByCategory(ModuleCategory.LlmAction));
        Assert.NotEmpty(ModuleCatalog.ByCategory(ModuleCategory.Training));
        Assert.NotEmpty(ModuleCatalog.ByCategory(ModuleCategory.Evaluation));
        Assert.NotEmpty(ModuleCatalog.ByCategory(ModuleCategory.Output));

        // All four scripting languages the specification lists.
        Assert.Equal(4, ModuleCatalog.ByCategory(ModuleCategory.Script).Count());

        foreach (var task in (MlTask[])
                 [MlTask.BinaryClassification, MlTask.MulticlassClassification, MlTask.Regression,
                  MlTask.Clustering, MlTask.AnomalyDetection, MlTask.Recommendation])
        {
            Assert.NotEmpty(ModuleCatalog.TrainersFor(task));
        }
    }
}
