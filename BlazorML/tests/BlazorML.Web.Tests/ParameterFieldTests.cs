using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.Web.Components.Shared;
using Bunit;

namespace BlazorML.Web.Tests;

/// <summary>
/// The whole inspector is generated from parameter specs, so a spec kind that this component
/// does not render correctly means a module that cannot be configured at all — with no compile
/// error and no exception to notice. That is exactly how the dataset picker shipped broken.
/// </summary>
public class ParameterFieldTests : TestContext
{
    private static ParameterSpec Spec(string name, string label, ParameterKind kind,
        string? def = null, params (string Value, string Label)[] choices) => new()
    {
        Name = name,
        Label = label,
        Kind = kind,
        Default = def,
        Choices = choices.Select(c => new ChoiceOption(c.Value, c.Label)).ToList()
    };

    private IRenderedComponent<ParameterField> Render(ParameterSpec spec, string? value = null,
        List<string>? columns = null, List<ChoiceOption>? datasets = null) =>
        RenderComponent<ParameterField>(p => p
            .Add(x => x.Spec, spec)
            .Add(x => x.Value, value)
            .Add(x => x.Columns, columns ?? [])
            .Add(x => x.Datasets, datasets ?? []));

    /// <summary>
    /// Regression test for the bug the unit suite caught. A DatasetRef must render the workspace
    /// datasets; rendering an empty select left Import Dataset impossible to configure.
    /// </summary>
    [Fact]
    public void DatasetRef_lists_the_workspace_datasets()
    {
        var datasets = new List<ChoiceOption>
        {
            new("id-1", "Churn Pelanggan (900 baris)"),
            new("id-2", "Harga Rumah (400 baris)")
        };

        var component = Render(Spec("datasetId", "Dataset", ParameterKind.DatasetRef), datasets: datasets);
        var options = component.FindAll("select option");

        // The placeholder plus one option per dataset.
        Assert.Equal(3, options.Count);
        Assert.Contains(options, o => o.TextContent.Contains("Churn Pelanggan"));
        Assert.Contains(options, o => o.TextContent.Contains("Harga Rumah"));
        Assert.Equal("id-2", options[2].GetAttribute("value"));
    }

    [Fact]
    public void DatasetRef_with_nothing_imported_says_so_and_points_at_the_import_page()
    {
        // An empty dropdown with no explanation is the failure mode this replaces.
        var component = Render(Spec("datasetId", "Dataset", ParameterKind.DatasetRef));

        Assert.Contains("Belum ada dataset", component.Markup);
        Assert.Equal("/dataset", component.Find(".hint a").GetAttribute("href"));
    }

    [Fact]
    public void Choice_renders_one_option_per_declared_choice_and_marks_the_current_value()
    {
        var spec = Spec("mode", "Action", ParameterKind.Choice, "keep",
            ("keep", "Keep selected"), ("drop", "Drop selected"));

        var component = Render(spec, "drop");
        var select = component.Find("select");

        Assert.Equal(2, component.FindAll("option").Count);
        Assert.Equal("drop", select.GetAttribute("value"));
    }

    [Fact]
    public void Boolean_renders_a_checkbox_reflecting_the_stored_string()
    {
        var on = Render(Spec("aktif", "Aktif", ParameterKind.Boolean, "true"), "true");
        var off = Render(Spec("aktif", "Aktif", ParameterKind.Boolean, "true"), "false");

        Assert.True(on.Find("input[type=checkbox]").HasAttribute("checked"));
        Assert.False(off.Find("input[type=checkbox]").HasAttribute("checked"));
    }

    [Fact]
    public void Boolean_emits_the_string_true_or_false_not_a_boxed_bool()
    {
        // Parameters are stored as strings; emitting a boxed bool would round-trip as "True"
        // and never match the "true" the executors compare against.
        string? captured = null;

        var component = RenderComponent<ParameterField>(p => p
            .Add(x => x.Spec, Spec("aktif", "Aktif", ParameterKind.Boolean, "false"))
            .Add(x => x.Value, "false")
            .Add(x => x.ValueChanged, (string? v) => captured = v));

        component.Find("input[type=checkbox]").Change(true);

        Assert.Equal("true", captured);
    }

    [Fact]
    public void ColumnName_offers_the_known_columns_as_suggestions_without_forcing_them()
    {
        // A datalist, not a select: a graph saved before the dataset was chosen may name a
        // column that is not in the current list, and that value must not be silently dropped.
        var component = Render(Spec("labelColumn", "Label column", ParameterKind.ColumnName),
            "kolom_lama", columns: ["a", "b"]);

        Assert.Equal(2, component.FindAll("datalist option").Count);
        Assert.Equal("kolom_lama", component.Find("input").GetAttribute("value"));
    }

    [Fact]
    public void ColumnList_offers_clickable_chips_that_toggle_a_column_in_and_out()
    {
        string? captured = null;

        var component = RenderComponent<ParameterField>(p => p
            .Add(x => x.Spec, Spec("columns", "Columns", ParameterKind.ColumnList))
            .Add(x => x.Value, "usia")
            .Add(x => x.Columns, ["usia", "kota"])
            .Add(x => x.ValueChanged, (string? v) => captured = v));

        component.FindAll("button.tag")[1].Click();
        Assert.Equal("usia, kota", captured);

        component.FindAll("button.tag")[0].Click();
        Assert.Equal(string.Empty, captured);
    }

    [Theory]
    [InlineData(ParameterKind.Integer, "input[type=number]")]
    [InlineData(ParameterKind.Number, "input[type=number]")]
    [InlineData(ParameterKind.Secret, "input[type=password]")]
    [InlineData(ParameterKind.MultilineText, "textarea")]
    [InlineData(ParameterKind.Code, "textarea.code")]
    [InlineData(ParameterKind.Text, "input")]
    public void Each_kind_renders_the_control_it_needs(ParameterKind kind, string selector)
    {
        var component = Render(Spec("p", "Parameter", kind));

        Assert.NotNull(component.Find(selector));
    }

    /// <summary>
    /// Every kind in the enum must render something. A kind added to the catalog without a case
    /// here would fall through to the default and silently render the wrong control.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllKinds))]
    public void No_parameter_kind_renders_an_empty_field(ParameterKind kind)
    {
        var component = Render(Spec("p", "Parameter", kind, "true", ("true", "Ya"), ("false", "Tidak")));

        Assert.True(
            component.FindAll("input, select, textarea").Count > 0,
            $"{kind} rendered no input control at all.");
    }

    public static TheoryData<ParameterKind> AllKinds()
    {
        var data = new TheoryData<ParameterKind>();

        foreach (var kind in Enum.GetValues<ParameterKind>())
        {
            data.Add(kind);
        }

        return data;
    }

    [Fact]
    public void Help_text_is_shown_when_the_spec_carries_it()
    {
        var spec = new ParameterSpec
        {
            Name = "fraction",
            Label = "Rows in the first output",
            Kind = ParameterKind.Number,
            Help = "0.8 sends 80% to training."
        };

        Assert.Contains("0.8 sends 80% to training.", Render(spec).Markup);
    }

    [Fact]
    public void The_label_is_tied_to_its_control_so_clicking_it_focuses_the_field()
    {
        var component = Render(Spec("p", "Nama parameter", ParameterKind.Text));

        var forAttribute = component.Find("label").GetAttribute("for");
        Assert.False(string.IsNullOrEmpty(forAttribute));
        Assert.Equal(forAttribute, component.Find("input").Id);
    }
}
