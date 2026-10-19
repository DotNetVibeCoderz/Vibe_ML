using ScienceAppGen.Services.Conversion;
using Xunit;

namespace ScienceAppGen.Tests;

public class CodeExtractionTests
{
    [Fact]
    public void The_longest_fenced_block_is_the_file()
    {
        var reply = """
            Usage: ```csharp
            Program.Run();
            ```
            Here is the file:
            ```csharp
            namespace X;
            public static class Program { public static void Main() { } }
            ```
            Note: nothing else.
            """;

        var code = ConversionSupport.ExtractCode(reply)!;

        Assert.StartsWith("namespace X;", code);
        Assert.Contains("Main()", code);
    }

    [Fact]
    public void A_reply_cut_off_inside_the_fence_still_yields_the_code()
    {
        var code = ConversionSupport.ExtractCode("```csharp\nusing System;\nnamespace X;\npublic class A {");
        Assert.NotNull(code);
        Assert.StartsWith("using System;", code);
    }

    [Fact]
    public void Prose_without_code_yields_nothing()
        => Assert.Null(ConversionSupport.ExtractCode("I cannot convert this file because it is empty."));
}

public class BuildErrorParserTests
{
    [Fact]
    public void Errors_are_project_relative_deduplicated_and_free_of_the_project_suffix()
    {
        var root = Path.Combine(Path.GetTempPath(), "proj");
        var file = Path.Combine(root, "Model.cs");
        // MSBuild prints every error twice, with \r\n on Windows and a [project] suffix.
        var output = $"{file}(12,5): error CS0117: 'DataFrame' does not contain a definition for 'Foo' [{root}\\p.csproj]\r\n"
                     + $"{file}(12,5): error CS0117: 'DataFrame' does not contain a definition for 'Foo' [{root}\\p.csproj]\r\n"
                     + $"{file}(3,1): warning CS8618: whatever [{root}\\p.csproj]\r\n";

        var errors = ConversionSupport.ParseBuildErrors(output, root);

        var error = Assert.Single(errors);
        Assert.Equal("Model.cs", error.File);
        Assert.Equal(12, error.Line);
        Assert.Equal("CS0117", error.Code);
        Assert.Equal("'DataFrame' does not contain a definition for 'Foo'", error.Message);
    }

    [Fact]
    public void A_runtime_exception_is_pinned_to_the_throw_site_and_the_caller()
    {
        var root = Path.Combine(Path.GetTempPath(), "proj");
        var output = "rows: 1095\r\n"
                     + "Unhandled exception. System.Collections.Generic.KeyNotFoundException: No column named 'mean'. Available: store, sales.\r\n"
                     + "   at Gravicode.Science.GraviFrame.DataFrame.get_Item(String name)\r\n"
                     + $"   at SalesForecast.DataLoader.MeanOf(DataFrame df) in {Path.Combine(root, "DataLoader.cs")}:line 31\r\n"
                     + $"   at SalesForecast.Program.Main(String[] args) in {Path.Combine(root, "Program.cs")}:line 28\r\n";

        var errors = ConversionSupport.ParseRuntimeFailure(output, root, ["Program.cs", "DataLoader.cs"]);

        Assert.Equal(2, errors.Count);
        Assert.Equal(("DataLoader.cs", 31, "RUNTIME"), (errors[0].File, errors[0].Line, errors[0].Code));
        Assert.Contains("No column named 'mean'", errors[0].Message);
        Assert.Equal(("Program.cs", 28), (errors[1].File, errors[1].Line));
    }

    [Fact]
    public void A_failure_that_is_not_an_exception_in_our_files_is_not_a_repair_job()
    {
        Assert.Empty(ConversionSupport.ParseRuntimeFailure("Could not find data/sales.csv\n", "C:\\p", ["Program.cs"]));
        Assert.Empty(ConversionSupport.ParseRuntimeFailure(
            "Unhandled exception. System.OutOfMemoryException: x\n   at System.Array.Resize()\n", "C:\\p", ["Program.cs"]));
    }

    [Fact]
    public void A_restore_failure_is_reported_even_though_it_has_no_file()
    {
        var output = "C:\\x\\p.csproj : error NU1101: Unable to find package Gravicode.Nope. [C:\\x\\p.csproj]\n";
        var error = Assert.Single(ConversionSupport.ParseBuildErrors(output, "C:\\x"));
        Assert.Equal("NU1101", error.Code);
        Assert.Equal("", error.File);
    }
}

public class UsingNormalizationTests
{
    [Fact]
    public void Missing_usings_are_added_and_invented_ones_removed()
    {
        var code = """
            using System;
            using Visual.Net.Matplotlib;
            using Stats;
            using Gravicode.Science.SciNet;

            namespace X;
            """;

        var fixedCode = ConversionSupport.NormalizeUsings(code, ["Gravicode.Science.SciNet", "Matplotlib.Net", "Visual.Net.Core.Colors"]);

        Assert.Contains("using System;", fixedCode);
        Assert.DoesNotContain("Visual.Net.Matplotlib", fixedCode);   // no such namespace
        Assert.DoesNotContain("using Stats;", fixedCode);            // a type, not a namespace
        Assert.Contains("using Matplotlib.Net;", fixedCode);
        Assert.Contains("using Visual.Net.Core.Colors;", fixedCode);
        Assert.Single(fixedCode.Split('\n'), l => l == "using Gravicode.Science.SciNet;");
        Assert.True(fixedCode.IndexOf("using Matplotlib.Net;", StringComparison.Ordinal) < fixedCode.IndexOf("namespace X;", StringComparison.Ordinal));
    }

    [Fact]
    public void Unused_library_usings_are_pruned_but_used_and_system_ones_stay()
    {
        var code = """
            using System;
            using Gravicode.Science.GraviNum;
            using Gravicode.Science.GraviLearn.Trees;
            using Gravicode.Science.GraviLearn.Clustering;
            using Matplotlib.Net;

            namespace X;
            public static class P { public static RandomForestRegressor M() => new(10); public static Figure F() => new(4, 3); }
            """;

        var pruned = ConversionSupport.PruneUnusedUsings(code, ["Gravicode.Science.GraviNum"]);

        Assert.Contains("using System;", pruned);
        Assert.Contains("using Gravicode.Science.GraviNum;", pruned);            // kept: extension methods live there
        Assert.Contains("using Gravicode.Science.GraviLearn.Trees;", pruned);     // RandomForestRegressor
        Assert.Contains("using Matplotlib.Net;", pruned);                         // Figure
        Assert.DoesNotContain("Clustering", pruned);
    }

    [Fact]
    public void A_stub_of_a_library_type_is_caught_even_though_it_compiles()
    {
        var code = """
            namespace X;
            public static class Features { }
            public class DataFrame { public int RowCount; }
            public record Bounds(double Lo, double Hi);
            """;

        var shadows = ConversionSupport.FindShadowedLibraryTypes("Features.cs", code,
            ["Gravicode.Science.GraviFrame"], new HashSet<string> { "Features" });

        var shadow = Assert.Single(shadows);   // Bounds is SciNet's, and SciNet is not imported here
        Assert.Equal("CONV001", shadow.Code);
        Assert.Equal(3, shadow.Line);
        Assert.Contains("Gravicode.Science.GraviFrame.DataFrame", shadow.Message);
    }

    [Fact]
    public void A_python_class_the_project_owns_may_share_a_library_name()
    {
        var shadows = ConversionSupport.FindShadowedLibraryTypes("Model.cs", "public class Pipeline { }",
            ["Gravicode.Science.GraviLearn"], new HashSet<string> { "Pipeline" });
        Assert.Empty(shadows);
    }

    [Fact]
    public void The_public_surface_keeps_signatures_and_drops_bodies()
    {
        var surface = ConversionSupport.PublicSurface("""
            public static class Features
            {
                public static readonly string[] Names = ["a"];
                public static (NdArray X, NdArray Y) Build(DataFrame df) => (null!, null!);
                public static double Share(DataFrame df)
                {
                    return 0;
                }
                private static int Hidden() => 1;
            }
            """);

        Assert.Contains("public static class Features", surface);
        Assert.Contains("public static readonly string[] Names", surface);
        Assert.Contains("public static (NdArray X, NdArray Y) Build(DataFrame df)", surface);
        Assert.Contains("public static double Share(DataFrame df)", surface);
        Assert.DoesNotContain("Hidden", surface);
        Assert.DoesNotContain("return", surface);
    }
}

public class ApiReferenceTests
{
    private static string Reference(string source)
    {
        var module = PythonImportAnalyzer.Analyze(new PythonSourceFile("m.py", source), new HashSet<string>());
        return ApiReferenceBuilder.Build(module, LibraryMap.MapAll([module]));
    }

    [Fact]
    public void Scipy_calls_bring_their_real_signatures_and_parameter_types()
    {
        var text = Reference("from scipy import signal\nb, a = signal.butter(4, [0.5, 15], btype='bandpass', fs=250)\n");

        Assert.Contains("static FilterBA Butter(int n, CriticalFrequencies wn, FilterType btype = FilterType.Lowpass", text);
        Assert.Contains("ButterSos(", text);                           // the variant scipy selects with output="sos"
        Assert.Contains("enum FilterType: Lowpass, Highpass, Bandpass, Bandstop", text);
        Assert.Contains("CriticalFrequencies: implicit from double", text);
        Assert.DoesNotContain("<Clone>$", text);                       // record plumbing is noise
    }

    [Fact]
    public void A_call_chain_is_followed_through_the_returned_type()
    {
        var text = Reference("from scipy import stats\np = stats.norm.pdf(1.0)\n");
        Assert.Contains("static NormalDistribution Norm(double loc = 0, double scale = 1)", text);
        Assert.Contains("double Pdf(double x)", text);
    }

    [Fact]
    public void Enum_defaults_print_as_names_not_numbers()
    {
        var text = Reference("from scipy import signal\nf, p = signal.welch(x)\n");
        Assert.Contains("DetrendType? detrend = DetrendType.Constant", text);
    }

    [Fact]
    public void Plotting_gets_the_overview_and_exactly_the_members_the_file_calls()
    {
        var text = Reference("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.set_xlabel('x')\nax.scatter(x, y)\nfig.savefig('a.png')\n");

        Assert.Contains("Axes.Axes SetXLabel(string label)", text);
        Assert.Contains("Axes.ScatterArtist Scatter(double[] x, double[] y", text);
        Assert.Contains("Figure.void Save(string path", text);
        Assert.DoesNotContain("Axes.ImShow", text);                    // not called, not sent
        Assert.Contains("savefig -> Save", text);
    }
}
