using ScienceAppGen.Services.Conversion;
using Xunit;

namespace ScienceAppGen.Tests;

/// <summary>A throwaway folder of files, deleted afterwards.</summary>
internal sealed class TempTree : IDisposable
{
    public TempTree() => Directory.CreateDirectory(Root);

    public string Root { get; } = Path.Combine(Path.GetTempPath(), "sag-tests-" + Guid.NewGuid().ToString("N")[..10]);

    public string Write(string relative, string content)
    {
        var path = Path.Combine(Root, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
        return path;
    }

    public void Dispose()
    {
        try { Directory.Delete(Root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }
}

public class PythonScannerTests
{
    [Fact]
    public void A_single_file_is_its_own_project()
    {
        using var tree = new TempTree();
        var path = tree.Write("signal_analysis.py", "import numpy as np\nprint(np.pi)\n");

        var project = PythonProjectScanner.Scan(path);

        Assert.Equal("SignalAnalysis", project.Name);
        Assert.Single(project.Files);
        Assert.Equal("signal_analysis", project.Files[0].ModuleName);
    }

    [Fact]
    public void A_folder_skips_environments_caches_and_packaging()
    {
        using var tree = new TempTree();
        tree.Write("main.py", "print(1)\n");
        tree.Write("pkg/__init__.py", "");                              // an empty marker: nothing to convert
        tree.Write("pkg/utils.py", "def f():\n    return 1\n");
        tree.Write(".venv/lib/site-packages/numpy/core.py", "x = 1\n");
        tree.Write("myenv/pyvenv.cfg", "home = /usr\n");                 // a venv by content, not by name
        tree.Write("myenv/lib/thing.py", "y = 2\n");
        tree.Write("__pycache__/main.cpython-312.py", "z = 3\n");
        tree.Write("setup.py", "from setuptools import setup\n");
        tree.Write("data/sales.csv", "a,b\n1,2\n");
        tree.Write("requirements.txt", "numpy\n");

        var project = PythonProjectScanner.Scan(tree.Root);

        Assert.Equal(["main.py", "pkg/utils.py"], project.Files.Select(f => f.RelativePath));
        Assert.Equal("pkg.utils", project.Files[1].ModuleName);
        Assert.Equal(["data/sales.csv"], project.DataFiles.Select(d => d.RelativePath));
        Assert.Equal("numpy\n", project.Requirements);
        Assert.Contains(project.Skipped, s => s.StartsWith("setup.py", StringComparison.Ordinal));
    }

    [Fact]
    public void A_notebook_becomes_its_code_cells_with_magics_commented_out()
    {
        const string notebook = """
            {"cells": [
              {"cell_type": "markdown", "source": ["# Title\n"]},
              {"cell_type": "code", "source": ["%matplotlib inline\n", "import numpy as np\n"]},
              {"cell_type": "code", "source": "!pip install scipy\nx = np.arange(3)"}
            ]}
            """;

        var python = PythonProjectScanner.NotebookToPython(notebook);

        Assert.DoesNotContain("# Title", python);
        Assert.Contains("# (notebook magic) %matplotlib inline", python);
        Assert.Contains("# (notebook magic) !pip install scipy", python);
        Assert.Contains("import numpy as np", python);
        Assert.Contains("# ---- cell 2", python);
    }

    [Theory]
    [InlineData("sales_forecast", "SalesForecast")]
    [InlineData("2024-report", "App2024Report")]
    [InlineData("my project!", "MyProject")]
    [InlineData("___", "Converted")]
    public void Project_names_are_valid_identifiers(string raw, string expected)
        => Assert.Equal(expected, PythonProjectScanner.ProjectNameFrom(raw));
}

public class PythonImportAnalyzerTests
{
    private static ModuleAnalysis Analyze(string source, params string[] localModules)
        => PythonImportAnalyzer.Analyze(new PythonSourceFile("m.py", source), localModules.ToHashSet(StringComparer.Ordinal));

    [Fact]
    public void Aliases_are_followed_to_fully_qualified_symbols()
    {
        var module = Analyze("""
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import signal, stats
            from scipy.optimize import curve_fit as fit

            x = np.linalg.solve(a, b)
            b, a = signal.butter(4, 0.1)
            p = stats.norm.pdf(x)
            plt.savefig("a.png")
            fit(f, x, y)
            """);

        Assert.Contains("numpy.linalg.solve", module.UsedSymbols);
        Assert.Contains("scipy.signal.butter", module.UsedSymbols);
        Assert.Contains("scipy.stats.norm.pdf", module.UsedSymbols);
        Assert.Contains("matplotlib.pyplot.savefig", module.UsedSymbols);
        Assert.Contains("scipy.optimize.curve_fit", module.UsedSymbols);
        Assert.Equal(["numpy", "matplotlib", "scipy"], module.ExternalPackages);
    }

    [Fact]
    public void Docstrings_and_comments_do_not_count_as_code()
    {
        var module = Analyze("""
            '''This module does not import pandas.
            import pandas as pd
            '''
            # import seaborn
            import numpy as np
            s = "np.fake_function"
            """);

        Assert.Equal(["numpy"], module.ExternalPackages);
        Assert.DoesNotContain("numpy.fake_function", module.UsedSymbols);
    }

    [Fact]
    public void Parenthesised_multi_line_imports_are_read()
    {
        var module = Analyze("""
            from sklearn.metrics import (
                mean_absolute_error,
                r2_score as r2,
            )
            """);

        Assert.Contains("sklearn.metrics.mean_absolute_error", module.UsedSymbols);
        Assert.Contains("sklearn.metrics.r2_score", module.UsedSymbols);
    }

    [Fact]
    public void Local_modules_are_told_apart_from_packages()
    {
        var module = Analyze("""
            from data_loader import load_sales
            import features
            from . import plots
            import pandas as pd
            """, "data_loader", "features", "plots");

        Assert.Equal(["data_loader", "features", "plots"], module.LocalDependencies.OrderBy(x => x));
        Assert.Equal(["pandas"], module.ExternalPackages);
    }

    [Fact]
    public void Definitions_and_the_main_guard_are_found()
    {
        var module = Analyze("""
            class DemandModel:
                def fit(self, x, y):
                    return self

            def load(path, sep=","):
                pass

            async def fetch(url):
                pass

            if __name__ == "__main__":
                load("x")
            """);

        Assert.Equal(["DemandModel", "load", "fetch"], module.Definitions.Select(d => d.Name));
        Assert.Equal(["class", "def", "def"], module.Definitions.Select(d => d.Kind));
        Assert.True(module.HasMainGuard);
    }
}

public class ConversionPlannerTests
{
    [Fact]
    public void Dependencies_come_first_and_the_entry_point_last()
    {
        using var tree = new TempTree();
        tree.Write("main.py", "from model import DemandModel\nfrom data_loader import load\n");
        tree.Write("model.py", "from features import build\nclass DemandModel:\n    pass\n");
        tree.Write("features.py", "import numpy as np\ndef build(df):\n    return df\n");
        tree.Write("data_loader.py", "import pandas as pd\ndef load():\n    return pd.read_csv('x')\n");

        var plan = ConversionPlanner.Plan(PythonProjectScanner.Scan(tree.Root));
        var order = plan.Modules.Select(m => m.PythonPath).ToList();

        Assert.Equal("main.py", order[^1]);
        Assert.True(order.IndexOf("features.py") < order.IndexOf("model.py"));
        Assert.Equal("Program.cs", plan.Modules[^1].CSharpPath);
        Assert.True(plan.Modules[^1].IsEntryPoint);

        // model.py defines class DemandModel, not Model, so the module keeps the plain name.
        Assert.Equal("Model", plan.Modules.Single(m => m.PythonPath == "model.py").ClassName);

        // Independent modules share a level, so they can go to the model in parallel.
        var levels = PythonConverter.Levels(plan);
        Assert.Contains(levels, level => level.Any(m => m.PythonPath == "features.py") && level.Any(m => m.PythonPath == "data_loader.py"));
        Assert.Equal("main.py", levels[^1].Single().PythonPath);
    }

    [Fact]
    public void A_module_named_after_its_own_class_takes_a_suffix_and_duplicates_are_qualified()
    {
        using var tree = new TempTree();
        tree.Write("app.py", "import model\n");
        tree.Write("model.py", "class Model:\n    pass\n");
        tree.Write("io_tools/utils.py", "def a():\n    pass\n");
        tree.Write("viz/utils.py", "def b():\n    pass\n");

        var plan = ConversionPlanner.Plan(PythonProjectScanner.Scan(tree.Root));
        var names = plan.Modules.ToDictionary(m => m.PythonPath, m => m.ClassName);

        Assert.Equal("ModelModule", names["model.py"]);
        Assert.Equal("Program", names["app.py"]);
        Assert.NotEqual(names["io_tools/utils.py"], names["viz/utils.py"]);
        Assert.Equal("IoTools/Utils.cs", plan.Modules.Single(m => m.PythonPath == "io_tools/utils.py").CSharpPath);
    }

    [Fact]
    public void Without_a_main_file_the_guarded_module_nothing_imports_is_the_entry_point()
    {
        using var tree = new TempTree();
        tree.Write("helpers.py", "def h():\n    pass\nif __name__ == '__main__':\n    h()\n");
        tree.Write("report.py", "import helpers\nif __name__ == '__main__':\n    helpers.h()\n");

        var plan = ConversionPlanner.Plan(PythonProjectScanner.Scan(tree.Root));

        Assert.Equal("report.py", plan.Modules.Single(m => m.IsEntryPoint).PythonPath);
    }
}

public class LibraryMapTests
{
    [Theory]
    [InlineData("numpy", "Gravicode.Science.GraviNum")]
    [InlineData("pandas", "Gravicode.Science.GraviFrame")]
    [InlineData("sklearn", "Gravicode.Science.GraviLearn")]
    [InlineData("scipy", "Gravicode.Science.SciNet")]
    [InlineData("matplotlib", "Gravicode.Matplotlib.Net")]
    [InlineData("seaborn", "Gravicode.Seaborn.Net")]
    [InlineData("plotly", "Gravicode.Plotly.Net")]
    [InlineData("networkx", "Gravicode.Science.GraviGraph")]
    [InlineData("pymc", "Gravicode.Science.GraviProb")]
    public void Python_packages_map_to_published_packages(string package, string packageId)
        => Assert.Equal(packageId, LibraryMap.Map(package).Target?.PackageId);

    [Fact]
    public void The_standard_library_needs_no_package_and_the_unknown_is_flagged()
    {
        Assert.Same(LibraryMap.Bcl, LibraryMap.Map("collections").Target);
        Assert.True(LibraryMap.Map("torch").IsUnsupported);
        Assert.True(LibraryMap.Map("some_obscure_package").IsUnsupported);
    }

    [Fact]
    public void Visual_net_pulls_in_the_science_packages_so_their_versions_line_up()
    {
        var packages = LibraryMap.PackagesFor([LibraryMap.Map("seaborn")]).Select(p => p.PackageId).ToList();

        Assert.Contains("Gravicode.Seaborn.Net", packages);
        Assert.Contains("Gravicode.Matplotlib.Net", packages);
        Assert.Contains("Gravicode.Science.GraviFrame", packages);
        Assert.All(LibraryMap.PackagesFor([LibraryMap.Map("seaborn")]).Where(p => p.Id.StartsWith("Gravi", StringComparison.Ordinal)),
            p => Assert.Equal(LibraryMap.ScienceVersion, p.Version));
    }

    [Fact]
    public void A_module_that_never_imports_pandas_still_gets_the_dataframe_namespace()
    {
        // features.py receives a DataFrame from data_loader.py; without the namespace the model
        // once declared its own class DataFrame, which shadowed the real one project-wide.
        var features = PythonImportAnalyzer.Analyze(new PythonSourceFile("features.py", "import numpy as np\n"), new HashSet<string>());
        var project = new[] { LibraryMap.Map("pandas"), LibraryMap.Map("numpy"), LibraryMap.Map("matplotlib") };

        var usings = LibraryMap.UsingsFor(features, project);

        Assert.Contains("Gravicode.Science.GraviFrame", usings);
        Assert.Contains("Gravicode.Science.GraviNum", usings);
        // Plotting namespaces stay with the modules that plot.
        Assert.DoesNotContain("Matplotlib.Net", usings);
    }
}
