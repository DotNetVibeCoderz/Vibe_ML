using System.Diagnostics;
using ScienceAppGen.Plugins;
using ScienceAppGen.Services;
using ScienceAppGen.Services.Conversion;
using Xunit;

namespace ScienceAppGen.Tests;

/// <summary>Dask, dask.distributed and dask-sql converting to Dask.Net, and the Dask.Net templates.</summary>
public class DaskSupportTests
{
    private static ModuleAnalysis Analyze(string source, string name = "m.py")
        => PythonImportAnalyzer.Analyze(new PythonSourceFile(name, source), new HashSet<string>());

    [Theory]
    [InlineData("dask")]
    [InlineData("distributed")]
    [InlineData("dask_sql")]
    public void Dask_packages_map_to_dasknet(string package)
    {
        var mapping = LibraryMap.Map(package);
        Assert.Same(LibraryMap.DaskNet, mapping.Target);
        Assert.Equal("Gravicode.Science.DaskNet", mapping.Target!.PackageId);
        Assert.NotNull(mapping.Note);
    }

    [Fact]
    public void Dask_submodules_bring_their_namespaces_and_nothing_more()
    {
        var frames = Analyze("import dask.dataframe as dd\ndf = dd.read_csv('a/*.csv')\n");
        var usings = LibraryMap.UsingsFor(frames, LibraryMap.MapAll([frames]));
        Assert.Contains("Gravicode.Science.GraviNum", usings);
        Assert.Contains("Gravicode.Science.DaskNet", usings);
        Assert.DoesNotContain("Gravicode.Science.DaskNet.Distributed", usings);
        Assert.DoesNotContain("Gravicode.Science.DaskNet.Sql", usings);

        var cluster = Analyze("from dask.distributed import Client, LocalCluster\nclient = Client(LocalCluster())\n");
        Assert.Contains("Gravicode.Science.DaskNet.Distributed", LibraryMap.UsingsFor(cluster, LibraryMap.MapAll([cluster])));

        var sql = Analyze("from dask_sql import Context\nc = Context()\n");
        Assert.Contains("Gravicode.Science.DaskNet.Sql", LibraryMap.UsingsFor(sql, LibraryMap.MapAll([sql])));
    }

    [Fact]
    public void A_module_that_never_imports_dask_still_gets_the_dasknet_namespace()
    {
        // report.py receives the lazy DaskFrame that pipeline.py built, like features.py and the DataFrame.
        var report = Analyze("import numpy as np\n", "report.py");
        var project = new[] { LibraryMap.Map("dask"), LibraryMap.Map("numpy") };

        Assert.Contains("Gravicode.Science.DaskNet", LibraryMap.UsingsFor(report, project));
        Assert.Contains(LibraryMap.DaskNet, LibraryMap.SharedTargets(project));
    }

    [Fact]
    public void Dask_pulls_in_graviframe_and_gravinum_at_one_version()
    {
        var packages = LibraryMap.PackagesFor([LibraryMap.Map("dask")]);
        var ids = packages.Select(p => p.PackageId).ToList();

        Assert.Contains("Gravicode.Science.DaskNet", ids);
        Assert.Contains("Gravicode.Science.GraviFrame", ids);
        Assert.Contains("Gravicode.Science.GraviNum", ids);
        // Dask.Net first shipped in 2.1.0; a converted project must not ask for an older science package.
        Assert.All(packages, p => Assert.Equal(LibraryMap.ScienceVersion, p.Version));
        Assert.True(Version.Parse(LibraryMap.ScienceVersion) >= new Version(2, 1, 0));
    }

    [Fact]
    public void Dataframe_calls_bring_the_curated_reference_and_real_signatures()
    {
        var text = ApiReferenceBuilder.Build(Analyze("""
            import pandas as pd
            import dask.dataframe as dd
            df = dd.read_csv("sales/*.csv")
            small = dd.from_pandas(pd.DataFrame({"a": [1]}), npartitions=1)
            out = df[df.qty > 0].merge(small, on="a").groupby("region").agg({"x": "sum"}).compute()
            """), LibraryMap.MapAll([Analyze("import pandas\nimport dask.dataframe\n")]));

        Assert.Contains("DaskNet - Dask for .NET", text);                   // curated
        Assert.Contains("Dask.Net signatures for the dask names this file uses", text);
        Assert.Contains("static DaskFrame ReadCsv(string pattern", text);    // reflected
        Assert.Contains("static DaskFrame FromFrame(DataFrame frame, int partitions = 4)", text);   // from_pandas
        Assert.Contains("DaskGroupBy GroupBy(", text);
        Assert.Contains("DaskFrame Agg(", text);
        Assert.Contains("DaskFrame Merge(", text);
        Assert.Contains("DataFrame Compute(", text);
        Assert.DoesNotContain("SqlContext.", text);                          // not imported, not described
    }

    [Fact]
    public void Bag_array_cluster_and_sql_names_resolve_to_their_types()
    {
        var module = Analyze("""
            import dask.bag as db
            import dask.array as da
            from dask.distributed import Client, LocalCluster
            from dask_sql import Context
            words = db.read_text("logs/*.txt").map(str.split).frequencies()
            x = da.from_array(arr, chunks=1000).mean(axis=0)
            client = Client(LocalCluster(n_workers=4))
            c = Context()
            c.create_table("t", df)
            r = c.sql("SELECT 1")
            """);
        var text = ApiReferenceBuilder.Build(module, LibraryMap.MapAll([module]));

        Assert.Contains("Bag.static Bag<string> ReadText(string pattern", text);
        Assert.Contains("Frequencies()", text);
        Assert.Contains("static DaskArray FromArray(NdArray array, params int[] chunkSize)", text);
        Assert.Contains("static Task<LocalCluster> StartAsync(int workers = 2", text);
        Assert.Contains("SqlContext.SqlContext Register(", text);
        Assert.Contains("DataFrame Execute(string query", text);
        Assert.Contains("[RemoteTask]", text);                               // the rule that a cluster enforces
    }

    [Fact]
    public void Invented_dask_namespaces_are_removed_and_real_ones_kept()
    {
        var code = """
            using Dask.Distributed;
            using Gravicode.Science.DaskNet.Sql;

            namespace X;
            """;

        var fixedCode = ConversionSupport.NormalizeUsings(code, ["Gravicode.Science.DaskNet", "Gravicode.Science.DaskNet.Distributed"]);

        Assert.DoesNotContain("using Dask.Distributed;", fixedCode);
        Assert.Contains("using Gravicode.Science.DaskNet.Sql;", fixedCode);
        Assert.Contains("using Gravicode.Science.DaskNet;", fixedCode);
        Assert.Contains("using Gravicode.Science.DaskNet.Distributed;", fixedCode);
    }

    [Fact]
    public void The_assistant_reference_lists_dasknet()
    {
        var plugin = new GravicodeReferencePlugin();
        Assert.Contains("DaskNet", plugin.GravicodeReference("all"));
        var reference = plugin.GravicodeReference("DaskNet");
        Assert.Contains("[RemoteTask]", reference);
        Assert.Contains("LocalCluster.StartAsync", reference);
    }

    [Theory]
    [InlineData("dask-pipeline")]
    [InlineData("dask-cluster")]
    public void Dask_templates_reference_dasknet(string id)
    {
        var template = TemplateService.Find(id);
        Assert.NotNull(template);
        Assert.Contains("DaskNet", template.Libraries);
        Assert.Contains(@"DaskNet\DaskNet.csproj", template.Files["$name$.csproj"]);
        Assert.Equal("#D9609E", template.LibraryColours[template.Libraries.ToList().IndexOf("DaskNet")]);
    }

    /// <summary>Every template id, for the build test.</summary>
    public static TheoryData<string> TemplateIds => new(TemplateService.All.Select(t => t.Id));

    /// <summary>
    /// Every template is generated next to the repository's libraries and built, so "every one of them
    /// builds and runs as written" is checked rather than claimed. Release, so it does not contend
    /// with Debug test builds of the libraries.
    /// </summary>
    [Theory]
    [MemberData(nameof(TemplateIds))]
    public void Every_template_builds(string id)
    {
        var template = TemplateService.Find(id)!;
        var root = Path.Combine(Path.GetTempPath(), "sag-template-" + Guid.NewGuid().ToString("N")[..8]);
        var name = "Probe" + new string(id.Where(char.IsLetterOrDigit).ToArray());
        var directory = Path.Combine(root, name);
        try
        {
            TemplateService.Create(template, directory, name);
            var project = Directory.GetFiles(directory, "*.csproj").SingleOrDefault();
            if (project is null) return;                                    // the notebook template has nothing to build

            var info = new ProcessStartInfo("dotnet", $"build \"{project}\" -c Release --nologo -v q")
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false,
            };
            using var process = Process.Start(info)!;
            var output = process.StandardOutput.ReadToEndAsync();
            var error = process.StandardError.ReadToEndAsync();
            process.WaitForExit();
            Assert.True(process.ExitCode == 0, $"{id} failed to build:\n{output.Result}\n{error.Result}");
        }
        finally
        {
            try { Directory.Delete(root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }
}
