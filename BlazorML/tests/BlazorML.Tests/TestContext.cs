using BlazorML.Core.Data;
using BlazorML.Core.Designer;
using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.ML.Execution;
using Microsoft.ML;

namespace BlazorML.Tests;

/// <summary>
/// Builds a <see cref="ModuleExecutionContext"/> for a module under test without standing up the
/// whole application. Executors that reach for a service will fail loudly rather than silently
/// receiving a null, which is what we want in a test.
/// </summary>
internal static class TestContext
{
    public static ModuleExecutionContext For(string moduleId,
        (string Name, string? Value)[]? parameters = null,
        params object?[] inputs) =>
        With(moduleId, EmptyServices.Instance, parameters, inputs);

    /// <summary>As <see cref="For"/>, with services the module under test needs to resolve.</summary>
    public static ModuleExecutionContext With(string moduleId, IServiceProvider services,
        (string Name, string? Value)[]? parameters = null,
        params object?[] inputs)
    {
        var module = ModuleCatalog.Find(moduleId)
            ?? throw new ArgumentException($"'{moduleId}' is not in the catalog.", nameof(moduleId));

        var node = new GraphNode
        {
            ModuleId = moduleId,
            Label = module.Name,
            Parameters = module.BuildDefaultParameters()
        };

        foreach (var (name, value) in parameters ?? [])
        {
            node.Parameters[name] = value;
        }

        return new ModuleExecutionContext
        {
            Ml = new MLContext(seed: 42),
            Node = node,
            Module = module,
            Services = services,
            Inputs = inputs,
            Ct = CancellationToken.None,
            Log = (_, _) => { }
        };
    }

    /// <summary>Runs an executor and returns the value on the given output port.</summary>
    public static T Output<T>(this IModuleExecutor executor, ModuleExecutionContext context, int port = 0)
        where T : class
    {
        var results = executor.ExecuteAsync(context).GetAwaiter().GetResult();

        Assert.True(port < results.Length,
            $"Module produced {results.Length} outputs, so port {port} does not exist.");

        return results[port] as T
            ?? throw new InvalidOperationException($"Port {port} did not carry a {typeof(T).Name}.");
    }

    /// <summary>A table with a categorical class column, for testing stratified behaviour.</summary>
    public static TabularData Classified(int perClass, params string[] classes)
    {
        var table = TabularData.WithColumns("fitur", "kelas");
        var random = new Random(7);

        foreach (var name in classes)
        {
            for (var i = 0; i < perClass; i++)
            {
                table.AddRow(random.Next(0, 100), name);
            }
        }

        return table;
    }

    private sealed class EmptyServices : IServiceProvider
    {
        public static readonly EmptyServices Instance = new();

        public object? GetService(Type serviceType) => null;
    }
}
