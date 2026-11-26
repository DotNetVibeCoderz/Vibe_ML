using System.Reflection;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// One node of a task graph: what it depends on and how to compute it from those results.
/// </summary>
/// <remarks>
/// Every task has a local <see cref="Function"/>. A task built from a
/// <see cref="RemoteTaskAttribute"/> method also has a <see cref="Remote"/> form, and only those
/// can be shipped to a cluster worker. Thread and synchronous schedulers ignore the distinction.
/// </remarks>
public sealed class TaskSpec
{
    private TaskSpec(TaskKey key, IReadOnlyList<TaskKey> dependencies, Func<object?[], object?> function,
        RemoteCall? remote, string label)
    {
        Key = key;
        Dependencies = dependencies;
        Function = function;
        Remote = remote;
        Label = label;
    }

    /// <summary>This task's key.</summary>
    public TaskKey Key { get; }

    /// <summary>The tasks whose results this one reads, in the order <see cref="Function"/> receives them.</summary>
    public IReadOnlyList<TaskKey> Dependencies { get; }

    /// <summary>Computes the result from the dependency results, given in <see cref="Dependencies"/> order.</summary>
    public Func<object?[], object?> Function { get; }

    /// <summary>The shippable form, or <c>null</c> when the task wraps a local delegate.</summary>
    public RemoteCall? Remote { get; }

    /// <summary>A short operation name for progress output and graph pictures (<c>"read-csv"</c>).</summary>
    public string Label { get; }

    /// <summary>Whether a cluster worker can run this task.</summary>
    public bool IsRemotable => Remote is not null;

    /// <summary>A task that produces a value already in hand.</summary>
    public static TaskSpec Value(TaskKey key, object? value, string label = "value")
        => new(key, [], _ => value, new RemoteCall(Builtins.IdentityMethod, value), label);

    /// <summary>A task that runs a local delegate over its dependencies' results.</summary>
    public static TaskSpec Create(TaskKey key, IReadOnlyList<TaskKey> dependencies,
        Func<object?[], object?> function, string? label = null)
    {
        ArgumentNullException.ThrowIfNull(function);
        return new TaskSpec(key, dependencies.ToArray(), function, null, label ?? key.Name);
    }

    /// <summary>A task that calls a <see cref="RemoteTaskAttribute"/> method; arguments may be
    /// <see cref="TaskRef"/>, <see cref="TaskRefList"/> or nested <see cref="RemoteCall"/>.</summary>
    public static TaskSpec Call(TaskKey key, MethodInfo method, object?[] arguments, string? label = null)
        => FromRemote(key, new RemoteCall(method, arguments), label ?? method.Name);

    /// <summary>A task whose body is an existing <see cref="RemoteCall"/>.</summary>
    public static TaskSpec FromRemote(TaskKey key, RemoteCall call, string? label = null)
    {
        var dependencies = call.Dependencies();
        return new TaskSpec(key, dependencies, values =>
        {
            var lookup = new Dictionary<TaskKey, object?>(dependencies.Count);
            for (var i = 0; i < dependencies.Count; i++) lookup[dependencies[i]] = values[i];
            return call.Invoke(k => lookup[k]);
        }, call, label ?? call.Method.Name);
    }

    /// <summary>The same computation under a different key.</summary>
    internal TaskSpec WithKey(TaskKey key) => new(key, Dependencies, Function, Remote, Label);

    /// <summary>
    /// Folds a single-use dependency into this task, so the pair runs as one (Dask's
    /// <c>fuse</c>). The result reads the child's dependencies in place of the child.
    /// </summary>
    internal TaskSpec Absorb(TaskSpec child)
    {
        var position = IndexOf(Dependencies, child.Key);
        if (position < 0) throw new InvalidOperationException($"{child.Key} is not a dependency of {Key}.");

        var merged = new List<TaskKey>();
        foreach (var d in Dependencies)
        {
            if (d == child.Key) { foreach (var c in child.Dependencies) if (!merged.Contains(c)) merged.Add(c); }
            else if (!merged.Contains(d)) merged.Add(d);
        }

        var label = $"{child.Label}+{Label}";

        if (Remote is not null && child.Remote is not null)
        {
            var inlined = (RemoteCall)Inline(Remote, child.Key, child.Remote)!;
            var spec = FromRemote(Key, inlined, label);
            return spec;
        }

        var parentDeps = Dependencies;
        var childDeps = child.Dependencies;
        var parentFn = Function;
        var childFn = child.Function;
        var mergedArray = merged.ToArray();
        return new TaskSpec(Key, mergedArray, values =>
        {
            var childArgs = new object?[childDeps.Count];
            for (var i = 0; i < childArgs.Length; i++) childArgs[i] = values[IndexOf(mergedArray, childDeps[i])];
            var childValue = childFn(childArgs);

            var parentArgs = new object?[parentDeps.Count];
            for (var i = 0; i < parentArgs.Length; i++)
                parentArgs[i] = parentDeps[i] == child.Key ? childValue : values[IndexOf(mergedArray, parentDeps[i])];
            return parentFn(parentArgs);
        }, null, label);
    }

    private static object? Inline(object? argument, TaskKey target, RemoteCall replacement) => argument switch
    {
        TaskRef r when r.Key == target => replacement,
        TaskRefList list when list.Keys.Contains(target) =>
            throw new InvalidOperationException("A gathered dependency cannot be fused."),
        RemoteCall call => new RemoteCall(call.Method, call.Arguments.Select(a => Inline(a, target, replacement)).ToArray()),
        _ => argument,
    };

    internal bool ReadsAsList(TaskKey key) => Remote is not null && ReadsAsList(Remote, key);

    private static bool ReadsAsList(object? argument, TaskKey key) => argument switch
    {
        TaskRefList list => list.Keys.Contains(key),
        RemoteCall call => call.Arguments.Any(a => ReadsAsList(a, key)),
        _ => false,
    };

    private static int IndexOf(IReadOnlyList<TaskKey> keys, TaskKey key)
    {
        for (var i = 0; i < keys.Count; i++) if (keys[i] == key) return i;
        return -1;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Key} <- [{string.Join(", ", Dependencies)}]";
}

/// <summary>Remote-callable helpers every worker has.</summary>
public static class Builtins
{
    internal static readonly MethodInfo IdentityMethod = typeof(Builtins).GetMethod(nameof(Identity))!;
    internal static readonly MethodInfo ItemMethod = typeof(Builtins).GetMethod(nameof(Item))!;
    internal static readonly MethodInfo ListMethod = typeof(Builtins).GetMethod(nameof(List))!;

    /// <summary>Returns its argument: how a literal value becomes a task.</summary>
    [RemoteTask]
    public static object? Identity(object? value) => value;

    /// <summary>One element of an array-valued result (Dask's <c>getitem</c>).</summary>
    [RemoteTask]
    public static object? Item(object?[] values, int index) => values[index];

    /// <summary>Gathers several results into one array.</summary>
    [RemoteTask]
    public static object?[] List(object?[] values) => values;
}
