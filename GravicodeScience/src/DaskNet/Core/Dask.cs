using System.Reflection;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// The entry point: building delayed computations, choosing a scheduler and computing several
/// collections together.
/// </summary>
/// <example>
/// <code>
/// var a = Dask.Delayed(() => Load("a.csv"));
/// var b = Dask.Delayed(() => Load("b.csv"));
/// var total = Dask.Delayed((x, y) => x + y, a.Then(Count), b.Then(Count));
/// Console.WriteLine(total.Compute());
/// </code>
/// </example>
public static class Dask
{
    private static readonly AsyncLocal<IScheduler?> Scoped = new();
    private static IScheduler _default = new ThreadScheduler();

    /// <summary>The scheduler used when none is given or scoped. A <see cref="ThreadScheduler"/> by default.</summary>
    public static IScheduler DefaultScheduler
    {
        get => _default;
        set => _default = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>The scheduler in effect here: the innermost <see cref="UseScheduler"/>, else the default.</summary>
    public static IScheduler CurrentScheduler => Scoped.Value ?? _default;

    /// <summary>Whether <c>Compute</c> fuses linear task chains first. On by default.</summary>
    public static bool FuseGraphs { get; set; } = true;

    /// <summary>
    /// Uses <paramref name="scheduler"/> for every compute in this async flow until the returned
    /// handle is disposed (Dask's <c>with dask.config.set(scheduler=...)</c>).
    /// </summary>
    public static IDisposable UseScheduler(IScheduler scheduler)
    {
        ArgumentNullException.ThrowIfNull(scheduler);
        var previous = Scoped.Value;
        Scoped.Value = scheduler;
        return new Restore(() => Scoped.Value = previous);
    }

    private sealed class Restore(Action action) : IDisposable
    {
        private int _done;
        public void Dispose() { if (Interlocked.Exchange(ref _done, 1) == 0) action(); }
    }

    // ------------------------------------------------------------------ delayed

    /// <summary>A value as a delayed computation.</summary>
    public static Delayed<T> Value<T>(T value, string name = "value")
    {
        var key = new TaskKey(TaskKey.NewName(name));
        return new Delayed<T>(new GraphLayer(key.Name, [TaskSpec.Value(key, value, name)]), key);
    }

    /// <summary>Defers a function call.</summary>
    public static Delayed<T> Delayed<T>(Func<T> function, string? name = null)
        => Build<T>(function, [], name);

    /// <summary>Defers a function call over one delayed argument.</summary>
    public static Delayed<TResult> Delayed<T1, TResult>(Func<T1, TResult> function, Delayed<T1> a, string? name = null)
        => Build<TResult>(function, [a], name);

    /// <summary>Defers a function call over two delayed arguments.</summary>
    public static Delayed<TResult> Delayed<T1, T2, TResult>(Func<T1, T2, TResult> function,
        Delayed<T1> a, Delayed<T2> b, string? name = null)
        => Build<TResult>(function, [a, b], name);

    /// <summary>Defers a function call over three delayed arguments.</summary>
    public static Delayed<TResult> Delayed<T1, T2, T3, TResult>(Func<T1, T2, T3, TResult> function,
        Delayed<T1> a, Delayed<T2> b, Delayed<T3> c, string? name = null)
        => Build<TResult>(function, [a, b, c], name);

    /// <summary>Defers a function call over four delayed arguments.</summary>
    public static Delayed<TResult> Delayed<T1, T2, T3, T4, TResult>(Func<T1, T2, T3, T4, TResult> function,
        Delayed<T1> a, Delayed<T2> b, Delayed<T3> c, Delayed<T4> d, string? name = null)
        => Build<TResult>(function, [a, b, c, d], name);

    /// <summary>Gathers many delayed values into one delayed array.</summary>
    public static Delayed<T[]> All<T>(IEnumerable<Delayed<T>> items, string name = "gather")
    {
        var list = items.ToArray();
        var key = new TaskKey(TaskKey.NewName(name));
        var spec = TaskSpec.FromRemote(key,
            new RemoteCall(Builtins.ListMethod, new TaskRefList(list.Select(d => d.Key).ToArray())), name);
        var typed = TaskSpec.Create(key, spec.Dependencies, values => values.Select(v => (T)v!).ToArray(), name);
        return new Delayed<T[]>(new GraphLayer(key.Name, [typed], list.Select(d => d.Layer)), key);
    }

    private static Delayed<T> Build<T>(Delegate function, IDaskCollection[] inputs, string? name)
    {
        ArgumentNullException.ThrowIfNull(function);
        var method = function.Method;
        var label = name ?? (IsRemotable(function) ? method.Name : "delayed");
        var key = new TaskKey(TaskKey.NewName(label));
        var deps = inputs.Select(i => i.OutputKeys[0]).ToArray();

        // A method group naming a [RemoteTask] static method keeps its shippable form; a lambda's
        // closure cannot cross a process boundary, so it stays local.
        var spec = IsRemotable(function)
            ? TaskSpec.Call(key, method, deps.Select(d => (object?)new TaskRef(d)).ToArray(), label)
            : TaskSpec.Create(key, deps, values =>
            {
                try { return function.DynamicInvoke(values); }
                catch (TargetInvocationException e) when (e.InnerException is not null)
                {
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                    throw;
                }
            }, label);

        return new Delayed<T>(new GraphLayer(key.Name, [spec], inputs.Select(i => i.Layer)), key);
    }

    internal static bool IsRemotable(Delegate function)
        => function.Target is null && function.Method.IsStatic && function.Method.IsPublic
           && function.Method.GetCustomAttribute<RemoteTaskAttribute>() is not null;

    // ------------------------------------------------------------------ compute

    /// <summary>
    /// The graph <c>Compute</c> would run for these collections: merged, culled to their outputs
    /// and (if <see cref="FuseGraphs"/>) fused.
    /// </summary>
    public static TaskGraph Graph(params IDaskCollection[] collections)
    {
        var outputs = collections.SelectMany(c => c.OutputKeys).ToArray();
        var graph = TaskGraph.FromCollections(collections).Cull(outputs);
        return FuseGraphs ? graph.Fuse(outputs) : graph;
    }

    /// <summary>Computes two delayed values in one graph, so shared ancestors run once.</summary>
    public static (T1, T2) Compute<T1, T2>(Delayed<T1> a, Delayed<T2> b, IScheduler? scheduler = null)
    {
        var results = Run([a, b], scheduler, CancellationToken.None);
        return ((T1)results[a.Key]!, (T2)results[b.Key]!);
    }

    /// <summary>Computes three delayed values in one graph, so shared ancestors run once.</summary>
    public static (T1, T2, T3) Compute<T1, T2, T3>(Delayed<T1> a, Delayed<T2> b, Delayed<T3> c, IScheduler? scheduler = null)
    {
        var results = Run([a, b, c], scheduler, CancellationToken.None);
        return ((T1)results[a.Key]!, (T2)results[b.Key]!, (T3)results[c.Key]!);
    }

    /// <summary>Computes several delayed values of one type in one graph.</summary>
    public static T[] Compute<T>(IEnumerable<Delayed<T>> items, IScheduler? scheduler = null)
    {
        var list = items.ToArray();
        var results = Run(list, scheduler, CancellationToken.None);
        return list.Select(d => (T)results[d.Key]!).ToArray();
    }

    /// <summary>Runs the graph of several collections and returns every output key's result.</summary>
    public static IReadOnlyDictionary<TaskKey, object?> Run(IReadOnlyList<IDaskCollection> collections,
        IScheduler? scheduler, CancellationToken cancellationToken)
        => RunAsync(collections, scheduler, cancellationToken).GetAwaiter().GetResult();

    /// <summary>Runs the graph of several collections and returns every output key's result.</summary>
    public static Task<IReadOnlyDictionary<TaskKey, object?>> RunAsync(IReadOnlyList<IDaskCollection> collections,
        IScheduler? scheduler, CancellationToken cancellationToken)
    {
        var outputs = collections.SelectMany(c => c.OutputKeys).Distinct().ToArray();
        var graph = Graph([.. collections]);
        return (scheduler ?? CurrentScheduler).ExecuteAsync(graph, outputs, cancellationToken);
    }
}

/// <summary>A lazily computed value of type <typeparamref name="T"/> (Dask's <c>Delayed</c>).</summary>
public sealed class Delayed<T> : IDaskCollection
{
    internal Delayed(GraphLayer layer, TaskKey key)
    {
        Layer = layer;
        Key = key;
    }

    /// <inheritdoc />
    public GraphLayer Layer { get; }

    /// <summary>The task that produces the value.</summary>
    public TaskKey Key { get; }

    /// <inheritdoc />
    public IReadOnlyList<TaskKey> OutputKeys => [Key];

    /// <summary>Applies a function to the value, lazily.</summary>
    public Delayed<TResult> Then<TResult>(Func<T, TResult> function, string? name = null)
        => Dask.Delayed(function, this, name);

    /// <summary>Computes the value.</summary>
    public T Compute(IScheduler? scheduler = null, CancellationToken cancellationToken = default)
        => (T)Dask.Run([this], scheduler, cancellationToken)[Key]!;

    /// <summary>Computes the value asynchronously.</summary>
    public async Task<T> ComputeAsync(IScheduler? scheduler = null, CancellationToken cancellationToken = default)
        => (T)(await Dask.RunAsync([this], scheduler, cancellationToken).ConfigureAwait(false))[Key]!;

    /// <summary>Computes now and returns a delayed value that holds the result, so later graphs
    /// built on it do not repeat the work.</summary>
    public Delayed<T> Persist(IScheduler? scheduler = null) => Dask.Value(Compute(scheduler), "persisted");

    /// <summary>The graph that would run, as Graphviz DOT.</summary>
    public string Visualize(bool highLevel = false) => Dask.Graph(this).ToDot(highLevel);

    /// <inheritdoc />
    public override string ToString() => $"Delayed<{typeof(T).Name}>({Key})";
}
