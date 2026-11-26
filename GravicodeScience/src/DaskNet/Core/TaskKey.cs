using System.Reflection;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// The name of one task in a graph: a layer name plus, for partitioned collections, the index of
/// the partition or block. Dask's <c>('read-csv-3f2a', 4)</c>.
/// </summary>
/// <param name="Name">The layer the task belongs to. Unique per collection operation.</param>
/// <param name="Index">Partition or flat block index, or -1 for a single-task layer.</param>
public readonly record struct TaskKey(string Name, int Index = -1)
{
    /// <inheritdoc />
    public override string ToString() => Index < 0 ? Name : $"('{Name}', {Index})";

    private static long _counter;

    /// <summary>
    /// A layer name that no other layer in this process shares: <paramref name="prefix"/>
    /// followed by a short counter.
    /// </summary>
    /// <remarks>
    /// Dask hashes a task's inputs to name it, so that two identical expressions share work. That
    /// requires hashing arbitrary .NET objects and delegates, which cannot be done reliably - a
    /// closure's captured state is invisible to a hash. A counter never merges two tasks that
    /// differ, which is the failure that matters; the cost is that identical expressions built
    /// twice are computed twice.
    /// </remarks>
    public static string NewName(string prefix) => $"{prefix}-{Interlocked.Increment(ref _counter):x}";
}

/// <summary>A placeholder argument: "the result of this task", resolved when the task runs.</summary>
/// <param name="Key">The task whose result is substituted.</param>
public sealed record TaskRef(TaskKey Key);

/// <summary>
/// A placeholder argument for several task results at once, delivered as one array typed to the
/// parameter it fills (a <c>DataFrame[]</c> parameter receives a <c>DataFrame[]</c>).
/// </summary>
/// <param name="Keys">The tasks whose results are gathered, in this order.</param>
public sealed record TaskRefList(IReadOnlyList<TaskKey> Keys);

/// <summary>
/// Marks a public static method that a cluster worker may execute on request.
/// </summary>
/// <remarks>
/// A worker that ran any method a client named would be a remote code execution service. This
/// attribute is the allow-list: a worker refuses a call to anything that does not carry it, after
/// authentication and in addition to it. It is also how a task becomes runnable on a cluster at
/// all - a lambda's closure cannot be sent to another process, a named static method can.
/// </remarks>
[AttributeUsage(AttributeTargets.Method, Inherited = false)]
public sealed class RemoteTaskAttribute : Attribute;

/// <summary>
/// A call to a <see cref="RemoteTaskAttribute"/> method whose arguments are literal values, task
/// placeholders or further calls. This is the form of a task that can leave the process.
/// </summary>
public sealed class RemoteCall
{
    /// <summary>Builds a call, checking that the method may be called remotely.</summary>
    /// <exception cref="ArgumentException">The method is not static or not marked
    /// <see cref="RemoteTaskAttribute"/>, or the argument count does not match.</exception>
    public RemoteCall(MethodInfo method, params object?[] arguments)
    {
        ArgumentNullException.ThrowIfNull(method);
        EnsureCallable(method);
        var parameters = method.GetParameters();
        if (parameters.Length != arguments.Length)
            throw new ArgumentException(
                $"{method.DeclaringType?.Name}.{method.Name} takes {parameters.Length} arguments, got {arguments.Length}.");
        Method = method;
        Arguments = arguments;
    }

    /// <summary>The method to run.</summary>
    public MethodInfo Method { get; }

    /// <summary>The arguments, possibly holding <see cref="TaskRef"/>, <see cref="TaskRefList"/>
    /// or nested <see cref="RemoteCall"/> values.</summary>
    public IReadOnlyList<object?> Arguments { get; }

    /// <summary>Throws unless <paramref name="method"/> is a public static method marked
    /// <see cref="RemoteTaskAttribute"/>.</summary>
    public static void EnsureCallable(MethodInfo method)
    {
        if (!method.IsStatic || !method.IsPublic)
            throw new ArgumentException($"{method.DeclaringType?.Name}.{method.Name} must be public and static to run as a remote task.");
        if (method.GetCustomAttribute<RemoteTaskAttribute>() is null)
            throw new ArgumentException($"{method.DeclaringType?.Name}.{method.Name} is not marked [RemoteTask].");
        if (method.ContainsGenericParameters)
            throw new ArgumentException($"{method.DeclaringType?.Name}.{method.Name} has open generic parameters.");
    }

    /// <summary>Every task this call reads, in first-seen order, nested calls included.</summary>
    public IReadOnlyList<TaskKey> Dependencies()
    {
        var seen = new HashSet<TaskKey>();
        var ordered = new List<TaskKey>();
        Collect(this, seen, ordered);
        return ordered;
    }

    private static void Collect(object? argument, HashSet<TaskKey> seen, List<TaskKey> ordered)
    {
        switch (argument)
        {
            case TaskRef r:
                if (seen.Add(r.Key)) ordered.Add(r.Key);
                break;
            case TaskRefList list:
                foreach (var key in list.Keys)
                    if (seen.Add(key)) ordered.Add(key);
                break;
            case RemoteCall call:
                foreach (var a in call.Arguments) Collect(a, seen, ordered);
                break;
        }
    }

    /// <summary>Runs the call, resolving placeholders through <paramref name="resolve"/>.</summary>
    public object? Invoke(Func<TaskKey, object?> resolve)
    {
        var parameters = Method.GetParameters();
        var values = new object?[parameters.Length];
        for (var i = 0; i < values.Length; i++)
            values[i] = Coerce(Evaluate(Arguments[i], parameters[i].ParameterType, resolve), parameters[i].ParameterType);

        try
        {
            return Method.Invoke(null, values);
        }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw();
            throw; // unreachable
        }
    }

    private static object? Evaluate(object? argument, Type target, Func<TaskKey, object?> resolve) => argument switch
    {
        TaskRef r => resolve(r.Key),
        TaskRefList list => list.Keys.Select(resolve).ToArray(),
        RemoteCall call => call.Invoke(resolve),
        _ => argument,
    };

    /// <summary>
    /// Converts an argument to its parameter type where the wire or a gather produced a looser
    /// one: <c>object[]</c> into a typed array, a boxed number into another numeric width.
    /// </summary>
    internal static object? Coerce(object? value, Type target)
    {
        if (value is null || target.IsInstanceOfType(value)) return value;

        if (target.IsArray && value is Array source)
        {
            var element = target.GetElementType()!;
            var typed = Array.CreateInstance(element, source.Length);
            for (var i = 0; i < source.Length; i++) typed.SetValue(Coerce(source.GetValue(i), element), i);
            return typed;
        }

        if (value is IConvertible && (target.IsPrimitive || target == typeof(decimal)))
            return Convert.ChangeType(value, target, System.Globalization.CultureInfo.InvariantCulture);

        return value;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Method.DeclaringType?.Name}.{Method.Name}({Arguments.Count} args)";
}
