using System.Collections.Concurrent;
using System.Globalization;
using System.Reflection;
using Gravicode.Science.GraviLearn;

namespace Gravicode.Science.DaskNet.ML;

/// <summary>
/// An estimator described by its type and constructor arguments rather than as an object, so it
/// can be built on a worker: only names and plain values cross the wire. Parameters are matched to
/// constructor parameters by name (case-insensitive), then to settable properties; anything left
/// over is an error, never silently ignored.
/// </summary>
public sealed class EstimatorSpec
{
    /// <summary>The estimator type; it must implement <see cref="IEstimator"/>.</summary>
    public Type Type { get; }

    /// <summary>The parameters given (the rest keep their defaults).</summary>
    public IReadOnlyDictionary<string, object?> Parameters { get; }

    /// <summary>Describes <paramref name="type"/> with <paramref name="parameters"/>, checking that
    /// they bind.</summary>
    public EstimatorSpec(Type type, IReadOnlyDictionary<string, object?>? parameters = null)
    {
        ArgumentNullException.ThrowIfNull(type);
        if (!typeof(IEstimator).IsAssignableFrom(type) || type.IsAbstract)
            throw new ArgumentException($"{type.Name} is not a concrete IEstimator.", nameof(type));
        Type = type;
        // Values are stored as the type they bind to, so "cosine" and DistanceMetric.Cosine are
        // the same candidate (and print the same).
        Parameters = (parameters ?? new Dictionary<string, object?>()).ToDictionary(
            p => p.Key, p => Normalize(type, p.Key, p.Value), StringComparer.OrdinalIgnoreCase);
        _ = Create();
    }

    /// <summary>Describes <typeparamref name="T"/> with the given parameters.</summary>
    public static EstimatorSpec Of<T>(params (string Name, object? Value)[] parameters) where T : IEstimator
        => new(typeof(T), parameters.ToDictionary(p => p.Name, p => p.Value, StringComparer.OrdinalIgnoreCase));

    /// <summary>Whether the estimator is a classifier (scored by accuracy unless told otherwise).</summary>
    public bool IsClassifier => typeof(IClassifier).IsAssignableFrom(Type);

    /// <summary>A new, unfitted estimator.</summary>
    public IEstimator Create() => Build(Type, [.. Parameters.Keys], [.. Parameters.Values]);

    /// <inheritdoc/>
    public override string ToString()
        => $"{Type.Name}({string.Join(", ", Parameters.Select(p => $"{p.Key}={Format(p.Value)}"))})";

    internal static string Format(object? v) => v switch
    {
        null => "null",
        string s => s,
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => v.ToString() ?? "",
    };

    /// <summary>The type as a worker resolves it.</summary>
    internal string TypeId => $"{Type.FullName}, {Type.Assembly.GetName().Name}";

    /// <summary>Parameter values as the wire carries them: enums become their names.</summary>
    internal object?[] WireValues => [.. Parameters.Values.Select(v => v is Enum e ? e.ToString() : v)];

    private static readonly ConcurrentDictionary<string, Type> Types = new();

    internal static Type Resolve(string typeId) => Types.GetOrAdd(typeId, static id =>
        Type.GetType(id, throwOnError: false)
        ?? AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType(id[..id.IndexOf(',')], throwOnError: false)).FirstOrDefault(t => t is not null)
        ?? throw new TypeLoadException($"Estimator type {id} is not loaded on this worker; preload its assembly."));

    internal static IEstimator Build(Type type, string[] names, object?[] values)
    {
        var ctor = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault()
            ?? throw new ArgumentException($"{type.Name} has no public constructor.");
        var parameters = ctor.GetParameters();
        var args = parameters.Select(p => p.HasDefaultValue ? p.DefaultValue : p.ParameterType.IsValueType ? Activator.CreateInstance(p.ParameterType) : null).ToArray();
        var given = new bool[parameters.Length];
        var leftover = new List<(string, object?)>();
        for (var i = 0; i < names.Length; i++)
        {
            var at = Array.FindIndex(parameters, p => string.Equals(p.Name, names[i], StringComparison.OrdinalIgnoreCase));
            if (at < 0) { leftover.Add((names[i], values[i])); continue; }
            args[at] = Convert(values[i], parameters[at].ParameterType, names[i]);
            given[at] = true;
        }
        var missing = parameters.Where((p, i) => !given[i] && !p.HasDefaultValue).Select(p => p.Name).ToArray();
        if (missing.Length > 0) throw new ArgumentException($"{type.Name} needs {string.Join(", ", missing)}.");
        var model = (IEstimator)ctor.Invoke(args);
        foreach (var (name, value) in leftover)
        {
            var property = type.GetProperty(name, BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase);
            if (property is not { CanWrite: true } || property.SetMethod is not { IsPublic: true })
                throw new ArgumentException($"{type.Name} has no parameter or settable property '{name}'. Constructor parameters: {string.Join(", ", parameters.Select(p => p.Name))}.");
            property.SetValue(model, Convert(value, property.PropertyType, name));
        }
        return model;
    }

    private static object? Normalize(Type type, string name, object? value)
    {
        var ctor = type.GetConstructors().OrderByDescending(c => c.GetParameters().Length).FirstOrDefault();
        var target = ctor?.GetParameters().FirstOrDefault(p => string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase))?.ParameterType
            ?? type.GetProperty(name, BindingFlags.Public | BindingFlags.Instance | BindingFlags.IgnoreCase)?.PropertyType;
        return target is null ? value : Convert(value, target, name);   // unknown names fail in Create
    }

    private static object? Convert(object? value, Type target, string name)
    {
        var t = Nullable.GetUnderlyingType(target) ?? target;
        try
        {
            if (value is null) return target.IsValueType && Nullable.GetUnderlyingType(target) is null
                ? throw new ArgumentException($"'{name}' cannot be null.") : null;
            if (t.IsInstanceOfType(value)) return value;
            if (t.IsEnum) return value is string s ? Enum.Parse(t, s, ignoreCase: true) : Enum.ToObject(t, value);
            return System.Convert.ChangeType(value, t, CultureInfo.InvariantCulture);
        }
        catch (Exception e) when (e is InvalidCastException or FormatException or OverflowException)
        {
            throw new ArgumentException($"'{name}' = {Format(value)} is not a {t.Name}.", e);
        }
    }
}
