using System.Diagnostics;

namespace MediaPipeNet.Framework;

/// <summary>
/// An immutable, timestamped unit of data flowing between graph nodes, mirroring
/// <c>mediapipe::Packet</c>. Use <see cref="Packet{T}"/> for typed access.
/// </summary>
public abstract class Packet
{
    private protected Packet(Timestamp timestamp) => Timestamp = timestamp;

    /// <summary>The packet timestamp.</summary>
    public Timestamp Timestamp { get; }

    /// <summary>The payload as <see cref="object"/>.</summary>
    public abstract object? UntypedValue { get; }

    /// <summary>The declared payload type.</summary>
    public abstract Type PayloadType { get; }

    /// <summary>Creates a typed packet.</summary>
    public static Packet<T> Create<T>(T value, Timestamp timestamp) => new(value, timestamp);

    /// <summary>Creates a typed packet with a millisecond timestamp.</summary>
    public static Packet<T> Create<T>(T value, long timestampMs) => new(value, Timestamp.FromMilliseconds(timestampMs));

    /// <summary>Returns this packet as <see cref="Packet{T}"/>, converting the view when the payload is compatible.</summary>
    /// <exception cref="InvalidCastException">The payload is not a <typeparamref name="T"/>.</exception>
    public Packet<T> As<T>()
    {
        if (this is Packet<T> typed) return typed;
        if (UntypedValue is T value) return new Packet<T>(value, Timestamp);
        if (UntypedValue is null && default(T) is null) return new Packet<T>(default!, Timestamp);
        throw new InvalidCastException($"Packet holds {PayloadType.Name}, which is not a {typeof(T).Name}.");
    }

    /// <summary>Returns a packet with the same payload at another timestamp.</summary>
    public abstract Packet At(Timestamp timestamp);
}

/// <summary>A typed, timestamped packet.</summary>
/// <typeparam name="T">Payload type.</typeparam>
[DebuggerDisplay("{Value} @ {Timestamp}")]
public sealed class Packet<T> : Packet
{
    /// <summary>Creates the packet.</summary>
    public Packet(T value, Timestamp timestamp) : base(timestamp) => Value = value;

    /// <summary>The payload.</summary>
    public T Value { get; }

    /// <inheritdoc />
    public override object? UntypedValue => Value;

    /// <inheritdoc />
    public override Type PayloadType => typeof(T);

    /// <inheritdoc />
    public override Packet At(Timestamp timestamp) => new Packet<T>(Value, timestamp);

    /// <summary>Returns a typed packet with the same payload at another timestamp.</summary>
    public Packet<T> WithTimestamp(Timestamp timestamp) => new(Value, timestamp);

    /// <inheritdoc />
    public override string ToString() => $"Packet<{typeof(T).Name}>({Value}) @ {Timestamp}";
}
