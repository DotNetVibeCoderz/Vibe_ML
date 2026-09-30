namespace Gravicode.HFNet.GraviTransformers;

/// <summary>The arithmetic the linear layers run in.</summary>
public enum Precision
{
    /// <summary>
    /// float32 weights, double activations and double sums - the default. Agrees with torch in float64
    /// to about 1e-13, and costs roughly twice the time of <see cref="Single"/>.
    /// </summary>
    Double,

    /// <summary>
    /// float32 activations and float32 sums in the linear layers, as torch computes by default.
    /// About twice as fast on the matrix products, and agrees with torch's float32 output to the
    /// relative 1e-6 that two float32 implementations summing in different orders do.
    /// </summary>
    Single,
}

/// <summary>Process-wide settings for the managed inference kernels.</summary>
/// <remarks>
/// <para>
/// <see cref="LinearLayers"/> is an opt-in speed setting. The linear layers are where inference
/// spends its time, and on a CPU single precision fits twice the values in each vector instruction.
/// Everything else - attention, norms, activations - stays double, since they are a small share of
/// the cost and the accuracy they keep is free.
/// </para>
/// <para>
/// Training is not affected: a LoRA or prefix-tuning step always runs its forward pass in double,
/// because its hand-written backward pass is the derivative of that function and no other.
/// </para>
/// </remarks>
public static class ComputeOptions
{
    [ThreadStatic]
    private static int _exact;

    /// <summary>The precision of every linear layer's product during inference. <see cref="Precision.Double"/> by default.</summary>
    public static Precision LinearLayers { get; set; } = Precision.Double;

    /// <summary>Whether the product about to run on this thread should be single precision.</summary>
    internal static bool UseSingle => LinearLayers == Precision.Single && _exact == 0;

    /// <summary>Keeps this thread's products in double until disposed, whatever <see cref="LinearLayers"/> says.</summary>
    internal static ExactScope Exact()
    {
        _exact++;
        return default;
    }

    internal readonly struct ExactScope : IDisposable
    {
        public void Dispose() => _exact--;
    }
}
