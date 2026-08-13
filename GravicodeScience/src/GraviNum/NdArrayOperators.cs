namespace Gravicode.Science.GraviNum;

/// <summary>
/// Operator overloads and fluent element-wise helpers, so array maths reads like maths.
/// </summary>
public sealed partial class NdArray
{
    /// <summary>Element-wise sum with broadcasting.</summary>
    public static NdArray operator +(NdArray a, NdArray b) => UFunc.Add(a, b);

    /// <summary>Element-wise difference with broadcasting.</summary>
    public static NdArray operator -(NdArray a, NdArray b) => UFunc.Subtract(a, b);

    /// <summary>Element-wise (Hadamard) product with broadcasting. Use <see cref="LinAlg.Dot"/> for matrix products.</summary>
    public static NdArray operator *(NdArray a, NdArray b) => UFunc.Multiply(a, b);

    /// <summary>Element-wise quotient with broadcasting.</summary>
    public static NdArray operator /(NdArray a, NdArray b) => UFunc.Divide(a, b);

    /// <summary>Element-wise remainder with broadcasting.</summary>
    public static NdArray operator %(NdArray a, NdArray b) => UFunc.Modulo(a, b);

    /// <summary>Adds a scalar to every element.</summary>
    public static NdArray operator +(NdArray a, double s) => UFunc.AddScalar(a, s);

    /// <summary>Adds a scalar to every element.</summary>
    public static NdArray operator +(double s, NdArray a) => UFunc.AddScalar(a, s);

    /// <summary>Subtracts a scalar from every element.</summary>
    public static NdArray operator -(NdArray a, double s) => UFunc.AddScalar(a, -s);

    /// <summary>Subtracts every element from a scalar.</summary>
    public static NdArray operator -(double s, NdArray a) => UFunc.Unary(a, x => s - x);

    /// <summary>Scales every element.</summary>
    public static NdArray operator *(NdArray a, double s) => UFunc.MultiplyScalar(a, s);

    /// <summary>Scales every element.</summary>
    public static NdArray operator *(double s, NdArray a) => UFunc.MultiplyScalar(a, s);

    /// <summary>Divides every element by a scalar.</summary>
    public static NdArray operator /(NdArray a, double s) => UFunc.MultiplyScalar(a, 1.0 / s);

    /// <summary>Divides a scalar by every element.</summary>
    public static NdArray operator /(double s, NdArray a) => UFunc.Unary(a, x => s / x);

    /// <summary>Element-wise negation.</summary>
    public static NdArray operator -(NdArray a) => UFunc.Negate(a);

    /// <summary>Element-wise unary plus.</summary>
    public static NdArray operator +(NdArray a) => a;

    // ------------------------------------------------------------ fluent maths

    /// <summary>Matrix product with <paramref name="other"/>.</summary>
    public NdArray Dot(NdArray other) => LinAlg.Dot(this, other);

    /// <summary>Element-wise absolute value.</summary>
    public NdArray Abs() => UFunc.Abs(this);

    /// <summary>Element-wise square root.</summary>
    public NdArray Sqrt() => UFunc.Sqrt(this);

    /// <summary>Element-wise square.</summary>
    public NdArray Square() => UFunc.Square(this);

    /// <summary>Element-wise natural exponential.</summary>
    public NdArray Exp() => UFunc.Exp(this);

    /// <summary>Element-wise natural logarithm.</summary>
    public NdArray Log() => UFunc.Log(this);

    /// <summary>Element-wise power.</summary>
    public NdArray Pow(double exponent) => UFunc.PowerScalar(this, exponent);

    /// <summary>Element-wise logistic sigmoid.</summary>
    public NdArray Sigmoid() => UFunc.Sigmoid(this);

    /// <summary>Element-wise rectified linear unit.</summary>
    public NdArray Relu() => UFunc.Relu(this);

    /// <summary>Element-wise hyperbolic tangent.</summary>
    public NdArray Tanh() => UFunc.Tanh(this);

    /// <summary>Clamps every element into <c>[min, max]</c>.</summary>
    public NdArray Clip(double min, double max) => UFunc.Clip(this, min, max);

    /// <summary>Applies an arbitrary function to every element.</summary>
    public NdArray Map(Func<double, double> f) => UFunc.Unary(this, f);

    // ------------------------------------------------------------ reductions

    /// <summary>Sum of every element.</summary>
    public double Sum() => Statistics.Sum(this);

    /// <summary>Arithmetic mean of every element.</summary>
    public double Mean() => Statistics.Mean(this);

    /// <summary>Smallest element.</summary>
    public double Min() => Statistics.Min(this);

    /// <summary>Largest element.</summary>
    public double Max() => Statistics.Max(this);

    /// <summary>Sample standard deviation.</summary>
    public double Std(int ddof = 0) => Statistics.Std(this, ddof);

    /// <summary>Sample variance.</summary>
    public double Var(int ddof = 0) => Statistics.Var(this, ddof);

    /// <summary>Reduction along a single axis.</summary>
    public NdArray Sum(int axis) => Statistics.Sum(this, axis);

    /// <summary>Mean along a single axis.</summary>
    public NdArray Mean(int axis) => Statistics.Mean(this, axis);
}
