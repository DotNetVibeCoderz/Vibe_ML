namespace Gravicode.Science.GraviNum.Autodiff;

/// <summary>
/// Operator and instance-method sugar over <see cref="TensorOps"/>, so a forward pass reads like
/// the formula it implements.
/// </summary>
public sealed partial class Tensor
{
    /// <summary>Element-wise sum.</summary>
    public static Tensor operator +(Tensor a, Tensor b) => TensorOps.Add(a, b);

    /// <summary>Element-wise difference.</summary>
    public static Tensor operator -(Tensor a, Tensor b) => TensorOps.Subtract(a, b);

    /// <summary>Element-wise product.</summary>
    public static Tensor operator *(Tensor a, Tensor b) => TensorOps.Multiply(a, b);

    /// <summary>Element-wise quotient.</summary>
    public static Tensor operator /(Tensor a, Tensor b) => TensorOps.Divide(a, b);

    /// <summary>Negation.</summary>
    public static Tensor operator -(Tensor a) => TensorOps.Negate(a);

    /// <summary>Sum of every element.</summary>
    public Tensor Sum() => TensorOps.Sum(this);

    /// <summary>Sum along one axis.</summary>
    public Tensor Sum(int axis) => TensorOps.Sum(this, axis);

    /// <summary>Mean of every element.</summary>
    public Tensor Mean() => TensorOps.Mean(this);

    /// <summary>Natural logarithm.</summary>
    public Tensor Log() => TensorOps.Log(this);

    /// <summary>Natural exponential.</summary>
    public Tensor Exp() => TensorOps.Exp(this);

    /// <summary>Square root.</summary>
    public Tensor Sqrt() => TensorOps.Sqrt(this);

    /// <summary>Raises each element to a constant power.</summary>
    public Tensor Pow(double exponent) => TensorOps.Pow(this, exponent);

    /// <summary>Hyperbolic tangent.</summary>
    public Tensor Tanh() => TensorOps.Tanh(this);

    /// <summary>Logistic sigmoid.</summary>
    public Tensor Sigmoid() => TensorOps.Sigmoid(this);

    /// <summary>Rectified linear unit.</summary>
    public Tensor Relu() => TensorOps.Relu(this);

    /// <summary>Absolute value.</summary>
    public Tensor Abs() => TensorOps.Abs(this);

    /// <summary><c>log(1 + exp(x))</c>.</summary>
    public Tensor Softplus() => TensorOps.Softplus(this);

    /// <summary><c>log(sum(exp(x)))</c> over every element.</summary>
    public Tensor LogSumExp() => TensorOps.LogSumExp(this);

    /// <summary>Matrix product.</summary>
    public Tensor MatMul(Tensor other) => TensorOps.MatMul(this, other);

    /// <summary>Matrix transpose.</summary>
    public Tensor T() => TensorOps.Transpose(this);

    /// <summary>Reshapes without moving data.</summary>
    public Tensor Reshape(params int[] shape) => TensorOps.Reshape(this, shape);

    /// <summary>One element, as a one-element tensor.</summary>
    public Tensor At(int flatIndex) => TensorOps.At(this, flatIndex);
}
