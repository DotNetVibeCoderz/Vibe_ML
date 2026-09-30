using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>
/// Matrix products for the linear layers: double activations against float32 weights stored in
/// panels of twelve outputs.
/// </summary>
/// <remarks>
/// <para>
/// The layout is the standard one for a register-blocked GEMM. The weights are kept as panels -
/// <c>panel[i][j] = W[12 p + j, i]</c> - and the inner kernel holds a 4-row by 12-output block of
/// the result in twelve AVX registers, broadcasting one input value and multiplying it into three
/// weight vectors per step. That is three fused multiply-adds per load, where the dot-product
/// formulation it replaced needed a load for every multiply-add and ran at a third of the speed.
/// </para>
/// <para>
/// The weights stay float32 in memory, which is exact for every checkpoint and halves what a short
/// input has to stream. They are widened into a small per-thread buffer one block of 256 inputs at
/// a time, and each widened block is reused by every group of four rows, so the conversion is paid
/// once per call rather than once per row. Widening inside the inner loop instead was measured at
/// 40% slower: the conversion competes with the multiply-adds for the same execution port.
/// </para>
/// <para>
/// Work is split across threads by panel, so each thread writes its own columns of the result.
/// Every block of the product is summed in the same order whatever the thread count, so results
/// do not depend on the machine.
/// </para>
/// </remarks>
internal static class Gemm
{
    /// <summary>Outputs per panel: three vectors of four doubles.</summary>
    internal const int Width = 12;

    /// <summary>Rows per kernel call.</summary>
    private const int Height = 4;

    /// <summary>Inputs widened at a time: 12 x 256 doubles is 24 KB, which stays in the L1 cache.</summary>
    private const int Depth = 256;

    /// <summary>Whether the vectorised kernel can run; otherwise a scalar one does the same sums.</summary>
    private static readonly bool Vectorised = Avx.IsSupported && Fma.IsSupported;

    /// <summary>Panels needed for <paramref name="outputs"/> outputs, the last padded with zeros.</summary>
    internal static int Panels(int outputs) => (outputs + Width - 1) / Width;

    /// <summary>Packs a row-major <c>(outputs, inputs)</c> matrix into panels.</summary>
    internal static float[] Pack(ReadOnlySpan<float> weights, int inputs, int outputs)
    {
        var packed = new float[Panels(outputs) * inputs * Width];
        for (var o = 0; o < outputs; o++)
        {
            var row = weights.Slice(o * inputs, inputs);
            var at = o / Width * inputs * Width + o % Width;
            for (var i = 0; i < inputs; i++) packed[at + i * Width] = row[i];
        }

        return packed;
    }

    /// <summary>One weight, <c>W[output, input]</c>, read back from the panels.</summary>
    internal static float At(float[] packed, int inputs, int output, int input)
        => packed[output / Width * inputs * Width + input * Width + output % Width];

    /// <summary><c>y = x W^T</c>: <paramref name="rows"/> rows of <paramref name="x"/> against the packed weights.</summary>
    /// <returns>Row-major <c>[rows, outputs]</c>.</returns>
    internal static double[] Multiply(double[] x, int rows, float[] packed, int inputs, int outputs)
        => Run(x, rows, inputs, outputs, (panel, k0, count, buffer) =>
            Widen(packed, panel * inputs * Width + k0 * Width, buffer, count * Width));

    /// <summary>
    /// <c>y = x W^T</c> in single precision: activations rounded to float32, float32 sums - the
    /// <see cref="Precision.Single"/> path.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The same packed panels, read without widening. The blocking turns around: eight input rows are
    /// transposed into one float vector per input position, and each of a panel's twelve weights is
    /// broadcast against it, so twelve accumulators hold an 8-row by 12-output block. That is eight
    /// lanes per multiply-add where the double kernel has four, which is where the factor of two
    /// comes from.
    /// </para>
    /// <para>
    /// Each block is summed over the whole depth in order, so results do not depend on the thread count.
    /// </para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static double[] MultiplySingle(double[] x, int rows, float[] packed, int inputs, int outputs)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return Multiply(x, rows, packed, inputs, outputs);

        const int Rows = 8;
        var groups = (rows + Rows - 1) / Rows;
        var panels = Panels(outputs);

        // [group][input][8 rows], float - the transposed, rounded activations, shared by every panel.
        var transposed = new float[groups * inputs * Rows];
        Parallel.For(0, groups, group =>
        {
            var baseRow = group * Rows;
            var target = group * inputs * Rows;
            for (var r = 0; r < Rows && baseRow + r < rows; r++)
            {
                var source = (baseRow + r) * inputs;
                for (var k = 0; k < inputs; k++) transposed[target + k * Rows + r] = (float)x[source + k];
            }
        });

        var result = new double[rows * outputs];
        var work = (long)rows * inputs * outputs;
        var workers = (int)Math.Min(Math.Min(Environment.ProcessorCount, panels), Math.Max(1, work / 400_000));

        Parallel.For(0, workers, worker =>
        {
            Span<float> block = stackalloc float[Rows * Width];
            var firstPanel = panels * worker / workers;
            var lastPanel = panels * (worker + 1) / workers;

            for (var panel = firstPanel; panel < lastPanel; panel++)
            {
                for (var group = 0; group < groups; group++)
                {
                    SingleKernel(transposed, group * inputs * Rows, packed, panel * inputs * Width, inputs, block);

                    var columns = Math.Min(Width, outputs - panel * Width);
                    for (var r = 0; r < Rows && group * Rows + r < rows; r++)
                    {
                        var at = (group * Rows + r) * outputs + panel * Width;
                        for (var j = 0; j < columns; j++) result[at + j] = block[j * Rows + r];
                    }
                }
            }
        });

        return result;
    }

    /// <summary>An 8-row by 12-output block, <c>block[j * 8 + r]</c>, summed over every input.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SingleKernel(float[] aArray, int aOffset, float[] wArray, int wOffset, int depth, Span<float> block)
    {
        ref var a = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(aArray), aOffset);
        ref var w = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(wArray), wOffset);

        Vector256<float> c0 = default, c1 = default, c2 = default, c3 = default, c4 = default, c5 = default,
            c6 = default, c7 = default, c8 = default, c9 = default, c10 = default, c11 = default;

        for (nuint k = 0; k < (nuint)depth; k++)
        {
            var column = Vector256.LoadUnsafe(ref a, k * 8);
            ref var row = ref Unsafe.Add(ref w, k * Width);

            c0 = Fma.MultiplyAdd(column, Vector256.Create(row), c0);
            c1 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 1)), c1);
            c2 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 2)), c2);
            c3 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 3)), c3);
            c4 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 4)), c4);
            c5 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 5)), c5);
            c6 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 6)), c6);
            c7 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 7)), c7);
            c8 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 8)), c8);
            c9 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 9)), c9);
            c10 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 10)), c10);
            c11 = Fma.MultiplyAdd(column, Vector256.Create(Unsafe.Add(ref row, 11)), c11);
        }

        ref var target = ref MemoryMarshal.GetReference(block);
        c0.StoreUnsafe(ref target, 0); c1.StoreUnsafe(ref target, 8); c2.StoreUnsafe(ref target, 16);
        c3.StoreUnsafe(ref target, 24); c4.StoreUnsafe(ref target, 32); c5.StoreUnsafe(ref target, 40);
        c6.StoreUnsafe(ref target, 48); c7.StoreUnsafe(ref target, 56); c8.StoreUnsafe(ref target, 64);
        c9.StoreUnsafe(ref target, 72); c10.StoreUnsafe(ref target, 80); c11.StoreUnsafe(ref target, 88);
    }

    /// <summary><c>dx = dy W</c>: the transposed product, from the same packed weights.</summary>
    /// <returns>Row-major <c>[rows, inputs]</c>.</returns>
    /// <remarks>
    /// Here the roles swap - the depth runs over outputs and the panels over inputs - so each
    /// widened block is gathered from the forward panels rather than copied. That costs one scalar
    /// read per weight per call, which the rows of a training step amortise, and it saves keeping a
    /// second, transposed copy of every weight.
    /// </remarks>
    internal static double[] MultiplyTransposed(double[] dy, int rows, float[] packed, int inputs, int outputs)
        => Run(dy, rows, outputs, inputs, (panel, k0, count, buffer) =>
        {
            var first = panel * Width;
            var width = Math.Min(Width, inputs - first);

            for (var k = 0; k < count; k++)
            {
                var output = k0 + k;
                var at = output / Width * inputs * Width + output % Width;
                var target = buffer.AsSpan(k * Width, Width);

                for (var j = 0; j < width; j++) target[j] = packed[at + (first + j) * Width];
                for (var j = width; j < Width; j++) target[j] = 0;
            }
        });

    /// <summary>Fills <c>buffer[k][j]</c> with <c>B[k0 + k][12 panel + j]</c> for <c>k &lt; count</c>.</summary>
    private delegate void Fill(int panel, int k0, int count, double[] buffer);

    /// <summary>
    /// <c>C = A B</c> for row-major <c>A [rows, depth]</c>, with <paramref name="fill"/> supplying the
    /// blocks of <c>B [depth, columns]</c> panel by panel.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double[] Run(double[] a, int rows, int depth, int columns, Fill fill)
    {
        var panels = Panels(columns);
        var height = (rows + Height - 1) / Height * Height;
        var stride = panels * Width;

        // The kernel reads four rows at a time and writes twelve columns, so a shape that is not a
        // multiple of either runs on a zero-padded copy and is trimmed at the end.
        var source = a;
        if (height != rows)
        {
            source = new double[height * depth];
            Array.Copy(a, source, rows * depth);
        }

        var product = new double[height * stride];
        // Below a few hundred thousand multiply-adds the threads cost more than they save.
        var work = (long)rows * depth * columns;
        var workers = (int)Math.Min(Math.Min(Environment.ProcessorCount, panels), Math.Max(1, work / 200_000));

        Parallel.For(0, workers, worker =>
        {
            var buffer = new double[Depth * Width];
            var firstPanel = panels * worker / workers;
            var lastPanel = panels * (worker + 1) / workers;

            for (var k0 = 0; k0 < depth; k0 += Depth)
            {
                var count = Math.Min(Depth, depth - k0);

                for (var panel = firstPanel; panel < lastPanel; panel++)
                {
                    fill(panel, k0, count, buffer);

                    for (var row = 0; row < height; row += Height)
                    {
                        if (Vectorised)
                        {
                            Kernel(source, row * depth + k0, depth, buffer, count, product, row * stride + panel * Width, stride, k0 > 0);
                        }
                        else
                        {
                            Scalar(source, row * depth + k0, depth, buffer, count, product, row * stride + panel * Width, stride, k0 > 0);
                        }
                    }
                }
            }
        });

        if (stride == columns && height == rows) return product;

        var result = new double[rows * columns];
        for (var row = 0; row < rows; row++) Array.Copy(product, row * stride, result, row * columns, columns);
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Widen(float[] source, int offset, double[] target, int count)
    {
        if (!Vectorised)
        {
            for (var i = 0; i < count; i++) target[i] = source[offset + i];
            return;
        }

        ref var s = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(source), offset);
        ref var t = ref MemoryMarshal.GetArrayDataReference(target);

        nuint i2 = 0;
        for (; i2 + 8 <= (nuint)count; i2 += 8)
        {
            var f = Vector256.LoadUnsafe(ref s, i2);
            Avx.ConvertToVector256Double(f.GetLower()).StoreUnsafe(ref t, i2);
            Avx.ConvertToVector256Double(f.GetUpper()).StoreUnsafe(ref t, i2 + 4);
        }

        for (; i2 < (nuint)count; i2++) Unsafe.Add(ref t, i2) = Unsafe.Add(ref s, i2);
    }

    /// <summary>A 4 x 12 block of <c>C</c>: twelve accumulators, three multiply-adds per weight load.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel(
        double[] aArray, int aOffset, int lda, double[] bArray, int count, double[] cArray, int cOffset, int ldc, bool accumulate)
    {
        ref var a0 = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(aArray), aOffset);
        ref var a1 = ref Unsafe.Add(ref a0, lda);
        ref var a2 = ref Unsafe.Add(ref a0, 2 * lda);
        ref var a3 = ref Unsafe.Add(ref a0, 3 * lda);
        ref var b = ref MemoryMarshal.GetArrayDataReference(bArray);
        ref var c = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(cArray), cOffset);

        var r1 = (nuint)ldc;
        var r2 = (nuint)(2 * ldc);
        var r3 = (nuint)(3 * ldc);

        Vector256<double> c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;
        if (accumulate)
        {
            c00 = Vector256.LoadUnsafe(ref c); c01 = Vector256.LoadUnsafe(ref c, 4); c02 = Vector256.LoadUnsafe(ref c, 8);
            c10 = Vector256.LoadUnsafe(ref c, r1); c11 = Vector256.LoadUnsafe(ref c, r1 + 4); c12 = Vector256.LoadUnsafe(ref c, r1 + 8);
            c20 = Vector256.LoadUnsafe(ref c, r2); c21 = Vector256.LoadUnsafe(ref c, r2 + 4); c22 = Vector256.LoadUnsafe(ref c, r2 + 8);
            c30 = Vector256.LoadUnsafe(ref c, r3); c31 = Vector256.LoadUnsafe(ref c, r3 + 4); c32 = Vector256.LoadUnsafe(ref c, r3 + 8);
        }
        else
        {
            c00 = c01 = c02 = c10 = c11 = c12 = c20 = c21 = c22 = c30 = c31 = c32 = Vector256<double>.Zero;
        }

        for (nuint k = 0; k < (nuint)count; k++)
        {
            var b0 = Vector256.LoadUnsafe(ref b, k * Width);
            var b1 = Vector256.LoadUnsafe(ref b, k * Width + 4);
            var b2 = Vector256.LoadUnsafe(ref b, k * Width + 8);

            var s = Vector256.Create(Unsafe.Add(ref a0, k));
            c00 = Fma.MultiplyAdd(s, b0, c00); c01 = Fma.MultiplyAdd(s, b1, c01); c02 = Fma.MultiplyAdd(s, b2, c02);
            s = Vector256.Create(Unsafe.Add(ref a1, k));
            c10 = Fma.MultiplyAdd(s, b0, c10); c11 = Fma.MultiplyAdd(s, b1, c11); c12 = Fma.MultiplyAdd(s, b2, c12);
            s = Vector256.Create(Unsafe.Add(ref a2, k));
            c20 = Fma.MultiplyAdd(s, b0, c20); c21 = Fma.MultiplyAdd(s, b1, c21); c22 = Fma.MultiplyAdd(s, b2, c22);
            s = Vector256.Create(Unsafe.Add(ref a3, k));
            c30 = Fma.MultiplyAdd(s, b0, c30); c31 = Fma.MultiplyAdd(s, b1, c31); c32 = Fma.MultiplyAdd(s, b2, c32);
        }

        c00.StoreUnsafe(ref c); c01.StoreUnsafe(ref c, 4); c02.StoreUnsafe(ref c, 8);
        c10.StoreUnsafe(ref c, r1); c11.StoreUnsafe(ref c, r1 + 4); c12.StoreUnsafe(ref c, r1 + 8);
        c20.StoreUnsafe(ref c, r2); c21.StoreUnsafe(ref c, r2 + 4); c22.StoreUnsafe(ref c, r2 + 8);
        c30.StoreUnsafe(ref c, r3); c31.StoreUnsafe(ref c, r3 + 4); c32.StoreUnsafe(ref c, r3 + 8);
    }

    /// <summary>The same block without AVX: correct everywhere, fast only where the other one runs.</summary>
    private static void Scalar(
        double[] a, int aOffset, int lda, double[] b, int count, double[] c, int cOffset, int ldc, bool accumulate)
    {
        for (var r = 0; r < Height; r++)
        {
            for (var j = 0; j < Width; j++)
            {
                var sum = accumulate ? c[cOffset + r * ldc + j] : 0.0;
                for (var k = 0; k < count; k++) sum += a[aOffset + r * lda + k] * b[k * Width + j];
                c[cOffset + r * ldc + j] = sum;
            }
        }
    }
}
