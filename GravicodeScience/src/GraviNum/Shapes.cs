namespace Gravicode.Science.GraviNum;

/// <summary>
/// Helpers for shape arithmetic: sizes, contiguous strides and NumPy-style broadcasting.
/// </summary>
public static class Shapes
{
    /// <summary>Total number of elements described by <paramref name="shape"/>.</summary>
    public static int Size(ReadOnlySpan<int> shape)
    {
        var size = 1;
        for (var i = 0; i < shape.Length; i++)
        {
            if (shape[i] < 0)
                throw new ArgumentException($"Negative dimension {shape[i]} at axis {i}.", nameof(shape));
            size = checked(size * shape[i]);
        }
        return size;
    }

    /// <summary>Row-major (C order) strides, measured in elements.</summary>
    public static int[] ContiguousStrides(ReadOnlySpan<int> shape)
    {
        var strides = new int[shape.Length];
        var acc = 1;
        for (var i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    /// <summary>
    /// Resulting shape of broadcasting <paramref name="a"/> against <paramref name="b"/>.
    /// Dimensions are aligned from the right; a dimension of 1 stretches to match.
    /// </summary>
    public static int[] Broadcast(ReadOnlySpan<int> a, ReadOnlySpan<int> b)
    {
        var rank = Math.Max(a.Length, b.Length);
        var result = new int[rank];
        for (var i = 0; i < rank; i++)
        {
            var da = i < rank - a.Length ? 1 : a[i - (rank - a.Length)];
            var db = i < rank - b.Length ? 1 : b[i - (rank - b.Length)];
            if (da != db && da != 1 && db != 1)
                throw new InvalidOperationException(
                    $"Shapes ({Describe(a)}) and ({Describe(b)}) are not broadcast compatible.");
            result[i] = Math.Max(da, db);
        }
        return result;
    }

    /// <summary>True when the two shapes can be broadcast together.</summary>
    public static bool CanBroadcast(ReadOnlySpan<int> a, ReadOnlySpan<int> b)
    {
        var rank = Math.Max(a.Length, b.Length);
        for (var i = 0; i < rank; i++)
        {
            var da = i < rank - a.Length ? 1 : a[i - (rank - a.Length)];
            var db = i < rank - b.Length ? 1 : b[i - (rank - b.Length)];
            if (da != db && da != 1 && db != 1) return false;
        }
        return true;
    }

    /// <summary>
    /// Strides to use when reading an array of <paramref name="shape"/>/<paramref name="strides"/>
    /// as if it had shape <paramref name="target"/>. Stretched axes get a stride of zero so the
    /// same element is re-read instead of copied.
    /// </summary>
    public static int[] BroadcastStrides(ReadOnlySpan<int> shape, ReadOnlySpan<int> strides, ReadOnlySpan<int> target)
    {
        var pad = target.Length - shape.Length;
        if (pad < 0)
            throw new InvalidOperationException($"Cannot broadcast rank {shape.Length} down to rank {target.Length}.");

        var result = new int[target.Length];
        for (var i = 0; i < target.Length; i++)
        {
            if (i < pad) { result[i] = 0; continue; }
            var dim = shape[i - pad];
            if (dim == target[i]) result[i] = strides[i - pad];
            else if (dim == 1) result[i] = 0;
            else throw new InvalidOperationException(
                $"Cannot broadcast axis {i - pad} of size {dim} to {target[i]}.");
        }
        return result;
    }

    /// <summary>Resolves a single <c>-1</c> placeholder in a reshape target.</summary>
    public static int[] ResolveReshape(ReadOnlySpan<int> requested, int size)
    {
        var shape = requested.ToArray();
        var unknown = -1;
        var known = 1;
        for (var i = 0; i < shape.Length; i++)
        {
            if (shape[i] == -1)
            {
                if (unknown >= 0) throw new ArgumentException("Only one dimension may be -1.");
                unknown = i;
            }
            else known *= shape[i];
        }

        if (unknown >= 0)
        {
            if (known == 0 || size % known != 0)
                throw new ArgumentException($"Cannot reshape {size} elements into ({Describe(requested)}).");
            shape[unknown] = size / known;
        }
        else if (known != size)
        {
            throw new ArgumentException($"Cannot reshape {size} elements into ({Describe(requested)}).");
        }

        return shape;
    }

    /// <summary>Formats a shape as <c>2, 3, 4</c> for diagnostics.</summary>
    public static string Describe(ReadOnlySpan<int> shape)
    {
        if (shape.Length == 0) return "scalar";
        return string.Join(", ", shape.ToArray());
    }

    /// <summary>Normalizes a possibly negative axis index against a rank.</summary>
    public static int NormalizeAxis(int axis, int rank)
    {
        var a = axis < 0 ? axis + rank : axis;
        if (a < 0 || a >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} out of range for rank {rank}.");
        return a;
    }
}
