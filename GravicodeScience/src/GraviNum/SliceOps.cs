using Sel = Gravicode.Science.GraviNum.Slice;

namespace Gravicode.Science.GraviNum;

/// <summary>
/// The half of NumPy's indexing that returns nothing: writing through a view, selecting along an
/// arbitrary axis, and choosing element-wise between two arrays.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="NdArray.Slice"/> already produces a zero-copy view and <see cref="NdArray.Assign"/>
/// already writes an array through one, broadcasting included. What was missing is everything
/// around them: selecting along an axis that is not the first, naming trailing axes without
/// counting the leading ones, and choosing element-wise between two arrays.
/// </para>
/// <para>
/// Everything here that returns an array returns a copy; everything that writes, writes through the
/// view into the original buffer. That split is deliberate and is the one thing to keep in mind:
/// <c>a.Slice(...).Assign(b)</c> changes <c>a</c>, while <c>a.Slice(...).Copy()</c> does not.
/// </para>
/// </remarks>
public static class SliceOps
{
    /// <summary>Fills a view with a constant and returns it, so the call chains.</summary>
    /// <remarks>
    /// <see cref="NdArray.Assign(NdArray)"/> already covers the array-to-array case, broadcasting
    /// included; this is only the scalar spelling of <see cref="NdArray.Fill"/> with a return value.
    /// </remarks>
    public static NdArray Assign(this NdArray destination, double value)
    {
        ArgumentNullException.ThrowIfNull(destination);
        destination.Fill(value);
        return destination;
    }

    private static void Decompose(int flat, ReadOnlySpan<int> shape, int[] into)
    {
        for (var axis = shape.Length - 1; axis >= 0; axis--)
        {
            into[axis] = flat % shape[axis];
            flat /= shape[axis];
        }
    }

    // ---------------------------------------------------------------- ellipsis

    /// <summary>
    /// Slices with an ellipsis, which expands to however many whole axes are left over.
    /// </summary>
    /// <param name="array">The array to view.</param>
    /// <param name="before">Selectors applied to the leading axes.</param>
    /// <param name="after">Selectors applied to the trailing axes.</param>
    /// <remarks>
    /// The point is selecting a trailing axis without counting the leading ones. Taking the last
    /// channel of a rank-4 batch is <c>SliceEllipsis(array, [], [Sel.At(0)])</c> and stays correct
    /// if the batch gains an axis — where <c>Slice(Sel.All, Sel.All, Sel.All, Sel.At(0))</c> does
    /// not.
    /// </remarks>
    public static NdArray SliceEllipsis(this NdArray array, Sel[] before, Sel[] after)
    {
        ArgumentNullException.ThrowIfNull(array);
        ArgumentNullException.ThrowIfNull(before);
        ArgumentNullException.ThrowIfNull(after);

        var filled = array.Rank - before.Length - after.Length;
        if (filled < 0)
            throw new ArgumentException(
                $"{before.Length + after.Length} selectors is more than the rank {array.Rank} array has axes.");

        var selectors = new Sel[array.Rank];
        Array.Copy(before, selectors, before.Length);
        for (var i = 0; i < filled; i++) selectors[before.Length + i] = Sel.All;
        Array.Copy(after, 0, selectors, before.Length + filled, after.Length);

        return array.Slice(selectors);
    }

    // ------------------------------------------------------------------ axes

    /// <summary>Selects positions along one axis, copying the result.</summary>
    /// <remarks>
    /// <see cref="NdArray.Take"/> only walks axis 0. This is the general form: picking columns is
    /// <c>TakeAlong(indices, axis: 1)</c>, and indices may repeat or reorder.
    /// </remarks>
    public static NdArray TakeAlong(this NdArray array, IReadOnlyList<int> indices, int axis = 0)
    {
        ArgumentNullException.ThrowIfNull(array);
        ArgumentNullException.ThrowIfNull(indices);

        if (axis < 0) axis += array.Rank;
        if (axis < 0 || axis >= array.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is outside a rank {array.Rank} array.");

        var shape = array.Shape.ToArray();
        shape[axis] = indices.Count;
        var result = NdArray.Zeros(shape);

        var index = new int[array.Rank];

        for (var flat = 0; flat < result.Size; flat++)
        {
            Decompose(flat, shape, index);

            var source = index[axis];
            var position = indices[source];
            if (position < 0) position += array.Shape[axis];
            if (position < 0 || position >= array.Shape[axis])
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {indices[source]} is outside axis {axis} of length {array.Shape[axis]}.");

            index[axis] = position;
            result.SetAt(flat, array[index]);
            index[axis] = source;
        }

        return result;
    }

    /// <summary>A single position along an axis, with that axis dropped.</summary>
    /// <remarks>The general form of <c>Row</c> and <c>Column</c>, and a zero-copy view like them.</remarks>
    public static NdArray AxisAt(this NdArray array, int axis, int index)
    {
        if (axis < 0) axis += array.Rank;
        if (axis < 0 || axis >= array.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is outside a rank {array.Rank} array.");

        var selectors = new Sel[axis + 1];
        for (var i = 0; i < axis; i++) selectors[i] = Sel.All;
        selectors[axis] = Sel.At(index);

        return array.Slice(selectors);
    }

    /// <summary>A contiguous span of an axis, keeping that axis.</summary>
    public static NdArray AxisRange(this NdArray array, int axis, int start, int? stop = null, int step = 1)
    {
        if (axis < 0) axis += array.Rank;
        if (axis < 0 || axis >= array.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis), $"Axis {axis} is outside a rank {array.Rank} array.");

        var selectors = new Sel[axis + 1];
        for (var i = 0; i < axis; i++) selectors[i] = Sel.All;
        selectors[axis] = Sel.Range(start, stop, step);

        return array.Slice(selectors);
    }

    // ----------------------------------------------------------------- masking

    /// <summary>
    /// Element-wise choice: <paramref name="ifTrue"/> where the condition holds, otherwise
    /// <paramref name="ifFalse"/>.
    /// </summary>
    /// <remarks>
    /// The three-argument form, distinct from <see cref="NdArray.Where(Func{double, bool})"/> which
    /// filters and returns the survivors. This one keeps the shape, which is what makes it
    /// composable — clipping, masking a loss, replacing a sentinel — where the filtering form is
    /// not.
    /// </remarks>
    public static NdArray Select(NdArray condition, NdArray ifTrue, NdArray ifFalse)
    {
        ArgumentNullException.ThrowIfNull(condition);

        var result = NdArray.Zeros([.. condition.Shape]);
        for (var i = 0; i < condition.Size; i++)
            result.SetAt(i, condition.At(i) != 0
                ? (ifTrue.Size == 1 ? ifTrue.At(0) : ifTrue.At(i))
                : (ifFalse.Size == 1 ? ifFalse.At(0) : ifFalse.At(i)));

        return result;
    }

    /// <summary>Element-wise choice between an array and a constant.</summary>
    public static NdArray Select(NdArray condition, NdArray ifTrue, double ifFalse)
    {
        var result = NdArray.Zeros([.. condition.Shape]);
        for (var i = 0; i < condition.Size; i++)
            result.SetAt(i, condition.At(i) != 0 ? ifTrue.At(i) : ifFalse);
        return result;
    }

    /// <summary>Writes <paramref name="value"/> wherever <paramref name="predicate"/> holds.</summary>
    /// <remarks>
    /// In place, and returns the same array, so it chains. The masked counterpart to
    /// <see cref="NdArray.Fill"/>; use it on a view to confine the write to a region.
    /// </remarks>
    public static NdArray SetWhere(this NdArray array, Func<double, bool> predicate, double value)
    {
        ArgumentNullException.ThrowIfNull(array);
        ArgumentNullException.ThrowIfNull(predicate);

        for (var i = 0; i < array.Size; i++)
            if (predicate(array.At(i))) array.SetAt(i, value);

        return array;
    }

    /// <summary>Writes <paramref name="value"/> wherever <paramref name="mask"/> is true.</summary>
    public static NdArray SetWhere(this NdArray array, bool[] mask, double value)
    {
        ArgumentNullException.ThrowIfNull(array);
        ArgumentNullException.ThrowIfNull(mask);

        if (mask.Length != array.Size)
            throw new ArgumentException($"Mask has {mask.Length} entries but the array has {array.Size} elements.");

        for (var i = 0; i < array.Size; i++)
            if (mask[i]) array.SetAt(i, value);

        return array;
    }

    /// <summary>The flat positions where <paramref name="predicate"/> holds.</summary>
    /// <remarks>
    /// The companion to <see cref="NdArray.Where(Func{double, bool})"/>, which returns the values.
    /// Positions are what you need to look the same rows up in another array.
    /// </remarks>
    public static int[] IndicesWhere(this NdArray array, Func<double, bool> predicate)
    {
        ArgumentNullException.ThrowIfNull(array);

        var found = new List<int>();
        for (var i = 0; i < array.Size; i++)
            if (predicate(array.At(i))) found.Add(i);

        return [.. found];
    }

    /// <summary>Rows of a matrix for which <paramref name="predicate"/> holds, copied out.</summary>
    /// <remarks>The array equivalent of filtering a table — "keep the samples where the label is 1".</remarks>
    public static NdArray FilterRows(this NdArray array, Func<NdArray, bool> predicate)
    {
        ArgumentNullException.ThrowIfNull(array);
        if (array.Rank < 1) throw new InvalidOperationException("Cannot filter the rows of a rank 0 array.");

        var kept = new List<int>();
        for (var i = 0; i < array.Shape[0]; i++)
            if (predicate(array.Row(i))) kept.Add(i);

        return array.Take(kept);
    }

    /// <summary>Clamps every element into <paramref name="low"/>..<paramref name="high"/>, copying.</summary>
    public static NdArray Clip(this NdArray array, double low, double high)
    {
        if (low > high) throw new ArgumentException($"The lower bound {low} is above the upper bound {high}.");

        var result = NdArray.Zeros([.. array.Shape]);
        for (var i = 0; i < array.Size; i++) result.SetAt(i, Math.Clamp(array.At(i), low, high));
        return result;
    }
}

/// <summary>Extra selector shorthands.</summary>
public static class SliceShorthand
{
    /// <summary>The last <paramref name="count"/> elements of an axis.</summary>
    /// <remarks>
    /// <c>Sel.Range(-count)</c> already does this; the named form exists because a bare negative
    /// start reads as a mistake at the call site.
    /// </remarks>
    public static Sel Last(int count)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        return Sel.Range(-count);
    }

    /// <summary>The first <paramref name="count"/> elements of an axis.</summary>
    public static Sel First(int count)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        return Sel.Range(0, count);
    }

    /// <summary>Every <paramref name="step"/>-th element of the whole axis.</summary>
    public static Sel Every(int step) => Sel.Range(0, null, step);

    /// <summary>Everything except the last <paramref name="count"/> elements.</summary>
    public static Sel DropLast(int count)
    {
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        return Sel.Range(0, -count);
    }
}
