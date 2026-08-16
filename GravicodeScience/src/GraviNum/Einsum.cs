namespace Gravicode.Science.GraviNum;

/// <summary>
/// Einstein summation: one notation for transposing, tracing, contracting and outer products.
/// </summary>
/// <remarks>
/// <para>
/// A subscript string names an axis of each operand with a letter, and the output names the axes
/// to keep. Every letter that appears in the inputs but not the output is summed over. That single
/// rule covers a surprising amount of tensor algebra:
/// </para>
/// <code>
/// "ij,jk->ik"   matrix product          "ij->ji"   transpose
/// "ij,ij->"     Frobenius inner product "ii->i"    diagonal
/// "ij->i"       row sums                "ii->"     trace
/// "i,j->ij"     outer product           "bij,bjk->bik"  batched matrix product
/// </code>
/// <para>
/// The value of the notation is that the intent is written down. <c>LinAlg.Dot(a.T, b)</c> makes
/// the reader reconstruct which axes met; <c>"ji,jk->ik"</c> says so. That matters most for the
/// operations that have no name — a contraction over two axes of a rank-4 tensor is unreadable as
/// a sequence of transposes and reshapes, and obvious as a subscript string.
/// </para>
/// <para>
/// This is a straightforward nested-loop evaluator, not an optimising one: it does not reorder a
/// chain of operands to minimise intermediate size. For two operands — which is nearly every real
/// use — there is no ordering choice to make.
/// </para>
/// </remarks>
public static class Einsum
{
    /// <summary>Evaluates a subscript expression over the given operands.</summary>
    /// <param name="subscripts">Something like <c>"ij,jk->ik"</c>. The arrow may be omitted.</param>
    /// <param name="operands">One array per comma-separated group.</param>
    public static NdArray Evaluate(string subscripts, params NdArray[] operands)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(subscripts);
        ArgumentNullException.ThrowIfNull(operands);

        var (inputs, output) = Parse(subscripts, operands);
        var extent = MeasureAxes(inputs, operands);

        // Letters kept in the output are the loop nest we build a result for; the rest are summed.
        var kept = output.ToCharArray();
        var summed = extent.Keys.Where(c => !output.Contains(c)).ToArray();

        var resultShape = kept.Length == 0 ? [1] : kept.Select(c => extent[c]).ToArray();
        var result = NdArray.Zeros(resultShape);

        var keptSizes = kept.Select(c => extent[c]).ToArray();
        var summedSizes = summed.Select(c => extent[c]).ToArray();

        var keptTotal = Total(keptSizes);
        var summedTotal = Total(summedSizes);

        var assignment = new Dictionary<char, int>(extent.Count);
        var keptIndex = new int[kept.Length];
        var summedIndex = new int[summed.Length];

        for (var outer = 0; outer < keptTotal; outer++)
        {
            Decompose(outer, keptSizes, keptIndex);
            for (var i = 0; i < kept.Length; i++) assignment[kept[i]] = keptIndex[i];

            var accumulator = 0.0;
            for (var inner = 0; inner < summedTotal; inner++)
            {
                Decompose(inner, summedSizes, summedIndex);
                for (var i = 0; i < summed.Length; i++) assignment[summed[i]] = summedIndex[i];

                var term = 1.0;
                for (var op = 0; op < operands.Length; op++) term *= Read(operands[op], inputs[op], assignment);
                accumulator += term;
            }

            result.SetAt(outer, accumulator);
        }

        return result;
    }

    /// <summary>
    /// Splits the subscript string, inferring the output when no arrow is given.
    /// </summary>
    /// <remarks>
    /// The implicit output is NumPy's rule: every letter appearing exactly once across all inputs,
    /// in alphabetical order. So <c>"ij,jk"</c> means <c>"ij,jk->ik"</c> — <c>j</c> is repeated and
    /// therefore summed. Spelling the output out is clearer and always allowed.
    /// </remarks>
    private static (string[] Inputs, string Output) Parse(string subscripts, NdArray[] operands)
    {
        var cleaned = subscripts.Replace(" ", "");
        var arrow = cleaned.IndexOf("->", StringComparison.Ordinal);

        var left = arrow >= 0 ? cleaned[..arrow] : cleaned;
        var inputs = left.Split(',');

        if (inputs.Length != operands.Length)
            throw new ArgumentException(
                $"'{subscripts}' names {inputs.Length} operand(s) but {operands.Length} were given.");

        for (var i = 0; i < inputs.Length; i++)
        {
            if (inputs[i].Any(c => !char.IsAsciiLetter(c)))
                throw new ArgumentException($"'{inputs[i]}' contains a non-letter subscript.");

            if (inputs[i].Length != operands[i].Rank)
                throw new ArgumentException(
                    $"Subscript '{inputs[i]}' names {inputs[i].Length} axes but operand {i} has rank {operands[i].Rank}.");
        }

        if (arrow >= 0) return (inputs, cleaned[(arrow + 2)..]);

        var counts = new Dictionary<char, int>();
        foreach (var c in left.Where(char.IsAsciiLetter))
            counts[c] = counts.GetValueOrDefault(c) + 1;

        var implicitOutput = string.Concat(counts.Where(kv => kv.Value == 1).Select(kv => kv.Key).Order());
        return (inputs, implicitOutput);
    }

    /// <summary>
    /// Works out the extent of every letter, checking that repeated letters agree.
    /// </summary>
    /// <remarks>
    /// This is where a genuine mistake surfaces. If <c>j</c> names a length-4 axis in one operand
    /// and a length-5 axis in another, the contraction is meaningless — and without the check it
    /// would quietly run over whichever came first.
    /// </remarks>
    private static Dictionary<char, int> MeasureAxes(string[] inputs, NdArray[] operands)
    {
        var extent = new Dictionary<char, int>();

        for (var op = 0; op < inputs.Length; op++)
            for (var axis = 0; axis < inputs[op].Length; axis++)
            {
                var letter = inputs[op][axis];
                var length = operands[op].Shape[axis];

                if (extent.TryGetValue(letter, out var known) && known != length)
                    throw new ArgumentException(
                        $"Subscript '{letter}' is {known} in one operand and {length} in another.");

                extent[letter] = length;
            }

        return extent;
    }

    /// <summary>Reads one element of an operand under the current letter assignment.</summary>
    private static double Read(NdArray operand, string subscript, Dictionary<char, int> assignment)
    {
        // A letter repeated within one operand — "ii" — selects the diagonal, because both axes
        // take the same index. That falls out of the assignment rather than needing a special case.
        var flat = 0;
        for (var axis = 0; axis < subscript.Length; axis++)
            flat = flat * operand.Shape[axis] + assignment[subscript[axis]];

        return operand.At(flat);
    }

    private static int Total(int[] sizes)
    {
        var total = 1;
        foreach (var size in sizes) total *= size;
        return total;
    }

    /// <summary>Turns a flat counter into one index per axis, row-major.</summary>
    private static void Decompose(int flat, int[] sizes, int[] into)
    {
        for (var axis = sizes.Length - 1; axis >= 0; axis--)
        {
            into[axis] = flat % sizes[axis];
            flat /= sizes[axis];
        }
    }
}
