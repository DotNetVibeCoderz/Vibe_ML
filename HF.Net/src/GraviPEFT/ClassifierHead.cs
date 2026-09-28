using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviPEFT;

/// <summary>
/// BERT's pooler: a dense layer and tanh over the <c>[CLS]</c> row. Frozen, from the checkpoint.
/// </summary>
/// <param name="Weight">Row-major <c>[hidden, hidden]</c>, in PyTorch's <c>(outputs, inputs)</c> order.</param>
/// <param name="Bias"><c>[hidden]</c>.</param>
internal sealed record Pooler(double[] Weight, double[] Bias)
{
    internal int Width => Bias.Length;
}

/// <summary>A trained linear layer named <c>classifier</c>, and the loss it is trained with.</summary>
/// <remarks>
/// Training targets come as one array per example: a single class index for a sequence head, a
/// label per token for a token head, where <see cref="Ignored"/> marks a position that does not
/// count - a special token, or a subword that continues a word.
/// </remarks>
internal abstract class LinearHead
{
    /// <summary>The target meaning "no label here", PyTorch's <c>ignore_index</c>.</summary>
    internal const int Ignored = -100;

    protected LinearHead(NdArray weight, NdArray bias)
    {
        Weight = weight;
        Bias = bias;
        WeightGradient = new double[Weight.Size];
        BiasGradient = new double[Bias.Size];
    }

    /// <summary>Draws the weights from a normal with standard deviation 0.02, the Transformers default.</summary>
    /// <remarks>
    /// Not zero: with LoRA's B at zero as well, every adapter gradient would be exactly zero on the
    /// first step.
    /// </remarks>
    protected static NdArray Initial(int classes, int hidden, int seed)
    {
        var weight = NdArray.Zeros(classes, hidden);
        var random = new GraviRandom(seed);
        var values = weight.AsSpan();
        for (var i = 0; i < values.Length; i++) values[i] = 0.02 * random.Normal();

        return weight;
    }

    protected static void Check(NdArray weight, NdArray bias)
    {
        if (weight.Rank != 2 || bias.Size != weight.Shape[0])
        {
            throw new InvalidDataException(
                $"A classifier weight of shape [{string.Join(", ", weight.Shape.ToArray())}] does not go "
                + $"with a bias of {bias.Size} values.");
        }
    }

    /// <summary><c>[classes, hidden]</c>. Named <c>classifier.weight</c> in the checkpoint.</summary>
    internal NdArray Weight { get; }

    /// <summary><c>[classes]</c>.</summary>
    internal NdArray Bias { get; }

    internal double[] WeightGradient { get; }

    internal double[] BiasGradient { get; }

    internal int Classes => Bias.Size;

    /// <summary>How many terms an example's targets add to the loss.</summary>
    internal abstract int Units(int[] targets);

    /// <summary>
    /// The summed cross-entropy of one example's targets, with its gradient - each term multiplied
    /// by <paramref name="weight"/> - accumulated into the head and returned for the hidden states.
    /// </summary>
    /// <returns>The unweighted loss sum, and <c>d loss / d hidden</c> shaped <c>[rows, width]</c>.</returns>
    internal abstract (double Loss, double[] HiddenGradient) Backward(
        double[] hidden, int rows, int width, int[] targets, double weight);

    /// <summary>Logits for one vector.</summary>
    internal double[] Logits(ReadOnlySpan<double> input)
    {
        var width = input.Length;
        var weights = Weight.AsSpan();
        var bias = Bias.AsSpan();
        var logits = new double[Classes];

        for (var c = 0; c < logits.Length; c++)
        {
            var sum = bias[c];
            for (var d = 0; d < width; d++) sum += weights[c * width + d] * input[d];
            logits[c] = sum;
        }

        return logits;
    }

    internal static double[] Softmax(double[] logits)
    {
        var max = logits.Max();
        var result = new double[logits.Length];
        var total = 0.0;

        for (var i = 0; i < logits.Length; i++)
        {
            result[i] = Math.Exp(logits[i] - max);
            total += result[i];
        }

        for (var i = 0; i < result.Length; i++) result[i] /= total;
        return result;
    }

    /// <summary>
    /// Cross-entropy of <paramref name="input"/> against <paramref name="target"/>, accumulating the
    /// head's gradients and adding the input's gradient, scaled by <paramref name="weight"/>, into
    /// <paramref name="inputGradient"/>.
    /// </summary>
    protected double Term(ReadOnlySpan<double> input, int target, double weight, Span<double> inputGradient)
    {
        var width = input.Length;
        var probabilities = Softmax(Logits(input));
        var weights = Weight.AsSpan();

        for (var c = 0; c < probabilities.Length; c++)
        {
            var dLogit = weight * (probabilities[c] - (c == target ? 1.0 : 0.0));
            BiasGradient[c] += dLogit;

            for (var d = 0; d < width; d++)
            {
                WeightGradient[c * width + d] += dLogit * input[d];
                inputGradient[d] += dLogit * weights[c * width + d];
            }
        }

        return -Math.Log(Math.Max(probabilities[target], double.Epsilon));
    }

    internal void ClearGradients()
    {
        Array.Clear(WeightGradient);
        Array.Clear(BiasGradient);
    }
}

/// <summary>
/// The sequence classifier trained alongside the adapters: one label per text.
/// </summary>
/// <remarks>
/// <para>
/// Two poolings, chosen by what the base checkpoint has.
/// </para>
/// <para>
/// <b>With a BERT pooler</b> the head is exactly <c>BertForSequenceClassification</c>'s: the
/// <c>[CLS]</c> row through the pretrained pooler, then <c>classifier</c>. The pooler stays frozen,
/// as PEFT leaves it, and only <c>classifier</c> is trained - so an adapter saved from here loads
/// into <c>AutoModelForSequenceClassification</c> in Python and predicts the same thing.
/// </para>
/// <para>
/// <b>Without one</b>, the rows are averaged, for the reason <c>Embed</c> averages them: on an
/// encoder that has not been fine-tuned, the bare <c>[CLS]</c> row is close to constant.
/// </para>
/// </remarks>
internal sealed class ClassifierHead : LinearHead
{
    internal ClassifierHead(int hidden, int classes, int seed, Pooler? pooler = null)
        : this(Initial(classes, hidden, seed), NdArray.Zeros(classes), pooler)
    {
    }

    private ClassifierHead(NdArray weight, NdArray bias, Pooler? pooler)
        : base(weight, bias)
    {
        if (pooler is not null && pooler.Width != weight.Shape[1])
        {
            throw new ArgumentException(
                $"The pooler is {pooler.Width} wide but the classifier reads {weight.Shape[1]} inputs.", nameof(pooler));
        }

        Pooler = pooler;
    }

    /// <summary>Wraps stored weights, from a saved adapter.</summary>
    internal static ClassifierHead FromWeights(NdArray weight, NdArray bias, Pooler? pooler)
    {
        Check(weight, bias);
        return new ClassifierHead(weight.AsContiguous(), bias.AsContiguous(), pooler);
    }

    /// <summary>The frozen BERT pooler, or <c>null</c> for mean pooling.</summary>
    internal Pooler? Pooler { get; }

    /// <summary>Mean of the rows of <paramref name="hidden"/>.</summary>
    internal static double[] MeanPool(double[] hidden, int rows, int width)
    {
        var pooled = new double[width];
        for (var r = 0; r < rows; r++)
        {
            for (var d = 0; d < width; d++) pooled[d] += hidden[r * width + d];
        }

        for (var d = 0; d < width; d++) pooled[d] /= rows;
        return pooled;
    }

    /// <summary>The vector the classifier reads: the pooler's output, or the mean row.</summary>
    internal double[] Pool(double[] hidden, int rows, int width)
    {
        if (Pooler is null) return MeanPool(hidden, rows, width);

        var first = hidden.AsSpan(0, width);
        var pooled = new double[width];

        for (var o = 0; o < width; o++)
        {
            var sum = Pooler.Bias[o];
            var row = Pooler.Weight.AsSpan(o * width, width);
            for (var i = 0; i < width; i++) sum += row[i] * first[i];
            pooled[o] = Math.Tanh(sum);
        }

        return pooled;
    }

    /// <summary>Class logits from the encoder's final hidden states.</summary>
    internal double[] Logits(double[] hidden, int rows, int width) => Logits(Pool(hidden, rows, width));

    internal override int Units(int[] targets) => 1;

    /// <summary>Cross-entropy of one example against one class.</summary>
    internal (double Loss, double[] HiddenGradient) Backward(double[] hidden, int rows, int width, int target, double weight)
        => Backward(hidden, rows, width, [target], weight);

    internal override (double Loss, double[] HiddenGradient) Backward(
        double[] hidden, int rows, int width, int[] targets, double weight)
    {
        var pooled = Pool(hidden, rows, width);
        var dPooled = new double[width];
        var loss = Term(pooled, targets[0], weight, dPooled);

        var dHidden = new double[rows * width];

        if (Pooler is null)
        {
            for (var r = 0; r < rows; r++)
            {
                for (var d = 0; d < width; d++) dHidden[r * width + d] = dPooled[d] / rows;
            }

            return (loss, dHidden);
        }

        // Through tanh, then back through the frozen dense layer into the [CLS] row alone.
        for (var o = 0; o < width; o++)
        {
            var dz = dPooled[o] * (1 - pooled[o] * pooled[o]);
            if (dz == 0) continue;

            var row = Pooler.Weight.AsSpan(o * width, width);
            for (var i = 0; i < width; i++) dHidden[i] += dz * row[i];
        }

        return (loss, dHidden);
    }
}

/// <summary>
/// The token classifier trained alongside the adapters: one label per token, for named entities.
/// </summary>
/// <remarks>
/// <c>classifier</c> over every row, which is all <c>BertForTokenClassification</c>,
/// <c>RobertaForTokenClassification</c> and <c>DistilBertForTokenClassification</c> have - no pooler
/// and no extra layer - so on every family a saved token head loads in Python.
/// </remarks>
internal sealed class TokenClassifierHead : LinearHead
{
    internal TokenClassifierHead(int hidden, int classes, int seed)
        : base(Initial(classes, hidden, seed), NdArray.Zeros(classes))
    {
    }

    private TokenClassifierHead(NdArray weight, NdArray bias)
        : base(weight, bias)
    {
    }

    /// <summary>Wraps stored weights, from a saved adapter.</summary>
    internal static TokenClassifierHead FromWeights(NdArray weight, NdArray bias)
    {
        Check(weight, bias);
        return new TokenClassifierHead(weight.AsContiguous(), bias.AsContiguous());
    }

    /// <summary>Logits for every row.</summary>
    internal double[][] Logits(double[] hidden, int rows, int width)
        => [.. Enumerable.Range(0, rows).Select(r => Logits(hidden.AsSpan(r * width, width)))];

    internal override int Units(int[] targets) => targets.Count(t => t != Ignored);

    internal override (double Loss, double[] HiddenGradient) Backward(
        double[] hidden, int rows, int width, int[] targets, double weight)
    {
        var dHidden = new double[rows * width];
        var loss = 0.0;

        for (var r = 0; r < rows && r < targets.Length; r++)
        {
            if (targets[r] == Ignored) continue;
            loss += Term(hidden.AsSpan(r * width, width), targets[r], weight, dHidden.AsSpan(r * width, width));
        }

        return (loss, dHidden);
    }
}

/// <summary>
/// The extractive question answering head trained alongside the adapters: a start and an end score
/// per token.
/// </summary>
/// <remarks>
/// <c>qa_outputs</c>, a linear layer from the hidden width to two, as in every family's
/// <c>ForQuestionAnswering</c> model. The loss is Transformers' own: a softmax over <i>positions</i>
/// for the start and another for the end, and the mean of the two cross-entropies. The targets are
/// token indices, with 0 - the <c>[CLS]</c> position - standing for an answer that is not in the
/// window, as Transformers' preprocessing marks it.
/// </remarks>
internal sealed class SpanHead : LinearHead
{
    internal SpanHead(int hidden, int seed)
        : base(Initial(2, hidden, seed), NdArray.Zeros(2))
    {
    }

    private SpanHead(NdArray weight, NdArray bias)
        : base(weight, bias)
    {
    }

    /// <summary>Wraps stored weights, from a saved adapter.</summary>
    internal static SpanHead FromWeights(NdArray weight, NdArray bias)
    {
        Check(weight, bias);
        if (weight.Shape[0] != 2)
        {
            throw new InvalidDataException($"qa_outputs has {weight.Shape[0]} outputs; a start and an end make two.");
        }

        return new SpanHead(weight.AsContiguous(), bias.AsContiguous());
    }

    /// <summary>Start and end scores for every row.</summary>
    internal (double[] Start, double[] End) Logits(double[] hidden, int rows, int width)
    {
        var start = new double[rows];
        var end = new double[rows];

        for (var r = 0; r < rows; r++)
        {
            var logits = Logits(hidden.AsSpan(r * width, width));
            start[r] = logits[0];
            end[r] = logits[1];
        }

        return (start, end);
    }

    internal override int Units(int[] targets) => 1;

    internal override (double Loss, double[] HiddenGradient) Backward(
        double[] hidden, int rows, int width, int[] targets, double weight)
    {
        var (startLogits, endLogits) = Logits(hidden, rows, width);
        var start = Softmax(startLogits);
        var end = Softmax(endLogits);
        var (first, last) = (targets[0], targets[1]);

        var loss = -(Math.Log(Math.Max(start[first], double.Epsilon)) + Math.Log(Math.Max(end[last], double.Epsilon))) / 2;

        var weights = Weight.AsSpan();
        var dHidden = new double[rows * width];

        for (var r = 0; r < rows; r++)
        {
            var dStart = weight * (start[r] - (r == first ? 1.0 : 0.0)) / 2;
            var dEnd = weight * (end[r] - (r == last ? 1.0 : 0.0)) / 2;

            BiasGradient[0] += dStart;
            BiasGradient[1] += dEnd;

            var row = hidden.AsSpan(r * width, width);
            var target = dHidden.AsSpan(r * width, width);

            for (var d = 0; d < width; d++)
            {
                WeightGradient[d] += dStart * row[d];
                WeightGradient[width + d] += dEnd * row[d];
                target[d] = dStart * weights[d] + dEnd * weights[width + d];
            }
        }

        return (loss, dHidden);
    }
}
