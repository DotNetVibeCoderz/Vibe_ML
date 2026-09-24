using Gravicode.Science.GraviNum;
using Encoder = Gravicode.Science.GraviText.Transformers.TransformerModel;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>
/// The forward pass of a loaded text encoder, compiled into the shared kernels.
/// </summary>
/// <remarks>
/// <para>
/// The foundation's <see cref="Encoder"/> stays the model of record: the loader fills it, PEFT
/// merges adapters into it, and <c>TransformerModel.Encoder</c> exposes it for inspection. This is
/// a copy of its parameters in the layout the kernels want - float32 weights in
/// <c>(outputs, inputs)</c> order, query, key and value stacked - and it is what inference runs.
/// </para>
/// <para>
/// Two differences from running the foundation's layers directly, both deliberate and both making
/// the result closer to the reference, not further. The activation comes from the config's
/// <c>hidden_act</c>, so <c>gelu</c> is the exact GELU rather than the foundation's tanh
/// approximation. And each norm's epsilon comes from <c>layer_norm_eps</c> rather than the
/// foundation's fixed 1e-12, which is right for BERT and wrong for RoBERTa.
/// </para>
/// <para>
/// Because it is a copy, it goes stale if the foundation's parameters change. Anything that changes
/// them has to call <c>TransformerModel.WeightsChanged</c>; <c>PeftModel.Merge</c> does.
/// </para>
/// </remarks>
internal sealed class CompiledEncoder
{
    private readonly Encoder _source;
    private readonly Norm _embeddingNorm;
    private readonly EncoderBlock[] _blocks;

    private CompiledEncoder(Encoder source, Norm embeddingNorm, EncoderBlock[] blocks)
    {
        _source = source;
        _embeddingNorm = embeddingNorm;
        _blocks = blocks;
    }

    /// <summary>Copies an encoder's parameters into the kernels' layout.</summary>
    /// <exception cref="NotSupportedException">The config names an activation this cannot run.</exception>
    internal static CompiledEncoder Build(Encoder source, PretrainedConfig config)
    {
        var activation = Activation.For(config.Activation);
        var epsilon = config.LayerNormEpsilon;

        var blocks = new EncoderBlock[source.Layers.Count];
        Parallel.For(0, blocks.Length, i =>
        {
            var layer = source.Layers[i];

            blocks[i] = new EncoderBlock(
                NormOrder.Post,
                source.Config.Heads,
                Linear.From(layer.Attention.Query),
                Linear.From(layer.Attention.Key),
                Linear.From(layer.Attention.Value),
                Linear.From(layer.Attention.Output),
                Norm.From(layer.AttentionNorm, epsilon),
                Linear.From(layer.Intermediate),
                Linear.From(layer.OutputProjection),
                Norm.From(layer.OutputNorm, epsilon),
                activation);
        });

        return new CompiledEncoder(source, Norm.From(source.EmbeddingNorm, epsilon), blocks);
    }

    /// <summary>Runs the encoder, returning one hidden state per position.</summary>
    /// <param name="ids">Token ids.</param>
    /// <param name="mask">1 for a real token, 0 for padding.</param>
    /// <param name="typeIds">Segment per position, or <c>null</c> when every position is segment 0.</param>
    /// <param name="segmentDelta">
    /// <c>token_type_embeddings[1] - token_type_embeddings[0]</c>, added where the segment is 1.
    /// Segment 0 is already folded into the word embeddings.
    /// </param>
    internal NdArray Forward(int[] ids, int[] mask, int[]? typeIds = null, NdArray? segmentDelta = null)
    {
        var rows = ids.Length;
        var width = _source.Config.HiddenSize;
        var positions = _source.PositionEmbeddings;
        var tokens = _source.TokenEmbeddings;

        if (rows > positions.Shape[0])
        {
            throw new ArgumentException(
                $"The sequence is {rows} tokens but the model has {positions.Shape[0]} position "
                + "embeddings. Truncate it with maxLength.", nameof(ids));
        }

        var vocabulary = tokens.Shape[0];
        var hidden = new double[rows * width];

        for (var i = 0; i < rows; i++)
        {
            var id = Math.Clamp(ids[i], 0, vocabulary - 1);
            var segment = typeIds is not null && segmentDelta is not null && typeIds[i] != 0;

            for (var d = 0; d < width; d++)
            {
                // Before the embedding norm, which is where the reference adds the segment too.
                hidden[i * width + d] = tokens[id, d] + positions[i, d]
                    + (segment ? segmentDelta!.At(d) : 0.0);
            }
        }

        hidden = _embeddingNorm.Apply(hidden, rows);
        foreach (var block in _blocks) hidden = block.Forward(hidden, rows, mask);

        return new NdArray(hidden, [rows, width]);
    }
}
