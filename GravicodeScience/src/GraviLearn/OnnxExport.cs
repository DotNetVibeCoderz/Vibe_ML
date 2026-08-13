using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviNum.Io;
using Gravicode.Science.GraviLearn.Decomposition;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.Preprocessing;

namespace Gravicode.Science.GraviLearn;

/// <summary>
/// Exports a fitted <see cref="Pipeline"/> to ONNX, so a model trained here can be served
/// anywhere that runs ONNX.
/// </summary>
/// <remarks>
/// <para>
/// Every step this supports is an affine map — subtract, divide, multiply by a matrix, add — which
/// is why the whole pipeline collapses into a handful of core ONNX operators. That is also the
/// limit: a decision tree or a k-nearest-neighbour model is not an affine map and cannot be
/// written this way, so those are refused rather than approximated.
/// </para>
/// <para>
/// The exported graph uses <c>float</c>, because ONNX runtimes support float32 universally and
/// float64 only patchily. Predictions therefore agree with this library to about single precision,
/// not to the last bit — see <see cref="Supports"/> and the tolerance used in the tests.
/// </para>
/// </remarks>
public static class OnnxExport
{
    /// <summary>Whether every step of <paramref name="pipeline"/> can be expressed in ONNX.</summary>
    public static bool Supports(Pipeline pipeline) => pipeline.Steps.All(IsSupported);

    /// <summary>The steps that cannot be exported, for an error message worth reading.</summary>
    public static IReadOnlyList<string> UnsupportedSteps(Pipeline pipeline)
        => [.. pipeline.Steps.Where(s => !IsSupported(s)).Select(s => s.GetType().Name)];

    private static bool IsSupported(object step) => step switch
    {
        StandardScaler => true,
        MinMaxScaler => true,
        PrincipalComponentAnalysis => true,

        // Ridge and Lasso derive from LinearRegression, so the base case covers them: the penalty
        // shapes the coefficients during fitting and leaves the prediction an ordinary affine map.
        LinearRegression => true,
        LogisticRegression => true,
        _ => false,
    };

    /// <summary>
    /// Writes <paramref name="pipeline"/> to an ONNX file.
    /// </summary>
    /// <param name="pipeline">A fitted pipeline.</param>
    /// <param name="path">Where to write the <c>.onnx</c> file.</param>
    /// <param name="features">Number of input columns the model expects.</param>
    /// <remarks>
    /// The batch dimension is left symbolic, so the exported model takes any number of rows —
    /// including one, which is what a serving endpoint actually sends.
    /// </remarks>
    public static void Save(Pipeline pipeline, string path, int features)
        => File.WriteAllBytes(path, Build(pipeline, features));

    /// <summary>Builds the ONNX model for <paramref name="pipeline"/> in memory.</summary>
    public static byte[] Build(Pipeline pipeline, int features)
    {
        if (!pipeline.IsFitted)
            throw new InvalidOperationException("The pipeline must be fitted before it can be exported.");

        var unsupported = UnsupportedSteps(pipeline);
        if (unsupported.Count > 0)
            throw new NotSupportedException(
                $"These steps have no ONNX form: {string.Join(", ", unsupported)}. "
                + "Only affine steps — scalers, PCA and linear models — can be exported.");

        var graph = new OnnxGraphBuilder("input", features);
        var current = graph.InputName;
        var columns = features;
        var outputIsInt64 = false;

        foreach (var step in pipeline.Steps)
        {
            switch (step)
            {
                case StandardScaler scaler:
                    // (x - mean) / scale, both broadcast across rows.
                    current = graph.AddNode("Sub",
                        [current, graph.AddInitializer("scaler_mean", scaler.Mean, scaler.Mean.Size)]);
                    current = graph.AddNode("Div",
                        [current, graph.AddInitializer("scaler_scale", scaler.Scale, scaler.Scale.Size)]);
                    break;

                case MinMaxScaler minMax:
                    current = ExportMinMax(graph, minMax, current);
                    break;

                case PrincipalComponentAnalysis pca:
                    // (x - mean) @ components^T. The transpose happens here rather than in the
                    // graph, because a constant is cheaper to store transposed than to transpose
                    // on every inference.
                    current = graph.AddNode("Sub",
                        [current, graph.AddInitializer("pca_mean", pca.Mean, pca.Mean.Size)]);
                    var componentsT = pca.ComponentVectors.T.Copy();
                    current = graph.AddNode("MatMul",
                        [current, graph.AddInitializer("pca_components", componentsT,
                            componentsT.Shape[0], componentsT.Shape[1])]);
                    columns = pca.ComponentVectors.Shape[0];
                    break;

                // Must come after LogisticRegression would, if it derived from this — it does not,
                // but Ridge and Lasso do, and they are handled here.
                case LinearRegression linear:
                    current = ExportLinear(graph, linear.Coefficients, linear.Intercept, current);
                    columns = 1;
                    break;

                case LogisticRegression logistic:
                    (current, columns) = ExportLogistic(graph, logistic, current);
                    outputIsInt64 = true;
                    break;
            }
        }

        return graph.Build(current, columns, outputIsInt64);
    }

    private static string ExportMinMax(OnnxGraphBuilder graph, MinMaxScaler scaler, string current)
    {
        // The fitted form is (x - dataMin) / range * (max - min) + min, which rearranges to a
        // single multiply and add — fewer nodes and identical arithmetic.
        var range = scaler.DataRange;
        var minimum = scaler.DataMinimum;
        var span = scaler.Maximum - scaler.Minimum;

        var slope = NdArray.Zeros(range.Size);
        var offset = NdArray.Zeros(range.Size);
        for (var i = 0; i < range.Size; i++)
        {
            var width = range.At(i) == 0 ? 1.0 : range.At(i);
            slope.SetAt(i, span / width);
            offset.SetAt(i, scaler.Minimum - minimum.At(i) * span / width);
        }

        current = graph.AddNode("Mul",
            [current, graph.AddInitializer("minmax_slope", slope, slope.Size)]);
        return graph.AddNode("Add",
            [current, graph.AddInitializer("minmax_offset", offset, offset.Size)]);
    }

    private static string ExportLinear(OnnxGraphBuilder graph, NdArray coefficients,
        double intercept, string current)
    {
        var column = coefficients.Reshape(coefficients.Size, 1).Copy();
        current = graph.AddNode("MatMul",
            [current, graph.AddInitializer("linear_coef", column, column.Shape[0], 1)]);

        return graph.AddNode("Add",
            [current, graph.AddInitializer("linear_intercept", NdArray.FromValues([intercept]), 1)]);
    }

    /// <summary>
    /// Exports a one-vs-rest logistic model as scores followed by an arg-max.
    /// </summary>
    /// <remarks>
    /// The sigmoid is deliberately omitted. It is monotonic, so it cannot change which class wins,
    /// and leaving it out saves a node and a little accuracy. A binary model is expanded to two
    /// columns — the negative class scoring zero — so that the same arg-max works for both cases
    /// and the exported output means the same thing either way.
    /// </remarks>
    private static (string Output, int Columns) ExportLogistic(
        OnnxGraphBuilder graph, LogisticRegression logistic, string current)
    {
        var coefficients = logistic.Classes.Count == 2
            ? logistic.Coefficients.Reshape(1, logistic.FeatureCount)
            : logistic.CoefficientMatrix;

        var weights = coefficients.T.Copy();
        current = graph.AddNode("MatMul",
            [current, graph.AddInitializer("logit_coef", weights, weights.Shape[0], weights.Shape[1])]);

        var intercepts = logistic.Intercepts;
        current = graph.AddNode("Add",
            [current, graph.AddInitializer("logit_intercept", intercepts, intercepts.Size)]);

        if (logistic.Classes.Count == 2)
        {
            // One score decides a binary problem; the arg-max needs two columns to choose between,
            // so a zero column is concatenated for the negative class.
            var zero = graph.AddInitializer("logit_zero", NdArray.Zeros(1), 1);
            var zeros = graph.AddNode("Mul", [current, zero]);
            current = graph.AddNode("Concat", [zeros, current], null,
                OnnxGraphBuilder.IntAttribute("axis", 1));
        }

        var output = graph.AddNode("ArgMax", [current], "label",
            OnnxGraphBuilder.IntAttribute("axis", 1),
            OnnxGraphBuilder.IntAttribute("keepdims", 0));

        return (output, 0);
    }
}
