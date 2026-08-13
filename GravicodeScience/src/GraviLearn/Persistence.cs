using System.Text.Json;
using System.Text.Json.Serialization;
using Gravicode.Science.GraviLearn.Linear;
using Gravicode.Science.GraviLearn.Clustering;
using Gravicode.Science.GraviLearn.Preprocessing;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn;

/// <summary>
/// Saves and reloads fitted model parameters as JSON.
/// </summary>
/// <remarks>
/// Parameters are written explicitly rather than by serialising the object graph. That keeps the
/// format readable and version-tolerant - a saved model can be inspected, diffed and loaded by a
/// later build - and it avoids binary serialisation, which is both a compatibility and a security
/// hazard. Models whose state is a full tree ensemble are saved through their own structure below.
/// </remarks>
public static class ModelPersistence
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>A saved model: its type, a schema version and its parameters.</summary>
    /// <param name="ModelType">Simple type name of the model.</param>
    /// <param name="Version">Format version.</param>
    /// <param name="FeatureCount">Features the model was fitted on.</param>
    /// <param name="Parameters">Named numeric parameters.</param>
    /// <param name="Matrices">Named matrices, stored as shape plus flattened data.</param>
    public sealed record ModelSnapshot(
        string ModelType,
        int Version,
        int FeatureCount,
        Dictionary<string, double[]> Parameters,
        Dictionary<string, MatrixPayload> Matrices);

    /// <summary>A matrix in a form JSON can hold.</summary>
    /// <param name="Shape">Dimensions.</param>
    /// <param name="Data">Row-major values.</param>
    public sealed record MatrixPayload(int[] Shape, double[] Data);

    /// <summary>Writes a snapshot to disk.</summary>
    public static void Save(ModelSnapshot snapshot, string path)
        => File.WriteAllText(path, JsonSerializer.Serialize(snapshot, Options));

    /// <summary>Reads a snapshot from disk.</summary>
    public static ModelSnapshot Load(string path)
        => JsonSerializer.Deserialize<ModelSnapshot>(File.ReadAllText(path), Options)
           ?? throw new InvalidDataException($"'{path}' does not contain a model snapshot.");

    /// <summary>Captures a linear regression's coefficients.</summary>
    public static ModelSnapshot Capture(LinearRegression model) => new(
        nameof(LinearRegression), 1, model.FeatureCount,
        new Dictionary<string, double[]>
        {
            ["coefficients"] = model.Coefficients.ToArray(),
            ["intercept"] = [model.Intercept],
        },
        []);

    /// <summary>Captures a logistic regression's coefficients and class list.</summary>
    public static ModelSnapshot Capture(LogisticRegression model) => new(
        nameof(LogisticRegression), 1, model.FeatureCount,
        new Dictionary<string, double[]>
        {
            ["classes"] = model.Classes.ToArray(),
            ["intercept"] = [model.Intercept],
        },
        new Dictionary<string, MatrixPayload>
        {
            ["coefficients"] = ToPayload(model.CoefficientMatrix),
        });

    /// <summary>Captures a scaler's learned mean and scale.</summary>
    public static ModelSnapshot Capture(StandardScaler scaler) => new(
        nameof(StandardScaler), 1, scaler.FeatureCount,
        new Dictionary<string, double[]>
        {
            ["mean"] = scaler.Mean.ToArray(),
            ["scale"] = scaler.Scale.ToArray(),
        },
        []);

    /// <summary>Captures a k-means model's centroids.</summary>
    public static ModelSnapshot Capture(KMeans model) => new(
        nameof(KMeans), 1, model.FeatureCount,
        new Dictionary<string, double[]> { ["inertia"] = [model.Inertia] },
        new Dictionary<string, MatrixPayload> { ["centroids"] = ToPayload(model.Centroids) });

    /// <summary>Rebuilds a matrix from a snapshot entry.</summary>
    public static NdArray FromPayload(MatrixPayload payload) => new(payload.Data, payload.Shape);

    /// <summary>Converts a matrix into a snapshot entry.</summary>
    public static MatrixPayload ToPayload(NdArray array) => new(array.Shape.ToArray(), array.ToArray());
}
