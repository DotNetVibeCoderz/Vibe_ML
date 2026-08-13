using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn;

/// <summary>
/// A stateful feature transformation: learn parameters from training data, then apply them.
/// </summary>
/// <remarks>
/// The fit/transform split exists to prevent leakage: a scaler must learn its mean from the
/// training split only and then apply that same mean to the test split. Calling
/// <see cref="Transform"/> before <see cref="Fit"/> is an error rather than a silent no-op.
/// </remarks>
public interface ITransformer
{
    /// <summary>Learns the transformation's parameters from <paramref name="x"/>.</summary>
    void Fit(NdArray x, NdArray? y = null);

    /// <summary>Applies the learned transformation.</summary>
    NdArray Transform(NdArray x);

    /// <summary>True once <see cref="Fit"/> has run.</summary>
    bool IsFitted { get; }
}

/// <summary>A supervised model that learns a mapping from features to a target.</summary>
public interface IEstimator
{
    /// <summary>Trains the model.</summary>
    void Fit(NdArray x, NdArray y);

    /// <summary>Predicts the target for each row of <paramref name="x"/>.</summary>
    NdArray Predict(NdArray x);

    /// <summary>True once <see cref="Fit"/> has run.</summary>
    bool IsFitted { get; }
}

/// <summary>A supervised model that can also report class probabilities.</summary>
public interface IClassifier : IEstimator
{
    /// <summary>Class probabilities, one row per sample and one column per class.</summary>
    NdArray PredictProbabilities(NdArray x);

    /// <summary>The distinct class labels seen during training, in ascending order.</summary>
    IReadOnlyList<double> Classes { get; }
}

/// <summary>A model that assigns each sample to a cluster without supervision.</summary>
public interface IClusterer
{
    /// <summary>Learns the cluster structure.</summary>
    void Fit(NdArray x);

    /// <summary>Assigns each row of <paramref name="x"/> to a cluster.</summary>
    NdArray Predict(NdArray x);

    /// <summary>Cluster assignment for the training data.</summary>
    NdArray Labels { get; }
}

/// <summary>Shared plumbing: fitted-state tracking and input validation.</summary>
public abstract class ModelBase
{
    /// <summary>True once the model has been fitted.</summary>
    public bool IsFitted { get; protected set; }

    /// <summary>Number of features seen during training.</summary>
    public int FeatureCount { get; protected set; }

    /// <summary>Throws when the model has not been fitted yet.</summary>
    protected void RequireFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException($"{GetType().Name} must be fitted before use. Call Fit first.");
    }

    /// <summary>Validates a feature matrix and returns its sample count.</summary>
    protected static int ValidateMatrix(NdArray x, string name = "x")
    {
        if (x.Rank != 2)
            throw new ArgumentException($"{name} must be a rank 2 array of shape (samples, features), got rank {x.Rank}.");
        if (x.Shape[0] == 0)
            throw new ArgumentException($"{name} has no samples.");
        return x.Shape[0];
    }

    /// <summary>Validates that features and targets line up, and that the feature count matches training.</summary>
    protected void ValidateForPrediction(NdArray x)
    {
        ValidateMatrix(x);
        if (x.Shape[1] != FeatureCount)
            throw new ArgumentException(
                $"Model was fitted on {FeatureCount} features but received {x.Shape[1]}.");
    }

    /// <summary>Validates that <paramref name="y"/> has one entry per row of <paramref name="x"/>.</summary>
    protected static void ValidateTarget(NdArray x, NdArray y)
    {
        if (y.Size != x.Shape[0])
            throw new ArgumentException($"Target has {y.Size} entries but the feature matrix has {x.Shape[0]} rows.");
    }

    /// <summary>The sorted, distinct labels in a target vector.</summary>
    protected static double[] DistinctClasses(NdArray y)
        => y.ToArray().Distinct().OrderBy(v => v).ToArray();
}
