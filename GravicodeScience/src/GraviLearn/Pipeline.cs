using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviLearn;

/// <summary>
/// Chains preprocessing steps and a final model into one fit/predict unit.
/// </summary>
/// <remarks>
/// The reason to use a pipeline rather than calling the steps by hand is leakage: inside
/// cross-validation the whole pipeline is refitted on each training fold, so the scaler learns
/// its mean from that fold alone. Scaling first and splitting afterwards quietly leaks test
/// statistics into training and produces scores that do not survive contact with new data.
/// </remarks>
public sealed class Pipeline
{
    private readonly List<(string Name, object Step)> _steps = [];

    /// <summary>Names of the steps, in order.</summary>
    public IReadOnlyList<string> StepNames => _steps.Select(s => s.Name).ToList();

    /// <summary>The steps themselves, in order.</summary>
    public IReadOnlyList<object> Steps => _steps.Select(s => s.Step).ToList();

    /// <summary>True once the pipeline has been fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>Appends a transformation step.</summary>
    public Pipeline Add(ITransformer transformer, string? name = null)
    {
        _steps.Add((name ?? transformer.GetType().Name, transformer));
        return this;
    }

    /// <summary>Appends the final model. Nothing may follow it.</summary>
    public Pipeline Add(IEstimator estimator, string? name = null)
    {
        _steps.Add((name ?? estimator.GetType().Name, estimator));
        return this;
    }

    /// <summary>Retrieves a step by name.</summary>
    public object this[string name] => _steps.FirstOrDefault(s => s.Name == name).Step
        ?? throw new KeyNotFoundException($"No step named '{name}'. Steps: {string.Join(", ", StepNames)}.");

    /// <summary>The final estimator, or <c>null</c> when the pipeline is transform-only.</summary>
    public IEstimator? FinalEstimator => _steps.Count > 0 ? _steps[^1].Step as IEstimator : null;

    /// <summary>Fits every step in order, feeding each one the output of the previous.</summary>
    public Pipeline Fit(NdArray x, NdArray y)
    {
        if (_steps.Count == 0) throw new InvalidOperationException("The pipeline has no steps.");

        var current = x;
        for (var i = 0; i < _steps.Count; i++)
        {
            switch (_steps[i].Step)
            {
                case IEstimator estimator when i == _steps.Count - 1:
                    estimator.Fit(current, y);
                    break;
                case ITransformer transformer:
                    transformer.Fit(current, y);
                    current = transformer.Transform(current);
                    break;
                case IEstimator:
                    throw new InvalidOperationException(
                        $"Step '{_steps[i].Name}' is an estimator but is not last; only the final step may be a model.");
            }
        }

        IsFitted = true;
        return this;
    }

    /// <summary>Fits a transform-only pipeline.</summary>
    public Pipeline Fit(NdArray x) => Fit(x, NdArray.Zeros(x.Shape[0]));

    /// <summary>Applies every transformation step, stopping before the final estimator.</summary>
    public NdArray Transform(NdArray x)
    {
        RequireFitted();
        var current = x;
        foreach (var (_, step) in _steps)
            if (step is ITransformer transformer) current = transformer.Transform(current);
        return current;
    }

    /// <summary>Fits, then transforms, in one call.</summary>
    public NdArray FitTransform(NdArray x, NdArray? y = null)
    {
        Fit(x, y ?? NdArray.Zeros(x.Shape[0]));
        return Transform(x);
    }

    /// <summary>Runs the transformations and then the final model.</summary>
    public NdArray Predict(NdArray x)
    {
        RequireFitted();
        var estimator = FinalEstimator
            ?? throw new InvalidOperationException("The pipeline's last step is not an estimator.");
        return estimator.Predict(Transform(x));
    }

    /// <summary>Class probabilities from the final model, when it supports them.</summary>
    public NdArray PredictProbabilities(NdArray x)
    {
        RequireFitted();
        if (FinalEstimator is not IClassifier classifier)
            throw new InvalidOperationException("The pipeline's final step is not a probabilistic classifier.");
        return classifier.PredictProbabilities(Transform(x));
    }

    /// <summary>Accuracy for classifiers, R-squared for regressors.</summary>
    public double Score(NdArray x, NdArray y)
    {
        var predictions = Predict(x);
        return FinalEstimator is IClassifier
            ? Metrics.Accuracy(y, predictions)
            : Metrics.R2Score(y, predictions);
    }

    /// <summary>
    /// A deep-enough copy for cross-validation: steps are rebuilt from their constructor
    /// arguments so each fold trains an independent model.
    /// </summary>
    public Pipeline CloneUnfitted(Func<object, object> stepFactory)
    {
        var clone = new Pipeline();
        foreach (var (name, step) in _steps)
        {
            var fresh = stepFactory(step);
            switch (fresh)
            {
                case ITransformer t: clone.Add(t, name); break;
                case IEstimator e: clone.Add(e, name); break;
                default: throw new InvalidOperationException($"Step factory returned an unusable object for '{name}'.");
            }
        }
        return clone;
    }

    private void RequireFitted()
    {
        if (!IsFitted) throw new InvalidOperationException("The pipeline must be fitted before use.");
    }

    /// <inheritdoc />
    public override string ToString() => $"Pipeline[{string.Join(" -> ", StepNames)}]";
}
