using Gravicode.Science.GraviNum;

namespace Gravicode.Science.GraviProb;

/// <summary>The filtered or smoothed estimate of a hidden state at one time step.</summary>
/// <param name="Mean">The state estimate.</param>
/// <param name="Covariance">Its uncertainty.</param>
public readonly record struct StateEstimate(NdArray Mean, NdArray Covariance)
{
    /// <summary>Per-component standard deviations, from the covariance diagonal.</summary>
    public NdArray StandardDeviation
    {
        get
        {
            var result = NdArray.Zeros(Mean.Size);
            for (var i = 0; i < Mean.Size; i++) result.SetAt(i, Math.Sqrt(Math.Max(0, Covariance[i, i])));
            return result;
        }
    }
}

/// <summary>
/// A linear Gaussian state-space model, filtered and smoothed by the Kalman recursions.
/// </summary>
/// <remarks>
/// <para>
/// The model is a hidden state that evolves and an observation that sees part of it, both linearly
/// and both with Gaussian noise:
/// </para>
/// <code>
/// xₜ = F xₜ₋₁ + wₜ,   wₜ ~ N(0, Q)     the state, which is never observed
/// yₜ = H xₜ   + vₜ,   vₜ ~ N(0, R)     what is actually measured
/// </code>
/// <para>
/// Within those assumptions the Kalman filter is not a good method, it is <em>the</em> method: the
/// exact posterior over the state given everything observed so far, and the minimum-variance
/// estimator among all estimators, not merely linear ones. That exactness is why it is worth
/// keeping the model linear if the problem allows it.
/// </para>
/// <para>
/// A great deal fits this shape once written down. A local level model — <c>F = H = 1</c> — is an
/// exponentially weighted moving average whose smoothing constant is derived from the noise ratio
/// rather than guessed. Adding a slope component gives a linear trend that adapts. Tracking, sensor
/// fusion, and seasonal decomposition are all the same recursion with a different <c>F</c>.
/// </para>
/// <para>
/// <b>Filtering and smoothing answer different questions.</b> <see cref="Filter"/> estimates each
/// state from the past only, which is what a real-time system can do; <see cref="Smooth"/> uses the
/// whole series, which is strictly better and only available after the fact. Using smoothed states
/// to evaluate a forecasting rule is a look-ahead error, and a common one.
/// </para>
/// </remarks>
public sealed class KalmanFilter
{
    private readonly NdArray _transition;
    private readonly NdArray _observation;
    private readonly NdArray _processNoise;
    private readonly NdArray _observationNoise;

    /// <summary>Creates a filter from the four system matrices.</summary>
    /// <param name="transition">F, how the state evolves. Square, of the state dimension.</param>
    /// <param name="observation">H, how the state is measured. (observations × state).</param>
    /// <param name="processNoise">Q, the state noise covariance.</param>
    /// <param name="observationNoise">R, the measurement noise covariance.</param>
    /// <remarks>
    /// The ratio between Q and R is what the filter actually responds to. A large Q relative to R
    /// says the state moves faster than the sensor lies, and the filter tracks the measurements
    /// closely; the reverse says the sensor is noisy and the filter smooths hard. Only the ratio
    /// matters, which is why a filter can be tuned with one number.
    /// </remarks>
    public KalmanFilter(NdArray transition, NdArray observation, NdArray processNoise, NdArray observationNoise)
    {
        ArgumentNullException.ThrowIfNull(transition);
        ArgumentNullException.ThrowIfNull(observation);
        ArgumentNullException.ThrowIfNull(processNoise);
        ArgumentNullException.ThrowIfNull(observationNoise);

        if (transition.Rank != 2 || transition.Shape[0] != transition.Shape[1])
            throw new ArgumentException("The transition matrix must be square.", nameof(transition));

        StateDimension = transition.Shape[0];
        ObservationDimension = observation.Shape[0];

        if (observation.Rank != 2 || observation.Shape[1] != StateDimension)
            throw new ArgumentException(
                $"The observation matrix must have {StateDimension} columns.", nameof(observation));

        RequireSquare(processNoise, StateDimension, nameof(processNoise));
        RequireSquare(observationNoise, ObservationDimension, nameof(observationNoise));

        _transition = transition.Copy();
        _observation = observation.Copy();
        _processNoise = processNoise.Copy();
        _observationNoise = observationNoise.Copy();
    }

    /// <summary>
    /// A local level model: a random walk observed with noise.
    /// </summary>
    /// <remarks>
    /// The simplest useful state-space model, and equivalent to an exponentially weighted moving
    /// average — except that the smoothing weight follows from the noise ratio rather than being
    /// chosen, and the model reports its own uncertainty.
    /// </remarks>
    public static KalmanFilter LocalLevel(double processVariance, double observationVariance)
        => new(NdArray.Eye(1), NdArray.Eye(1),
            NdArray.Full(processVariance, 1, 1), NdArray.Full(observationVariance, 1, 1));

    /// <summary>
    /// A local linear trend: a level and a slope, both drifting.
    /// </summary>
    /// <remarks>
    /// The state is (level, slope) and the level accumulates the slope each step, so this
    /// extrapolates a trend rather than flattening out the way a local level model does. Only the
    /// level is observed — the slope is inferred entirely from how the level moves, which is what
    /// makes it a genuine latent variable rather than a computed difference.
    /// </remarks>
    public static KalmanFilter LocalLinearTrend(double levelVariance, double slopeVariance,
        double observationVariance)
    {
        var transition = NdArray.FromArray(new double[,] { { 1, 1 }, { 0, 1 } });
        var observation = NdArray.FromArray(new double[,] { { 1, 0 } });

        var processNoise = NdArray.Zeros(2, 2);
        processNoise[0, 0] = levelVariance;
        processNoise[1, 1] = slopeVariance;

        return new KalmanFilter(transition, observation, processNoise,
            NdArray.Full(observationVariance, 1, 1));
    }

    /// <summary>Dimension of the hidden state.</summary>
    public int StateDimension { get; }

    /// <summary>Dimension of an observation.</summary>
    public int ObservationDimension { get; }

    /// <summary>The result of a filtering pass.</summary>
    /// <param name="Filtered">The state estimate at each step, given observations up to that step.</param>
    /// <param name="Predicted">The one-step-ahead prediction made before each observation arrived.</param>
    /// <param name="LogLikelihood">
    /// The log likelihood of the whole series, accumulated from the one-step-ahead prediction errors.
    /// </param>
    public readonly record struct FilterResult(
        IReadOnlyList<StateEstimate> Filtered,
        IReadOnlyList<StateEstimate> Predicted,
        double LogLikelihood);

    /// <summary>
    /// Runs the forward pass, estimating each state from the observations up to it.
    /// </summary>
    /// <param name="observations">One observation per row.</param>
    /// <param name="initialMean">The prior state mean. Defaults to zero.</param>
    /// <param name="initialCovariance">
    /// The prior state covariance. Defaults to a large diagonal, which says the initial state is
    /// essentially unknown and lets the data determine it within a few steps.
    /// </param>
    /// <remarks>
    /// Each step is predict-then-update. The prediction pushes the state through F and adds Q, which
    /// always increases uncertainty; the update folds in the observation, which always decreases it.
    /// The Kalman gain in between is the ratio that decides how much to believe the measurement, and
    /// it is derived rather than tuned.
    /// </remarks>
    public FilterResult Filter(NdArray observations, NdArray? initialMean = null, NdArray? initialCovariance = null)
    {
        ArgumentNullException.ThrowIfNull(observations);
        if (observations.Rank != 2)
            throw new ArgumentException("Observations must be a rank 2 array of shape (time, observation).");
        if (observations.Shape[1] != ObservationDimension)
            throw new ArgumentException(
                $"Each observation must have {ObservationDimension} components, not {observations.Shape[1]}.");

        var steps = observations.Shape[0];

        var mean = initialMean?.Copy() ?? NdArray.Zeros(StateDimension);
        var covariance = initialCovariance?.Copy() ?? Diagonal(StateDimension, 1e6);

        var filtered = new List<StateEstimate>(steps);
        var predicted = new List<StateEstimate>(steps);
        var logLikelihood = 0.0;

        for (var t = 0; t < steps; t++)
        {
            // Predict: push the state forward and let the uncertainty grow.
            var priorMean = MatrixVector(_transition, mean);
            var priorCovariance = Add(Sandwich(_transition, covariance), _processNoise);

            predicted.Add(new StateEstimate(priorMean.Copy(), priorCovariance.Copy()));

            // The innovation: how far the observation fell from where it was expected.
            var expected = MatrixVector(_observation, priorMean);
            var innovation = NdArray.Zeros(ObservationDimension);
            for (var i = 0; i < ObservationDimension; i++)
                innovation.SetAt(i, observations[t, i] - expected.At(i));

            var innovationCovariance = Add(Sandwich(_observation, priorCovariance), _observationNoise);

            // The likelihood contribution, which is what makes parameter fitting possible: the
            // series is decomposed into independent one-step-ahead prediction errors.
            logLikelihood += new MultivariateNormal(
                NdArray.Zeros(ObservationDimension), innovationCovariance).LogDensity(innovation);

            // Gain: K = P Hᵀ S⁻¹, computed as a solve. Sᵀ is used because Solve works on the left.
            var crossCovariance = LinAlg.Dot(priorCovariance, Transpose(_observation));
            var gain = Transpose(LinAlg.Solve(innovationCovariance, Transpose(crossCovariance)));

            var correction = MatrixVector(gain, innovation);
            mean = NdArray.Zeros(StateDimension);
            for (var i = 0; i < StateDimension; i++) mean.SetAt(i, priorMean.At(i) + correction.At(i));

            // Joseph form: (I − KH) P (I − KH)ᵀ + K R Kᵀ. Algebraically the same as the short
            // P − KHP, and numerically far better behaved — the short form can drift into an
            // asymmetric or negative-definite covariance over a long series, and then the filter
            // diverges with no warning.
            var factor = Subtract(NdArray.Eye(StateDimension), LinAlg.Dot(gain, _observation));
            covariance = Add(Sandwich(factor, priorCovariance), Sandwich(gain, _observationNoise));
            Symmetrise(covariance);

            filtered.Add(new StateEstimate(mean.Copy(), covariance.Copy()));
        }

        return new FilterResult(filtered, predicted, logLikelihood);
    }

    /// <summary>
    /// Runs the backward pass, re-estimating every state using the whole series.
    /// </summary>
    /// <remarks>
    /// The Rauch–Tung–Striebel recursion. It walks backwards from the last filtered state,
    /// correcting each earlier estimate by how much the future disagreed with the prediction made
    /// from it. Smoothed estimates are never worse than filtered ones and are usually visibly
    /// better in the early steps, where the filter had almost no data.
    /// </remarks>
    public IReadOnlyList<StateEstimate> Smooth(NdArray observations,
        NdArray? initialMean = null, NdArray? initialCovariance = null)
    {
        var result = Filter(observations, initialMean, initialCovariance);
        var steps = result.Filtered.Count;
        if (steps == 0) return [];

        var smoothed = new StateEstimate[steps];
        smoothed[^1] = result.Filtered[^1];

        for (var t = steps - 2; t >= 0; t--)
        {
            var filtered = result.Filtered[t];
            var prior = result.Predicted[t + 1];

            // J = P_t Fᵀ P_{t+1|t}⁻¹, again as a solve rather than an inverse.
            var cross = LinAlg.Dot(filtered.Covariance, Transpose(_transition));
            var gain = Transpose(LinAlg.Solve(prior.Covariance, Transpose(cross)));

            var meanGap = NdArray.Zeros(StateDimension);
            for (var i = 0; i < StateDimension; i++)
                meanGap.SetAt(i, smoothed[t + 1].Mean.At(i) - prior.Mean.At(i));

            var correction = MatrixVector(gain, meanGap);
            var mean = NdArray.Zeros(StateDimension);
            for (var i = 0; i < StateDimension; i++) mean.SetAt(i, filtered.Mean.At(i) + correction.At(i));

            var covarianceGap = Subtract(smoothed[t + 1].Covariance, prior.Covariance);
            var covariance = Add(filtered.Covariance, Sandwich(gain, covarianceGap));
            Symmetrise(covariance);

            smoothed[t] = new StateEstimate(mean, covariance);
        }

        return smoothed;
    }

    /// <summary>
    /// Projects the state forward beyond the observed series.
    /// </summary>
    /// <remarks>
    /// No observations arrive, so only the predict step runs and the uncertainty grows monotonically
    /// — which is the honest answer, and the reason a state-space forecast comes with a fan rather
    /// than a line.
    /// </remarks>
    public IReadOnlyList<StateEstimate> Forecast(NdArray observations, int horizon,
        NdArray? initialMean = null, NdArray? initialCovariance = null)
    {
        if (horizon <= 0) throw new ArgumentOutOfRangeException(nameof(horizon));

        var filtered = Filter(observations, initialMean, initialCovariance).Filtered;
        if (filtered.Count == 0) throw new ArgumentException("There is nothing to forecast from.");

        var mean = filtered[^1].Mean.Copy();
        var covariance = filtered[^1].Covariance.Copy();

        var forecasts = new List<StateEstimate>(horizon);

        for (var h = 0; h < horizon; h++)
        {
            mean = MatrixVector(_transition, mean);
            covariance = Add(Sandwich(_transition, covariance), _processNoise);
            forecasts.Add(new StateEstimate(mean.Copy(), covariance.Copy()));
        }

        return forecasts;
    }

    /// <summary>The observations a sequence of states would produce, without noise.</summary>
    public NdArray Observe(IReadOnlyList<StateEstimate> states)
    {
        var result = NdArray.Zeros(states.Count, ObservationDimension);

        for (var t = 0; t < states.Count; t++)
        {
            var value = MatrixVector(_observation, states[t].Mean);
            for (var i = 0; i < ObservationDimension; i++) result[t, i] = value.At(i);
        }

        return result;
    }

    /// <summary>
    /// Simulates a series from the model.
    /// </summary>
    /// <returns>The hidden states and the noisy observations they produced.</returns>
    /// <remarks>
    /// Mostly useful for checking a filter against a truth it cannot see, which is the only way to
    /// tell a well-tuned filter from a confidently wrong one.
    /// </remarks>
    public (NdArray States, NdArray Observations) Simulate(int steps, GraviRandom rng, NdArray? initialState = null)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (steps <= 0) throw new ArgumentOutOfRangeException(nameof(steps));

        var stateNoise = new MultivariateNormal(NdArray.Zeros(StateDimension), _processNoise);
        var measurementNoise = new MultivariateNormal(NdArray.Zeros(ObservationDimension), _observationNoise);

        var states = NdArray.Zeros(steps, StateDimension);
        var observations = NdArray.Zeros(steps, ObservationDimension);

        var state = initialState?.Copy() ?? NdArray.Zeros(StateDimension);

        for (var t = 0; t < steps; t++)
        {
            var moved = MatrixVector(_transition, state);
            var noise = stateNoise.Sample(rng);

            state = NdArray.Zeros(StateDimension);
            for (var i = 0; i < StateDimension; i++) state.SetAt(i, moved.At(i) + noise.At(i));

            var measured = MatrixVector(_observation, state);
            var error = measurementNoise.Sample(rng);

            for (var i = 0; i < StateDimension; i++) states[t, i] = state.At(i);
            for (var i = 0; i < ObservationDimension; i++) observations[t, i] = measured.At(i) + error.At(i);
        }

        return (states, observations);
    }

    // ------------------------------------------------------------------ helpers

    private static void RequireSquare(NdArray matrix, int dimension, string name)
    {
        if (matrix.Rank != 2 || matrix.Shape[0] != dimension || matrix.Shape[1] != dimension)
            throw new ArgumentException($"{name} must be {dimension}x{dimension}.", name);
    }

    private static NdArray Diagonal(int dimension, double value)
    {
        var result = NdArray.Zeros(dimension, dimension);
        for (var i = 0; i < dimension; i++) result[i, i] = value;
        return result;
    }

    private static NdArray MatrixVector(NdArray matrix, NdArray vector)
    {
        var result = NdArray.Zeros(matrix.Shape[0]);
        for (var i = 0; i < matrix.Shape[0]; i++)
        {
            var sum = 0.0;
            for (var j = 0; j < matrix.Shape[1]; j++) sum += matrix[i, j] * vector.At(j);
            result.SetAt(i, sum);
        }
        return result;
    }

    /// <summary>The congruence <c>A B Aᵀ</c>, which is how a covariance moves through a linear map.</summary>
    private static NdArray Sandwich(NdArray a, NdArray b) => LinAlg.Dot(LinAlg.Dot(a, b), Transpose(a));

    private static NdArray Add(NdArray a, NdArray b)
    {
        var result = NdArray.Zeros(a.Shape[0], a.Shape[1]);
        for (var i = 0; i < a.Shape[0]; i++)
            for (var j = 0; j < a.Shape[1]; j++)
                result[i, j] = a[i, j] + b[i, j];
        return result;
    }

    private static NdArray Subtract(NdArray a, NdArray b)
    {
        var result = NdArray.Zeros(a.Shape[0], a.Shape[1]);
        for (var i = 0; i < a.Shape[0]; i++)
            for (var j = 0; j < a.Shape[1]; j++)
                result[i, j] = a[i, j] - b[i, j];
        return result;
    }

    private static NdArray Transpose(NdArray a)
    {
        var result = NdArray.Zeros(a.Shape[1], a.Shape[0]);
        for (var i = 0; i < a.Shape[0]; i++)
            for (var j = 0; j < a.Shape[1]; j++)
                result[j, i] = a[i, j];
        return result;
    }

    /// <summary>Averages a covariance with its transpose, undoing rounding asymmetry.</summary>
    private static void Symmetrise(NdArray a)
    {
        for (var i = 0; i < a.Shape[0]; i++)
            for (var j = i + 1; j < a.Shape[1]; j++)
            {
                var average = 0.5 * (a[i, j] + a[j, i]);
                a[i, j] = average;
                a[j, i] = average;
            }
    }
}
