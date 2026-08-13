# GraviProb

*[Bahasa Indonesia](id/GraviProb.md)* · Probabilistic programming for .NET — distributions, MCMC, variational inference and probabilistic models.

## Distributions

Every distribution exposes `LogDensity` rather than a plain density, because inference multiplies
many densities together: in linear space a few hundred observations underflow to zero, while in
log space they simply add.

```csharp
using Gravicode.Science.GraviProb;

var normal = Distribution.Normal(mean: 0, stdDev: 1);
normal.LogDensity(1.5);   normal.Density(1.5);
normal.Sample(rng);       normal.Sample(rng, 10_000);
normal.Mean;  normal.Variance;  normal.Cdf(1.96);
normal.LogLikelihood(data);
```

Available: `Normal`, `Uniform`, `Bernoulli`, `Binomial`, `Poisson`, `Gamma`, `Beta`,
`Exponential`, `StudentT`, `LogNormal`, `HalfNormal`, `Categorical`.

`HalfNormal` is the usual weakly informative prior for a scale parameter. `Beta` exposes its
conjugacy directly:

```csharp
var exact = Distribution.Beta(1, 1).PosteriorAfter(successes: 125, failures: 75);
// Beta(126, 76) — the closed-form posterior, used to check the samplers
```

## Building a model

```csharp
var model = new BayesianModel()
    .AddDistribution("theta", Distribution.Beta(1, 1))
    .AddObservation("data", DistributionSpec.Binomial(200, "theta"), 125);
```

`DistributionSpec.Binomial(200, "theta")` records a **dependency** on the latent variable rather
than a fixed probability, so the sampler re-evaluates the likelihood as `theta` moves. Available
specs: `Binomial`, `Bernoulli`, `Normal`, `Poisson`, plus `DistributionSpec.From` for anything
else:

```csharp
DistributionSpec.From(v => new Gamma(v["shape"], v["rate"]), "shape", "rate");
```

The model is a joint log density: prior plus likelihood. It is known only **up to a constant**,
because the marginal likelihood in Bayes' rule is never computed — which is exactly why MCMC
works, since Metropolis-Hastings only looks at ratios and the constant cancels.

```csharp
model.LogPosterior(values);
model.LogPrior(values);
model.SampleFromPrior(rng);
model.PriorPredictive(draws: 1000);   // does the prior imply plausible data?
```

## Sampling

```csharp
var posterior = model.SampleMCMC(iterations: 20_000, chains: 4, warmup: 10_000, thin: 1, seed: 42);
var gibbs     = model.SampleGibbs(iterations: 20_000, chains: 4);
```

**Metropolis-Hastings** proposes a Gaussian step and accepts with probability
`min(1, exp(Δ logPosterior))`. During warmup the proposal scale adapts toward a **0.234**
acceptance rate — the asymptotically optimal value for a random walk. Too large a step is always
rejected and the chain stalls; too small a step is always accepted but explores nothing. Adaptation
**stops** when warmup ends, because a proposal that keeps changing breaks the Markov property.

**Metropolis within Gibbs** updates one coordinate at a time, targeting 0.44 acceptance. Higher
acceptance per move, but slow mixing when parameters are correlated, since single-coordinate steps
cannot travel along a diagonal ridge.

Chains run in parallel and are independent, which is both why multi-chain sampling is nearly free
and why R-hat is meaningful.

### Gradient-based sampling

```csharp
var nuts = model.SampleNUTS(iterations: 2000, chains: 4);   // reach for this one
var hmc  = model.SampleHMC(iterations: 2000, chains: 4, leapfrogSteps: 20);

model.IsDifferentiable;                    // false means these will throw
model.LogPosteriorGradient(values);        // (density, gradient) in ParameterNames order
```

**Hamiltonian Monte Carlo** simulates a physical trajectory using the gradient of the log density,
so proposals travel *across* the distribution instead of diffusing around it. **NUTS** removes the
last tuning knob by doubling the trajectory until the two ends start moving back towards each
other. Both adapt the step size by dual averaging toward 0.8 acceptance — far higher than a random
walk's 0.234, because a rejected trajectory wastes much more work than a rejected step.

The gradients come from the tape in [`GraviNum.Autodiff`](GraviNum.md#automatic-differentiation).
Every parameter is mapped to the whole real line first, and the transform is built on the tape too,
so the chain rule through it is never derived by hand. That is also what keeps every draw inside
the support.

### Which sampler, honestly

Measured on this machine, 4 chains, worst parameter reported:

| Model | Sampler | Wall clock | ESS/draw | ESS/sec | Worst R-hat |
|---|---|---:|---:|---:|---:|
| coin, 1 parameter | random walk | 58 ms | 16.5% | **11,340** | 1.006 |
| | NUTS | 669 ms | 44.5% | 2,663 | 1.000 |
| 20 parameters | random walk | 44 ms | 0.4% | 161 | **1.773** |
| | NUTS | 9,915 ms | **100.0%** | **202** | **1.000** |

On a one-parameter model the random walk is four times faster *per effective sample* and there is
no reason to reach for a gradient. Read the 20-parameter row carefully, though: the random walk
returns an R-hat of **1.773**, and anything above about 1.01 means the chains disagree and the
draws are not from the posterior. It is not producing worse samples quickly — it is producing
nothing usable, quickly. NUTS returns perfectly independent draws (100% of them effective) and
still wins on samples per second.

So: **random walk below a handful of parameters, NUTS above.** The crossover is the point where
diffusion stops being able to cross the posterior, not a point on a speed curve.

## Reading the posterior

```csharp
posterior["theta"];                  // all draws, pooled
posterior.Chain("theta", 0);         // one chain
posterior.Mean("theta");  posterior.StandardDeviation("theta");  posterior.Median("theta");

posterior.CredibleInterval("theta", 0.95);           // equal-tailed
posterior.HighestDensityInterval("theta", 0.95);     // shortest interval holding 95%

posterior.RHat("theta");                   // near 1 means converged; above ~1.01 does not
posterior.EffectiveSampleSize("theta");    // how many independent draws it is worth
posterior.AcceptanceRate;

Console.WriteLine(posterior.Summary());
```

Prefer the **HDI** on skewed posteriors: the equal-tailed interval can exclude the mode, the single
most probable value, while the HDI cannot.

### Posterior predictive checks

```csharp
var replicated = posterior.PosteriorPredictive(
    (values, rng) => rng.Binomial(200, values["theta"]), draws: 5000);
```

Can the fitted model reproduce the data it was fitted to? If the observed value sits far in the
tail of the replications, the model is wrong regardless of how well the chains converged.

## Variational inference

```csharp
var fit = model.FitVariational(iterations: 2000, learningRate: 0.05, monteCarloSamples: 8);
fit.Means["theta"];  fit.StandardDeviations["theta"];  fit.EvidenceLowerBound;
```

VI turns integration into optimisation, so it converges in milliseconds where MCMC takes seconds.
The price is the mean-field assumption: parameters are treated as independent, so it systematically
**understates posterior variance** and cannot represent correlations. Use it to iterate, then
confirm with MCMC.

Constrained parameters are mapped onto the whole real line before fitting — a logit for a Beta
parameter on (0,1), a log for a scale on (0,∞) — with the log Jacobian included in the objective.
Without that transform the optimiser walks straight out of the support and the fit is meaningless.

Gradients come from the autodiff tape once the model has at least 32 parameters, and from central
differences below that. That threshold is worth explaining, because the intuitive answer is
backwards: on a two-parameter model the tape measured **11× slower**, since building and walking a
graph costs more than four cheap scalar evaluations. Reverse mode's advantage is that one backward
pass covers every parameter, while finite differences need `2d` extra evaluations — so it wins by
growing dimension, not by being faster per call. The measured crossover is near 50 parameters.

## Bayesian networks

```csharp
using Gravicode.Science.GraviProb.Models;

var network = new BayesianNetwork()
    .AddVariable("rain", 0.8, 0.2)
    .AddVariable("sprinkler", 2, ["rain"], [[0.6, 0.4], [0.99, 0.01]])
    .AddVariable("wet", 2, ["rain", "sprinkler"],
    [
        [1.00, 0.00],   // no rain, no sprinkler
        [0.10, 0.90],   // no rain, sprinkler
        [0.20, 0.80],   // rain, no sprinkler
        [0.01, 0.99],   // both
    ]);

network.Infer("wet");                                                  // 0.4484
network.Infer("rain", new Dictionary<string, int> { ["wet"] = 1 });     // 0.3577
network.Infer("rain", new() { ["wet"] = 1, ["sprinkler"] = 1 });        // 0.0068
network.Sample(rng);
```

That last pair is **explaining away**: wet grass raises the probability of rain from 0.20 to 0.36,
but once the sprinkler accounts for it, rain drops to 0.007.

The factorisation is the point — a joint over *n* binary variables has `2^n - 1` free parameters,
while the product of per-variable conditionals is exponentially smaller when the graph is sparse.
Inference is exact enumeration: correct for small networks, exponential beyond them. Rows that do
not sum to one are rejected at construction.

## Hidden Markov models

```csharp
var hmm = new HiddenMarkovModel(initial, transitions, emissions);

hmm.LogLikelihood(observations);      // forward recursion, per-step scaling
hmm.Viterbi(observations);            // most probable path, log space
hmm.StatePosteriors(observations);    // forward-backward
hmm.Generate(length, rng);

var learned = HiddenMarkovModel.Random(states: 2, symbols: 3, seed: 42);
learned.Fit(sequences, iterations: 50);   // Baum-Welch (EM)
```

All three algorithms work in log space or with per-step normalisation, because the raw
probabilities underflow within a few dozen time steps.

Viterbi is **not** the same as taking the most probable state at each step independently: it
maximises the joint probability of the whole path, so it can never return a sequence containing a
transition the model forbids.

Baum-Welch guarantees the likelihood will not decrease, but finds only a **local** optimum — which
is why the initial parameters matter and a real fit uses several random restarts.

## Bayesian linear regression

```csharp
var model = new BayesianLinearRegression(priorPrecision: 1e-3, noisePrecision: 1.0).Fit(x, y);

model.CoefficientMeans;  model.CoefficientCovariance;
model.Predict(newX);
var (mean, deviation) = model.PredictWithUncertainty(newX);
var (lower, upper) = model.PredictInterval(newX, mass: 0.95);
model.SampleCoefficients(draws: 1000);
```

The normal-inverse-gamma prior is conjugate, so the posterior is closed form — no sampling, and the
answer is exact. The payoff over least squares is that every coefficient carries a full posterior,
and predictions carry an interval that **widens where the model has seen little data**, which a
point estimate cannot express.

## Common mistakes

| Symptom | Cause |
|---|---|
| "Could not find a starting point with finite posterior density" | The prior puts no mass where the likelihood is defined |
| `RHat` well above 1 | Not converged — sample longer, or reparameterise |
| Acceptance near 0 or 1 | Warmup was too short for the scale to adapt |
| VI variance looks too small | Expected — mean-field understates spread by construction |
| VI result outside the parameter's range | Fixed: parameters are transformed; check `SupportBounds` on a custom distribution |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
