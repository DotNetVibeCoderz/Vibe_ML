# GraviDiffusers

**Denoising diffusion: schedulers, and text-to-image over ONNX.**

Mirrors `diffusers`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviDiffusers;
```

## The noise schedule

Every sampler is a different way of walking the forward noise process backwards, so the schedule has
to match the one the weights were trained with or nothing else matters.

```csharp
NoiseSchedule.StableDiffusion;   // 1000 steps, 0.00085 -> 0.012, ScaledLinear
NoiseSchedule.Ddpm;              // 1000 steps, 1e-4 -> 0.02, Linear

new NoiseSchedule(trainTimesteps: 1000, betaStart: 0.00085, betaEnd: 0.012,
                  BetaSchedule.ScaledLinear);
```

| Schedule | Shape |
|---|---|
| `Linear` | Betas spaced linearly |
| `ScaledLinear` | The square of a linear ramp in `sqrt(beta)` - what Stable Diffusion trains with |
| `SquaredCosine` | Cosine, adding noise more gently at both ends |

**Using plain linear betas with Stable Diffusion's endpoints** produces images that are recognisably
structured and consistently washed out - easy to mistake for a bad prompt.

```csharp
schedule.Betas; schedule.Alphas; schedule.AlphasCumulative;
schedule.AlphaBar(t);    // 1.0 for t < 0, which lands the last step on a clean sample
schedule.Sigma(t);
```

## Samplers

```csharp
IScheduler scheduler = new DdimScheduler(eta: 0);   // deterministic
                     = new DdpmScheduler();          // the reference
                     = new EulerScheduler();         // good at 20-30 steps
```

**DDIM** reconstructs an estimate of the clean sample at every step and re-noises it to the next
level. That is what lets it skip timesteps: it stays consistent visiting 25 of 1000, where DDPM's
update assumes adjacent steps and degrades badly. At `eta = 0` it is fully deterministic, so a seed
and a prompt reproduce an image exactly.

**DDPM** is faithful and slow, one step per training timestep. It is here as the reference the
faster samplers are checked against.

**Euler** treats the reverse process as an ODE in the noise level and takes plain Euler steps along
it. That framing is why it produces usable images in twenty or thirty steps - the step count becomes
an integration-accuracy choice rather than a property of the trained schedule.

### How the coefficients are verified

Build `x_t` from a known `x₀` and a known noise, then let the model return that exact noise at every
step. DDIM's reconstruction is then exact at each step, so the trajectory must land back on `x₀` -
and it does so to **1e-9** only if `sqrt(abar)`, `sqrt(1 - abar)` and the direction term are all in
their right places. That test is in `tests/GraviDiffusers.Tests`.

Note that a *constant* noise prediction is not a contraction: DDIM amplifies by
`sqrt(abar₀ / abar_T)` along the way, roughly fifteen-fold on the Stable Diffusion schedule. That is
correct behaviour, not a bug - real models predict noise proportional to what is in the sample.

## Text to image

```csharp
using var pipeline = DiffusionPipeline.FromPretrained(
    "some-user/stable-diffusion-onnx",
    scheduler: new EulerScheduler());

using var image = pipeline.Generate(
    "a watercolour of a mountain village at dawn",
    new GenerationOptions(Steps: 25, GuidanceScale: 7.5, Width: 512, Height: 512, Seed: 42),
    onStep: (step, total) => Console.Write($"\r  {step}/{total}"));

image.Save("output.png");
```

### What it needs

A repository laid out the way the ONNX exports are:

```
text_encoder/model.onnx
unet/model.onnx
vae_decoder/model.onnx
```

A repository holding only the PyTorch weights **will not work** and says so on load rather than
failing later. Convert one with `optimum-cli export onnx --model <id> <out>`.

The three networks run through ONNX Runtime because a diffusion run evaluates the UNet tens of
times, and the managed `double` path is the wrong tool for that by two orders of magnitude.

### Things that will catch you

**Width and height must be multiples of 8.** The latent grid is eight times smaller than the image
in each dimension - that is where all the `/ 8` arithmetic comes from.

**A guidance scale above 1 runs the UNet twice per step**, once conditioned and once not. A scale of
exactly 1 is not merely a weak setting: it halves the work.

**The latent scaling factor is 0.18215.** It is applied on the way in and divided out before the VAE
decodes; forgetting either direction gives a washed-out or a saturated image.

## Reproducibility

```csharp
new GenerationOptions(Seed: 42)
```

The initial latent is drawn from a seeded `GraviRandom`, and DDIM at `eta = 0` and Euler are both
deterministic. The same prompt and seed give the same image every time - which is the whole basis of
sharing a generation rather than just a picture of one.

## See also

[GraviOptimum](GraviOptimum.md) · [GraviTokenizers](GraviTokenizers.md)
