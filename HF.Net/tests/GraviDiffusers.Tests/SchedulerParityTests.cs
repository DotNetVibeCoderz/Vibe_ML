using Gravicode.HFNet.GraviDiffusers;
using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.HFNet.GraviDiffusers.Tests;

/// <summary>
/// The configured schedulers against diffusers 0.40 itself.
/// </summary>
/// <remarks>
/// Every expected value came from diffusers' <c>DDIMScheduler</c> and <c>EulerDiscreteScheduler</c>
/// with Stable Diffusion 1.x's configuration (scaled-linear betas 0.00085-0.012,
/// <c>steps_offset</c> 1, <c>set_alpha_to_one</c> false). diffusers keeps its betas in float32, so
/// agreement is to about 1e-6 relative rather than to the last digit.
/// </remarks>
public sealed class SchedulerParityTests
{
    private static readonly SchedulerConfig StableDiffusion = SchedulerConfig.StableDiffusion;

    [Theory]
    [InlineData(TimestepSpacing.Leading, new[] { 901, 801, 701, 601, 501, 401, 301, 201, 101, 1 })]
    [InlineData(TimestepSpacing.Linspace, new[] { 999, 888, 777, 666, 555, 444, 333, 222, 111, 0 })]
    [InlineData(TimestepSpacing.Trailing, new[] { 999, 899, 799, 699, 599, 499, 399, 299, 199, 99 })]
    public void Ddim_visits_the_reference_timesteps(TimestepSpacing spacing, int[] expected)
        => Assert.Equal(expected, new DdimScheduler(StableDiffusion with { Spacing = spacing }).SetTimesteps(10));

    [Fact]
    public void Euler_keeps_fractional_linspace_timesteps_and_interpolates_their_sigmas()
    {
        var euler = new EulerScheduler(StableDiffusion with { Spacing = TimestepSpacing.Linspace });
        euler.SetTimesteps(7);

        double[] timesteps = [999.0, 832.5, 666.0, 499.5, 333.0, 166.5, 0.0];
        double[] sigmas = [14.614646912, 5.948918343, 2.918308496, 1.61558342, 0.932358325, 0.493553519, 0.029167533, 0.0];

        for (var i = 0; i < timesteps.Length; i++) Assert.Equal(timesteps[i], euler.ModelTimestep(i), 9);
        for (var i = 0; i < sigmas.Length; i++) Assert.True(Math.Abs(euler.Sigmas[i] - sigmas[i]) < 2e-6 * Math.Max(1, sigmas[i]), $"sigma[{i}] {euler.Sigmas[i]} against {sigmas[i]}");

        Assert.True(Math.Abs(euler.InitialNoiseScale - 14.614646911621094) < 2e-5, "linspace starts at sigma_max");
    }

    [Fact]
    public void Euler_with_leading_spacing_starts_at_the_hypotenuse_of_sigma_max()
    {
        var euler = new EulerScheduler(StableDiffusion);
        euler.SetTimesteps(10);

        Assert.True(Math.Abs(euler.Sigmas[0] - 8.3906869888) < 2e-5);
        Assert.True(Math.Abs(euler.InitialNoiseScale - 8.4500665665) < 2e-5, $"{euler.InitialNoiseScale}");
    }

    private static readonly double[] Sample = [1.5409961082440433, -0.2934289057609464, -2.1787893820745574, 0.5684312772806678, -1.084522342424021, -1.3985953953708767, 0.4033468476292993, 0.8380263329976598];
    private static readonly double[] Output = [-0.7192575784693592, -0.40334352493217457, -0.5966353626151273, 0.18203648506130554, -0.8566745932963743, 1.100604170903427, -1.0711873631091473, 0.12270123916331216];

    private static NdArray Tensor(double[] values) => new([.. values], 1, 2, 2, 2);

    private static void Close(double[] expected, NdArray actual, string what)
    {
        for (var i = 0; i < expected.Length; i++)
        {
            var tolerance = 5e-6 * Math.Max(1, Math.Abs(expected[i]));
            Assert.True(Math.Abs(actual.At(i) - expected[i]) < tolerance, $"{what}[{i}]: {actual.At(i)} against {expected[i]}");
        }
    }

    [Theory]
    [InlineData(PredictionType.Epsilon, new[] { 19.027515669948, 0.892691273976, -13.416432950434, 3.279786087284, -2.000334467285, -21.012013244735, 12.359806276166, 6.052836014767 })]
    [InlineData(PredictionType.VPrediction, new[] { 2.30920596302, 0.275863632038, -1.127930965812, 0.259334113118, 0.179808315733, -2.688895626464, 1.773762796339, 0.57525706438 })]
    [InlineData(PredictionType.Sample, new[] { -0.671206719965, -0.410386066456, -0.658281041758, 0.198016551458, -0.885176892383, 1.055247229253, -1.055167092253, 0.146828405873 })]
    public void Ten_ddim_steps_land_where_diffusers_lands(PredictionType prediction, double[] expected)
    {
        // A fixed model output at every step: each of the ten steps compounds any error in the
        // timesteps, the final alpha or the prediction-type algebra.
        var ddim = new DdimScheduler(StableDiffusion with { Prediction = prediction });
        ddim.SetTimesteps(10);

        var sample = Tensor(Sample);
        for (var step = 0; step < 10; step++) sample = ddim.Step(Tensor(Output), step, sample);

        Close(expected, sample, $"ddim {prediction}");
    }

    [Theory]
    [InlineData(PredictionType.Epsilon, new[] { 33.032812160017, 1.606363544102, -23.1226218036, 5.647023227575, -3.329913990961, -36.524918833436, 21.549794765481, 10.454224833764 })]
    [InlineData(PredictionType.VPrediction, new[] { 2.322691622108, 0.305019750194, -1.05225602524, 0.238042194903, 0.250339074157, -2.727428654228, 1.829133907689, 0.552420060524 })]
    public void Ten_euler_steps_land_where_diffusers_lands(PredictionType prediction, double[] expected)
    {
        // EulerDiscreteScheduler's own default spacing is linspace, which is what produced these.
        var euler = new EulerScheduler(StableDiffusion with { Prediction = prediction, Spacing = TimestepSpacing.Linspace });
        euler.SetTimesteps(10);

        var sample = Tensor([.. Sample.Select(v => v * euler.InitialNoiseScale)]);
        for (var step = 0; step < 10; step++) sample = euler.Step(Tensor(Output), step, sample);

        Close(expected, sample, $"euler {prediction}");
    }

    [Fact]
    public void Adding_noise_matches_the_forward_process_in_both_parameterisations()
    {
        var ddim = new DdimScheduler(StableDiffusion);
        ddim.SetTimesteps(10);
        Close([-0.043237989322, -0.487015034281, -1.417900495701, 0.39409939899, -1.218800936214, 0.449713705903, -0.820620658999, 0.447487909674],
            ddim.AddNoise(Tensor(Sample), Tensor(Output), 3), "ddim add_noise at t=601");

        var euler = new EulerScheduler(StableDiffusion with { Spacing = TimestepSpacing.Linspace });
        euler.SetTimesteps(10);
        Close([-0.558019394157, -1.470509741569, -3.919955430092, 1.099669898304, -3.584563086755, 1.813307107833, -2.722708335449, 1.196106401776],
            euler.AddNoise(Tensor(Sample), Tensor(Output), 3), "euler add_noise at t=666");
    }

    [Fact]
    public void A_diffusers_scheduler_config_is_read_field_by_field()
    {
        var path = Path.Combine(Path.GetTempPath(), $"scheduler-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, """
            {
              "_class_name": "PNDMScheduler",
              "beta_end": 0.012, "beta_schedule": "scaled_linear", "beta_start": 0.00085,
              "clip_sample": false, "num_train_timesteps": 1000, "prediction_type": "v_prediction",
              "set_alpha_to_one": false, "steps_offset": 1, "timestep_spacing": "trailing"
            }
            """);

        try
        {
            var config = SchedulerConfig.Load(path);

            Assert.Equal(1, config.StepsOffset);
            Assert.False(config.SetAlphaToOne);
            Assert.Equal(PredictionType.VPrediction, config.Prediction);
            Assert.Equal(TimestepSpacing.Trailing, config.Spacing);
            Assert.Equal(BetaSchedule.ScaledLinear, config.BetaSchedule);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
