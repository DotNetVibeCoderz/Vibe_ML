using FluentAssertions;
using LocalGen.Core.Engines;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// Multi-GPU splitting is configured on machines that cannot be used to check it — the split is
/// written once and then only observed when a large model loads on the host that has the cards.
/// These tests are what stands in for that: they pin the arithmetic and, more importantly, the
/// behaviour when the configuration and the hardware disagree, which is the case that otherwise
/// fails silently.
/// </summary>
public class TensorSplitPlanTests
{
    [Fact]
    public void An_empty_split_leaves_the_decision_to_the_backend()
    {
        var plan = TensorSplitPlan.Create([], deviceCount: 2);

        plan.IsAutomatic.Should().BeTrue();
        plan.Warnings.Should().BeEmpty();
        plan.Describe().Should().Contain("automatic");
    }

    [Fact]
    public void Weights_are_normalised_to_fractions()
    {
        var plan = TensorSplitPlan.Create([3, 1], deviceCount: 2);

        plan.Fractions.Should().Equal(0.75f, 0.25f);
        plan.Warnings.Should().BeEmpty();
    }

    [Fact]
    public void Percentages_and_fractions_describe_the_same_split()
    {
        var asFractions = TensorSplitPlan.Create([0.6f, 0.4f], deviceCount: 2);
        var asPercentages = TensorSplitPlan.Create([60, 40], deviceCount: 2);

        asPercentages.Fractions.Should().Equal(asFractions.Fractions);
    }

    [Fact]
    public void A_split_naming_more_devices_than_exist_drops_the_extras_and_says_so()
    {
        // The case this whole type exists for: llama.cpp accepts the third weight and ignores it,
        // so without this the user's model quietly loads on two cards while they believe it is
        // spread over three.
        var plan = TensorSplitPlan.Create([1, 1, 1], deviceCount: 2);

        plan.Fractions.Should().Equal(0.5f, 0.5f);
        plan.Warnings.Should().ContainSingle()
            .Which.Should().Contain("names 3 devices but only 2 are visible");
    }

    [Fact]
    public void A_split_shorter_than_the_device_list_leaves_the_rest_idle_and_says_so()
    {
        var plan = TensorSplitPlan.Create([1, 1], deviceCount: 4);

        plan.Fractions.Should().Equal(0.5f, 0.5f, 0f, 0f);
        plan.Warnings.Should().ContainSingle()
            .Which.Should().Contain("device 2, 3");
    }

    [Fact]
    public void A_split_on_a_single_gpu_machine_is_reported_rather_than_applied()
    {
        var plan = TensorSplitPlan.Create([0.6f, 0.4f], deviceCount: 1);

        plan.IsAutomatic.Should().BeTrue();
        plan.Warnings.Should().ContainSingle()
            .Which.Should().Contain("only one GPU is visible");
    }

    [Fact]
    public void A_split_with_no_gpu_at_all_is_reported_rather_than_applied()
    {
        var plan = TensorSplitPlan.Create([0.6f, 0.4f], deviceCount: 0);

        plan.IsAutomatic.Should().BeTrue();
        plan.Warnings.Should().ContainSingle()
            .Which.Should().Contain("no GPU is registered");
    }

    [Fact]
    public void Negative_and_non_finite_weights_are_read_as_zero_and_reported()
    {
        var plan = TensorSplitPlan.Create([-1, 2, float.NaN], deviceCount: 3);

        plan.Fractions.Should().Equal(0f, 1f, 0f);
        plan.Warnings.Should().HaveCount(2);
    }

    [Fact]
    public void An_all_zero_split_falls_back_to_the_backend()
    {
        var plan = TensorSplitPlan.Create([0, 0], deviceCount: 2);

        plan.IsAutomatic.Should().BeTrue();
        plan.Warnings.Should().Contain(w => w.Contains("all zeros"));
    }

    [Fact]
    public void A_zero_weight_keeps_one_card_free_without_complaint()
    {
        // Deliberately leaving a card empty is a legitimate configuration — it is how you keep a
        // display GPU out of inference — so it must not be corrected or warned about.
        var plan = TensorSplitPlan.Create([0, 1], deviceCount: 2);

        plan.Fractions.Should().Equal(0f, 1f);
        plan.Warnings.Should().BeEmpty();
    }

    [Fact]
    public void Describe_names_the_devices_the_split_lands_on()
    {
        var plan = TensorSplitPlan.Create([3, 1], deviceCount: 2);

        var described = plan.Describe(
        [
            new AcceleratorDevice { Index = 0, Name = "CUDA0" },
            new AcceleratorDevice { Index = 1, Name = "CUDA1" }
        ]);

        described.Should().Be("CUDA0 75% · CUDA1 25%");
    }

    [Theory]
    [InlineData("0.6, 0.4")]
    [InlineData("0.6 0.4")]
    [InlineData(" 0.6 ,0.4 ")]
    public void Weights_are_read_however_they_are_typed(string text)
    {
        TensorSplitPlan.TryParseWeights(text, out var weights).Should().BeTrue();
        weights.Should().Equal(0.6f, 0.4f);
    }

    [Fact]
    public void An_unreadable_weight_rejects_the_whole_list()
    {
        // Partial acceptance would shift every later weight onto the wrong device.
        TensorSplitPlan.TryParseWeights("0.6, half", out var weights).Should().BeFalse();
        weights.Should().BeEmpty();
    }

    [Fact]
    public void No_text_is_a_valid_empty_split()
    {
        TensorSplitPlan.TryParseWeights("   ", out var weights).Should().BeTrue();
        weights.Should().BeEmpty();
    }
}
