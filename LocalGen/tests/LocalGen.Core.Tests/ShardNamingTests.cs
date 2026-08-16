using FluentAssertions;
using LocalGen.Runtime.Downloads;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// A model published in parts only works if every part is fetched — llama.cpp opens the first
/// shard and expects its siblings beside it. Getting this wrong fails at load time with a missing
/// tensor, long after the download appeared to succeed.
/// </summary>
public class ShardNamingTests
{
    [Theory]
    [InlineData("model-00001-of-00003.gguf", 1, 3)]
    [InlineData("Meta-Llama-3-70B-Q4_K_M-00002-of-00009.gguf", 2, 9)]
    [InlineData("MODEL-00010-of-00010.GGUF", 10, 10)]
    public void A_shard_name_yields_its_position(string fileName, int index, int total)
    {
        ShardNaming.IsShard(fileName).Should().BeTrue();
        ShardNaming.Parse(fileName).Should().Be((index, total));
    }

    [Theory]
    [InlineData("model.gguf")]
    [InlineData("model-Q4_K_M.gguf")]
    [InlineData("model-1-of-3.gguf")]      // not zero-padded, so not the convention
    [InlineData("model-00001-of-00003.bin")]
    public void Anything_else_is_not_a_shard(string fileName)
    {
        ShardNaming.IsShard(fileName).Should().BeFalse();
        ShardNaming.Parse(fileName).Should().BeNull();
    }

    [Fact]
    public void Expanding_a_shard_lists_every_part_in_order()
    {
        var parts = ShardNaming.Expand("model-00001-of-00003.gguf");

        parts.Should().Equal(
            "model-00001-of-00003.gguf",
            "model-00002-of-00003.gguf",
            "model-00003-of-00003.gguf");
    }

    [Fact]
    public void Expanding_from_a_later_shard_still_lists_the_whole_set()
    {
        // A user may name any part; the download has to cover all of them either way.
        ShardNaming.Expand("model-00002-of-00003.gguf").Should().HaveCount(3);
    }

    [Fact]
    public void An_unsharded_name_expands_to_itself()
    {
        ShardNaming.Expand("model-Q4_K_M.gguf").Should().Equal("model-Q4_K_M.gguf");
    }

    [Fact]
    public void An_absurd_shard_count_is_not_expanded()
    {
        // Guards against a malformed name producing an unbounded download list.
        ShardNaming.Expand("model-00001-of-00000.gguf").Should().HaveCount(1);
    }

    [Fact]
    public void Collapsing_leaves_one_entry_per_model()
    {
        var files = new[]
        {
            "small-Q4_K_M.gguf",
            "big-Q4_K_M-00001-of-00003.gguf",
            "big-Q4_K_M-00002-of-00003.gguf",
            "big-Q4_K_M-00003-of-00003.gguf",
            "other-Q8_0.gguf"
        };

        ShardNaming.CollapseShards(files).Should().Equal(
            "small-Q4_K_M.gguf",
            "big-Q4_K_M-00001-of-00003.gguf",
            "other-Q8_0.gguf");
    }

    [Fact]
    public void Collapsing_keeps_a_set_whose_first_shard_is_absent_out_of_the_listing()
    {
        // Offering a set that cannot be completed would produce a download that fails at load.
        var files = new[] { "big-00002-of-00003.gguf", "big-00003-of-00003.gguf" };

        ShardNaming.CollapseShards(files).Should().BeEmpty();
    }
}
