using FluentAssertions;
using LocalGen.Core.Engines;
using LocalGen.Core.Inference;
using LocalGen.Core.Modelfiles;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// The Modelfile is a user-authored format, so the parser's job is as much about rejecting bad
/// input with a useful message as it is about accepting good input.
/// </summary>
public class ModelfileParserTests
{
    [Fact]
    public void Parse_reads_a_minimal_file()
    {
        var modelfile = ModelfileParser.Parse("FROM ./model.gguf");

        modelfile.From.Should().Be("./model.gguf");
        modelfile.System.Should().BeNull();
        modelfile.Parameters.Temperature.Should().BeNull();
    }

    [Fact]
    public void Parse_requires_a_from_instruction()
    {
        var act = () => ModelfileParser.Parse("PARAMETER temperature 0.5");

        act.Should().Throw<ModelfileException>()
            .WithMessage("*FROM instruction is required*");
    }

    [Fact]
    public void Parse_separates_sampling_from_load_parameters()
    {
        var modelfile = ModelfileParser.Parse(
            """
            FROM ./model.gguf
            PARAMETER temperature 0.35
            PARAMETER top_p 0.85
            PARAMETER num_predict 512
            PARAMETER num_ctx 8192
            PARAMETER num_gpu 24
            """);

        modelfile.Parameters.Temperature.Should().Be(0.35f);
        modelfile.Parameters.TopP.Should().Be(0.85f);
        modelfile.Parameters.MaxTokens.Should().Be(512);

        // Context size and GPU offload are load-time concerns, not sampling ones.
        modelfile.LoadOptions.ContextSize.Should().Be(8192);
        modelfile.LoadOptions.GpuLayerCount.Should().Be(24);
    }

    [Fact]
    public void Parse_reads_multiline_triple_quoted_blocks()
    {
        var modelfile = ModelfileParser.Parse(
            """"
            FROM ./model.gguf
            SYSTEM """
            You are a careful assistant.
            Answer in Indonesian.
            """
            """");

        modelfile.System.Should().Be("You are a careful assistant.\nAnswer in Indonesian.");
    }

    [Fact]
    public void Parse_reads_a_single_line_triple_quoted_block()
    {
        var modelfile = ModelfileParser.Parse(
            """"
            FROM ./model.gguf
            SYSTEM """You are terse."""
            """");

        modelfile.System.Should().Be("You are terse.");
    }

    [Fact]
    public void Parse_rejects_an_unterminated_block()
    {
        var act = () => ModelfileParser.Parse(
            """"
            FROM ./model.gguf
            SYSTEM """
            never closed
            """");

        act.Should().Throw<ModelfileException>().WithMessage("*unterminated*");
    }

    [Fact]
    public void Parse_collects_repeated_stop_sequences()
    {
        var modelfile = ModelfileParser.Parse(
            """
            FROM ./model.gguf
            PARAMETER stop "<|im_end|>"
            PARAMETER stop "<|endoftext|>"
            """);

        modelfile.Parameters.StopSequences.Should()
            .BeEquivalentTo(["<|im_end|>", "<|endoftext|>"]);
    }

    [Fact]
    public void Parse_keeps_unknown_parameters_as_extras()
    {
        // Forward compatibility: an unrecognised key should not fail the whole file.
        var modelfile = ModelfileParser.Parse(
            """
            FROM ./model.gguf
            PARAMETER some_future_knob 42
            """);

        modelfile.Extras.Should().ContainKey("some_future_knob").WhoseValue.Should().Be("42");
    }

    [Fact]
    public void Parse_reports_the_line_number_of_a_bad_instruction()
    {
        var act = () => ModelfileParser.Parse(
            """
            FROM ./model.gguf
            NONSENSE value
            """);

        act.Should().Throw<ModelfileException>()
            .Which.Line.Should().Be(2);
    }

    [Fact]
    public void Parse_reads_messages_and_device()
    {
        var modelfile = ModelfileParser.Parse(
            """
            FROM ./model.gguf
            PARAMETER device cuda
            MESSAGE user "Hello"
            MESSAGE assistant "Hi there"
            EMBEDDING false
            """);

        modelfile.LoadOptions.Device.Should().Be(DeviceKind.Cuda);
        modelfile.Messages.Should().HaveCount(2);
        modelfile.Messages[0].Role.Should().Be(ChatRole.User);
        modelfile.Messages[1].Text.Should().Be("Hi there");
        modelfile.IsEmbedding.Should().BeFalse();
    }

    [Fact]
    public void Parse_reads_the_multi_gpu_parameters()
    {
        var modelfile = ModelfileParser.Parse(
            """
            FROM ./model.gguf
            PARAMETER tensor_split 0.6, 0.4
            PARAMETER split_mode row
            PARAMETER main_gpu 1
            """);

        modelfile.LoadOptions.TensorSplit.Should().Equal(0.6f, 0.4f);
        modelfile.LoadOptions.SplitMode.Should().Be(GpuSplitMode.Row);
        modelfile.LoadOptions.MainGpu.Should().Be(1);
    }

    [Fact]
    public void Parse_rejects_an_unknown_split_mode()
    {
        var act = () => ModelfileParser.Parse(
            """
            FROM ./model.gguf
            PARAMETER split_mode sideways
            """);

        act.Should().Throw<ModelfileException>()
            .WithMessage("*expected auto, none, layer or row*");
    }

    [Fact]
    public void Render_round_trips_the_multi_gpu_parameters()
    {
        // `localgen show --modelfile` is how a split gets copied to the machine that has the
        // cards, so a split that does not survive rendering would be lost exactly there.
        var original = ModelfileParser.Parse(
            """
            FROM ./model.gguf
            PARAMETER tensor_split 0.6, 0.4
            PARAMETER split_mode layer
            PARAMETER main_gpu 1
            """);

        var reparsed = ModelfileParser.Parse(ModelfileParser.Render(original));

        reparsed.LoadOptions.TensorSplit.Should().Equal(0.6f, 0.4f);
        reparsed.LoadOptions.SplitMode.Should().Be(GpuSplitMode.Layer);
        reparsed.LoadOptions.MainGpu.Should().Be(1);
    }

    [Fact]
    public void Render_round_trips_through_Parse()
    {
        var original = ModelfileParser.Parse(
            """"
            FROM ./model.gguf
            PARAMETER temperature 0.4
            PARAMETER num_ctx 4096
            PARAMETER stop "<|im_end|>"
            SYSTEM """Be brief."""
            """");

        var reparsed = ModelfileParser.Parse(ModelfileParser.Render(original));

        reparsed.From.Should().Be(original.From);
        reparsed.System.Should().Be(original.System);
        reparsed.Parameters.Temperature.Should().Be(original.Parameters.Temperature);
        reparsed.LoadOptions.ContextSize.Should().Be(original.LoadOptions.ContextSize);
        reparsed.Parameters.StopSequences.Should().BeEquivalentTo(original.Parameters.StopSequences);
    }

    [Fact]
    public void Parse_ignores_comments_and_blank_lines()
    {
        var modelfile = ModelfileParser.Parse(
            """
            # this is a comment

            FROM ./model.gguf

            # another comment
            PARAMETER temperature 1
            """);

        modelfile.From.Should().Be("./model.gguf");
        modelfile.Parameters.Temperature.Should().Be(1f);
    }
}
