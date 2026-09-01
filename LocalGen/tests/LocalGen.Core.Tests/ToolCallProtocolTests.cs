using FluentAssertions;
using LocalGen.Core.Prompting;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// Tool calling rides on a prompt convention, so the parser has to cope with whatever the model
/// actually produces — including malformed JSON and generation cut off mid-call.
/// </summary>
public class ToolCallProtocolTests
{
    [Fact]
    public void Parse_returns_plain_text_untouched()
    {
        var (calls, text) = ToolCallProtocol.Parse("The answer is 42.");

        calls.Should().BeEmpty();
        text.Should().Be("The answer is 42.");
    }

    [Fact]
    public void Parse_extracts_a_call_and_strips_its_block()
    {
        var (calls, text) = ToolCallProtocol.Parse(
            """Let me check. <tool_call>{"name": "Math-evaluate", "arguments": {"expression": "2+2"}}</tool_call>""");

        calls.Should().HaveCount(1);
        calls[0].Name.Should().Be("Math-evaluate");
        calls[0].ArgumentsJson.Should().Contain("2+2");
        text.Should().Be("Let me check.");
    }

    [Fact]
    public void Parse_extracts_several_calls()
    {
        var (calls, _) = ToolCallProtocol.Parse(
            """<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b"}</tool_call>""");

        calls.Select(c => c.Name).Should().BeEquivalentTo(["a", "b"]);
    }

    [Fact]
    public void Parse_keeps_malformed_json_as_visible_text()
    {
        // Silently dropping it would leave the user staring at an empty reply.
        var (calls, text) = ToolCallProtocol.Parse(
            "<tool_call>{not json}</tool_call>");

        calls.Should().BeEmpty();
        text.Should().Contain("not json");
    }

    [Fact]
    public void Parse_recovers_a_call_whose_closing_tag_never_arrived()
    {
        // Observed with Qwen2.5: the model emits the opening tag and the JSON, then stops.
        // Treating that as prose would show the user raw JSON and lose the call entirely.
        var (calls, text) = ToolCallProtocol.Parse(
            """<tool_call>{"name": "get_weather", "arguments": {"city": "Jakarta"}}""");

        calls.Should().HaveCount(1);
        calls[0].Name.Should().Be("get_weather");
        calls[0].ArgumentsJson.Should().Contain("Jakarta");
        text.Should().BeEmpty();
    }

    [Fact]
    public void Parse_recovers_an_unterminated_call_that_has_trailing_prose()
    {
        var (calls, _) = ToolCallProtocol.Parse(
            """<tool_call>{"name": "a", "arguments": {}} I hope that helps""");

        calls.Should().HaveCount(1);
        calls[0].Name.Should().Be("a");
    }

    [Fact]
    public void Parse_keeps_genuinely_truncated_json_as_text()
    {
        // The object never closes, so there is nothing to recover — the user should at least
        // see what the model produced.
        var (calls, text) = ToolCallProtocol.Parse(
            """Working on it <tool_call>{"name": "a" """);

        calls.Should().BeEmpty();
        text.Should().Contain("Working on it");
    }

    [Fact]
    public void BuildInstructions_is_empty_when_there_are_no_tools()
    {
        ToolCallProtocol.BuildInstructions([]).Should().BeEmpty();
    }

    [Fact]
    public void BuildInstructions_lists_each_tool()
    {
        var instructions = ToolCallProtocol.BuildInstructions(
        [
            new Inference.ToolDefinition { Name = "Math-evaluate", Description = "Does maths" }
        ]);

        instructions.Should().Contain("Math-evaluate").And.Contain("Does maths");
        instructions.Should().Contain(ToolCallProtocol.OpenTag);
    }
}

/// <summary>
/// The stream filter is where a tag split across token boundaries either works or leaks a stray
/// angle bracket into the user's transcript, so it is tested at every split position.
/// </summary>
public class ToolCallStreamFilterTests
{
    [Fact]
    public void Plain_text_passes_straight_through()
    {
        var filter = new ToolCallStreamFilter();

        var output = filter.Push("Hello ") + filter.Push("world") + filter.Flush();

        output.Should().Be("Hello world");
        filter.HasCalls.Should().BeFalse();
    }

    [Fact]
    public void A_call_split_across_tokens_is_reassembled_and_hidden()
    {
        var filter = new ToolCallStreamFilter();

        var tokens = new[]
        {
            "Sure. ", "<tool", "_call>", "{\"name\":", " \"Math-evaluate\",",
            " \"arguments\": {\"expression\": \"1+1\"}}", "</tool", "_call>"
        };

        var visible = string.Concat(tokens.Select(filter.Push)) + filter.Flush();

        visible.Should().Be("Sure. ");
        filter.Calls.Should().HaveCount(1);
        filter.Calls[0].Name.Should().Be("Math-evaluate");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(8)]
    public void No_tag_fragment_leaks_at_any_chunk_size(int chunkSize)
    {
        const string generated =
            """before <tool_call>{"name": "t", "arguments": {}}</tool_call> after""";

        var filter = new ToolCallStreamFilter();
        var visible = new System.Text.StringBuilder();

        for (var i = 0; i < generated.Length; i += chunkSize)
        {
            visible.Append(filter.Push(generated.Substring(i, Math.Min(chunkSize, generated.Length - i))));
        }

        visible.Append(filter.Flush());

        visible.ToString().Should().Be("before  after");
        visible.ToString().Should().NotContain("<tool");
        filter.Calls.Should().HaveCount(1);
    }

    [Fact]
    public void Text_that_merely_looks_like_a_tag_is_still_emitted()
    {
        var filter = new ToolCallStreamFilter();

        var output = filter.Push("Use the <tool") + filter.Push("box> element.") + filter.Flush();

        output.Should().Be("Use the <toolbox> element.");
        filter.HasCalls.Should().BeFalse();
    }

    [Fact]
    public void A_call_missing_its_closing_tag_is_recovered_on_flush()
    {
        var filter = new ToolCallStreamFilter();

        var visible = filter.Push("""<tool_call>{"name": "get_weather", "arguments": {"city": "Jakarta"}}""")
                      + filter.Flush();

        visible.Should().BeEmpty();
        filter.Calls.Should().HaveCount(1);
        filter.Calls[0].Name.Should().Be("get_weather");
    }

    [Fact]
    public void Genuinely_truncated_json_is_released_as_text_on_flush()
    {
        var filter = new ToolCallStreamFilter();

        // Text before the tag is emitted immediately; only the truncated call waits for the flush.
        var output = filter.Push("""text <tool_call>{"name": "a" """) + filter.Flush();

        // Nothing may be swallowed: a truncated call still has to reach the user somehow.
        output.Should().Contain("text").And.Contain("name");
        filter.HasCalls.Should().BeFalse();
    }
}
