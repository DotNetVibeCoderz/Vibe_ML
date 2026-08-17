using FluentAssertions;
using LocalGen.Core.Inference;
using LocalGen.Core.Prompting;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// Each model family expresses a tool call in its own form. These cover the detection of that
/// form from the chat template and the parsing of each one, including the ways generation goes
/// wrong — a missing terminator, an array where an object was expected, the other spelling of
/// the arguments property.
/// </summary>
public class ToolDialectTests
{
    [Theory]
    [InlineData(null, "hermes")]
    [InlineData("", "hermes")]
    [InlineData("{% for message in messages %}<|im_start|>{{ message.role }}", "hermes")]
    [InlineData("""{%- if tools %}<tool_call>{"name":...}</tool_call>{%- endif %}""", "hermes")]
    [InlineData("{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}", "llama3")]
    [InlineData("""{%- if tool_calls %}[TOOL_CALLS] [{"name": ...}]{%- endif %}""", "mistral")]
    public void Detect_reads_the_family_from_the_chat_template(string? template, string expected) =>
        ToolDialect.Detect(template).Name.Should().Be(expected);

    [Fact]
    public void A_template_naming_both_conventions_prefers_the_more_specific_marker()
    {
        // Mistral's own template mentions the generic tag in passing; the marker it actually
        // emits is what decides, so the specific one has to win.
        const string template = """{%- if tools %}<tool_call>{%- endif %}[TOOL_CALLS] [...]""";

        ToolDialect.Detect(template).Should().Be(ToolDialect.Mistral);
    }

    [Fact]
    public void A_llama3_finetune_that_speaks_chatml_is_read_as_hermes()
    {
        // Hermes 3 is a Llama 3 model, but it was retrained onto ChatML and <tool_call>. Reading
        // the architecture rather than the convention would pick exactly the wrong dialect.
        const string hermes3 =
            """{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role']}}""" +
            """{%- if tools %}<tool_call>{"name": ...}</tool_call>{%- endif %}""";

        ToolDialect.Detect(hermes3).Should().Be(ToolDialect.Hermes);
    }

    [Fact]
    public void A_llama3_template_without_a_tools_branch_is_still_llama3()
    {
        // Meta's published 3.1 template is sometimes mirrored without its tools section. The
        // model was still fine-tuned on bare JSON with `parameters`, so the family decides.
        const string bare =
            """{% for message in messages %}{% set content = '<|start_header_id|>' + """ +
            """message['role'] + '<|end_header_id|>' + message['content'] + '<|eot_id|>' %}{% endfor %}""";

        ToolDialect.Detect(bare).Should().Be(ToolDialect.Llama3);
    }

    [Fact]
    public void Llama3_marks_its_calls_with_nothing_at_all()
    {
        // Verified against Meta's published templates: the reply *is* the JSON object, with no
        // tag on either side, and `<|eot_id|>` is consumed as end-of-sequence.
        ToolDialect.Llama3.HasOpenTag.Should().BeFalse();

        var (calls, text) = ToolDialect.Llama3.Parse(
            """{"name": "get_weather", "parameters": {"city": "Bandung"}}""");

        calls.Should().HaveCount(1);
        calls[0].Name.Should().Be("get_weather");
        calls[0].ArgumentsJson.Should().Contain("Bandung");
        text.Should().BeEmpty();
    }

    [Fact]
    public void Llama3_accepts_the_python_tag_prefix_when_the_model_uses_it()
    {
        var (calls, _) = ToolDialect.Llama3.Parse(
            """<|python_tag|>{"name": "get_weather", "parameters": {"city": "Bandung"}}""");

        calls.Should().ContainSingle().Which.Name.Should().Be("get_weather");
    }

    [Fact]
    public void Llama3_does_not_mistake_prose_that_mentions_json_for_a_call()
    {
        const string prose = """You could call it with {"name": "get_weather"} if you wanted.""";

        var (calls, text) = ToolDialect.Llama3.Parse(prose);

        calls.Should().BeEmpty();
        text.Should().Be(prose);
    }

    [Fact]
    public void Mistral_parses_an_array_of_calls_from_one_marker()
    {
        var (calls, _) = ToolDialect.Mistral.Parse(
            """[TOOL_CALLS] [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]""");

        calls.Should().HaveCount(2);
        calls.Select(static c => c.Name).Should().Equal("a", "b");
        calls.Select(static c => c.Id).Should().OnlyHaveUniqueItems();
    }

    [Fact]
    public void Mistral_has_no_closing_marker_so_the_call_runs_to_the_end_of_the_turn()
    {
        ToolDialect.Mistral.HasCloseTag.Should().BeFalse();

        var (calls, _) = ToolDialect.Mistral.Parse(
            """[TOOL_CALLS] [{"name": "a", "arguments": {}}]""");

        calls.Should().HaveCount(1);
    }

    [Fact]
    public void A_model_answering_in_the_other_spelling_is_still_understood()
    {
        // Asked for Hermes, the model replied with Llama's property name. The call is
        // well-formed in every way that matters, so refusing it would help nobody.
        var (calls, _) = ToolDialect.Hermes.Parse(
            """<tool_call>{"name": "t", "parameters": {"a": 1}}</tool_call>""");

        calls.Should().HaveCount(1);
        calls[0].ArgumentsJson.Should().Contain("\"a\"");
    }

    [Fact]
    public void Instructions_show_the_example_in_the_families_own_form()
    {
        var tools = new[]
        {
            new ToolDefinition { Name = "get_weather", Description = "Looks up the weather." }
        };

        ToolDialect.Llama3.BuildInstructions(tools)
            .Should().Contain("\"parameters\"").And.NotContain("<tool_call>");

        ToolDialect.Mistral.BuildInstructions(tools)
            .Should().Contain("[TOOL_CALLS]").And.Contain("[{");

        ToolDialect.Hermes.BuildInstructions(tools)
            .Should().Contain("<tool_call>").And.Contain("\"arguments\"");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(8)]
    public void No_llama3_json_fragment_leaks_at_any_chunk_size(int chunkSize)
    {
        const string generated = """{"name": "t", "parameters": {"q": "x"}}""";

        var filter = new ToolCallStreamFilter(ToolDialect.Llama3);
        var visible = new System.Text.StringBuilder();

        for (var i = 0; i < generated.Length; i += chunkSize)
        {
            visible.Append(filter.Push(generated.Substring(i, Math.Min(chunkSize, generated.Length - i))));
        }

        visible.Append(filter.Flush());

        visible.ToString().Should().BeEmpty();
        filter.Calls.Should().ContainSingle().Which.Name.Should().Be("t");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(9)]
    public void Llama3_prose_streams_rather_than_being_buffered_whole(int chunkSize)
    {
        const string generated = "The weather in Bandung is warm today.";

        var filter = new ToolCallStreamFilter(ToolDialect.Llama3);
        var visible = new System.Text.StringBuilder();

        for (var i = 0; i < generated.Length; i += chunkSize)
        {
            visible.Append(filter.Push(generated.Substring(i, Math.Min(chunkSize, generated.Length - i))));
        }

        // Everything has already been released before the flush: once the reply is known to be
        // prose there is nothing left to decide, and holding it would stall the stream.
        visible.ToString().Should().Be(generated);

        filter.Flush().Should().BeEmpty();
        filter.HasCalls.Should().BeFalse();
    }

    [Fact]
    public void A_llama3_reply_that_opens_like_json_but_is_not_is_shown_as_text()
    {
        var filter = new ToolCallStreamFilter(ToolDialect.Llama3);

        filter.Push("""{"unfinished": """);
        var visible = filter.Flush();

        filter.HasCalls.Should().BeFalse();
        visible.Should().Be("""{"unfinished": """);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    public void A_mistral_call_with_no_terminator_is_recovered_on_flush(int chunkSize)
    {
        const string generated =
            """[TOOL_CALLS] [{"name": "t", "arguments": {"q": "x"}}]""";

        var filter = new ToolCallStreamFilter(ToolDialect.Mistral);
        var visible = new System.Text.StringBuilder();

        for (var i = 0; i < generated.Length; i += chunkSize)
        {
            visible.Append(filter.Push(generated.Substring(i, Math.Min(chunkSize, generated.Length - i))));
        }

        visible.Append(filter.Flush());

        filter.Calls.Should().HaveCount(1);
        filter.Calls[0].Name.Should().Be("t");
        visible.ToString().Trim().Should().BeEmpty();
    }

    [Fact]
    public void Genuinely_truncated_json_still_falls_through_as_text()
    {
        var filter = new ToolCallStreamFilter(ToolDialect.Hermes);

        filter.Push("""<tool_call>{"name": "t", "argumen""");
        var visible = filter.Flush();

        filter.HasCalls.Should().BeFalse();
        visible.Should().Contain("<tool_call>");
    }

    [Fact]
    public void Prepare_renders_a_previous_call_in_the_dialect_the_model_speaks()
    {
        var request = new ChatRequest
        {
            Model = "m",
            Messages =
            [
                ChatMessage.User("weather?"),
                new ChatMessage
                {
                    Role = ChatRole.Assistant,
                    ToolCalls = [new ToolCall("call_0", "get_weather", """{"city":"Bandung"}""")]
                }
            ]
        };

        var rendered = ChatPrompt.Prepare(request, ToolDialect.Mistral)
            .Single(static m => m.Role == ChatRole.Assistant).Text;

        rendered.Should().StartWith("[TOOL_CALLS][").And.EndWith("]");
        rendered.Should().Contain("\"arguments\"");

        // What is rendered has to be what the parser reads back, or a multi-turn tool
        // conversation would degrade on every replay of the transcript.
        ToolDialect.Mistral.Parse(rendered).Calls.Should().ContainSingle()
            .Which.Name.Should().Be("get_weather");
    }
}
