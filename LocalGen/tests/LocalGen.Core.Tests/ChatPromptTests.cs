using FluentAssertions;
using LocalGen.Core.Inference;
using LocalGen.Core.Prompting;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// How a conversation is flattened for a text-completion backend. The image handling matters
/// most: the marker's position in the rendered text is what decides where the picture lands in
/// the token sequence, so it has to survive flattening intact and in order.
/// </summary>
public class ChatPromptTests
{
    private static ChatRequest WithImage(params ContentPart[] parts) => new()
    {
        Model = "m",
        Messages = [new ChatMessage { Role = ChatRole.User, Content = parts }]
    };

    [Fact]
    public void A_text_only_session_describes_the_image_instead_of_marking_it()
    {
        var request = WithImage(
            new ContentPart.Text("What is this?"),
            new ContentPart.Image(new byte[] { 1, 2, 3 }, "image/png", "http://host/a.png"));

        var text = ChatPrompt.Prepare(request).Single().Text;

        text.Should().Contain("What is this?");
        text.Should().Contain("cannot process images");
        text.Should().Contain("http://host/a.png");
    }

    [Fact]
    public void A_vision_session_gets_the_marker_where_the_image_was()
    {
        var request = WithImage(
            new ContentPart.Text("Before."),
            new ContentPart.Image(new byte[] { 1 }, "image/png"),
            new ContentPart.Text("After."));

        var text = ChatPrompt.Prepare(request, imageMarker: "<__media__>").Single().Text;

        text.Should().Be("Before.<__media__>After.");
        text.Should().NotContain("cannot process");
    }

    [Fact]
    public void Several_images_keep_their_order()
    {
        // The nth marker takes the nth image, so a reordering here would silently pair each
        // question with the wrong picture.
        var request = WithImage(
            new ContentPart.Text("first:"),
            new ContentPart.Image(new byte[] { 1 }, "image/png"),
            new ContentPart.Text(" second:"),
            new ContentPart.Image(new byte[] { 2 }, "image/jpeg"));

        var text = ChatPrompt.Prepare(request, imageMarker: "[M]").Single().Text;

        text.Should().Be("first:[M] second:[M]");
    }

    [Fact]
    public void A_message_of_plain_text_is_passed_through_untouched()
    {
        var request = new ChatRequest { Model = "m", Messages = [ChatMessage.User("plain")] };

        ChatPrompt.Prepare(request, imageMarker: "[M]").Single().Text.Should().Be("plain");
    }

    [Fact]
    public void Documents_are_inlined_as_text_regardless_of_vision()
    {
        var request = WithImage(
            new ContentPart.Text("Summarise."),
            new ContentPart.Document("notes.md", "the body", "http://host/notes.md"));

        var text = ChatPrompt.Prepare(request, imageMarker: "[M]").Single().Text;

        text.Should().Contain("notes.md").And.Contain("the body");
    }

    [Fact]
    public void Tool_instructions_are_merged_into_an_existing_system_turn()
    {
        var request = new ChatRequest
        {
            Model = "m",
            Messages = [ChatMessage.System("Be brief."), ChatMessage.User("hi")],
            Tools = [new ToolDefinition { Name = "get_weather" }]
        };

        var prepared = ChatPrompt.Prepare(request);

        prepared.Should().HaveCount(2);
        prepared[0].Role.Should().Be(ChatRole.System);
        prepared[0].Text.Should().StartWith("Be brief.").And.Contain("get_weather");
    }

    [Fact]
    public void A_system_turn_is_inserted_when_there_was_none_to_merge_into()
    {
        var request = new ChatRequest
        {
            Model = "m",
            Messages = [ChatMessage.User("hi")],
            Tools = [new ToolDefinition { Name = "get_weather" }]
        };

        var prepared = ChatPrompt.Prepare(request);

        prepared.Should().HaveCount(2);
        prepared[0].Role.Should().Be(ChatRole.System);
        prepared[0].Text.Should().Contain("get_weather");
    }
}
