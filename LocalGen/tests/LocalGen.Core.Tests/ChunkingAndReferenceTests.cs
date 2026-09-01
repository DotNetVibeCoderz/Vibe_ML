using FluentAssertions;
using LocalGen.Rag.Documents;
using LocalGen.Runtime.Models;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// Chunking decides whether retrieval returns a usable passage, so the properties that matter are
/// coverage (nothing is lost) and boundaries (chunks do not end mid-sentence when avoidable).
/// </summary>
public class TextChunkerTests
{
    [Fact]
    public void Short_text_becomes_one_chunk()
    {
        var chunks = TextChunker.Split("A short document.", chunkSize: 1000);

        chunks.Should().HaveCount(1);
        chunks[0].Text.Should().Be("A short document.");
    }

    [Fact]
    public void Empty_text_produces_nothing()
    {
        TextChunker.Split("   ").Should().BeEmpty();
    }

    [Fact]
    public void Long_text_is_split_into_overlapping_chunks()
    {
        var text = string.Join(" ", Enumerable.Repeat("word", 800));

        var chunks = TextChunker.Split(text, chunkSize: 500, overlap: 100);

        chunks.Should().HaveCountGreaterThan(1);
        chunks.Should().OnlyContain(chunk => chunk.Text.Length <= 500);
        chunks.Select(c => c.Index).Should().BeInAscendingOrder();
    }

    [Fact]
    public void Splitting_prefers_sentence_boundaries()
    {
        var sentence = new string('a', 120) + ". ";
        var text = string.Concat(Enumerable.Repeat(sentence, 20));

        var chunks = TextChunker.Split(text, chunkSize: 300, overlap: 50);

        // Every chunk but the last should land on a full stop rather than mid-word.
        chunks.Take(chunks.Count - 1).Should().OnlyContain(chunk => chunk.Text.EndsWith('.'));
    }

    [Fact]
    public void No_content_is_lost_when_there_is_no_overlap()
    {
        var text = string.Join(" ", Enumerable.Range(0, 300).Select(i => $"token{i}"));

        var chunks = TextChunker.Split(text, chunkSize: 200, overlap: 0);
        var reassembled = string.Join(" ", chunks.Select(c => c.Text));

        foreach (var index in (int[])[0, 42, 150, 299])
        {
            reassembled.Should().Contain($"token{index}");
        }
    }

    [Fact]
    public void An_overlap_at_or_above_the_chunk_size_still_terminates()
    {
        // Clamping matters: without it the cursor would never advance and this would hang.
        var text = new string('x', 5000);

        var chunks = TextChunker.Split(text, chunkSize: 100, overlap: 500);

        chunks.Should().NotBeEmpty();
        chunks.Count.Should().BeLessThan(1000);
    }

    [Fact]
    public void Markdown_chunks_carry_their_heading()
    {
        var markdown =
            """
            # Title

            Intro paragraph.

            ## Installation

            Run the installer and follow the prompts.

            ## Configuration

            Edit appsettings.json.
            """;

        var chunks = TextChunker.SplitMarkdown(markdown, chunkSize: 200, overlap: 20);

        chunks.Should().NotBeEmpty();
        chunks.Should().Contain(chunk => chunk.Text.Contains("Installation"));
        chunks.Should().Contain(chunk => chunk.Text.Contains("Configuration"));
    }
}

/// <summary>Model references are user input from the CLI and the API, so parsing must be exact.</summary>
public class ModelReferenceTests
{
    [Theory]
    [InlineData("huggingface:owner/repo", "owner/repo", "")]
    [InlineData("hf:owner/repo", "owner/repo", "")]
    [InlineData("huggingface:owner/repo/file.gguf", "owner/repo", "file.gguf")]
    [InlineData("huggingface:owner/repo/nested/file.gguf", "owner/repo", "nested/file.gguf")]
    public void HuggingFace_references_split_into_repository_and_file(
        string reference, string repository, string file)
    {
        var parsed = ModelReference.Parse(reference);

        parsed.Provider.Should().Be(ModelReference.HuggingFace);
        parsed.Repository.Should().Be(repository);
        parsed.File.Should().Be(file);
    }

    [Fact]
    public void Foundry_references_keep_their_alias()
    {
        var parsed = ModelReference.Parse("foundry:phi-4-mini");

        parsed.Provider.Should().Be(ModelReference.Foundry);
        parsed.Path.Should().Be("phi-4-mini");
    }

    [Fact]
    public void A_bare_name_with_a_tag_is_local_not_a_scheme()
    {
        // "qwen2.5:7b" must not be read as scheme "qwen2.5".
        var parsed = ModelReference.Parse("qwen2.5:7b");

        parsed.Provider.Should().Be(ModelReference.Local);
        parsed.Tag.Should().Be("7b");
    }

    [Fact]
    public void A_windows_drive_letter_is_not_mistaken_for_a_scheme()
    {
        var parsed = ModelReference.Parse(@"C:\models\model.gguf");

        parsed.Provider.Should().Be(ModelReference.File_);
        parsed.Path.Should().EndWith("model.gguf");
    }

    [Fact]
    public void An_incomplete_huggingface_reference_is_rejected()
    {
        var act = () => ModelReference.Parse("huggingface:owner");

        act.Should().Throw<LocalGenException>().WithMessage("*not a valid HuggingFace reference*");
    }

    [Fact]
    public void An_empty_reference_is_rejected()
    {
        var act = () => ModelReference.Parse("   ");

        act.Should().Throw<LocalGenException>();
    }
}
