using FluentAssertions;
using LocalGen.Kernel;
using LocalGen.Kernel.Orchestration;
using Xunit;

namespace LocalGen.Core.Tests;

/// <summary>
/// The parts of team orchestration that do not need a model: how an agent's name becomes the
/// function a coordinator calls, and what a member is allowed to touch by default.
/// </summary>
public class AgentTeamTests
{
    private static AgentDefinition Named(string name) => new()
    {
        Name = name,
        Description = "d",
        Instructions = "i"
    };

    [Theory]
    [InlineData("Researcher", "researcher")]
    [InlineData("Data Analyst", "data_analyst")]
    [InlineData("QA/Review", "qa_review")]
    [InlineData("report-writer", "report_writer")]
    [InlineData("  Spaced  ", "spaced")]
    // Non-ASCII becomes an underscore, and leading ones are trimmed rather than left dangling.
    [InlineData("Ünïcode", "n_code")]
    public void An_agent_name_becomes_a_safe_function_name(string name, string expected) =>
        Named(name).FunctionSafeName.Should().Be(expected);

    [Fact]
    public void A_name_with_nothing_usable_still_produces_a_callable_function()
    {
        // The delegation function's name is built from this, so an empty result would produce
        // `ask_` — which the model cannot distinguish from any other agent's.
        Named("***").FunctionSafeName.Should().Be("agent");
    }

    [Fact]
    public void Hyphens_do_not_survive_into_a_function_name()
    {
        // A qualified tool name is split on its first hyphen, so a hyphen inside the agent's own
        // name would be read as the boundary between plugin and function.
        Named("report-writer").FunctionSafeName.Should().NotContain("-");
    }

    [Fact]
    public void An_agent_gets_no_tools_unless_it_is_given_some()
    {
        // A team member should not inherit code execution because a teammate needed it.
        var definition = Named("Writer");

        definition.Tools.Should().BeEquivalentTo(ToolSelection.None);
        definition.Tools.CodeExecution.Should().BeFalse();
        definition.Tools.FileSystem.Should().BeFalse();
    }

    [Fact]
    public async Task A_team_needs_at_least_one_member()
    {
        var act = async () => await AgentTeam.CreateAsync(null!, "model", []);

        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*at least one agent*");
    }
}
