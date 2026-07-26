using BlazorML.Core.Abstractions;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using Microsoft.Extensions.DependencyInjection;

namespace BlazorML.Agents.Tests;

/// <summary>
/// <see cref="IDatasetService.PreviewAsync"/>, against a real workspace.
/// <para>
/// It had been on the interface and implemented since the beginning and never once called. The
/// Dataset page's row preview is the first caller, and a preview that quietly reads a whole
/// 512 MB upload into a Blazor circuit to show twenty-five rows would be a bad way to find that
/// out in production.
/// </para>
/// </summary>
public class DatasetPreviewTests(WorkspaceFixture workspace) : IClassFixture<WorkspaceFixture>
{
    private IDatasetService Datasets(IServiceScope scope) =>
        scope.ServiceProvider.GetRequiredService<IDatasetService>();

    [Fact]
    public async Task A_preview_returns_the_first_rows_and_all_the_columns()
    {
        using var scope = workspace.Scopes.CreateScope();

        var preview = await Datasets(scope).PreviewAsync(workspace.DatasetId, 10);

        Assert.Equal(10, preview.RowCount);
        Assert.Equal(3, preview.ColumnCount);
        Assert.Equal(["umur", "kota", "churn"], preview.Columns.Select(c => c.Name));
    }

    [Fact]
    public async Task The_rows_are_the_ones_at_the_top_of_the_file()
    {
        using var scope = workspace.Scopes.CreateScope();
        var service = Datasets(scope);

        var everything = await service.LoadAsync(workspace.DatasetId);
        var preview = await service.PreviewAsync(workspace.DatasetId, 5);

        // A preview is only useful if it is the same data, not a differently ordered sample.
        for (var row = 0; row < 5; row++)
        {
            Assert.Equal(everything.Text(row, "umur"), preview.Text(row, "umur"));
            Assert.Equal(everything.Text(row, "kota"), preview.Text(row, "kota"));
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(25)]
    [InlineData(100)]
    [InlineData(500)]
    public async Task Asking_for_more_rows_than_there_are_returns_what_exists(int requested)
    {
        using var scope = workspace.Scopes.CreateScope();

        var preview = await Datasets(scope).PreviewAsync(workspace.DatasetId, requested);

        // The fixture's dataset has 60 rows, so the larger sizes the page offers land here.
        Assert.Equal(Math.Min(requested, 60), preview.RowCount);
    }

    /// <summary>
    /// End to end through storage and the settings-derived cap. That the reader genuinely stops
    /// early rather than parsing everything and trimming is measured in <c>SerializerTests</c>,
    /// where the stream can be watched.
    /// </summary>
    [Theory]
    [InlineData(DatasetFormat.Csv)]
    [InlineData(DatasetFormat.Json)]
    public async Task Previewing_a_large_dataset_returns_only_the_rows_asked_for(DatasetFormat format)
    {
        using var scope = workspace.Scopes.CreateScope();
        var service = Datasets(scope);

        var table = TabularData.WithColumns("n", "catatan");

        for (var i = 0; i < 20_000; i++)
        {
            table.AddRow(i, $"baris ke-{i} dengan cukup teks supaya berkasnya besar");
        }

        var big = await service.SaveAsync($"Besar {format} {Guid.NewGuid():n}"[..20], table, format, null);

        var preview = await service.PreviewAsync(big.Id, 25);

        Assert.Equal(25, preview.RowCount);
        Assert.Equal("0", preview.Text(0, "n"));
        Assert.Equal("24", preview.Text(24, "n"));
    }

    [Fact]
    public async Task A_preview_of_a_dataset_that_no_longer_exists_says_which_one()
    {
        using var scope = workspace.Scopes.CreateScope();

        var error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => Datasets(scope).PreviewAsync("tidak-ada", 25));

        Assert.Contains("tidak-ada", error.Message);
    }

    [Fact]
    public async Task Gaps_in_the_data_survive_into_the_preview()
    {
        using var scope = workspace.Scopes.CreateScope();
        var service = Datasets(scope);

        var table = TabularData.WithColumns("kota", "jumlah");
        table.AddRow("Bandung", 10);
        table.AddRow("Jakarta", null);
        table.AddRow(null, 30);

        var saved = await service.SaveAsync($"Gap {Guid.NewGuid():n}"[..12], table,
            DatasetFormat.Csv, null);

        var preview = await service.PreviewAsync(saved.Id, 25);

        // The page prints a gap as an em dash rather than an empty cell, which only works if a
        // gap is still a gap by the time it gets there.
        Assert.True(TabularData.IsBlank(preview.Value(1, "jumlah")));
        Assert.True(TabularData.IsBlank(preview.Value(2, "kota")));
        Assert.Equal("30", preview.Text(2, "jumlah"));
    }
}
