using AngleSharp.Dom;
using System.Text.Json;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Analysis;
using BlazorML.Core.Data;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using Bunit;
using Bunit.TestDoubles;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Page = BlazorML.Web.Components.Pages.Dataset;

namespace BlazorML.Web.Tests;

/// <summary>
/// The row preview on the Dataset page.
/// <para>
/// <see cref="IDatasetService.PreviewAsync"/> is covered against a real workspace elsewhere; what
/// is only true here is how the rows reach the screen — that a gap reads as a gap, that numbers
/// line up, and that asking for more rows asks the service for more rather than slicing what it
/// already had.
/// </para>
/// </summary>
public sealed class DatasetPreviewPageTests : TestContext
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"blazorml-dspage-{Guid.NewGuid():n}");
    private readonly StubDatasets _datasets = new();

    public DatasetPreviewPageTests()
    {
        Directory.CreateDirectory(_root);

        Services.AddDbContext<AppDbContext>(b =>
            b.UseSqlite($"Data Source={Path.Combine(_root, "page.db")}"));

        Services.AddSingleton<IDatasetService>(_datasets);

        this.AddTestAuthorization().SetAuthorized("admin@gravicode.com");
        JSInterop.Mode = JSRuntimeMode.Loose;

        using var scope = Services.CreateScope();
        scope.ServiceProvider.GetRequiredService<AppDbContext>().Database.EnsureCreated();
    }

    /// <summary>Records what the page asked for, and hands back whatever the test set up.</summary>
    private sealed class StubDatasets : IDatasetService
    {
        public TabularData Rows { get; set; } = new();
        public List<int> Requested { get; } = [];
        public Exception? Throws { get; set; }

        public Task<TabularData> PreviewAsync(string datasetId, int rows = 50, CancellationToken ct = default)
        {
            Requested.Add(rows);

            if (Throws is not null)
            {
                return Task.FromException<TabularData>(Throws);
            }

            // Honour the limit the way the real service does, so "asked for 25, showed 25" is
            // being tested rather than assumed.
            var trimmed = Rows.Clone();
            while (trimmed.RowCount > rows)
            {
                trimmed.Rows.RemoveAt(trimmed.RowCount - 1);
            }

            return Task.FromResult(trimmed);
        }

        public Task<TabularData> LoadAsync(string datasetId, int rowLimit = 0, CancellationToken ct = default) =>
            PreviewAsync(datasetId, rowLimit == 0 ? int.MaxValue : rowLimit, ct);

        public Task<Dataset> ImportAsync(string name, Stream content, DatasetFormat format,
            DatasetSourceKind source, string? ownerId, CancellationToken ct = default) =>
            throw new NotSupportedException();

        public Task<Dataset> SaveAsync(string name, TabularData data, DatasetFormat format,
            string? ownerId, CancellationToken ct = default) => throw new NotSupportedException();
    }

    private static TabularData Sample()
    {
        var table = TabularData.WithColumns("kota", "jumlah");
        table.AddRow("Bandung", 10);
        table.AddRow("Jakarta", null);
        table.AddRow("Surabaya", 30);

        return table;
    }

    private Dataset Seed(int rowCount = 3, bool withProfile = true)
    {
        using var scope = Services.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        var profile = new DatasetProfile
        {
            RowCount = rowCount,
            Columns =
            [
                new ColumnProfile { Name = "kota", DataType = ColumnDataType.Categorical },
                new ColumnProfile { Name = "jumlah", DataType = ColumnDataType.Numeric, MissingCount = 1 }
            ]
        };

        var dataset = new Dataset
        {
            Name = "Pelanggan",
            Format = DatasetFormat.Csv,
            Source = DatasetSourceKind.Upload,
            RowCount = rowCount,
            ColumnCount = 2,
            SizeBytes = 2048,
            StorageKey = "datasets/pelanggan.csv",
            ProfileJson = withProfile ? JsonSerializer.Serialize(profile) : null
        };

        db.Datasets.Add(dataset);
        db.SaveChanges();

        _datasets.Rows = Sample();
        return dataset;
    }

    private static IElement Button(IRenderedComponent<Page> page, string text) =>
        page.FindAll("button").First(b => b.TextContent.Contains(text));

    // -------------------------------------------------------------- opening

    [Fact]
    public void The_card_offers_the_rows_and_the_columns_separately()
    {
        Seed();

        var page = RenderComponent<Page>();
        var labels = page.FindAll("article button").Select(b => b.TextContent.Trim()).ToList();

        Assert.Contains("Lihat data", labels);
        Assert.Contains("Lihat kolom", labels);
    }

    [Fact]
    public void Lihat_data_opens_on_the_rows_and_asks_for_the_smallest_page()
    {
        Seed();

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();

        Assert.Equal([25], _datasets.Requested);
        Assert.Equal(3, page.FindAll(".dialog tbody tr").Count);
    }

    [Fact]
    public void Lihat_kolom_opens_on_the_profile_and_reads_no_rows()
    {
        Seed();

        var page = RenderComponent<Page>();
        Button(page, "Lihat kolom").Click();

        // Opening the profile must not pull the file off storage for nothing.
        Assert.Empty(_datasets.Requested);
        Assert.Contains("Rata-rata", page.Find(".dialog").TextContent);
    }

    [Fact]
    public void The_two_views_are_tabs_on_one_dialog()
    {
        Seed();

        var page = RenderComponent<Page>();
        Button(page, "Lihat kolom").Click();

        page.FindAll(".dialog .tab").First(t => t.TextContent.Trim() == "Data").Click();

        Assert.Single(_datasets.Requested);
        Assert.Equal(3, page.FindAll(".dialog tbody tr").Count);
    }

    // ---------------------------------------------------------------- rows

    [Fact]
    public void The_header_names_every_column_and_numbers_the_rows()
    {
        Seed();

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();

        var headers = page.FindAll(".dialog thead th").Select(h => h.TextContent.Trim()).ToList();

        Assert.Equal(["#", "kota", "jumlah"], headers);
        Assert.Equal("1", page.FindAll(".dialog tbody tr").First().Children[0].TextContent.Trim());
    }

    [Fact]
    public void A_gap_is_shown_as_a_gap_rather_than_an_empty_cell()
    {
        Seed();

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();

        var missing = page.FindAll(".dialog tbody tr")[1].Children[2];

        // An empty cell reads as a rendering fault; a missing value is a fact about the data.
        Assert.Equal("—", missing.TextContent.Trim());
        Assert.Contains("faint", missing.ClassName);
    }

    [Fact]
    public void Numbers_are_right_aligned_and_text_is_not()
    {
        Seed();

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();

        var row = page.FindAll(".dialog tbody tr").First();

        Assert.DoesNotContain("num", row.Children[1].ClassName);
        Assert.Contains("num", row.Children[2].ClassName);
    }

    [Fact]
    public void The_footnote_says_how_much_of_the_dataset_is_on_screen()
    {
        Seed(rowCount: 4_000);

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();

        // The thousands separator follows the server's culture, which the app does not pin, so
        // the assertion is on the sentence rather than on how the number happens to be grouped.
        var text = page.Find(".dialog").TextContent;

        Assert.Contains("3 baris pertama dari", text);
        Assert.Contains(4_000.ToString("N0"), text);
    }

    [Fact]
    public void A_dataset_smaller_than_the_page_says_it_is_all_there()
    {
        Seed();

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();

        Assert.Contains("Seluruh 3 baris", page.Find(".dialog").TextContent);
    }

    // ------------------------------------------------------------- row count

    [Fact]
    public void Choosing_more_rows_asks_the_service_again_rather_than_reslicing()
    {
        Seed(rowCount: 4_000);

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();
        page.Find(".dialog select").Change("500");

        // Re-reading is the point: the first call only ever fetched 25 rows.
        Assert.Equal([25, 500], _datasets.Requested);
    }

    // ----------------------------------------------------------- unhappy paths

    [Fact]
    public void A_file_that_cannot_be_read_explains_itself_instead_of_showing_nothing()
    {
        Seed();
        _datasets.Throws = new InvalidOperationException("Berkasnya hilang dari storage.");

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();

        var notice = page.Find(".dialog .notice--danger").TextContent;

        Assert.Contains("Tidak bisa membaca", notice);
        Assert.Contains("Berkasnya hilang", notice);
    }

    [Fact]
    public void An_empty_file_says_so_rather_than_drawing_an_empty_table()
    {
        Seed();
        _datasets.Rows = TabularData.WithColumns("kota");

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();

        Assert.Contains("tidak ada satu baris pun", page.Find(".dialog").TextContent);
        Assert.Empty(page.FindAll(".dialog tbody tr"));
    }

    [Fact]
    public void A_dataset_with_no_stored_profile_still_previews_its_rows()
    {
        Seed(withProfile: false);

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();

        // Alignment comes from the profile, so its absence must cost the alignment and nothing else.
        Assert.Equal(3, page.FindAll(".dialog tbody tr").Count);

        page.FindAll(".dialog .tab").First(t => t.TextContent.Trim() == "Kolom").Click();

        Assert.Contains("belum tersimpan", page.Find(".dialog").TextContent);
    }

    [Fact]
    public void Closing_the_dialog_drops_the_rows_it_was_holding()
    {
        Seed();

        var page = RenderComponent<Page>();
        Button(page, "Lihat data").Click();
        page.Find(".dialog button[aria-label=Tutup]").Click();

        Assert.Empty(page.FindAll(".dialog"));

        // Reopening reads again rather than showing a table that may be stale.
        Button(page, "Lihat data").Click();

        Assert.Equal([25, 25], _datasets.Requested);
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);

        try
        {
            Directory.Delete(_root, recursive: true);
        }
        catch (IOException)
        {
            // A file SQLite still holds is not worth failing a run over.
        }
    }
}
