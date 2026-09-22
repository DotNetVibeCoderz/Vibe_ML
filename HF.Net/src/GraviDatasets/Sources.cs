using System.Text.Json;
using Gravicode.HFNet.GraviHub;
using Gravicode.Science.GraviFrame;

namespace Gravicode.HFNet.GraviDatasets;

/// <summary>
/// The datasets that ship with the repository, so that <c>Dataset.Load("titanic")</c> works with no
/// network and no credentials.
/// </summary>
/// <remarks>
/// The files live in the repository's <c>datasets/</c> directory. Resolution walks up from both the
/// running assembly and the current directory, because a sample run from the repository root and a
/// test run from deep inside <c>bin/</c> both have to find them.
/// </remarks>
public static class BuiltinDatasets
{
    private static readonly Dictionary<string, string> Files = new(StringComparer.OrdinalIgnoreCase)
    {
        ["titanic"] = "titanic.csv",
        ["iris"] = "iris.csv",
        ["imdb"] = "imdb_reviews.csv",
        ["imdb_reviews"] = "imdb_reviews.csv",
        ["finance"] = "finance_timeseries.csv",
        ["finance_timeseries"] = "finance_timeseries.csv",
        ["sms_spam"] = "sms_spam.csv",
    };

    /// <summary>The names this registry knows.</summary>
    public static IReadOnlyCollection<string> Names => Files.Keys;

    /// <summary>Finds a built-in dataset's file, if it is present.</summary>
    /// <param name="name">The dataset name.</param>
    /// <param name="path">The resolved path when the method returns true.</param>
    public static bool TryResolve(string name, out string path)
    {
        path = "";
        if (!Files.TryGetValue(name, out var file)) return false;

        var directory = FindDatasetsDirectory();
        if (directory is null) return false;

        var candidate = Path.Combine(directory, file);
        if (!File.Exists(candidate)) return false;

        path = candidate;
        return true;
    }

    /// <summary>The repository's <c>datasets/</c> directory, or null when it cannot be found.</summary>
    public static string? FindDatasetsDirectory()
    {
        foreach (var start in (string?[])[AppContext.BaseDirectory, Environment.CurrentDirectory])
        {
            if (start is null) continue;

            var directory = new DirectoryInfo(start);
            while (directory is not null)
            {
                var candidate = Path.Combine(directory.FullName, "datasets");
                if (Directory.Exists(candidate)) return candidate;

                directory = directory.Parent;
            }
        }

        return null;
    }
}

/// <summary>Loads datasets from the Hugging Face Hub.</summary>
/// <remarks>
/// <para>
/// A dataset repository has no single layout. Most now publish Parquet under a per-config
/// directory, older ones publish CSV or JSON Lines at the root, and some publish only a loading
/// script - which is Python and cannot run here. The strategy is therefore: ask the Hub's dataset
/// server which Parquet files back each split, and fall back to pattern-matching the file listing
/// when the server has no answer.
/// </para>
/// <para>
/// A script-only dataset is refused with a message saying so. That is a real limit and it is better
/// stated plainly than worked around by guessing at the script's intent.
/// </para>
/// </remarks>
public static class HubDatasets
{
    private const string ServerEndpoint = "https://datasets-server.huggingface.co";

    /// <summary>Loads one split of a Hub dataset.</summary>
    /// <param name="repoId">A dataset id such as <c>stanfordnlp/imdb</c>.</param>
    /// <param name="split">The split name.</param>
    public static Dataset Load(string repoId, string split = "train")
    {
        var all = LoadAll(repoId, split);
        return all.Contains(split) ? all[split] : all[all.Splits.First()];
    }

    /// <summary>Loads every split of a Hub dataset.</summary>
    /// <param name="repoId">A dataset id.</param>
    public static DatasetDict LoadAll(string repoId) => LoadAll(repoId, null);

    private static DatasetDict LoadAll(string repoId, string? onlySplit)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var fromServer = TryLoadViaServer(repoId, onlySplit);
        if (fromServer is not null) return fromServer;

        return LoadByFilePattern(repoId, onlySplit);
    }

    /// <summary>
    /// Asks the dataset server for the Parquet conversion of a dataset.
    /// </summary>
    /// <remarks>
    /// The Hub converts most public datasets to Parquet automatically and indexes the result. Using
    /// that index is what makes a script-based dataset loadable at all, and it avoids having to
    /// guess which of a repository's files are data and which are documentation.
    /// </remarks>
    private static DatasetDict? TryLoadViaServer(string repoId, string? onlySplit)
    {
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };

            var token = new HubOptions().ResolveToken();
            if (!string.IsNullOrWhiteSpace(token))
            {
                http.DefaultRequestHeaders.Authorization = new("Bearer", token);
            }

            var url = $"{ServerEndpoint}/parquet?dataset={Uri.EscapeDataString(repoId)}";
            var response = http.GetAsync(url).GetAwaiter().GetResult();
            if (!response.IsSuccessStatusCode) return null;

            var json = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            using var document = JsonDocument.Parse(json);

            if (!document.RootElement.TryGetProperty("parquet_files", out var files)) return null;

            var bySplit = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
            foreach (var entry in files.EnumerateArray())
            {
                var split = entry.GetProperty("split").GetString() ?? "train";
                if (onlySplit is not null && !string.Equals(split, onlySplit, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (!bySplit.TryGetValue(split, out var list)) bySplit[split] = list = [];
                list.Add(entry.GetProperty("url").GetString()!);
            }

            if (bySplit.Count == 0) return null;

            var splits = new Dictionary<string, Dataset>(StringComparer.OrdinalIgnoreCase);
            foreach (var (split, urls) in bySplit)
            {
                var frames = new List<DataFrame>();
                foreach (var fileUrl in urls)
                {
                    var local = DownloadToCache(http, fileUrl, repoId, split, frames.Count);
                    frames.Add(Gravicode.Science.GraviFrame.Io.ParquetIO.Read(local));
                }

                splits[split] = new Dataset(
                    frames.Count == 1 ? frames[0] : DataFrame.Concat(frames),
                    $"{repoId}/{split}");
            }

            return new DatasetDict(splits);
        }
        catch (Exception exception) when (exception is HttpRequestException or JsonException or TaskCanceledException)
        {
            // The server is an optimisation, not the source of truth. Falling through to the file
            // listing keeps a dataset loadable when the conversion is missing or the server is down.
            return null;
        }
    }

    private static string DownloadToCache(HttpClient http, string url, string repoId, string split, int index)
    {
        var directory = Path.Combine(
            new HubCache().Root, "datasets", repoId.Replace('/', '_'), "parquet", split);
        Directory.CreateDirectory(directory);

        var path = Path.Combine(directory, $"part-{index:D5}.parquet");
        if (File.Exists(path)) return path;

        using var response = http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();

        var partial = path + ".part";
        using (var output = File.Create(partial))
        {
            response.Content.CopyToAsync(output).GetAwaiter().GetResult();
        }

        File.Move(partial, path, overwrite: true);
        return path;
    }

    /// <summary>Falls back to reading whatever data files the repository publishes.</summary>
    private static DatasetDict LoadByFilePattern(string repoId, string? onlySplit)
    {
        var info = Hub.DatasetInfo(repoId);

        var dataFiles = info.Files
            .Where(f => f.Extension is ".csv" or ".tsv" or ".parquet" or ".json" or ".jsonl")
            .ToList();

        if (dataFiles.Count == 0)
        {
            var scripts = info.Files.Count(f => f.Extension == ".py");
            throw new HubException(
                $"'{repoId}' publishes no CSV, Parquet or JSON files"
                + (scripts > 0
                    ? " - only a Python loading script, which HF.Net cannot run. Datasets the Hub has converted to Parquet load fine; this one has not been converted."
                    : ".")) { RepoId = repoId };
        }

        var splits = new Dictionary<string, Dataset>(StringComparer.OrdinalIgnoreCase);

        foreach (var file in dataFiles)
        {
            var split = GuessSplit(file.Path);
            if (onlySplit is not null && !string.Equals(split, onlySplit, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var local = Hub.DownloadDatasetFile(repoId, file.Path);
            var dataset = Dataset.FromFile(local, $"{repoId}/{split}");

            // Several files can back one split; concatenating keeps the split whole.
            splits[split] = splits.TryGetValue(split, out var existing)
                ? new Dataset(DataFrame.Concat([existing.Frame, dataset.Frame]), existing.Name)
                : dataset;
        }

        if (splits.Count == 0)
        {
            throw new HubException($"'{repoId}' has no files for split '{onlySplit}'.") { RepoId = repoId };
        }

        return new DatasetDict(splits);
    }

    /// <summary>Guesses a split name from a file path.</summary>
    /// <remarks>
    /// Substring matching on the whole path, because the split usually appears as a directory
    /// (<c>data/train/0000.parquet</c>) as often as it does in the file name.
    /// </remarks>
    private static string GuessSplit(string path)
    {
        var lower = path.ToLowerInvariant();

        if (lower.Contains("valid") || lower.Contains("/dev")) return "validation";
        if (lower.Contains("test")) return "test";
        if (lower.Contains("train")) return "train";

        return "train";
    }
}

/// <summary>Reads JSON and JSON Lines into a <see cref="DataFrame"/>.</summary>
/// <remarks>
/// Both spellings are common on the Hub and they are told apart by inspecting the first
/// non-whitespace character rather than by the extension: plenty of repositories publish a JSON
/// array in a file named <c>.jsonl</c> and the reverse.
/// </remarks>
public static class JsonLines
{
    /// <summary>Reads a JSON array or a JSON Lines file.</summary>
    /// <param name="path">The file.</param>
    public static DataFrame Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var rows = new List<Dictionary<string, JsonElement>>();

        using (var stream = File.OpenRead(path))
        using (var probe = new StreamReader(stream, leaveOpen: true))
        {
            int first;
            while ((first = probe.Peek()) >= 0 && char.IsWhiteSpace((char)first)) probe.Read();

            if (first == '[')
            {
                probe.DiscardBufferedData();
                stream.Position = 0;

                using var document = JsonDocument.Parse(stream);
                foreach (var element in document.RootElement.EnumerateArray()) rows.Add(Flatten(element));
            }
            else
            {
                probe.DiscardBufferedData();
                stream.Position = 0;

                using var reader = new StreamReader(stream, leaveOpen: true);
                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    if (line.Length == 0) continue;

                    using var document = JsonDocument.Parse(line);
                    rows.Add(Flatten(document.RootElement));
                }
            }
        }

        return Build(rows);
    }

    private static Dictionary<string, JsonElement> Flatten(JsonElement element)
    {
        var row = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        if (element.ValueKind != JsonValueKind.Object) return row;

        // Cloned, not stored by reference. A JsonElement is a window onto its JsonDocument's
        // buffer and stops being readable the moment that document is disposed - which here is at
        // the end of the line being parsed, long before the rows are turned into columns.
        foreach (var property in element.EnumerateObject()) row[property.Name] = property.Value.Clone();

        return row;
    }

    private static DataFrame Build(List<Dictionary<string, JsonElement>> rows)
    {
        if (rows.Count == 0) return DataFrame.Empty();

        // Column order follows first appearance so a round trip through JSON Lines preserves the
        // shape a reader expects, rather than sorting the columns alphabetically.
        var names = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var row in rows)
        {
            foreach (var key in row.Keys)
            {
                if (seen.Add(key)) names.Add(key);
            }
        }

        var columns = new List<Series>(names.Count);

        foreach (var name in names)
        {
            var numeric = rows.All(r =>
                !r.TryGetValue(name, out var value)
                || value.ValueKind is JsonValueKind.Number or JsonValueKind.Null or JsonValueKind.Undefined);

            if (numeric)
            {
                var values = new double[rows.Count];
                for (var i = 0; i < rows.Count; i++)
                {
                    values[i] = rows[i].TryGetValue(name, out var value) && value.ValueKind == JsonValueKind.Number
                        ? value.GetDouble()
                        : double.NaN;
                }
                columns.Add(new NumericSeries(name, values));
            }
            else
            {
                var values = new string?[rows.Count];
                for (var i = 0; i < rows.Count; i++)
                {
                    values[i] = rows[i].TryGetValue(name, out var value)
                        ? value.ValueKind switch
                        {
                            JsonValueKind.String => value.GetString(),
                            JsonValueKind.Null or JsonValueKind.Undefined => null,
                            _ => value.GetRawText(),
                        }
                        : null;
                }
                columns.Add(new TextSeries(name, values));
            }
        }

        return new DataFrame(columns);
    }
}
