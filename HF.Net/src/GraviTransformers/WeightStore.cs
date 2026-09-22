using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviTransformers;

/// <summary>
/// Uniform, lazy access to a checkpoint's parameters, whatever files it happens to be stored in.
/// </summary>
/// <remarks>
/// <para>
/// A repository's weights may be one safetensors file, a set of sharded safetensors with an index,
/// or a PyTorch pickle - and the same model is often published in more than one of those at once.
/// Callers should not have to care, so this resolves the layout once and then answers by parameter
/// name.
/// </para>
/// <para>
/// Tensors are read on demand and not cached. Loading a model touches each parameter once, and
/// caching would double peak memory for no benefit; a caller that genuinely wants a tensor twice
/// can hold onto it.
/// </para>
/// </remarks>
public sealed class WeightStore : IDisposable
{
    private readonly List<SafeTensorsReader> _safeTensors = [];
    private readonly List<PyTorchCheckpoint> _pickles = [];
    private readonly Dictionary<string, object> _owners = new(StringComparer.Ordinal);
    private bool _disposed;

    private WeightStore(string description) => Description = description;

    /// <summary>A human-readable note about which files back this store.</summary>
    public string Description { get; }

    /// <summary>Every parameter name available, sorted.</summary>
    public IReadOnlyList<string> Names => [.. _owners.Keys.Order(StringComparer.Ordinal)];

    /// <summary>Number of parameters available.</summary>
    public int Count => _owners.Count;

    /// <summary>Opens the weights in a local directory.</summary>
    /// <param name="directory">A directory holding the checkpoint files.</param>
    /// <exception cref="FileNotFoundException">The directory holds no weights this can read.</exception>
    public static WeightStore Open(string directory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (!Directory.Exists(directory)) throw new DirectoryNotFoundException(directory);

        // Preference order is deliberate. safetensors needs no interpretation, so it is both faster
        // to open and safe on an untrusted file; the pickle is the fallback for the many
        // repositories that never published one.
        var index = Path.Combine(directory, "model.safetensors.index.json");
        if (File.Exists(index))
        {
            var store = new WeightStore($"sharded safetensors ({Path.GetFileName(directory)})");
            foreach (var shard in SafeTensors.ReadShardIndex(index).Values.Distinct(StringComparer.Ordinal))
            {
                if (File.Exists(shard)) store.AddSafeTensors(shard);
            }

            if (store.Count > 0) return store;
            store.Dispose();
        }

        var single = Directory.GetFiles(directory, "*.safetensors");
        if (single.Length > 0)
        {
            var store = new WeightStore($"safetensors ({single.Length} file(s))");
            foreach (var file in single) store.AddSafeTensors(file);
            return store;
        }

        var pickles = Directory.GetFiles(directory, "*.bin")
            .Concat(Directory.GetFiles(directory, "*.pt"))
            .Concat(Directory.GetFiles(directory, "*.pth"))
            .ToArray();

        if (pickles.Length > 0)
        {
            var store = new WeightStore($"PyTorch checkpoint ({pickles.Length} file(s))");
            foreach (var file in pickles) store.AddPickle(file);
            return store;
        }

        throw new FileNotFoundException(
            $"'{directory}' holds no .safetensors and no .bin weights.", directory);
    }

    /// <summary>Downloads a Hub model's weights and opens them.</summary>
    /// <param name="repoId">A model id.</param>
    /// <param name="revision">A branch, tag or commit.</param>
    /// <param name="progress">Receives transfer progress.</param>
    /// <remarks>
    /// safetensors is requested first and the PyTorch pickle is fetched only if the repository has
    /// no safetensors at all - otherwise a routine model load pulls the same weights twice.
    /// </remarks>
    public static WeightStore FromPretrained(
        string repoId, string revision = "main", IProgress<TransferProgress>? progress = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repoId);

        var info = Hub.ModelInfo(repoId, revision);
        var client = Hub.Shared;

        var wanted = info.HasSafeTensors
            ? (string[])["*.safetensors", "*.safetensors.index.json", "config.json"]
            : ["pytorch_model.bin", "pytorch_model-*.bin", "config.json"];

        var directory = client
            .SnapshotAsync(repoId, RepoKind.Model, revision, wanted, null, progress)
            .GetAwaiter().GetResult();

        return Open(directory);
    }

    private void AddSafeTensors(string path)
    {
        var reader = SafeTensors.Open(path);
        _safeTensors.Add(reader);
        foreach (var tensor in reader.Tensors) _owners.TryAdd(tensor.Name, reader);
    }

    private void AddPickle(string path)
    {
        var checkpoint = PyTorchCheckpoint.Open(path);
        _pickles.Add(checkpoint);
        foreach (var tensor in checkpoint.Tensors) _owners.TryAdd(tensor.Name, checkpoint);
    }

    /// <summary>Whether a parameter of that name is present.</summary>
    public bool Contains(string name) => _owners.ContainsKey(name);

    /// <summary>Reads a parameter.</summary>
    /// <param name="name">The parameter's name.</param>
    /// <exception cref="KeyNotFoundException">No parameter of that name is present.</exception>
    public NdArray Read(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_owners.TryGetValue(name, out var owner))
        {
            throw new KeyNotFoundException($"'{name}' is not in this checkpoint ({Description}).");
        }

        return owner switch
        {
            SafeTensorsReader reader => reader.Read(name),
            PyTorchCheckpoint checkpoint => checkpoint.Read(name),
            _ => throw new InvalidOperationException("Unknown weight owner."),
        };
    }

    /// <summary>Reads a parameter if present.</summary>
    public bool TryRead(string name, out NdArray tensor)
    {
        if (Contains(name))
        {
            tensor = Read(name);
            return true;
        }

        tensor = null!;
        return false;
    }

    /// <summary>
    /// Reads the first parameter present among several candidate names.
    /// </summary>
    /// <param name="tensor">The tensor when one was found.</param>
    /// <param name="candidates">Names to try, in order.</param>
    /// <remarks>
    /// The same parameter is spelled differently across families - a classifier head is
    /// <c>classifier.weight</c> on BERT and <c>classifier.out_proj.weight</c> on RoBERTa - and
    /// trying a list is simpler than resolving the family first.
    /// </remarks>
    public bool TryReadAny(out NdArray tensor, params string[] candidates)
    {
        foreach (var candidate in candidates)
        {
            if (TryRead(candidate, out tensor)) return true;
        }

        tensor = null!;
        return false;
    }

    /// <summary>Names that begin with a prefix.</summary>
    public IReadOnlyList<string> NamesStartingWith(string prefix)
        => [.. _owners.Keys.Where(n => n.StartsWith(prefix, StringComparison.Ordinal)).Order(StringComparer.Ordinal)];

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var reader in _safeTensors) reader.Dispose();
        foreach (var checkpoint in _pickles) checkpoint.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"WeightStore: {Description}, {Count} tensors";
}
