using System.IO.Compression;
using Gravicode.Science.GraviNum;

namespace Gravicode.HFNet.GraviHub.Io;

/// <summary>One parameter inside a <c>pytorch_model.bin</c>.</summary>
/// <param name="Name">The parameter path, for example <c>bert.encoder.layer.0.output.dense.weight</c>.</param>
/// <param name="DType">The stored element type.</param>
/// <param name="Shape">The dimensions, outermost first.</param>
public readonly record struct TorchTensorInfo(string Name, SafeTensorDType DType, int[] Shape)
{
    /// <summary>Number of elements the shape implies.</summary>
    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (var d in Shape) n *= d;
            return n;
        }
    }

    /// <inheritdoc />
    public override string ToString() => $"{Name} {DType} [{string.Join("x", Shape)}]";
}

/// <summary>
/// Reads a PyTorch <c>.bin</c> / <c>.pt</c> checkpoint - a ZIP holding a pickled state dict beside
/// raw tensor storages, or the legacy pre-1.6 layout of bare pickles followed by the storages.
/// </summary>
/// <remarks>
/// <para>
/// This exists because most of the Hub is still shipped this way. A great many popular models -
/// <c>prajjwal1/bert-tiny</c> and every <c>hf-internal-testing</c> fixture among them - carry only
/// <c>pytorch_model.bin</c>, so a loader that understands safetensors alone cannot open them.
/// </para>
/// <para>
/// The pickle is read by <see cref="PickleReader"/>, which resolves names against an allow-list
/// rather than calling them. Nothing in a checkpoint is executed. That is a real restriction and
/// not a theoretical one: a <c>.bin</c> is arbitrary code by construction, and this reader is
/// exactly as useful on a well-formed checkpoint and inert on a hostile one. Where both formats
/// are published, prefer safetensors anyway - it needs no interpretation at all.
/// </para>
/// <para>
/// The legacy pre-1.6 format is still common on the Hub (<c>ProsusAI/finbert</c> ships only that).
/// It is five pickles in a row - magic number, protocol version (1001), system info, the state
/// dict, and the list of storage keys - followed by each storage in that order as a little-endian
/// int64 element count and the raw elements. Opening one reads the pickles and walks the storage
/// headers to index where each storage's data begins; tensor data is read only on demand.
/// </para>
/// </remarks>
public sealed class PyTorchCheckpoint : IDisposable
{
    /// <summary>The protocol version torch's legacy serializer writes as its second pickle.</summary>
    private const long LegacyProtocolVersion = 1001;

    private readonly ZipArchive? _archive;
    private readonly Dictionary<string, (long Offset, long Count)>? _legacyStorages;
    private readonly Dictionary<string, TorchTensor> _tensors = new(StringComparer.Ordinal);
    private readonly string _prefix = "";
    private bool _disposed;

    private PyTorchCheckpoint(string path, bool legacy)
    {
        Path = path;
        if (legacy)
        {
            _legacyStorages = OpenLegacy(path, out var legacyRoot);
            Flatten(Unwrap(legacyRoot), prefix: "", _tensors);
            Tensors = [.. _tensors.Select(p => new TorchTensorInfo(p.Key, p.Value.DType, p.Value.Shape))
                .OrderBy(t => t.Name, StringComparer.Ordinal)];
            return;
        }

        _archive = ZipFile.OpenRead(path);

        var pickleEntry = _archive.Entries.FirstOrDefault(e =>
            e.FullName.EndsWith("data.pkl", StringComparison.Ordinal))
            ?? throw new InvalidDataException(
                $"'{path}' is a ZIP but has no data.pkl. It does not look like a torch checkpoint.");

        _prefix = pickleEntry.FullName[..^"data.pkl".Length];

        using var stream = pickleEntry.Open();
        using var buffered = new MemoryStream();
        stream.CopyTo(buffered);
        buffered.Position = 0;

        var reader = new PickleReader(buffered, ResolveStorage);
        var root = reader.Load();

        Flatten(Unwrap(root), prefix: "", _tensors);
        Tensors = [.. _tensors.Select(p => new TorchTensorInfo(p.Key, p.Value.DType, p.Value.Shape))
            .OrderBy(t => t.Name, StringComparer.Ordinal)];
    }

    /// <summary>Opens a checkpoint, reading its index but none of its tensor data.</summary>
    /// <param name="path">Path to a <c>.bin</c>, <c>.pt</c> or <c>.pth</c> file.</param>
    public static PyTorchCheckpoint Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path)) throw new FileNotFoundException("Checkpoint not found.", path);

        // A ZIP starts with 'PK'. Anything else here is the pre-1.6 bare-pickle layout, whose
        // storages are concatenated after the pickles with no index at all.
        Span<byte> magic = stackalloc byte[2];
        bool legacy;
        using (var probe = File.OpenRead(path))
        {
            legacy = probe.Read(magic) == 2 && !(magic[0] == 'P' && magic[1] == 'K');
        }

        return new PyTorchCheckpoint(path, legacy);
    }

    /// <summary>
    /// Reads a legacy checkpoint's pickles and indexes its storages: for each key, where its elements
    /// start in the file and how many there are.
    /// </summary>
    private static Dictionary<string, (long Offset, long Count)> OpenLegacy(string path, out object? root)
    {
        var name = System.IO.Path.GetFileName(path);
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);

        // The magic number is an 80-bit integer the pickle reader cannot hold, so the protocol
        // version that follows it is what identifies the format.
        new PickleReader(file).Load();
        var protocol = new PickleReader(file).Load();
        if (protocol is not long version || version != LegacyProtocolVersion)
        {
            throw new InvalidDataException(
                $"'{name}' is neither a ZIP nor a legacy torch checkpoint (protocol {protocol ?? "none"}, expected {LegacyProtocolVersion}).");
        }

        new PickleReader(file).Load(); // system info: endianness and type sizes
        if (!BitConverter.IsLittleEndian)
        {
            throw new NotSupportedException("Legacy torch checkpoints are read on little-endian hosts only.");
        }

        var storages = new Dictionary<string, StorageRef>(StringComparer.Ordinal);
        root = new PickleReader(file, id =>
        {
            // Legacy ids carry a sixth element, view metadata, which is null unless a storage was
            // saved as a view of another - something torch stopped producing long before 1.6.
            if (id.Count > 5 && id[5] is not null)
            {
                throw new NotSupportedException($"'{name}' stores a storage view, which is not supported.");
            }
            var storage = (StorageRef)ResolveStorage(id);
            storages[storage.Key] = storage;
            return storage;
        }).Load();

        if (new PickleReader(file).Load() is not List<object?> keys)
        {
            throw new InvalidDataException($"'{name}' has no storage key list after its state dict.");
        }

        var index = new Dictionary<string, (long Offset, long Count)>(StringComparer.Ordinal);
        Span<byte> header = stackalloc byte[8];
        foreach (var keyObject in keys)
        {
            var key = Convert.ToString(keyObject) ?? "";
            if (!storages.TryGetValue(key, out var storage))
            {
                throw new InvalidDataException($"'{name}' lists storage '{key}' that no tensor refers to.");
            }

            file.ReadExactly(header);
            var count = System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(header);
            var bytes = checked(count * SafeTensors.ElementSize(storage.DType));
            if (count < 0 || file.Position + bytes > file.Length)
            {
                throw new InvalidDataException($"Storage '{key}' in '{name}' runs past the end of the file.");
            }

            index[key] = (file.Position, count);
            file.Seek(bytes, SeekOrigin.Current);
        }

        return index;
    }

    /// <summary>Lists the parameters in a checkpoint without reading their data.</summary>
    public static IReadOnlyList<TorchTensorInfo> Inspect(string path)
    {
        using var checkpoint = Open(path);
        return checkpoint.Tensors;
    }

    /// <summary>Reads every parameter in a checkpoint into memory.</summary>
    public static Dictionary<string, NdArray> ReadAll(string path)
    {
        using var checkpoint = Open(path);
        var result = new Dictionary<string, NdArray>(StringComparer.Ordinal);
        foreach (var tensor in checkpoint.Tensors) result[tensor.Name] = checkpoint.Read(tensor.Name);
        return result;
    }

    /// <summary>The file this checkpoint was opened on.</summary>
    public string Path { get; }

    /// <summary>Every parameter in the checkpoint, ordered by name.</summary>
    public IReadOnlyList<TorchTensorInfo> Tensors { get; }

    /// <summary>Whether a parameter of that name is present.</summary>
    public bool Contains(string name) => _tensors.ContainsKey(name);

    /// <summary>Reads one parameter, widening it to <see cref="double"/>.</summary>
    /// <param name="name">The parameter path.</param>
    /// <exception cref="KeyNotFoundException">No parameter of that name is present.</exception>
    public NdArray Read(string name)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_tensors.TryGetValue(name, out var tensor))
        {
            throw new KeyNotFoundException($"'{name}' is not in {System.IO.Path.GetFileName(Path)}.");
        }

        var storage = ReadStorage(tensor.StorageKey, tensor.DType);
        var count = checked((int)tensor.ElementCount);
        var data = new double[count];

        if (tensor.IsContiguous)
        {
            Array.Copy(storage, tensor.StorageOffset, data, 0, count);
        }
        else
        {
            // A transposed or sliced parameter shares its storage with the tensor it came from, so
            // the strides are what say which elements are actually ours. Walking them in row-major
            // order is what turns the view back into a dense array.
            var rank = tensor.Shape.Length;
            var index = new int[rank];

            for (var i = 0; i < count; i++)
            {
                long offset = tensor.StorageOffset;
                for (var d = 0; d < rank; d++) offset += (long)index[d] * tensor.Stride[d];
                data[i] = storage[offset];

                for (var d = rank - 1; d >= 0; d--)
                {
                    if (++index[d] < tensor.Shape[d]) break;
                    index[d] = 0;
                }
            }
        }

        return new NdArray(data, tensor.Shape);
    }

    /// <summary>Reads a parameter if present, without throwing when it is not.</summary>
    public bool TryRead(string name, out NdArray tensor)
    {
        if (_tensors.ContainsKey(name))
        {
            tensor = Read(name);
            return true;
        }

        tensor = null!;
        return false;
    }

    // ------------------------------------------------------------------ internals

    /// <summary>Reads a storage entry out of the ZIP and widens it.</summary>
    private double[] ReadStorage(string key, SafeTensorDType dtype)
    {
        var width = SafeTensors.ElementSize(dtype);
        Stream stream;
        int count;

        if (_legacyStorages is not null)
        {
            if (!_legacyStorages.TryGetValue(key, out var location))
            {
                throw new InvalidDataException($"Storage '{key}' is missing from {System.IO.Path.GetFileName(Path)}.");
            }
            count = checked((int)location.Count);
            var file = new FileStream(Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
            file.Seek(location.Offset, SeekOrigin.Begin);
            stream = file;
        }
        else
        {
            var entry = _archive!.GetEntry(_prefix + "data/" + key)
                ?? throw new InvalidDataException($"Storage '{key}' is missing from {System.IO.Path.GetFileName(Path)}.");
            count = checked((int)(entry.Length / width));
            stream = entry.Open();
        }

        var values = new double[count];
        using var _ = stream;

        // Torch stores these uncompressed, but the entry still has to be read sequentially through
        // the ZIP stream rather than mapped, so it goes through a staging buffer.
        const int ChunkElements = 1 << 16;
        var staging = new byte[ChunkElements * width];

        for (var i = 0; i < count; i += ChunkElements)
        {
            var take = Math.Min(ChunkElements, count - i);
            var bytes = take * width;
            stream.ReadExactly(staging, 0, bytes);
            SafeTensors.DecodeBlock(staging.AsSpan(0, bytes), values.AsSpan(i, take), dtype);
        }

        return values;
    }

    /// <summary>Turns a storage persistent id into the reference the rebuild step will use.</summary>
    private static object ResolveStorage(IReadOnlyList<object?> id)
    {
        // ('storage', <torch.FloatStorage>, key, location, numel)
        if (id.Count < 5 || id[0] as string != "storage")
        {
            throw new NotSupportedException(
                $"Unsupported persistent id of {id.Count} parts; expected a 5-part storage tuple.");
        }

        var storageType = id[1] as PickleGlobal
            ?? throw new InvalidDataException("Storage tuple has no storage type.");

        return new StorageRef(
            Key: Convert.ToString(id[2]) ?? "",
            DType: DTypeFor(storageType.Name),
            Count: Convert.ToInt64(id[4]));
    }

    private static SafeTensorDType DTypeFor(string storageType) => storageType switch
    {
        "FloatStorage" => SafeTensorDType.F32,
        "DoubleStorage" => SafeTensorDType.F64,
        "HalfStorage" => SafeTensorDType.F16,
        "BFloat16Storage" => SafeTensorDType.BF16,
        "LongStorage" => SafeTensorDType.I64,
        "IntStorage" => SafeTensorDType.I32,
        "ShortStorage" => SafeTensorDType.I16,
        "CharStorage" => SafeTensorDType.I8,
        "ByteStorage" => SafeTensorDType.U8,
        "BoolStorage" => SafeTensorDType.Bool,
        _ => throw new NotSupportedException($"Unsupported torch storage type '{storageType}'."),
    };

    /// <summary>
    /// Picks the state dict out of whatever the checkpoint's top level turned out to be.
    /// </summary>
    /// <remarks>
    /// A checkpoint saved for inference is the state dict itself; one saved mid-training is a dict
    /// with the weights under <c>state_dict</c> or <c>model</c> beside an optimiser and an epoch
    /// counter. Both shapes are common on the Hub.
    /// </remarks>
    private static object? Unwrap(object? root)
    {
        if (root is not Dictionary<object, object?> dictionary) return root;

        foreach (var key in (string[])["state_dict", "model", "module"])
        {
            if (dictionary.TryGetValue(key, out var inner) && inner is Dictionary<object, object?>)
            {
                return inner;
            }
        }

        return dictionary;
    }

    /// <summary>Walks the state dict, recording every rebuilt tensor under its dotted path.</summary>
    private static void Flatten(object? node, string prefix, Dictionary<string, TorchTensor> into)
    {
        switch (node)
        {
            case Dictionary<object, object?> dictionary:
                foreach (var pair in dictionary)
                {
                    var name = Convert.ToString(pair.Key) ?? "";
                    Flatten(pair.Value, prefix.Length == 0 ? name : $"{prefix}.{name}", into);
                }
                return;

            case PickleReduce reduce:
                if (TryBuildTensor(reduce, out var tensor)) into[prefix] = tensor;
                return;
        }
    }

    /// <summary>Interprets a captured <c>_rebuild_tensor_v2</c> call.</summary>
    private static bool TryBuildTensor(PickleReduce reduce, out TorchTensor tensor)
    {
        tensor = default!;

        // _rebuild_parameter wraps a tensor; unwrap one level and carry on.
        if (reduce.Callable?.Name == "_rebuild_parameter" && reduce.Arguments.Count > 0)
        {
            return reduce.Arguments[0] is PickleReduce inner && TryBuildTensor(inner, out tensor);
        }

        if (reduce.Callable?.Name is not ("_rebuild_tensor" or "_rebuild_tensor_v2")) return false;
        if (reduce.Arguments.Count < 4) return false;

        if (reduce.Arguments[0] is not StorageRef storage) return false;

        var offset = Convert.ToInt64(reduce.Arguments[1]);
        var shape = ToIntArray(reduce.Arguments[2]);
        var stride = ToIntArray(reduce.Arguments[3]);

        // A zero-dimensional tensor is a scalar; NdArray has no rank 0, so it becomes length one.
        if (shape.Length == 0)
        {
            shape = [1];
            stride = [1];
        }

        tensor = new TorchTensor(storage.Key, storage.DType, shape, stride, offset);
        return true;
    }

    private static int[] ToIntArray(object? value) => value switch
    {
        List<object?> list => [.. list.Select(v => Convert.ToInt32(v))],
        null => [],
        _ => [Convert.ToInt32(value)],
    };

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _archive?.Dispose();
    }

    /// <inheritdoc />
    public override string ToString()
        => $"{System.IO.Path.GetFileName(Path)}: {Tensors.Count} tensors";

    /// <summary>A storage a persistent id pointed at, before any tensor claims part of it.</summary>
    private sealed record StorageRef(string Key, SafeTensorDType DType, long Count);

    /// <summary>A tensor's view over a storage: where it starts and how it steps.</summary>
    private sealed record TorchTensor(
        string StorageKey, SafeTensorDType DType, int[] Shape, int[] Stride, long StorageOffset)
    {
        public long ElementCount
        {
            get
            {
                long n = 1;
                foreach (var d in Shape) n *= d;
                return n;
            }
        }

        /// <summary>Whether the strides are exactly row-major, so the data can be block-copied.</summary>
        public bool IsContiguous
        {
            get
            {
                if (Stride.Length != Shape.Length) return false;

                var expected = 1;
                for (var d = Shape.Length - 1; d >= 0; d--)
                {
                    if (Shape[d] != 1 && Stride[d] != expected) return false;
                    expected *= Shape[d];
                }
                return true;
            }
        }
    }
}
