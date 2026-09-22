using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;

// GraviHub sample - talking to the Hugging Face Hub from .NET.
// Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.

Console.WriteLine("=== GraviHub: Hugging Face Hub from .NET ===");
Console.WriteLine();

// A token is optional for public repositories and raises the rate limit everywhere.
// It is read from HF_TOKEN, so nothing here has to hold a secret.
var account = Hub.WhoAmI();
Console.WriteLine(account is null
    ? "Anonymous (set HF_TOKEN for private repositories and a higher rate limit)."
    : $"Signed in as {account}.");
Console.WriteLine($"Cache: {Hub.Cache.Root}");
Console.WriteLine();

// ---------------------------------------------------------------- 1. metadata
Console.WriteLine("--- 1. Repository metadata ---");
var info = Hub.ModelInfo("prajjwal1/bert-tiny");
Console.WriteLine(info);
Console.WriteLine($"Pipeline : {info.PipelineTag ?? "(none declared)"}");
Console.WriteLine($"Tags     : {string.Join(", ", info.Tags.Take(8))}");
Console.WriteLine($"Weights  : {(info.HasSafeTensors ? "safetensors present" : "no safetensors")}");
Console.WriteLine();

Console.WriteLine("Files:");
foreach (var file in info.Files.Take(12)) Console.WriteLine($"  {file}");
Console.WriteLine();

// ---------------------------------------------------------------- 2. search
Console.WriteLine("--- 2. Search ---");
foreach (var hit in Hub.SearchModels("sentiment", limit: 5, task: "text-classification"))
{
    Console.WriteLine($"  {hit.Id,-52} {hit.Downloads,12:N0} downloads");
}
Console.WriteLine();

// ---------------------------------------------------------------- 3. download
Console.WriteLine("--- 3. Download ---");
var progress = new Progress<TransferProgress>(p =>
{
    if (p.Fraction is { } f && (p.BytesTransferred == p.TotalBytes || f == 0))
    {
        Console.WriteLine($"  {p}");
    }
});

var directory = await Hub.DownloadModelAsync("prajjwal1/bert-tiny", progress: progress);
Console.WriteLine($"Snapshot at: {directory}");

foreach (var file in Directory.GetFiles(directory))
{
    Console.WriteLine($"  {Path.GetFileName(file),-34} {new FileInfo(file).Length,12:N0} bytes");
}
Console.WriteLine();

// ---------------------------------------------------------------- 4. reading weights
// Two formats are in play on the Hub. safetensors is preferred where it exists: it needs no
// interpretation, so it is both faster and safe to open. A great many repositories - this one
// included - still publish only the PyTorch pickle, so both readers earn their place.
Console.WriteLine("--- 4. Reading the weights ---");

var safe = Path.Combine(directory, "model.safetensors");

if (File.Exists(safe))
{
    using var reader = SafeTensors.Open(safe);
    Console.WriteLine(reader);
    Describe(reader.Tensors.Select(t => (t.Name, t.Shape, t.DType.ToString())).ToList(), reader.Read);
}
else
{
    Console.WriteLine("  No safetensors here; falling back to the PyTorch checkpoint.");

    var bin = await Hub.Shared.DownloadFileAsync("prajjwal1/bert-tiny", "pytorch_model.bin", progress: progress);
    using var checkpoint = PyTorchCheckpoint.Open(bin);

    Console.WriteLine($"  {checkpoint}");
    Describe(checkpoint.Tensors.Select(t => (t.Name, t.Shape, t.DType.ToString())).ToList(), checkpoint.Read);
}

static void Describe(
    IReadOnlyList<(string Name, int[] Shape, string DType)> tensors,
    Func<string, Gravicode.Science.GraviNum.NdArray> read)
{
    foreach (var tensor in tensors.Take(6))
    {
        Console.WriteLine($"    {tensor.Name,-52} {tensor.DType,-5} [{string.Join(" x ", tensor.Shape)}]");
    }
    Console.WriteLine($"    ... {tensors.Count} tensors in total");
    Console.WriteLine();

    // Pull one tensor out and look at it. Nothing else in the file is touched.
    var name = tensors.First(t => t.Name.Contains("word_embeddings", StringComparison.Ordinal)).Name;
    var matrix = read(name);

    Console.WriteLine($"  {name}");
    Console.WriteLine($"    shape  : [{string.Join(" x ", matrix.Shape.ToArray())}]");
    Console.WriteLine($"    row 101: {string.Join(", ", matrix.Row(101).ToArray().Take(6).Select(v => v.ToString("F4")))} ...");
}

Console.WriteLine();
Console.WriteLine("Selesai. Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.");
