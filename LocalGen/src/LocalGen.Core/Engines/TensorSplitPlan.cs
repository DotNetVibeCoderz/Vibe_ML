using System.Globalization;
using System.Text.Json.Serialization;

namespace LocalGen.Core.Engines;

/// <summary>How weights are divided when more than one accelerator is present.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<GpuSplitMode>))]
public enum GpuSplitMode
{
    /// <summary>Leave the decision to the backend, which splits by layer.</summary>
    Auto,

    /// <summary>Keep the whole model on one device — the one named by <c>MainGpu</c>.</summary>
    None,

    /// <summary>Give each device a contiguous run of layers. The safe multi-GPU default.</summary>
    Layer,

    /// <summary>
    /// Split individual tensors by row across devices, so every device works on every layer.
    /// </summary>
    /// <remarks>
    /// This is tensor parallelism proper. It lowers latency for a single request because the
    /// devices compute together rather than in turn, but it exchanges activations on every layer
    /// and so needs a fast link between the cards — on consumer boards without NVLink it is
    /// usually slower than <see cref="Layer"/>, not faster.
    /// </remarks>
    Row
}

/// <summary>One accelerator ggml registered, as reported by the loaded native backend.</summary>
/// <remarks>
/// This is the authoritative count of GPUs available to inference, which is not the same as the
/// number of cards in the machine: it reflects the backend that was actually built (a CPU-only
/// build sees none) and any device masking such as <c>CUDA_VISIBLE_DEVICES</c>.
/// </remarks>
public sealed record AcceleratorDevice
{
    /// <summary>Position in the backend's device list. This is the index a tensor split addresses.</summary>
    public required int Index { get; init; }

    /// <summary>Backend device name — <c>CUDA0</c>, <c>Vulkan1</c>, <c>Metal</c>.</summary>
    public required string Name { get; init; }

    public override string ToString() => Name;
}

/// <summary>
/// A validated per-device weight split, resolved against the accelerators actually present.
/// </summary>
/// <remarks>
/// Kept apart from the backend because the failure modes here are configuration mistakes rather
/// than inference problems, and they are silent ones: a split naming three GPUs on a two-GPU host
/// loads without complaint and simply ignores the third weight, and a split written as percentages
/// adding to 100 behaves identically to one written as fractions adding to 1 — so a user who
/// believes those differ has no way to find out from the result. Resolving the split here means
/// the mistakes can be reported and unit-tested on a machine with no GPU at all.
/// </remarks>
public sealed record TensorSplitPlan
{
    /// <summary>
    /// Fraction of the weights placed on each device, in device order, summing to 1. Empty means
    /// the backend decides, which it does in proportion to free VRAM.
    /// </summary>
    public IReadOnlyList<float> Fractions { get; init; } = [];

    /// <summary>Configuration problems found while resolving, for the caller to log.</summary>
    public IReadOnlyList<string> Warnings { get; init; } = [];

    /// <summary>Whether the backend's own VRAM-proportional split is being used.</summary>
    public bool IsAutomatic => Fractions.Count == 0;

    public static readonly TensorSplitPlan Automatic = new();

    /// <summary>
    /// Resolves a configured split against the devices present, normalising the weights and
    /// reporting anything that will not do what the configuration says.
    /// </summary>
    /// <param name="requested">
    /// Relative weights per device. Any positive scale works — <c>0.6, 0.4</c>, <c>6, 4</c> and
    /// <c>60, 40</c> all mean the same thing.
    /// </param>
    /// <param name="deviceCount">Accelerators the backend has registered.</param>
    /// <remarks>
    /// Never throws. A model that would load with a wrong split should still load: refusing to
    /// serve because the second GPU was renamed is a worse outcome than serving from the first one
    /// and saying so.
    /// </remarks>
    public static TensorSplitPlan Create(IReadOnlyList<float>? requested, int deviceCount)
    {
        if (requested is null || requested.Count == 0)
        {
            return Automatic;
        }

        var warnings = new List<string>();
        var weights = new List<float>(requested.Count);

        foreach (var weight in requested)
        {
            if (float.IsNaN(weight) || float.IsInfinity(weight) || weight < 0)
            {
                warnings.Add(
                    $"tensor_split contains '{weight.ToString(CultureInfo.InvariantCulture)}', which is not a " +
                    "usable weight; it was read as 0.");
                weights.Add(0);
            }
            else
            {
                weights.Add(weight);
            }
        }

        if (deviceCount <= 0)
        {
            return new TensorSplitPlan
            {
                Warnings =
                [
                    .. warnings,
                    "tensor_split is set but no GPU is registered with this backend, so it has no effect."
                ]
            };
        }

        if (deviceCount == 1)
        {
            return new TensorSplitPlan
            {
                Warnings =
                [
                    .. warnings,
                    "tensor_split is set but only one GPU is visible, so the whole model goes on it."
                ]
            };
        }

        if (weights.Count > deviceCount)
        {
            warnings.Add(
                $"tensor_split names {weights.Count} devices but only {deviceCount} are visible; " +
                "the extra entries were dropped.");
            weights.RemoveRange(deviceCount, weights.Count - deviceCount);
        }

        var total = weights.Sum();

        if (total <= 0)
        {
            return new TensorSplitPlan
            {
                Warnings = [.. warnings, "tensor_split is all zeros; the backend's own split is used instead."]
            };
        }

        if (weights.Count < deviceCount)
        {
            // Left short rather than padded evenly: a shorter list is how you say "use the first
            // two of my four cards", and quietly spreading onto the others would override that.
            var idle = string.Join(", ", Enumerable.Range(weights.Count, deviceCount - weights.Count));
            warnings.Add(
                $"tensor_split covers {weights.Count} of {deviceCount} visible GPUs; " +
                $"device {idle} will hold no weights.");

            weights.AddRange(Enumerable.Repeat(0f, deviceCount - weights.Count));
        }

        return new TensorSplitPlan
        {
            Fractions = [.. weights.Select(w => w / total)],
            Warnings = warnings
        };
    }

    /// <summary>
    /// Reads a split written the way a person types it — <c>0.6, 0.4</c> or <c>6 4</c>.
    /// </summary>
    /// <returns>False when any entry is not a number, leaving <paramref name="weights"/> empty.</returns>
    /// <remarks>
    /// All-or-nothing on purpose: skipping the entries that failed to parse would shift every
    /// later weight onto the wrong device, which is worse than saying the text is unreadable.
    /// </remarks>
    public static bool TryParseWeights(string? text, out IReadOnlyList<float> weights)
    {
        weights = [];

        if (string.IsNullOrWhiteSpace(text))
        {
            return true;
        }

        var parsed = new List<float>();

        foreach (var part in text.Split(
                     [',', ' ', ';'],
                     StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (!float.TryParse(part, NumberStyles.Float, CultureInfo.InvariantCulture, out var value))
            {
                return false;
            }

            parsed.Add(value);
        }

        weights = parsed;
        return true;
    }

    /// <summary>Renders the split for logs and the Engine screen, naming the devices it applies to.</summary>
    public string Describe(IReadOnlyList<AcceleratorDevice>? devices = null)
    {
        if (IsAutomatic)
        {
            return "automatic (proportional to free VRAM)";
        }

        return string.Join(
            " · ",
            Fractions.Select((fraction, index) =>
            {
                var name = devices is not null && index < devices.Count
                    ? devices[index].Name
                    : $"GPU{index}";

                return $"{name} {fraction * 100:0.#}%";
            }));
    }
}
