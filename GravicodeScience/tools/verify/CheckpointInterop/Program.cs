using System.Globalization;
using Gravicode.Science.GraviText.Transformers;

// The C# half of tools/verify/checkpoint_interop.py. The Python side builds a checkpoint with known
// weights and recomputes the encoder in NumPy; this runs the same input through GraviText so the
// two can be compared elementwise. A load that is nearly right agrees on nothing.

if (args.Length < 1)
{
    Console.Error.WriteLine("usage: CheckpointInterop <forward|report|wrong-transpose|strict|lenient> <path> [tokens]");
    return 1;
}

// Must match the constants in checkpoint_interop.py.
var config = new TransformerConfig(
    VocabularySize: 40, HiddenSize: 16, Layers: 2, Heads: 4,
    IntermediateSize: 32, MaxPositions: 24);

switch (args[0])
{
    case "forward":
    {
        var model = new TransformerModel(config);
        TransformerCheckpoint.Load(model, args[1]);

        var tokens = args[2].Split(',').Select(int.Parse).ToArray();
        var hidden = model.Forward(tokens);

        for (var i = 0; i < hidden.Shape[0]; i++)
        {
            var row = Enumerable.Range(0, hidden.Shape[1])
                .Select(j => hidden[i, j].ToString("R", CultureInfo.InvariantCulture));

            Console.WriteLine($"row {i}: {string.Join(",", row)}");
        }

        return 0;
    }

    case "report":
    {
        var model = new TransformerModel(config);
        var report = TransformerCheckpoint.Load(model, args[1]);

        Console.WriteLine($"{report}  pretrained {model.HasPretrainedWeights}");
        return 0;
    }

    case "wrong-transpose":
    {
        // The feed-forward weight is hidden x intermediate, so its orientation is unambiguous and
        // the wrong convention has to be caught there rather than absorbed silently.
        var model = new TransformerModel(config);

        try
        {
            TransformerCheckpoint.Load(model, args[1],
                CheckpointNames.HuggingFaceBert with { Transposed = false });

            Console.WriteLine("ACCEPTED — the wrong transpose was not caught");
        }
        catch (InvalidDataException error)
        {
            Console.WriteLine($"REFUSED — {error.Message}");
        }

        return 0;
    }

    case "strict":
    {
        var model = new TransformerModel(config);

        try
        {
            TransformerCheckpoint.Load(model, args[1], strict: true);
            Console.WriteLine("ACCEPTED — a partial checkpoint loaded without complaint");
        }
        catch (InvalidDataException error)
        {
            Console.WriteLine($"REFUSED — {error.Message}");
        }

        return 0;
    }

    case "lenient":
    {
        var model = new TransformerModel(config);
        var report = TransformerCheckpoint.Load(model, args[1], strict: false);

        Console.WriteLine($"{report}  pretrained {model.HasPretrainedWeights}");
        return 0;
    }

    default:
        Console.Error.WriteLine($"unknown command '{args[0]}'");
        return 1;
}
