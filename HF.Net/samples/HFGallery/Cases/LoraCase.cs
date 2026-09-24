using Gravicode.HFNet.GraviPEFT;
using Gravicode.HFNet.GraviTransformers;
using HFGallery.Controls;

namespace HFGallery.Cases;

/// <summary>Trains LoRA adapters and a head on a handful of sentences, then folds them in.</summary>
/// <remarks>
/// The sentences turn on negation - "good" and "not good" share almost every token - so a head over
/// the frozen encoder's mean-pooled features has little to go on, and the adapters inside attention
/// are what make the difference.
/// </remarks>
internal sealed class LoraCase : GalleryCase
{
    public override string Title => "Teach it a task";
    public override string Blurb => "Train LoRA adapters on 32 sentences, then merge them into the weights.";
    public override string Library => "GraviPEFT";
    public override string Model => "bert-base-uncased";
    public override string InputLabel => "A review to classify after training";

    public override string? DefaultInput => "The staff were not friendly at all.";

    private static readonly (string Text, string Label)[] Train =
    [
        ("The food was good.", "positive"),
        ("The food was not good.", "negative"),
        ("The service was friendly.", "positive"),
        ("The service was not friendly.", "negative"),
        ("I liked the room.", "positive"),
        ("I did not like the room.", "negative"),
        ("The film was great fun.", "positive"),
        ("The film was not fun at all.", "negative"),
        ("It arrived on time and works well.", "positive"),
        ("It did not arrive on time and does not work.", "negative"),
        ("A lovely, quiet place to stay.", "positive"),
        ("Not a lovely place, and never quiet.", "negative"),
        ("The staff were helpful.", "positive"),
        ("The staff were not helpful.", "negative"),
        ("I would recommend it.", "positive"),
        ("I would not recommend it.", "negative"),
        ("The price was fair.", "positive"),
        ("The price was not fair.", "negative"),
        ("Everything was clean.", "positive"),
        ("Nothing was clean.", "negative"),
        ("The book held my attention.", "positive"),
        ("The book never held my attention.", "negative"),
        ("Setup was easy.", "positive"),
        ("Setup was not easy.", "negative"),
        ("We were happy with the result.", "positive"),
        ("We were not happy with the result.", "negative"),
        ("The music was beautiful.", "positive"),
        ("The music was far from beautiful.", "negative"),
        ("Delivery was quick.", "positive"),
        ("Delivery was anything but quick.", "negative"),
        ("The coffee tasted fresh.", "positive"),
        ("The coffee did not taste fresh.", "negative"),
    ];

    public override string Code => """
        using Gravicode.HFNet.GraviPEFT;
        using Gravicode.HFNet.GraviTransformers;

        using var model = TransformerModel.Load("bert-base-uncased");

        // B starts at zero, so this is the base model exactly until training moves it.
        var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

        var report = peft.Train(texts, labels, new TrainingOptions
        {
            Epochs = 8,
            BatchSize = 8,
            LearningRate = 1e-3,
        });

        Console.WriteLine(report);                 // loss per epoch
        Console.WriteLine(peft.Predict(review)[0]);

        peft.SaveAdapter("my-adapter");            // PEFT layout
        peft.Merge();                              // fold into the weights to serve
        """;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        // Its own copy, not the shared cache's: merging changes the weights in place, and the
        // fill-mask case uses the same checkpoint.
        progress.Report($"Loading a private copy of {Model}.");
        using var model = await Task.Run(() => TransformerModel.Load(Model), token).ConfigureAwait(false);

        var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));
        var (adapter, encoder, fraction) = peft.ParameterEfficiency();
        progress.Report($"{peft.Adapters.Adapters.Count} adapters, {adapter:N0} trainable values ({fraction:P2} of the encoder).");

        var texts = Train.Select(t => t.Text).ToArray();
        var labels = Train.Select(t => t.Label).ToArray();

        var report = await Task.Run(() => peft.Train(texts, labels, new TrainingOptions
        {
            Epochs = 8,
            BatchSize = 8,
            LearningRate = 1e-3,
            Progress = new Progress<TrainingProgress>(p =>
            {
                if (p.Step % 4 == 0 || p.Step == p.TotalSteps)
                {
                    progress.Report($"step {p.Step}/{p.TotalSteps}  loss {p.Loss:F4}");
                }
            }),
        }), token).ConfigureAwait(false);

        var (trainAccuracy, predictions) = await Task.Run(
            () => (peft.Score(texts, labels), peft.Predict(input)), token).ConfigureAwait(false);

        // Merged, the adapters are gone and inference runs the base model's own fast path on the
        // updated weights. It should agree with the adapter-in-the-loop answer to float32 rounding.
        progress.Report("Merging the adapters into the weights.");
        var merged = await Task.Run(() => peft.Merge().Predict(input), token).ConfigureAwait(false);
        var drift = predictions.Max(p => Math.Abs(p.Score - merged.First(m => m.Label == p.Label).Score));

        var best = predictions[0];

        return new CaseResult
        {
            Summary = $"{best.Label} at {best.Score:P1}, after {report.Steps} steps in "
                + $"{report.Elapsed.TotalSeconds:F0} s. Loss {report.EpochLosses[0]:F3} → {report.EpochLosses[^1]:F3}.",

            Lines = [new Series("training loss", report.StepLosses, 0)],
            XLabel = "optimizer step",

            Bars = [.. predictions.Select(p => new Datum(p.Label, p.Score, p.Label == "positive" ? 0 : 1))],

            Facts =
            [
                ("trainable", $"{adapter:N0} of {encoder:N0} ({fraction:P2})"),
                ("train accuracy", $"{trainAccuracy:P0}"),
                ("steps", report.Steps.ToString()),
                ("merged vs not", $"{drift:E1}"),
            ],
        };
    }
}
