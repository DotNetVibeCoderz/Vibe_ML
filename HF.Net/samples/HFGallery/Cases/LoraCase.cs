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

        // Adapters and classifier, in the layout PEFT writes for SEQ_CLS: it loads in
        // Python with AutoModelForSequenceClassification, and here with LoadAdapter.
        peft.SaveAdapter("my-adapter");
        var again = PEFT.LoadAdapter(model, "my-adapter");

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
                ("head", peft.HeadLoadsInPython ? "BERT pooler + classifier, loads in Python" : "mean-pooled, HF.Net only"),
            ],
        };
    }
}

/// <summary>Trains LoRA adapters and a token classifier to find people and places.</summary>
/// <remarks>
/// Sixteen sentences, eight names and eight cities, and a test sentence that uses neither. What it
/// has to learn is where a name goes in a sentence, not a list of names.
/// </remarks>
internal sealed class NamesCase : GalleryCase
{
    public override string Title => "Teach it names";
    public override string Blurb => "Train LoRA adapters to tag people and places, from 16 tagged sentences.";
    public override string Library => "GraviPEFT";
    public override string Model => "bert-base-uncased";
    public override string InputLabel => "A sentence to tag after training";

    public override string? DefaultInput => "Kartini moved to Yogyakarta, and Hendra stayed in Semarang.";

    private static readonly string[] People = ["Ani", "Budi", "Siti", "Joko", "Maria", "Ahmad", "Dewi", "Rudi"];
    private static readonly string[] Places = ["Bandung", "Jakarta", "Surabaya", "Medan", "Bogor", "Malang", "Depok", "Solo"];

    public override string Code => """
        using Gravicode.HFNet.GraviPEFT;
        using Gravicode.HFNet.GraviTransformers;

        using var model = TransformerModel.Load("bert-base-uncased");
        var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

        // Words and one tag per word, the way CoNLL-style datasets come.
        IReadOnlyList<string>[] words = [["Ani", "lives", "in", "Bandung", "."], ...];
        IReadOnlyList<string>[] tags  = [["B-PER", "O", "O", "B-LOC", "O"], ...];

        peft.TrainTokenClassifier(words, tags, new TrainingOptions
        {
            Epochs = 8,
            BatchSize = 4,
            LearningRate = 2e-3,
        });

        foreach (var entity in peft.FindEntities(sentence))
            Console.WriteLine($"{entity.Label} {entity.Text}");

        peft.SaveAdapter("ner-adapter");    // task_type TOKEN_CLS: loads in Python too
        """;

    private static int CategoryOf(string label) => label == "PER" ? 0 : 1;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        // Its own copy: the shared one must stay the plain pretrained model for the other cases.
        progress.Report($"Loading a private copy of {Model}.");
        using var model = await Task.Run(() => TransformerModel.Load(Model), token).ConfigureAwait(false);

        var words = new List<IReadOnlyList<string>>();
        var tags = new List<IReadOnlyList<string>>();

        for (var i = 0; i < 16; i++)
        {
            var who = People[i % People.Length];
            var where = Places[i * 3 % Places.Length];

            if (i % 2 == 0)
            {
                words.Add([who, "lives", "in", where, "."]);
                tags.Add(["B-PER", "O", "O", "B-LOC", "O"]);
            }
            else
            {
                words.Add(["Yesterday", who, "drove", "to", where, "."]);
                tags.Add(["O", "B-PER", "O", "O", "B-LOC", "O"]);
            }
        }

        var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));
        var report = await Task.Run(() => peft.TrainTokenClassifier(words, tags, new TrainingOptions
        {
            Epochs = 8,
            BatchSize = 4,
            LearningRate = 2e-3,
            Progress = new Progress<TrainingProgress>(p =>
            {
                if (p.Step % 4 == 0 || p.Step == p.TotalSteps) progress.Report($"step {p.Step}/{p.TotalSteps}  loss {p.Loss:F4}");
            }),
        }), token).ConfigureAwait(false);

        var entities = await Task.Run(() => peft.FindEntities(input), token).ConfigureAwait(false);
        var unseen = entities.Count(e => !People.Contains(e.Text) && !Places.Contains(e.Text));

        return new CaseResult
        {
            Summary = $"{entities.Count} entities, {unseen} of them names it was never shown. "
                + $"{report.Steps} steps in {report.Elapsed.TotalSeconds:F0} s.",

            Spans = new SpanText(input, [.. entities.Select(
                e => new SpanMark(e.Start, e.End, e.Label, e.Score, CategoryOf(e.Label)))]),

            Legend = ["PER", "LOC"],

            Lines = [new Series("training loss", report.StepLosses, 0)],
            XLabel = "optimizer step",

            Facts =
            [
                ("sentences", words.Count.ToString()),
                ("labels", string.Join(", ", peft.HeadLabels)),
                ("loss", $"{report.EpochLosses[0]:F3} → {report.EpochLosses[^1]:F4}"),
                ("head", "classifier per token, loads in Python"),
            ],
        };
    }
}

/// <summary>Trains LoRA adapters and an answer-span head from a couple of dozen question and answer pairs.</summary>
internal sealed class AnswerTrainingCase : GalleryCase
{
    public override string Title => "Teach it to answer";
    public override string Blurb => "Train LoRA adapters to extract answers, from 24 SQuAD-style examples.";
    public override string Library => "GraviPEFT";
    public override string Model => "bert-base-uncased";
    public override string InputLabel => "A question about the passage";
    public override string SecondInputLabel => "The passage";

    public override string? DefaultInput => "What does Hendra do?";
    public override string? DefaultSecondInput => "Hendra lives in Semarang. Hendra works as a chef.";

    private static readonly string[] People = ["Ani", "Budi", "Siti", "Joko", "Maria", "Ahmad", "Dewi", "Rudi"];
    private static readonly string[] Places = ["Bandung", "Jakarta", "Surabaya", "Medan", "Bogor", "Malang", "Depok", "Solo"];
    private static readonly string[] Jobs = ["a teacher", "a nurse", "a farmer", "an engineer", "a driver", "a baker", "a pilot", "a tailor"];

    public override string Code => """
        using Gravicode.HFNet.GraviPEFT;
        using Gravicode.HFNet.GraviTransformers;

        using var model = TransformerModel.Load("bert-base-uncased");
        var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

        // SQuAD's shape: a question, the passage, and the answer as it appears there.
        AnswerExample[] examples =
        [
            new("Where does Ani live?", "Ani lives in Bandung. Ani works as a teacher.", "Bandung"),
            // ...
        ];

        peft.TrainQuestionAnswering(examples, new TrainingOptions
        {
            Epochs = 10,
            BatchSize = 4,
            LearningRate = 2e-3,
        });

        Console.WriteLine(peft.Answer(question, passage)[0]);
        peft.SaveAdapter("qa-adapter");     // task_type QUESTION_ANS
        """;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        progress.Report($"Loading a private copy of {Model}.");
        using var model = await Task.Run(() => TransformerModel.Load(Model), token).ConfigureAwait(false);

        var examples = new List<AnswerExample>();
        for (var i = 0; i < 8; i++)
        {
            var place = Places[i * 3 % 8];
            var job = Jobs[i * 5 % 8];
            var context = $"{People[i]} lives in {place}. {People[i]} works as {job}.";

            examples.Add(new AnswerExample($"Where does {People[i]} live?", context, place));
            examples.Add(new AnswerExample($"What does {People[i]} do?", context, job));
            examples.Add(new AnswerExample($"Who lives in {place}?", context, People[i]));
        }

        var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));
        var report = await Task.Run(() => peft.TrainQuestionAnswering(examples, new TrainingOptions
        {
            Epochs = 10,
            BatchSize = 4,
            LearningRate = 2e-3,
            Progress = new Progress<TrainingProgress>(p =>
            {
                if (p.Step % 6 == 0 || p.Step == p.TotalSteps) progress.Report($"step {p.Step}/{p.TotalSteps}  loss {p.Loss:F4}");
            }),
        }), token).ConfigureAwait(false);

        var answers = await Task.Run(() => peft.Answer(input, second, topK: 3), token).ConfigureAwait(false);
        var best = answers[0];

        return new CaseResult
        {
            Summary = best.IsEmpty
                ? "No answer found in the passage."
                : $"\"{best.Text}\" at {best.Score:P1}, after {report.Steps} steps in {report.Elapsed.TotalSeconds:F0} s.",

            Spans = best.IsEmpty ? null : new SpanText(second, [new SpanMark(best.Start, best.End, "answer", best.Score, 0)]),
            Legend = ["answer"],

            Bars = [.. answers.Where(a => !a.IsEmpty).Select(a => new Datum(a.Text, a.Score, 0))],

            Lines = [new Series("training loss", report.StepLosses, 0)],
            XLabel = "optimizer step",

            Facts =
            [
                ("examples", examples.Count.ToString()),
                ("loss", $"{report.EpochLosses[0]:F3} → {report.EpochLosses[^1]:F4}"),
                ("head", "qa_outputs, start and end per token"),
            ],
        };
    }
}
