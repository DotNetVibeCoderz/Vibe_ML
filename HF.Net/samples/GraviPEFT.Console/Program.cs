using Gravicode.HFNet.GraviPEFT;
using Gravicode.HFNet.GraviTransformers;

// GraviPEFT sample. Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
// Runs on prajjwal1/bert-tiny so it finishes in seconds; swap in bert-base-uncased for real work.
Console.WriteLine("=== GraviPEFT ===\n");

var modelId = args.FirstOrDefault() ?? "prajjwal1/bert-tiny";
using var model = TransformerModel.Load(modelId);
Console.WriteLine(model);

// --- 1. The blueprint's example: PEFT.ApplyLoRA(model) ---------------------------
// Every B matrix starts at zero, so this is exactly the base model until training moves it.
var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16, TargetModules: ["query", "key", "value"]));
var (adapter, encoder, fraction) = peft.ParameterEfficiency();
Console.WriteLine($"\n{peft.Adapters.Adapters.Count} adapters: {adapter:N0} trainable of {encoder:N0} ({fraction:P2})");

// --- 2. Train adapters and a classifier together ----------------------------------
string[] texts =
[
    "The food was good.", "The food was not good.", "The staff were helpful.", "The staff were not helpful.",
    "I would recommend it.", "I would not recommend it.", "Delivery was quick.", "Delivery was anything but quick.",
    "The room was clean.", "The room was not clean.", "Setup was easy.", "Setup was not easy.",
];
string[] labels = [.. texts.Select((_, i) => i % 2 == 0 ? "positive" : "negative")];

var report = peft.Train(texts, labels, new TrainingOptions
{
    Epochs = 30,
    BatchSize = 4,
    LearningRate = 5e-3,
    Progress = new Progress<TrainingProgress>(p =>
    {
        if (p.Step % 30 == 0) Console.WriteLine($"  step {p.Step,3}/{p.TotalSteps}  loss {p.Loss:F4}");
    }),
});

Console.WriteLine(report);
Console.WriteLine($"train accuracy {peft.Score(texts, labels):P0}");
foreach (var probe in new[] { "The coffee was good.", "The coffee was not good." })
{
    Console.WriteLine($"  {probe,-28} -> {peft.Predict(probe)[0]}");
}

// --- 3. Save in the PEFT layout, load it back, merge to serve ---------------------
var directory = Path.Combine(Path.GetTempPath(), "hfnet-peft-sample");
peft.SaveAdapter(directory);
Console.WriteLine($"\nsaved to {directory}: {string.Join(", ", Directory.GetFiles(directory).Select(Path.GetFileName))}");
Console.WriteLine($"head loads in Python too: {peft.HeadLoadsInPython}");

using var fresh = TransformerModel.Load(modelId);
var served = PEFT.LoadAdapter(fresh, directory, merge: true);
Console.WriteLine($"reloaded and merged: {served}");
Console.WriteLine($"  'The coffee was not good.' -> {served.Predict("The coffee was not good.")[0]}");

// --- 4. The same adapters for named entities --------------------------------------
var tagger = PEFT.ApplyLoRA(fresh, new LoraConfig(Rank: 8, Alpha: 16));
IReadOnlyList<string>[] words =
[
    ["Ani", "lives", "in", "Bandung", "."], ["Budi", "moved", "to", "Jakarta", "."],
    ["Siti", "lives", "in", "Medan", "."], ["Joko", "moved", "to", "Bogor", "."],
];
IReadOnlyList<string>[] tags = [.. words.Select(_ => (IReadOnlyList<string>)["B-PER", "O", "O", "B-LOC", "O"])];

tagger.TrainTokenClassifier(words, tags, new TrainingOptions { Epochs = 60, BatchSize = 4, LearningRate = 5e-3 });
Console.WriteLine($"\nentities in 'Dewi lives in Solo.': {string.Join("; ", tagger.FindEntities("Dewi lives in Solo."))}");

// --- 5. Prefix tuning: no weight changes at all -----------------------------------
// Every attention layer reads a few learned keys and values ahead of the text; only those and the
// classifier train. The saved directory loads in Python with PeftModel.from_pretrained.
// On `model`, not `fresh`: section 3 merged the sentiment LoRA into fresh's weights.
var prefix = PEFT.ApplyPrefixTuning(model, new PrefixTuningConfig(VirtualTokens: 8));
Console.WriteLine($"\n{prefix}");
Console.WriteLine(prefix.Train(texts, labels, new TrainingOptions { Epochs = 40, BatchSize = 4, LearningRate = 2e-2 }));
Console.WriteLine($"train accuracy {prefix.Score(texts, labels):P0}");

var prefixDirectory = Path.Combine(Path.GetTempPath(), "hfnet-prefix-sample");
prefix.Save(prefixDirectory);
var reloaded = PrefixTuningModel.Load(model, prefixDirectory);
Console.WriteLine($"reloaded from {prefixDirectory}: 'The coffee was good.' -> {reloaded.Predict("The coffee was good.")[0]}");
