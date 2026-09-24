using Gravicode.HFNet.GraviTransformers;

// GraviTransformers sample. Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
var id = args.Length > 0 ? args[0] : "prajjwal1/bert-tiny";

Console.WriteLine($"=== GraviTransformers: {id} ===\n");

using var model = TransformerModel.Load(id);
Console.WriteLine(model);
Console.WriteLine($"  {model.Report}");
Console.WriteLine($"  tokenizer vocab = {model.Tokenizer.VocabularySize}");
Console.WriteLine();

// --- embeddings ---------------------------------------------------------------
var vector = model.Embed("HF.Net brings Hugging Face models to .NET.");
Console.WriteLine($"embedding: [{vector.Size}] first values {string.Join(", ", vector.ToArray().Take(5).Select(v => v.ToString("F4")))}");
Console.WriteLine();

Console.WriteLine("similarity:");
Console.WriteLine($"  cat/dog       {model.Similarity("the cat sat on the mat", "the dog sat on the rug"):F4}");
Console.WriteLine($"  cat/finance   {model.Similarity("the cat sat on the mat", "quarterly earnings exceeded analyst expectations"):F4}");
Console.WriteLine();

// --- fill-mask ----------------------------------------------------------------
if (model.HasMaskedLanguageHead)
{
    foreach (var prompt in (string[])
    [
        "The capital of France is [MASK].",
        "He was a [MASK] player in the national team.",
    ])
    {
        Console.WriteLine($"fill-mask: {prompt}");
        foreach (var fill in model.FillMask(prompt, topK: 5))
        {
            Console.WriteLine($"    {fill.Token,-16} {fill.Score:P2}");
        }
        Console.WriteLine();
    }
}
else Console.WriteLine("(no masked language head in this checkpoint)\n");

// --- classification -----------------------------------------------------------
if (model.HasClassificationHead)
{
    Console.WriteLine($"labels: {string.Join(", ", model.Labels)}");
    foreach (var text in (string[])["I absolutely loved this film.", "A complete waste of time."])
    {
        var best = model.Predict(text, topK: 2);
        Console.WriteLine($"  \"{text}\" -> {string.Join(", ", best)}");
    }
}
else Console.WriteLine("(no classification head in this checkpoint)");

// --- named entities -----------------------------------------------------------
if (model.HasTokenClassificationHead)
{
    const string Story = "Kang Fadhil founded Gravicode Studios in Bandung, and later worked with Microsoft.";

    Console.WriteLine($"\nentities in: {Story}");
    foreach (var entity in model.FindEntities(Story))
        Console.WriteLine($"    {entity.Label,-8} {entity.Text,-20} {entity.Score:P1}  [{entity.Start}..{entity.End})");
}

// --- question answering -------------------------------------------------------
if (model.HasQuestionAnsweringHead)
{
    const string Context =
        "HF.Net is a Hugging Face style machine learning stack for .NET, built by Gravicode Studios. "
        + "It reads both safetensors and PyTorch checkpoints, and its tokenizers produce the same ids "
        + "as the reference implementation.";

    foreach (var question in (string[])
    [
        "What does HF.Net read?",
        "Who built HF.Net?",
    ])
    {
        Console.WriteLine($"\nQ: {question}");
        foreach (var answer in model.Answer(question, Context, topK: 2))
            Console.WriteLine($"    {answer}");
    }
}
