using System.Diagnostics;
using Gravicode.HFNet.GraviDiffusers;
using Gravicode.Science.GraviNum;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// GraviDiffusers sample. Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
// Pass an ONNX Stable Diffusion repository to draw a real picture, for example
//   dotnet run -- nmkd/stable-diffusion-1.5-onnx-fp16 "a lighthouse at dawn, oil painting"
// With no argument it runs a 9 MB test pipeline with random weights: the image is noise, but every
// stage - tokenizer, text encoder, UNet, scheduler, VAE - runs for real.
Console.WriteLine("=== GraviDiffusers ===\n");

// --- 1. Noise schedules ------------------------------------------------------------
foreach (var (name, schedule) in new[] { ("Stable Diffusion", NoiseSchedule.StableDiffusion), ("DDPM", NoiseSchedule.Ddpm) })
{
    Console.WriteLine($"{name,-17} signal left at t=0: {schedule.AlphaBar(0):F4}, t=500: {schedule.AlphaBar(500):F4}, t=999: {schedule.AlphaBar(999):F6}");
}

Console.WriteLine();

// --- 2. A scheduler checked against a perfect noise oracle -------------------------
// Noise a known sample to the last timestep, then denoise it with the true noise as the "model
// output". A correct DDIM step recovers the sample exactly; this is how the schedulers are tested.
var random = new GraviRandom(3);
var clean = new NdArray([.. Enumerable.Range(0, 64).Select(_ => random.Normal(0, 1))], 64);
var noise = new NdArray([.. Enumerable.Range(0, 64).Select(_ => random.Normal(0, 1))], 64);

var ddim = new DdimScheduler();
var timesteps = ddim.SetTimesteps(25);
var alphaBar = ddim.Schedule.AlphaBar(timesteps[0]);
var sample = clean * Math.Sqrt(alphaBar) + noise * Math.Sqrt(1 - alphaBar);

for (var step = 0; step < timesteps.Count; step++) sample = ddim.Step(noise, step, sample);

var error = Enumerable.Range(0, 64).Max(i => Math.Abs(sample.At(i) - clean.At(i)));
Console.WriteLine($"DDIM, 25 steps with a perfect oracle: largest error {error:E1}");
Console.WriteLine();

// --- 3. The blueprint's example: DiffusionPipeline.Generate(prompt) -----------------
var repo = args.FirstOrDefault() ?? "optimum-internal-testing/tiny-stable-diffusion-onnx";
var prompt = args.Skip(1).FirstOrDefault() ?? "a lighthouse at dawn, oil painting";
var tiny = args.Length == 0;

// The sampler is built from the repository's scheduler_config.json, so its timestep spacing and
// offset are the ones the model shipped with.
using var pipeline = DiffusionPipeline.FromPretrained(repo, sampler: Sampler.Euler);
Console.WriteLine($"VAE factor {pipeline.VaeScaleFactor}, UNet channels {pipeline.UnetInputChannels}, {pipeline.Scheduler}");
var options = tiny
    ? new GenerationOptions(Steps: 10, GuidanceScale: 7.5, Width: 64, Height: 64, Seed: 7, NegativePrompt: "blurry")
    : new GenerationOptions(Steps: 25, GuidanceScale: 7.5, Width: 512, Height: 512, Seed: 7, NegativePrompt: "blurry, low quality");

var watch = Stopwatch.StartNew();
using var image = pipeline.Generate(prompt, options, (step, total) =>
{
    if (step % 5 == 0 || step == total) Console.WriteLine($"  step {step}/{total}");
});

var output = Path.Combine(Path.GetTempPath(), "hfnet-diffusion.png");
image.Save(output);
Console.WriteLine($"{repo}: {image.Width}x{image.Height} in {watch.Elapsed.TotalSeconds:F1} s -> {output}");

// --- 4. Image to image: redraw the picture just made, keeping its composition ------------
if (pipeline.SupportsImageToImage)
{
    using var redrawn = pipeline.ImageToImage(prompt + ", at night", image, strength: 0.6, options);
    var redrawnPath = Path.Combine(Path.GetTempPath(), "hfnet-img2img.png");
    redrawn.Save(redrawnPath);
    Console.WriteLine($"image to image, strength 0.6 -> {redrawnPath}");

    // --- 5. Inpainting: repaint the centre. An ordinary UNet blends; a 9-channel one is used directly.
    using var mask = new Image<Rgb24>(image.Width, image.Height);
    mask.ProcessPixelRows(rows =>
    {
        for (var y = image.Height / 4; y < image.Height * 3 / 4; y++)
        {
            var row = rows.GetRowSpan(y);
            for (var x = image.Width / 4; x < image.Width * 3 / 4; x++) row[x] = new Rgb24(255, 255, 255);
        }
    });

    using var repainted = pipeline.Inpaint("a full moon", image, mask, options);
    var repaintedPath = Path.Combine(Path.GetTempPath(), "hfnet-inpaint.png");
    repainted.Save(repaintedPath);
    Console.WriteLine($"inpainting ({(pipeline.UnetInputChannels == 9 ? "inpainting UNet" : "blended")}) -> {repaintedPath}");
}

// --- 6. Emphasis and LoRA -----------------------------------------------------------------
pipeline.PromptWeighting = true;
foreach (var (text, weight) in PromptWeights.Parse("a (lighthouse:1.3) at [dawn], ((oil painting))"))
{
    Console.WriteLine($"  {weight,6:F3}  \"{text}\"");
}

// A LoRA is a .safetensors file, a folder or a Hub repository, in kohya's or diffusers' layout:
//   var report = pipeline.LoadLora("some-user/some-sd15-lora", scale: 0.8);
//   Console.WriteLine(report);      // LoRA: 128 UNet and 72 text-encoder layers
//   pipeline.UnloadLoras();
