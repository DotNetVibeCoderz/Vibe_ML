using System.ComponentModel;
using System.Text;
using HFAppGen.Services;
using Microsoft.SemanticKernel;

namespace HFAppGen.Plugins;

/// <summary>
/// The HF.Net API surface, as a tool the assistant can consult.
/// </summary>
/// <remarks>
/// Without this the model writes plausible-looking calls that do not exist - the failure mode is
/// invented method names that compile in its head and not in the project. HF.Net is newer than any
/// training corpus, so for these libraries the reference is not an optimisation, it is the only
/// source of truth available.
/// </remarks>
public sealed class HFNetReferencePlugin
{
    private static readonly Dictionary<string, string> Libraries = new(StringComparer.OrdinalIgnoreCase)
    {
        ["GraviHub"] = """
            GraviHub - Hugging Face Hub client, and readers for the formats the Hub serves.
            using Gravicode.HFNet.GraviHub;
            using Gravicode.HFNet.GraviHub.Io;

            Hub (static facade; blocks, so use HubClient in a UI or a server)
              Hub.DownloadModel("bert-base-uncased")            -> local directory
              Hub.DownloadModel(id, revision, weightsOnly: true) weightsOnly skips duplicate formats
              Hub.DownloadFile(id, "config.json")               -> local file path
              Hub.ModelInfo(id) / Hub.DatasetInfo(id)           -> RepoInfo
              Hub.SearchModels("sentiment", limit: 10, task: "text-classification")
              Hub.SearchDatasets(query), Hub.DownloadDataset(id), Hub.DownloadDatasetFile(id, file)
              Hub.Upload(id, localPath, pathInRepo)             small files only, see below
              Hub.Login(token), Hub.WhoAmI(), Hub.Cache
              Hub.Configure(new HubOptions { Token = ..., CacheRoot = ..., OfflineMode = true })

            HubClient (async, owns one HttpClient - create ONE per process)
              new HubClient(options) or new HubClient(httpClient, options)
              await client.GetRepoInfoAsync(id, RepoKind.Model, "main")
              await client.ListFilesAsync(id, kind, revision)   -> RepoFile[] with sizes
              await client.DownloadFileAsync(id, file, kind, revision, progress, ct)
              await client.SnapshotAsync(id, kind, revision, allowPatterns, ignorePatterns, progress)
              await client.SearchAsync(query, kind, limit, filter)
              await client.UploadFileAsync(id, localPath, pathInRepo, kind, revision, message)
              client.ResolveUrl(id, file), client.IsAuthenticated, client.Cache

            RepoInfo: .Id .Kind .Sha .LastModified .Downloads .Likes .Tags .PipelineTag .Files
                      .Private .Gated .HasSafeTensors .File(path) .FilesWithExtension(".safetensors")
            RepoFile: .Path .Size .Sha .IsLfs .Extension
            TransferProgress: .Path .BytesTransferred .TotalBytes .Fraction

            HubCache - readable directory tree, not a symlink farm
              cache.Root, .PathFor(id, kind, revision, file), .Contains(...), .SizeInBytes()
              .Evict(id, kind, revision), .Clear()
              Honours HF_HUB_CACHE then HF_HOME, so it shares a machine with the Python tooling.

            SafeTensors - the format the Hub serves. Memory-mapped, read lazily.
              SafeTensors.Inspect(path)          -> SafeTensorInfo[] (name, dtype, shape) - no data
              SafeTensors.Open(path)             -> SafeTensorsReader, dispose it
              reader.Read("name"), reader.TryRead(name, out t), reader.Contains(n), reader.Tensors
              SafeTensors.ReadAll(path)          eager; a 7B F16 model is 56 GB as double
              SafeTensors.ReadShardIndex("model.safetensors.index.json")
              SafeTensors.Write(path, tensors, SafeTensorDType.F32, metadata)
              SafeTensors.Write(path, tensors, name => dtypeFor(name))   per-tensor dtype
              Dtypes: Bool U8 I8 F8E4M3 F8E5M2 U16 I16 F16 BF16 U32 I32 F32 U64 I64 F64
              Writing is limited to F16/BF16/F32/F64 - NdArray holds double.

            PyTorchCheckpoint - reads pytorch_model.bin, which most of the Hub still ships
              PyTorchCheckpoint.Inspect(path)    -> TorchTensorInfo[]
              PyTorchCheckpoint.Open(path)       -> dispose it; .Read(name), .TryRead, .Contains
              PyTorchCheckpoint.ReadAll(path)
              Resolves names against an allow-list, so nothing in the file is executed.
              The pre-1.6 bare-pickle format is refused with a message saying so.

            GOTCHAS
              HF_TOKEN is read from the environment; without one, private and gated repos 404.
              Upload goes through the inline commit API and is capped at 10 MB - real weights need
                Git LFS, which this client does not implement.
              A cached file is revalidated with one HEAD against its ETag, not re-downloaded.
            """,

        ["GraviTokenizers"] = """
            GraviTokenizers - tokenizers that reproduce what a model was trained with.
            using Gravicode.HFNet.GraviTokenizers;
            using Gravicode.HFNet.GraviTokenizers.Components;

            HfTokenizer
              HfTokenizer.FromPretrained("bert-base-uncased")   tries tokenizer.json, then
                                                                vocab.json+merges.txt, then vocab.txt
              HfTokenizer.Load("tokenizer.json")
              HfTokenizer.FromBertVocabulary("vocab.txt", lowercase: true)
              HfTokenizer.FromGpt2Files("vocab.json", "merges.txt")
              .Encode(text), .Encode(text, addSpecialTokens: false)
              .Encode(first, second)                            sentence pair, sets TypeIds
              .EncodeBatch(texts, BatchOptions.Default)         pads to the longest
              .EncodeBatch(texts, BatchOptions.Exactly(128))    pads and truncates to a fixed width
              .Decode(ids), .Decode(ids, skipSpecialTokens: false)
              .TokenToId(t), .IdToToken(id), .VocabularySize
              .PadId .UnknownId .ClassifierId .SeparatorId .MaskId     -1 when absent

            Encoding: .Ids .Tokens .AttentionMask .TypeIds .SpecialTokensMask .Offsets .Length
              .ToIdArray() .ToMaskArray()
              .Span(text, i) and .Span(text, first, last)  -> the SOURCE substring, not rejoined
                                                              pieces, so whitespace survives
            EncodingBatch: .Ids .AttentionMask .TypeIds are [batch, width] NdArrays

            Tokenizer (static shortcuts; each downloads a well-known tokenizer once)
              Tokenizer.BPE(text)            GPT-2 byte-level
              Tokenizer.WordPiece(text)      bert-base-uncased
              Tokenizer.SentencePiece(text)  xlm-roberta-base
              Tokenizer.FromPretrained(id), Tokenizer.Load(path)
              Tokenizer.TrainBpe(corpus, vocabularySize, minFrequency)
              Tokenizer.TrainWordPiece(corpus, vocabularySize, minFrequency)
              Tokenizer.TrainUnigram(corpus, vocabularySize)

            Components, to assemble a tokenizer by hand
              Models:        WordPieceModel(vocab, unk, "##", maxChars), BpeModel(vocab, merges, ...),
                             UnigramModel(pieces, unkId)
              Normalizers:   BertNormalizer(lowercase, stripAccents), LowercaseNormalizer,
                             StripNormalizer, ReplaceNormalizer, SequenceNormalizer
              PreTokenizers: BertPreTokenizer, WhitespacePreTokenizer, PunctuationPreTokenizer,
                             ByteLevelPreTokenizer, MetaspacePreTokenizer, SplitPreTokenizer,
                             SequencePreTokenizer
              Decoders:      WordPieceDecoder, ByteLevelDecoder, MetaspaceDecoder, WhitespaceDecoder
              PostProcessor.Bert(clsId, sepId), PostProcessor.None
              ByteAlphabet.Encode(text) / .Decode(mapped)   the GPT-2 byte mapping

            GOTCHAS
              BatchOptions is a record struct: `new BatchOptions()` zeroes it and means NO padding.
                Use BatchOptions.Default or BatchOptions.Exactly(n).
              Normalization runs per pre-token, so offsets stay anchored to the untouched input.
                Subword offsets are exact where normalization preserved length (lowercasing) and
                fall back to the whole word where it did not (accents, the byte alphabet).
              A word WordPiece cannot segment becomes entirely [UNK], not a partial segmentation.
              Use the model's OWN tokenizer. A mismatched one produces ids the model never saw.
            """,

        ["GraviDatasets"] = """
            GraviDatasets - dataset loading and shaping over GraviFrame.
            using Gravicode.HFNet.GraviDatasets;

            Dataset
              Dataset.Load("titanic")             built-in: titanic iris imdb finance sms_spam
              Dataset.Load("stanfordnlp/imdb", "train")     anything with a slash is a Hub id
              Dataset.LoadAll(nameOrId)           -> DatasetDict of every split
              Dataset.FromFile(path)              .csv .tsv .parquet .json .jsonl
              Dataset.FromCsvMemoryMapped(path)   for a file large relative to RAM
              Dataset.FromFrame(dataFrame, name)
              .Count .Columns .Features .Frame
              .Head(n) .Slice(start, count) .Select(indices) .SelectColumns(...) .RemoveColumns(...)
              .RenameColumn(from, to) .Shuffle(seed) .Filter(row => ...) .Map("col", i => value)
              .TrainTestSplit(testSize: 0.2, seed: 42, shuffle: true)   -> DatasetDict
              .Rows()          streams one RowView at a time
              .Batches(size, dropLast)
              .TextColumn(name)    works on a text OR numeric column
              .NumericColumn(name) .ToMatrix(cols...) .Describe()
              .WriteCsv(path) .WriteParquet(path)

            DatasetDict: dict["train"], .Train .Test .Validation, .Splits, .Contains(n), .TotalRows
              A missing split throws with the available names listed.

            HubDatasets.Load(id, split) / .LoadAll(id)
              Asks the Hub's dataset server for the Parquet conversion first, then falls back to
              pattern-matching the repository's own files. A script-only dataset is refused, since
              the script is Python.

            JsonLines.Read(path)   handles a JSON array or one object per line, decided by content
            BuiltinDatasets.Names, .TryResolve(name, out path), .FindDatasetsDirectory()

            GOTCHAS
              Every operation returns a NEW dataset; nothing mutates in place.
              TrainTestSplit shuffles by default. Many published CSVs are sorted by label, and an
                unshuffled split of one puts every positive example on one side.
            """,

        ["GraviTransformers"] = """
            GraviTransformers - run pretrained Hugging Face encoders.
            using Gravicode.HFNet.GraviTransformers;

            TransformerModel (dispose it - it holds the weight files open)
              TransformerModel.Load("bert-base-uncased")
              TransformerModel.Load(id, revision, progress)
              .Config .Tokenizer .Encoder .Report .RepoId
              .HasClassificationHead .HasMaskedLanguageHead .Labels
              .Hidden(text, maxLength)   -> [tokens, hidden] final hidden states
              .Embed(text)               -> [hidden] mean-pooled vector
              .EmbedBatch(texts)         -> [batch, hidden], parallel across inputs
              .Predict(text, topK)       -> Prediction[] - needs a classification head
              .FillMask("The capital of France is [MASK].", topK: 5) -> MaskFill[]
              .Similarity(a, b)          cosine similarity of the two embeddings
              TransformerModel.SupportedModelTypes   bert roberta xlm-roberta distilbert electra
                                                     camembert mpnet deberta

            Prediction: .Label .Score .Index          MaskFill: .Token .Score .Sequence

            PretrainedConfig
              PretrainedConfig.FromPretrained(id) / .Load("config.json")
              .ModelType .HiddenSize .Layers .Heads .IntermediateSize .VocabularySize
              .MaxPositions .Activation .IdToLabel .LabelCount .HeadSize .PositionOffset
              Reads each dimension from a list of aliases, because DistilBERT writes n_layers where
              BERT writes num_hidden_layers.

            WeightStore (dispose it)
              WeightStore.FromPretrained(id, revision, progress)
              WeightStore.Open(directory)     prefers safetensors, falls back to pytorch_model.bin
              .Read(name) .TryRead(name, out t) .TryReadAny(out t, names...) .Contains(n)
              .Names .NamesStartingWith(prefix) .Count .Description

            CheckpointLoader.Load(config, weights, strict: true) -> (Encoder, LoadReport)
              Detects the name prefix, the DistilBERT block layout, and the transpose convention.
              LoadReport: .Loaded .Missing .Prefix .IsComplete

            GOTCHAS
              Decoder-only models (GPT, Llama, Mistral) are REFUSED, not half-loaded: they need
                causal masking and rotary positions this encoder does not have.
              Token type embeddings are folded into the word embeddings, which is exact for a
                single sequence and approximate for a sentence pair.
              Inference is double precision on the CPU. For throughput, export to ONNX and use
                GraviOptimum.
              Embed() mean-pools rather than taking [CLS]: on a model that was not fine-tuned,
                [CLS] is nearly constant and makes every sentence look similar.
            """,

        ["GraviPEFT"] = """
            GraviPEFT - LoRA adapters, in the Hugging Face PEFT format.
            using Gravicode.HFNet.GraviPEFT;

            PEFT (facade)
              PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16))
              PEFT.LoadAdapter(model, "some-user/some-lora", merge: true)
              PEFT.ReadAdapter("adapter_model.safetensors")

            LoraConfig(Rank = 8, Alpha = 16, TargetModules = null, Dropout = 0)
              .Targets defaults to ["query", "value"]; .Scaling is Alpha / Rank

            LoraAdapter
              new LoraAdapter(inputs, outputs, config, seed)
              LoraAdapter.FromMatrices(a, b, config)
              .A [rank, inputs]  .B [outputs, rank]  .Delta()  .MergeInto(weight)  .ParameterCount
              B starts at ZERO, so an adapted model is identical to the base model until trained.

            LoraAdapterSet
              LoraAdapterSet.FromPretrained("some-user/some-lora")
              LoraAdapterSet.Load(path, config)
              .Save(directory, baseModelId)    writes adapter_model.safetensors + adapter_config.json
              .Adapters .ParameterCount .Config

            PeftModel
              .Merge()                folds adapters into the weights, in place and exactly
              .FitHead(texts, labels) trains a classifier on the frozen embeddings
              .Predict(text) .Score(texts, labels) .HeadLabels
              .SaveAdapter(directory) .ParameterEfficiency() .IsMerged .Model

            GOTCHAS
              PeftModel.SupportsAdapterTraining is FALSE. Adapters can be applied, merged, saved and
                loaded here, and the task head can be trained - but the adapter matrices themselves
                are not backpropagated into. Train those with PEFT in Python and serve them here.
              Merge cannot be undone from the merged weights. Merge to serve, not to swap adapters.
            """,

        ["GraviAccelerate"] = """
            GraviAccelerate - device selection and data-parallel primitives.
            using Gravicode.HFNet.GraviAccelerate;

            Accelerator
              new Accelerator(DeviceKind.Auto, workers: null)   workers defaults to ProcessorCount
              .Backend(elementCount) .Dot(a, b) .Add(a, b) .Multiply(a, b)
              .Train(estimator, dataset, labelColumn, featureColumns, epochs) -> TrainingReport
              .Partition(itemCount) -> Shard[]
              .ParallelGradient(itemCount, shard => gradient)   averages WEIGHTED by shard size
              Accelerator.Devices() .Describe() .Measure(work, iterations, warmup) .ReleaseGpu()
              Accelerator.GpuThreshold                           1 << 20 elements

            DeviceKind: Cpu | Gpu | Auto
            TrainingReport: .Epochs .Examples .Elapsed .Workers .Device .FinalScore .Throughput

            GOTCHAS
              The GPU is NOT the default. Everything here is float64, and consumer and integrated
                GPUs run double precision at a fraction of their single-precision rate - measured
                5 to 8 times SLOWER than the CPU on an integrated device.
              Measure() warms up 40 times by default. Tiered JIT recompiles a hot method after
                about thirty calls, so a shorter warm-up times the wrong code.
              ParallelGradient collects in worker order, not arrival order: floating-point addition
                is not associative, so arrival order would make results depend on scheduling.
            """,

        ["GraviOptimum"] = """
            GraviOptimum - ONNX Runtime inference and weight quantisation.
            using Gravicode.HFNet.GraviOptimum;

            Optimum
              Optimum.Optimize("model-id", target: "CUDA")   -> OptimizedModel (dispose it)
              Optimum.OptimizeFile("model.onnx", "auto")
              Optimum.ParseTarget("cuda")                    CPU CUDA DirectML oneDNN auto
              Optimum.AvailableTargets(probeModelPath)
              Optimum.Compare(modelPath, inputs, targets)    -> LatencyReport[]
              Optimum.Quantize(source, destination, QuantizationLevel.BFloat16, keepFullPrecision)
              Optimum.BuildEncoderInputs(tokenizer, text, session)

            OnnxSession (dispose it)
              OnnxSession.Open(path, ExecutionTarget.Auto)
              OnnxSession.FromPretrained(id, fileName, target, revision)
              .Run(inputs) .Run(singleInput) .Measure(inputs, iterations, warmup)
              .Inputs .Outputs (TensorSpec: Name, Shape with -1 for symbolic, ElementType)
              .RequestedTarget .ActualTarget .RanOnRequestedTarget

            QuantizationLevel: None | BFloat16 | Float16 | Int8 (Int8 is refused, see below)
            QuantizationReport: .OriginalBytes .QuantizedBytes .SizeRatio
                                .MaxAbsoluteError .MeanAbsoluteError

            GOTCHAS
              Naming a provider that is not installed THROWS rather than falling back silently -
                that silent fallback is how a "CUDA" deployment runs on the CPU for months. Use
                ExecutionTarget.Auto when a fallback is genuinely wanted.
              The base Microsoft.ML.OnnxRuntime package ships the CPU provider only. CUDA needs
                Microsoft.ML.OnnxRuntime.Gpu; DirectML needs Microsoft.ML.OnnxRuntime.DirectML.
              Quantising an index buffer corrupts it: bfloat16 has 8 mantissa bits, so 511 becomes
                512. Integer-typed tensors are kept at full precision automatically; pass
                keepFullPrecision for a source already widened to float.
              Int8 is refused rather than approximated - it needs per-tensor scales that the
                safetensors format has nowhere to put.
            """,

        ["GraviDiffusers"] = """
            GraviDiffusers - schedulers and a Stable Diffusion pipeline over ONNX.
            using Gravicode.HFNet.GraviDiffusers;

            NoiseSchedule
              new NoiseSchedule(trainTimesteps, betaStart, betaEnd, BetaSchedule.ScaledLinear)
              NoiseSchedule.StableDiffusion   (1000, 0.00085, 0.012, ScaledLinear)
              NoiseSchedule.Ddpm              (1000, 1e-4, 0.02, Linear)
              .Betas .Alphas .AlphasCumulative .AlphaBar(t) .Sigma(t) .TrainTimesteps
              BetaSchedule: Linear | ScaledLinear | SquaredCosine

            IScheduler: .SetTimesteps(steps) .Timesteps .ScaleInput(sample, i)
                        .Step(modelOutput, i, sample) .InitialNoiseScale
              DdimScheduler(schedule, eta: 0, seed)   deterministic at eta = 0, skips timesteps well
              DdpmScheduler(schedule, seed)           one step per training timestep, the reference
              EulerScheduler(schedule)                sigma parameterisation, good at 20-30 steps

            DiffusionPipeline (dispose it)
              DiffusionPipeline.FromPretrained(id, scheduler, target, revision, progress)
              .Generate(prompt, new GenerationOptions(Steps: 25, GuidanceScale: 7.5,
                                                      Width: 512, Height: 512, Seed: 42),
                        onStep: (i, total) => ...)    -> Image<Rgb24>
              .Scheduler (settable) .Tokenizer .RepoId .MaxPromptTokens
              DiffusionPipeline.RequiredFiles          text_encoder/ unet/ vae_decoder/ model.onnx

            GOTCHAS
              Needs the ONNX export of a diffusion model, not the PyTorch weights. Convert with
                `optimum-cli export onnx --model <id> <out>`.
              Width and height must be multiples of 8 - the latent grid is 8x smaller.
              GuidanceScale > 1 runs the UNet TWICE per step; a scale of 1 halves the work.
              The schedule must match what the weights were trained with. Stable Diffusion is
                ScaledLinear; plain Linear with the same endpoints gives washed-out images that
                look like a bad prompt.
            """,

        ["GraviNum"] = """
            GraviNum - the array foundation HF.Net is built on (Gravicode.Science).
            using Gravicode.Science.GraviNum;

            NdArray - reshape, transpose and slice return VIEWS over a shared buffer.
              NdArray.Zeros(3, 4) / Ones / Full(v, shape) / Eye(n) / Arange(0, 10, 2) / Linspace
              new NdArray(double[] data, params int[] shape)
              a[i, j], a.At(flat), a.SetAt(flat, v), a.Shape, a.Rank, a.Size, a.T
              a.Reshape(...), a.Row(i), a.Column(j), a.Copy(), a.AsContiguous(), a.ToArray()
              Operators + - * / ; * is ELEMENT-WISE. Use a.Dot(b) or LinAlg.Dot for a matrix product.
              a.Sum() .Mean() .Max() .Min() .Std() .Exp() .Log() .Sqrt() .Map(f)

            GraviRandom(seed): .NextDouble() .Next(n) .Normal(mean, sd) .Uniform(lo, hi)
              (there is no NextGaussian - it is Normal)
            LinAlg: Dot Solve Inverse Determinant Norm
            Compute: Compute.Cpu, Compute.Gpu, Compute.IsGpuAvailable, Compute.DescribeDevices()
            """,

        ["GraviFrame"] = """
            GraviFrame - dataframes (Gravicode.Science). GraviDatasets wraps this.
            using Gravicode.Science.GraviFrame;
            using Gravicode.Science.GraviFrame.Io;

            DataFrame.ReadCsv(path), .ReadCsv(path, new CsvOptions { Delimiter = "\\t" })
              (Delimiter is a STRING, not a char)
            DataFrame.ReadCsvMemoryMapped(path), ParquetIO.Read(path) / .Write(frame, path)
            frame.RowCount .ColumnCount .ColumnNames .Columns frame["name"]
            frame.Numeric(n) -> NumericSeries (.Values .ToNdArray() .Mean() .Sum())
            frame.Text(n) -> TextSeries (indexer returns string?)
            frame.Head(n) .Rows(start, count) .Take(indices) .Filter(row => ...) .SortBy(col)
            frame.GroupBy(keys) .Pivot(...) .Join(other, on, JoinKind.Inner) .Describe()
            frame.WithColumn(name, i => value) .Drop(...) .SelectColumns(...) .ToNdArray()
            DataFrame.Concat(frames), frame.Row(i) -> RowView, row.Number("col"), row.Text("col")
            """,

        ["GraviLearn"] = """
            GraviLearn - classical machine learning (Gravicode.Science).
            using Gravicode.Science.GraviLearn;
            using Gravicode.Science.GraviLearn.Linear;
            using Gravicode.Science.GraviLearn.Trees;
            using Gravicode.Science.GraviLearn.Preprocessing;
            using Gravicode.Science.GraviLearn.ModelSelection;

            new LogisticRegression(learningRate, maxIterations, l2Penalty).Fit(x, y)
              .Predict(x) .PredictProbabilities(x) .Classes .Coefficients
            LinearRegression, Ridge, Lasso, DecisionTree, RandomForest, KMeans, PCA, KNeighbors
            StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
            Selection.Split(x, y, testSize, seed, stratify), CrossValidation, Metrics
            IEstimator: void Fit(NdArray x, NdArray y); NdArray Predict(NdArray x)
            IClassifier adds NdArray PredictProbabilities(NdArray x)
            """,
    };

    /// <summary>Returns the real API surface of one HF.Net library.</summary>
    [KernelFunction("HFNetReference")]
    [Description("Returns the real API surface of an HF.Net library: namespaces, types, method " +
                 "signatures and the gotchas. Call this BEFORE writing code against a library " +
                 "rather than guessing method names. HF.Net is newer than your training data, so " +
                 "this is the only reliable source for these APIs.")]
    public string HFNetReference(
        [Description("Library: GraviHub, GraviTokenizers, GraviDatasets, GraviTransformers, " +
                     "GraviPEFT, GraviAccelerate, GraviOptimum, GraviDiffusers, or the foundation " +
                     "libraries GraviNum, GraviFrame, GraviLearn. Use 'all' for an index.")]
        string library)
    {
        if (string.Equals(library, "all", StringComparison.OrdinalIgnoreCase))
        {
            var builder = new StringBuilder("HF.Net libraries (a Hugging Face style stack for .NET):\n\n");
            builder.AppendLine("  GraviHub          Hub download/upload, safetensors, pytorch_model.bin");
            builder.AppendLine("  GraviTokenizers   WordPiece, BPE, Unigram, tokenizer.json");
            builder.AppendLine("  GraviDatasets     CSV/Parquet/JSON, Hub datasets, splits, streaming");
            builder.AppendLine("  GraviTransformers load and run pretrained BERT-family encoders");
            builder.AppendLine("  GraviPEFT         LoRA adapters in the Hugging Face PEFT format");
            builder.AppendLine("  GraviAccelerate   device selection, sharding, throughput measurement");
            builder.AppendLine("  GraviOptimum      ONNX Runtime inference, quantisation");
            builder.AppendLine("  GraviDiffusers    DDPM/DDIM/Euler schedulers, Stable Diffusion");
            builder.AppendLine();
            builder.AppendLine("Built on Gravicode.Science:");
            builder.AppendLine("  GraviNum          NdArray, linear algebra, random, statistics");
            builder.AppendLine("  GraviFrame        dataframes");
            builder.AppendLine("  GraviLearn        classical machine learning");
            builder.AppendLine();
            builder.AppendLine("Call HFNetReference with one name for its full API surface.");
            return builder.ToString();
        }

        if (Libraries.TryGetValue(library, out var reference)) return reference;

        return $"No library called '{library}'. Available: {string.Join(", ", Libraries.Keys)}, or 'all'.";
    }

    /// <summary>Returns the exact project references a generated project needs.</summary>
    [KernelFunction("HFNetProjectReferences")]
    [Description("Returns the exact <ItemGroup> XML a generated .csproj needs in order to use HF.Net " +
                 "libraries, with the paths already resolved for this machine. ALWAYS call this " +
                 "before writing a .csproj that uses HF.Net. The libraries are NOT on nuget.org, so " +
                 "a PackageReference to them fails with NU1101.")]
    public string HFNetProjectReferences(
        [Description("Comma-separated library names, for example \"GraviTransformers,GraviTokenizers\". " +
                     "Dependencies come along automatically, so naming the ones you actually use is enough.")]
        string libraries,
        [Description("The directory the .csproj will be written to; the paths are resolved against it.")]
        string projectDirectory)
    {
        var names = (libraries ?? "")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(n => Libraries.ContainsKey(n))
            .ToList();

        if (names.Count == 0)
        {
            return "Name at least one HF.Net library: " + string.Join(", ", Libraries.Keys);
        }

        var source = TemplateService.ResolveLibraryPath(projectDirectory);

        if (string.IsNullOrEmpty(source))
        {
            return "The HF.Net source tree could not be found from this machine. Tell the user that "
                + "generated projects need the HF.Net repository present, and that its src directory "
                + "could not be located from " + projectDirectory + ".";
        }

        var references = string.Join(
            Environment.NewLine,
            names.Select(n => $"    <ProjectReference Include=\"{Path.Combine(source, n, n + ".csproj")}\" />"));

        return $"""
            Put this in the .csproj. These are ProjectReferences on purpose: HF.Net is not published
            to nuget.org, so a PackageReference to Gravicode.HFNet.* fails with NU1101. Do NOT write
            a shim or a stub instead - the real libraries are right here.

              <ItemGroup>
            {references}
              </ItemGroup>

            The Gravicode.Science packages these depend on come from nuget.org automatically.
            """;
    }

    /// <summary>Returns a complete worked example for a common task.</summary>
    [KernelFunction("HFNetExample")]
    [Description("Returns a complete, working example program for a common HF.Net task. Use it as " +
                 "the starting shape for generated code rather than inventing structure.")]
    public string HFNetExample(
        [Description("Task: sentiment, embeddings, fillmask, semanticsearch, tokenize, hubsearch, " +
                     "dataset, finetune, lora, onnx, quantize, diffusion, safetensors, batch.")]
        string task)
        => task.ToLowerInvariant() switch
        {
            "sentiment" => """
                using Gravicode.HFNet.GraviTransformers;

                // A fine-tuned checkpoint carries its own classification head and its label names,
                // so nothing here has to be told what the classes are.
                using var model = TransformerModel.Load("distilbert-base-uncased-finetuned-sst-2-english");

                Console.WriteLine($"labels: {string.Join(", ", model.Labels)}");

                foreach (var text in new[] { "I loved this film.", "A complete waste of time." })
                {
                    var best = model.Predict(text, topK: 1)[0];
                    Console.WriteLine($"{best.Label,-9} {best.Score:P2}  {text}");
                }
                """,

            "embeddings" => """
                using Gravicode.HFNet.GraviTransformers;

                using var model = TransformerModel.Load("bert-base-uncased");

                // Mean-pooled over the tokens rather than the [CLS] vector: on a model that has
                // not been fine-tuned, [CLS] is close to constant.
                var vector = model.Embed("HF.Net brings Hugging Face models to .NET.");
                Console.WriteLine($"[{vector.Size}] {string.Join(", ", vector.ToArray().Take(5))}");

                Console.WriteLine(model.Similarity("the cat sat on the mat",
                                                   "the dog sat on the rug"));   // high
                Console.WriteLine(model.Similarity("the cat sat on the mat",
                                                   "quarterly earnings beat expectations"));  // low

                // EmbedBatch parallelises across inputs, which is where the throughput is.
                var matrix = model.EmbedBatch(["first document", "second document"]);
                Console.WriteLine($"[{matrix.Shape[0]} x {matrix.Shape[1]}]");
                """,

            "fillmask" => """
                using Gravicode.HFNet.GraviTransformers;

                // The sharpest check that a checkpoint loaded correctly. A transposed weight or a
                // shifted position embedding still produces plausible vectors, but it does not
                // answer this with "paris".
                using var model = TransformerModel.Load("bert-base-uncased");

                foreach (var fill in model.FillMask("The capital of France is [MASK].", topK: 5))
                    Console.WriteLine($"{fill.Token,-14} {fill.Score:P2}");
                """,

            "semanticsearch" => """
                using Gravicode.HFNet.GraviDatasets;
                using Gravicode.HFNet.GraviTransformers;

                using var model = TransformerModel.Load("bert-base-uncased");

                string[] documents =
                [
                    "The cat sat on the mat.",
                    "Quarterly revenue exceeded analyst expectations.",
                    "A golden retriever played in the park.",
                ];

                // Embed the corpus once, then every query is a handful of dot products.
                var corpus = model.EmbedBatch(documents);
                var query = model.Embed("a dog outside");

                var ranked = Enumerable.Range(0, documents.Length)
                    .Select(i => (Document: documents[i], Score: Cosine(corpus.Row(i), query)))
                    .OrderByDescending(r => r.Score);

                foreach (var (document, score) in ranked)
                    Console.WriteLine($"{score:F4}  {document}");

                static double Cosine(Gravicode.Science.GraviNum.NdArray a,
                                     Gravicode.Science.GraviNum.NdArray b)
                {
                    double dot = 0, na = 0, nb = 0;
                    for (var i = 0; i < a.Size; i++)
                    {
                        dot += a.At(i) * b.At(i);
                        na += a.At(i) * a.At(i);
                        nb += b.At(i) * b.At(i);
                    }
                    return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
                }
                """,

            "tokenize" => """
                using Gravicode.HFNet.GraviTokenizers;

                var tokenizer = HfTokenizer.FromPretrained("bert-base-uncased");

                const string Text = "Tokenizers are unbelievable.";
                var encoding = tokenizer.Encode(Text);

                Console.WriteLine(string.Join(' ', encoding.Tokens));
                Console.WriteLine(string.Join(' ', encoding.Ids));

                // Offsets point back into the ORIGINAL text, capitalisation and all - which is what
                // makes an entity span reportable as a substring rather than as token indices.
                for (var i = 0; i < encoding.Length; i++)
                    Console.WriteLine($"{encoding.Tokens[i],-12} '{encoding.Span(Text, i)}'");

                // A padded batch, ready for a model.
                var batch = tokenizer.EncodeBatch(["short", "a considerably longer sentence"],
                                                  BatchOptions.Default);
                Console.WriteLine(batch.Ids);   // [2 x width]
                """,

            "hubsearch" => """
                using Gravicode.HFNet.GraviHub;

                Console.WriteLine(Hub.WhoAmI() ?? "anonymous (set HF_TOKEN for a higher rate limit)");

                foreach (var hit in Hub.SearchModels("sentiment", limit: 5, task: "text-classification"))
                    Console.WriteLine($"{hit.Id,-55} {hit.Downloads,12:N0}");

                var info = Hub.ModelInfo("bert-base-uncased");
                Console.WriteLine($"{info.Id}: {info.Files.Count} files, safetensors={info.HasSafeTensors}");

                foreach (var file in info.FilesWithExtension(".safetensors"))
                    Console.WriteLine($"  {file}");

                // weightsOnly skips the duplicate formats most repositories carry.
                var directory = Hub.DownloadModel("prajjwal1/bert-tiny");
                Console.WriteLine(directory);
                """,

            "dataset" => """
                using Gravicode.HFNet.GraviDatasets;

                var data = Dataset.Load("titanic");
                Console.WriteLine(data);
                Console.WriteLine(data.Describe());

                var prepared = data
                    .SelectColumns("survived", "pclass", "age", "fare")
                    .Filter(row => !double.IsNaN(row.Number("age")));

                var split = prepared.TrainTestSplit(testSize: 0.2, seed: 42);
                Console.WriteLine(split);   // DatasetDict(train: ..., test: ...)

                foreach (var batch in split.Train.Batches(32))
                    Console.WriteLine($"batch of {batch.Count}");
                """,

            "finetune" => """
                using Gravicode.HFNet.GraviPEFT;
                using Gravicode.HFNet.GraviTransformers;

                // The cheap and effective path: freeze the encoder, train a head on its
                // embeddings. The transformer forward pass dominates the cost, and it runs once
                // per example rather than once per epoch.
                using var model = TransformerModel.Load("bert-base-uncased");

                string[] texts = ["great product", "terrible quality", "works perfectly", "broke immediately"];
                string[] labels = ["positive", "negative", "positive", "negative"];

                var peft = PEFT.ApplyLoRA(model).FitHead(texts, labels);

                Console.WriteLine(peft);
                foreach (var prediction in peft.Predict("this is excellent"))
                    Console.WriteLine($"  {prediction}");
                """,

            "lora" => """
                using Gravicode.HFNet.GraviPEFT;
                using Gravicode.HFNet.GraviTransformers;

                using var model = TransformerModel.Load("bert-base-uncased");

                // Adapters start as exactly no change - every B matrix is zero.
                var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

                var (adapter, encoder, fraction) = peft.ParameterEfficiency();
                Console.WriteLine($"{adapter:N0} of {encoder:N0} parameters = {fraction:P3}");

                // Load an adapter trained with PEFT in Python and fold it in exactly.
                // var served = PEFT.LoadAdapter(model, "some-user/some-lora", merge: true);

                peft.Merge();               // after this, inference costs what the base model costs
                peft.SaveAdapter("./my-adapter");   // adapter_model.safetensors + adapter_config.json
                """,

            "onnx" => """
                using Gravicode.HFNet.GraviOptimum;
                using Gravicode.HFNet.GraviTokenizers;

                // The fast inference path: single-precision kernels written for the hardware.
                using var model = Optimum.Optimize("hf-internal-testing/tiny-random-BertModel",
                                                   target: "auto");

                Console.WriteLine($"running on {model.Session.ActualTarget}");
                foreach (var spec in model.Session.Inputs) Console.WriteLine($"  {spec}");

                var tokenizer = HfTokenizer.FromPretrained("hf-internal-testing/tiny-random-BertModel");
                var feeds = Optimum.BuildEncoderInputs(tokenizer, "hello world", model.Session);

                foreach (var (name, tensor) in model.Run(feeds))
                    Console.WriteLine($"{name}: [{string.Join(" x ", tensor.Shape.ToArray())}]");

                Console.WriteLine(model.Measure(feeds, iterations: 30));
                """,

            "quantize" => """
                using Gravicode.HFNet.GraviOptimum;

                // The error is measured by reading back what was written, not predicted from the
                // format - which is the only way the number reflects the rounding performed.
                foreach (var level in new[] { QuantizationLevel.BFloat16, QuantizationLevel.Float16 })
                {
                    var report = Optimum.Quantize("model.safetensors", $"model-{level}.safetensors", level);
                    Console.WriteLine($"{level,-10} {report}");
                }

                // bfloat16 keeps float32's exponent range, so small weights stay representable;
                // float16 underflows below about 6e-5 and quietly loses the tail of the
                // distribution. Integer index buffers are kept exact automatically.
                """,

            "diffusion" => """
                using Gravicode.HFNet.GraviDiffusers;
                using SixLabors.ImageSharp;

                // Needs a repository with an ONNX layout: text_encoder/, unet/, vae_decoder/.
                using var pipeline = DiffusionPipeline.FromPretrained(
                    "some-user/stable-diffusion-onnx",
                    scheduler: new EulerScheduler());

                using var image = pipeline.Generate(
                    "a watercolour of a mountain village at dawn",
                    new GenerationOptions(Steps: 25, GuidanceScale: 7.5, Seed: 42),
                    onStep: (step, total) => Console.WriteLine($"  step {step}/{total}"));

                image.Save("output.png");

                // Same prompt and seed give the same image: DDIM at eta = 0 and Euler are both
                // deterministic, so a generation can be shared rather than just its result.
                """,

            "safetensors" => """
                using Gravicode.HFNet.GraviHub;
                using Gravicode.HFNet.GraviHub.Io;

                var path = Hub.DownloadFile("bert-base-uncased", "model.safetensors");

                // Memory-mapped: listing the tensors reads the header and nothing else.
                using var reader = SafeTensors.Open(path);
                Console.WriteLine(reader);

                foreach (var tensor in reader.Tensors.Take(5)) Console.WriteLine($"  {tensor}");

                // Pull one tensor out; the rest of the file is never touched.
                var embeddings = reader.Read("bert.embeddings.word_embeddings.weight");
                Console.WriteLine($"[{string.Join(" x ", embeddings.Shape.ToArray())}]");

                // Most of the Hub still ships the PyTorch pickle instead:
                //   using var checkpoint = PyTorchCheckpoint.Open("pytorch_model.bin");
                //   var same = checkpoint.Read("bert.embeddings.word_embeddings.weight");
                """,

            "batch" => """
                using Gravicode.HFNet.GraviDatasets;
                using Gravicode.HFNet.GraviTransformers;
                using Gravicode.HFNet.GraviAccelerate;

                using var model = TransformerModel.Load("distilbert-base-uncased-finetuned-sst-2-english");
                var data = Dataset.Load("imdb");

                Console.WriteLine(Accelerator.Describe());

                var texts = data.Head(64).TextColumn("review");
                var elapsed = Accelerator.Measure(() => model.EmbedBatch(texts), iterations: 3, warmup: 1);

                Console.WriteLine($"{texts.Count} documents in {elapsed.TotalSeconds:F2}s "
                    + $"({texts.Count / elapsed.TotalSeconds:N0}/s)");
                """,

            _ => "Unknown task. Try: sentiment, embeddings, fillmask, semanticsearch, tokenize, " +
                 "hubsearch, dataset, finetune, lora, onnx, quantize, diffusion, safetensors, batch.",
        };
}
