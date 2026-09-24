# GraviTransformers

**Load pretrained Hugging Face encoders and run them.**

Mirrors `transformers`. The flagship library.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviTransformers;
```

## The whole thing in five lines

```csharp
using var model = TransformerModel.Load("bert-base-uncased");

foreach (var fill in model.FillMask("The capital of France is [MASK].", topK: 3))
    Console.WriteLine($"{fill.Token,-10} {fill.Score:P2}");
```

```
paris      41.53 %
lille       7.16 %
lyon        6.31 %
```

`Load` ties together four things that all have to agree: `config.json`, the weights, the mapping of
checkpoint parameter names onto the encoder, and the matching tokenizer. Dispose it - it holds the
weight files open.

## What it runs

```csharp
TransformerModel.SupportedModelTypes
// bert  roberta  xlm-roberta  distilbert  electra  camembert  mpnet  deberta
```

Anything else is **refused**, not half-loaded:

```csharp
TransformerModel.Load("gpt2");
// NotSupportedException: 'gpt2' is a 'gpt2' model. GraviTransformers runs BERT-family
// encoders (...); decoder-only and encoder-decoder architectures need causal masking
// and cross-attention, which this encoder does not have.
```

Filling an encoder from a decoder's weights produces a model that runs happily and returns nonsense,
which is the worst possible outcome.

## Tasks

### Classification

```csharp
using var model = TransformerModel.Load("distilbert-base-uncased-finetuned-sst-2-english");

Console.WriteLine(string.Join(", ", model.Labels));     // NEGATIVE, POSITIVE

foreach (var prediction in model.Predict("I absolutely loved this film.", topK: 2))
    Console.WriteLine(prediction);                      // POSITIVE: 99.99 %
```

Needs a checkpoint with a classification head; a base checkpoint has none and `Predict` says so
rather than inventing one. `model.HasClassificationHead` is the check.

### Fill-mask

```csharp
model.FillMask("He was a [MASK] player in the national team.", topK: 5);
// regular 55.0 %, key 19.4 %, former 5.8 %, capped 2.1 %, prominent 1.4 %
```

This is also the sharpest available check that a checkpoint loaded correctly. A model with a
transposed weight or a shifted position embedding still produces plausible-looking vectors - but it
does not answer the France question with *paris*.

### Named entities

```csharp
using var model = TransformerModel.Load("dslim/bert-base-NER");

foreach (var entity in model.FindEntities(text))
    Console.WriteLine($"{entity.Label,-5} {entity.Text}  [{entity.Start}..{entity.End})  {entity.Score:P1}");

// PER   Kang Fadhil        [0..11)     98.8 %
// ORG   Gravicode Studios  [20..37)    99.5 %
// LOC   Bandung            [41..48)    99.7 %
```

BIO tags are decoded into whole entities, and each one is returned as a **span of the original
string** rather than as rejoined subword pieces. That keeps the casing and any punctuation inside
the entity, and it leaves nothing for a caller to clean up by guessing at `##` prefixes.
`model.HasTokenClassificationHead` is the check.

A token classifier stores `classifier.weight` with the same shape a sequence classifier does, so the
loader claims the token head first; the other way round, every NER checkpoint would load as a
sentence classifier with several hundred nonsense classes.

### Question answering

```csharp
using var model = TransformerModel.Load("distilbert-base-cased-distilled-squad");

foreach (var answer in model.Answer(question, passage, topK: 3))
    Console.WriteLine($"{answer.Text}  ({answer.Score:P1})");

// safetensors and PyTorch checkpoints  (52.4 %)
// both safetensors and PyTorch checkpoints  (37.8 %)
```

The question and the passage go in as a pair, which is what exercises `SegmentDelta`. The span
search is constrained rather than a pair of independent argmaxes: only positions in segment 1 are
eligible, the end may not precede the start, and the length is capped. An unconstrained search
happily answers with a span that starts in the question and ends in the passage.
`model.HasQuestionAnsweringHead` is the check.

### Embeddings

```csharp
var vector  = model.Embed("HF.Net brings Hugging Face models to .NET.");   // [hidden]
var matrix  = model.EmbedBatch(texts);                                     // [batch, hidden]
var hidden  = model.Hidden(text);                                          // [tokens, hidden]

model.Similarity("the cat sat on the mat", "the dog sat on the rug");      // 0.8949
model.Similarity("the cat sat on the mat", "quarterly earnings beat");     // 0.4936
```

`Embed` **mean-pools** over the tokens rather than taking `[CLS]`. On a model that has not been
fine-tuned for sentence similarity, `[CLS]` is close to constant and makes every pair of sentences
look alike. `EmbedBatch` parallelises across inputs, which is where the throughput is.

## Configuration

```csharp
var config = PretrainedConfig.FromPretrained("bert-base-uncased");

config.ModelType; config.HiddenSize; config.Layers; config.Heads;
config.IntermediateSize; config.VocabularySize; config.MaxPositions;
config.IdToLabel; config.LabelCount; config.HeadSize; config.PositionOffset;
```

Each dimension is read from a **list of aliases**, because families disagree about names for the
same quantity: BERT writes `num_hidden_layers`, DistilBERT `n_layers`, GPT-2 `n_layer`. A missing
dimension **throws** rather than defaulting - a model built with twelve layers where the checkpoint
has six loads "successfully" and returns noise.

`PositionOffset` is 2 for RoBERTa and its relatives, which reserve the first two position slots for
padding - which is why their `max_position_embeddings` is 514 rather than 512. Ignoring it shifts
every position embedding by two, with no shape mismatch to catch it.

## Weights

```csharp
using var weights = WeightStore.FromPretrained("bert-base-uncased");

weights.Names;
weights.Contains("bert.embeddings.word_embeddings.weight");
weights.Read(name);
weights.TryReadAny(out var tensor, "classifier.weight", "classifier.out_proj.weight");
```

Resolves the layout once - one safetensors file, sharded safetensors with an index, or a PyTorch
pickle - and then answers by parameter name. safetensors is preferred where both exist: it needs no
interpretation, so it is both faster to open and safe on an untrusted file.

Tensors are read on demand and not cached. Loading a model touches each parameter once, and caching
would double peak memory for no benefit.

## Loading a checkpoint by hand

```csharp
var (encoder, report) = CheckpointLoader.Load(config, weights, strict: true);

report.Prefix;       // "bert." — detected, not assumed
report.Loaded;       // 196
report.Missing;      // []
report.IsComplete;
```

Three things vary between families and all three are **detected**:

1. **The parameter prefix** - `bert.`, `roberta.`, `distilbert.`, or nothing at all. Fine-tuned
   checkpoints are routinely re-exported with a different prefix from the one their architecture
   implies.
2. **The block layout** - DistilBERT names its parts `q_lin`, `k_lin`, `ffn.lin1` and so on.
3. **The transpose convention** - checked against the **feed-forward weight**, the only non-square
   matrix in a block. The attention projections are square and accept either reading silently.

### Two conventions worth knowing

**Layer norms come in two spellings.** The original BERT checkpoints came from TensorFlow and name
them `gamma` and `beta`; modern PyTorch exports name them `weight` and `bias`. `bert-base-uncased`
itself is still published with the old spelling, so a loader that knows only the new one fails on
the most-downloaded model on the Hub. Both are accepted.

**Token type embeddings are folded into the word embeddings.** For a single-sequence input every
position gets segment 0, so adding `token_type_embeddings[0]` to every row of the word embedding
matrix is *exactly* equivalent. Dropping the term instead would shift every hidden state by a
constant vector that the first layer norm does not remove, because it is added before the norm
rather than after.

A sentence pair needs segment 1 as well, and folding cannot supply it. The **difference**
`token_type_embeddings[1] - token_type_embeddings[0]` is carried on `LoadReport.SegmentDelta` and
added per position for the rows that belong to the second sequence, which reproduces both segments
exactly. `CheckpointLoader.SupportsPairs` is `true`.

## Task heads

Three layouts cover nearly every fine-tuned encoder on the Hub, and they differ in both their
parameter names and their activation:

| Family | Layers | Activation |
|---|---|---|
| BERT | pooler dense, then `classifier` | tanh |
| DistilBERT | `pre_classifier`, then `classifier` | ReLU |
| RoBERTa | `classifier.dense`, then `classifier.out_proj` | tanh |

Using the wrong activation is not cosmetic: tanh saturates and ReLU does not, so the logits are
wrong in a way that still ranks classes plausibly.

The masked-language head's output projection is usually **tied** to the input word embeddings and
therefore absent from the checkpoint. Falling back to the embedding matrix is not a shortcut, it is
the model - treating the missing tensor as "no head" would make fill-mask unavailable on most base
checkpoints, which are exactly the ones that have it.

## Performance

Inference is `double` on the CPU. This library exists so a model can be **loaded, inspected and
understood** in pure .NET - not to serve requests. For throughput, export to ONNX and use
[GraviOptimum](GraviOptimum.md).

There is no KV cache because this is an encoder: every position attends to every other one anyway.

## Limits

- Decoder-only and encoder-decoder architectures are refused.
- Vision models (ViT, CLIP) are on the roadmap; see [PLAN.md](../PLAN.md).

## See also

[GraviTokenizers](GraviTokenizers.md) · [GraviPEFT](GraviPEFT.md) · [GraviOptimum](GraviOptimum.md)
