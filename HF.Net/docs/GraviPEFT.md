# GraviPEFT

**LoRA adapters, in the Hugging Face PEFT format.**

Mirrors `peft`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviPEFT;
```

## What this does

It **trains** LoRA adapters, together with a classification head, by backpropagating through a
pretrained encoder whose own weights stay frozen. It **applies** adapters, including ones trained
with PEFT in Python, and **merges** them into the weights exactly. It **saves and loads** the Hugging
Face PEFT layout, and an adapter trained here loads in Python. And it fits a head alone over the
frozen encoder, which is the cheap baseline to try first.

```csharp
PeftModel.SupportsAdapterTraining   // true, since 0.3
```

## Attaching adapters

```csharp
using var model = TransformerModel.Load("bert-base-uncased");

var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

var (adapter, encoder, fraction) = peft.ParameterEfficiency();
Console.WriteLine($"{adapter:N0} of {encoder:N0} = {fraction:P3}");
```

The returned model behaves **identically** to the input, because every adapter's `B` matrix is zero.
That is the defining property of LoRA, not an implementation detail: training starts from the
pretrained behaviour rather than from a perturbation of it. Initialising both matrices randomly
trains, converges, and reaches a measurably worse place.

`A` is Kaiming-uniform and non-zero, or no gradient would ever flow.

## Configuration

```csharp
new LoraConfig(Rank: 8, Alpha: 16, TargetModules: ["query", "value"], Dropout: 0);
```

`Scaling` is `Alpha / Rank`. That normalisation is what makes the rank a **capacity** knob rather
than a learning-rate knob: without it, doubling the rank doubles the size of the update at
initialisation.

Adapting only the query and value projections is the original paper's recommendation and the default
here. Adding key and output roughly doubles the trainable parameters for a change usually within
noise.

## Loading a published adapter

```csharp
var peft = PEFT.LoadAdapter(model, "some-user/some-lora", merge: true);
```

Reads `adapter_model.safetensors` (or `.bin`) beside `adapter_config.json`, in exactly the layout
PEFT writes:

```
base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.weight
base_model.model.bert.encoder.layer.0.attention.self.query.lora_B.weight
```

An adapter with only half of a pair is **refused**: applying it would add a zero-rank update that
looks like a working adapter doing nothing at all.

## Merging

```csharp
peft.Merge();
```

Folds every adapter into the base weights, in place and exactly. After merging, inference costs
precisely what the base model costs - there is no adapter left to evaluate.

This is the right thing to do before serving and the wrong thing before swapping adapters, because
the fold **cannot be undone** from the merged weights alone.

Adapter paths are matched on the layer index plus a suffix rather than on the full path, because the
same adapter is published with several prefixes. If nothing matches, `Merge` throws rather than
silently applying no change.

## Training a head

```csharp
peft.FitHead(texts, labels);

peft.Predict("this is excellent");      // -> Prediction[]
peft.Score(heldOutTexts, heldOutLabels);
peft.HeadLabels;
```

Trains a logistic regression on the adapted encoder's frozen embeddings. This is cheap because the
encoder is evaluated **once per example** and reused: the transformer forward pass dominates the
cost, and training the head is then a regression over a few hundred dimensions.

It is the baseline worth running before `Train`. When the frozen features already separate the
classes, it is all you need. When they do not, as with negation, where "good" and "not good" share
almost every token, the adapters have something to add.

## Training adapters

```csharp
using var model = TransformerModel.Load("bert-base-uncased");
var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

var report = peft.Train(texts, labels, new TrainingOptions
{
    Epochs = 8,
    BatchSize = 8,
    LearningRate = 1e-3,
    Progress = new Progress<TrainingProgress>(p => Console.WriteLine($"{p.Step}/{p.TotalSteps} {p.Loss:F4}")),
});

Console.WriteLine(report);                 // 32 steps in about a minute, loss 0.70 -> ... -> 0.003
peft.Predict("The staff were not friendly at all.");
```

Each step runs every example through the encoder with the adapters in the loop, puts a linear head
over the mean-pooled final hidden states, and backpropagates a cross-entropy loss into every
adapter's `A` and `B` and into the head. Then it takes one AdamW step. The base weights never change.
The adapters are updated in place, so `SaveAdapter` and `Merge` afterwards see the trained values.

| Option | Default | |
|---|---|---|
| `Epochs` | 3 | |
| `BatchSize` | 8 | examples whose mean loss makes one micro-batch |
| `GradientAccumulation` | 1 | micro-batches per optimizer step |
| `LearningRate` | 5e-4 | peak; LoRA wants roughly ten times full fine-tuning's rate |
| `WeightDecay` | 0 | decoupled, AdamW-style; biases are exempt |
| `WarmupFraction` | 0 | linear warm-up, then linear decay to zero |
| `MaxGradientNorm` | 1.0 | global norm clipping; 0 turns it off |
| `MaxLength` | 128 | truncation, in tokens |
| `Seed` | 42 | shuffling, head initialisation, adapter dropout |

The optimizer, the schedule and the clipping are the Hugging Face `Trainer`'s, written out from
`torch.optim.AdamW`, `get_linear_schedule_with_warmup` and `clip_grad_norm_`, and the tests pin all
three to those formulas. One consequence surprises people: with any warm-up at all, the first step
is taken at a learning rate of exactly zero, as it is in Python. `LoraConfig.Dropout` applies during
training only.

### Why it has its own backward pass

The foundation already has an autodiff encoder, and it is not used, for one specific reason: it has
no biases on its query, key and value projections, and every pretrained BERT has them. A gradient
taken through it would be the gradient of a *slightly different model*. It would train, converge,
and produce adapters that are quietly wrong for the model they get applied to.

The backward pass here is written out by hand over the same kernels inference uses. With the base
weights frozen, the only gradients wanted are the adapters'. Everything else is propagated and
thrown away, and nothing is propagated below the lowest adapted layer at all.

### How it is checked

- **Gradients.** Every entry of every adapter, on all six projections of a two-layer model with
  every bias and norm moved off its initial value, is checked against central differences. The
  agreement is within 1e-6 for the exact GELU and for the tanh GELU, and it still holds with
  dropout on. A deliberately wrong backward pass (the attention scale left out) fails this by
  a factor of two.
- **Forward.** With no adapters, the training forward pass matches the inference encoder to 1e-12.
  On `bert-base-uncased` after training, predictions from the adapters in the loop and from the
  merged weights agree to 6e-12.
- **Round trip.** An adapter trained and saved here, loaded into PEFT 0.21 in Python, gives the same
  hidden states as HF.Net's merged model to 7e-7. That is the float32 rounding of the merged
  weights. The adapter itself moves those hidden states by up to 4.

### What it costs, and where it stops

- **CPU, one sequence at a time, unpadded.** The batch is a unit of averaging, not of vectorisation.
  A step costs about three forward passes per example. Here, bert-base on 32 short sentences for
  8 epochs took 58 to 79 s across runs.
- **Memory.** Training holds a float32 copy of the encoder's weights beside the model of record,
  about 340 MB for bert-base.
- **Sequence classification only**, over a mean-pooled head. Token classification and question
  answering heads are not trained here.
- **The head is not saved.** `SaveAdapter` writes the adapters, in the PEFT layout. The head that
  `Train` fits stays in memory.
- **`Train` after `Merge` is refused.** The update would be counted twice. To continue training a
  merged adapter, load the base model again and attach the adapter unmerged with
  `PeftModel.WithAdapters(model, adapters, merge: false)`.

For more than a few thousand examples, train with PEFT in Python and load the result here. That
path is exact too.

## Saving

```csharp
peft.SaveAdapter("./my-adapter");
```

Writes `adapter_model.safetensors` and `adapter_config.json` in the PEFT layout, so the result loads
in Python.

Adapters made by `ApplyLoRA` are keyed by **the checkpoint's own module paths**, for example
`bert.encoder.layer.0.attention.self.query`, because PEFT matches tensors to modules by name and
leaves out, without an error, any it cannot place. Load the adapter onto a model class that uses
the same names. For `bert-base-uncased`, whose checkpoint names start with `bert.`, that means
`AutoModelForSequenceClassification` or `BertForMaskedLM`, not a bare `BertModel`. Before 0.3 the
keys were `layer.0.query`, which PEFT in Python loaded as no adapter at all.

## Why the parameter count matters

```
768 x 768 weight        589,824 values
rank 8 adapter           12,288 values     ~2 %
```

A rank-8 adapter over one projection is about two percent of the weight it adapts. That is the whole
argument: a task-specific adapter is a few megabytes, ships alongside one shared base model, and can
be swapped without reloading it.

## See also

[GraviTransformers](GraviTransformers.md) · [GraviHub](GraviHub.md)
