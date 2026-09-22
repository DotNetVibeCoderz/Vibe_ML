# GraviPEFT

**LoRA adapters, in the Hugging Face PEFT format.**

Mirrors `peft`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviPEFT;
```

## What this does, and what it does not

**It does:** apply adapters, merge them into weights exactly, save and load the Hugging Face PEFT
layout, and train a task head over a frozen encoder.

**It does not:** backpropagate into the adapter matrices themselves.

```csharp
PeftModel.SupportsAdapterTraining   // false
```

The reason is specific. The autodiff encoder available in the foundation omits the biases on its
Q/K/V projections, while every pretrained BERT has them - so a gradient taken through it would be
the gradient of a *slightly different model*. It would train, converge, and produce weights that are
quietly wrong. That is stated here rather than approximated; [PLAN.md](../PLAN.md) has the route to
fixing it.

**The honest path today:** train adapters with PEFT in Python, serve them here.

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

For a few thousand labelled examples this is the effective and honest option.

## Saving

```csharp
peft.SaveAdapter("./my-adapter");
```

Writes `adapter_model.safetensors` and `adapter_config.json` in the PEFT layout, so the result loads
in Python.

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
