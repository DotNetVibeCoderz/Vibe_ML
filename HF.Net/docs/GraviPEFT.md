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
Face PEFT layout. On BERT an adapter trained here, classifier included, loads in Python and
predicts the same thing there. And it fits a head alone over the frozen encoder, which is the
cheap baseline to try first.

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

Console.WriteLine(report);                 // 32 steps in a minute or so, loss 0.74 -> ... -> 0.008
peft.Predict("The staff were not friendly at all.");
```

Each step runs every example through the encoder with the adapters in the loop, puts a
classification head over the final hidden states, and backpropagates a cross-entropy loss into every
adapter's `A` and `B` and into the head. Then it takes one AdamW step. The base weights never change.
The adapters are updated in place, so `SaveAdapter` and `Merge` afterwards see the trained values.

The head depends on the checkpoint:

- **On BERT** it is `BertForSequenceClassification`'s own: the `[CLS]` row through the checkpoint's
  pretrained pooler (dense and tanh, kept frozen, as PEFT keeps it), then a trained `classifier`.
  `peft.HeadLoadsInPython` is `true`, and the saved adapter predicts the same thing in Python.
- **Elsewhere** it is a `classifier` over the mean of the rows. RoBERTa's and DistilBERT's own heads
  have layers that PEFT does not save, so a head copied from them would not reproduce in Python
  either. This head is saved and reloaded by HF.Net only.

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

### Training a token classifier

For named entities, tag words and train a classifier over every token:

```csharp
IReadOnlyList<string>[] words = [["Ani", "lives", "in", "Bandung", "."], /* ... */];
IReadOnlyList<string>[] tags  = [["B-PER", "O", "O", "B-LOC", "O"], /* ... */];

peft.TrainTokenClassifier(words, tags, new TrainingOptions { Epochs = 8, BatchSize = 4, LearningRate = 2e-3 });

peft.FindEntities("Kartini moved to Yogyakarta.");   // PER: Kartini, LOC: Yogyakarta
```

Words and one tag per word is the shape CoNLL-style datasets come in. The words are joined with
spaces and tokenized, and only the first piece of each word is trained on. That is Transformers'
`label_all_tokens=False`, and the loss is the mean over those pieces across a step, as PyTorch's
cross-entropy computes it on a batch. `FindEntities` reads each word's label from its first piece,
as `aggregation_strategy="first"` does, and returns spans of the original text with its casing
intact.

The head is one `classifier` over every token. That is all each family's `ForTokenClassification`
model has, so a saved token classifier loads in Python on RoBERTa and DistilBERT too, not only on
BERT. Training one kind of head replaces the other, because both depend on the same adapters.

### Training question answering

For extractive question answering, give questions, the passages that answer them, and the answers
as they appear there, in SQuAD's shape:

```csharp
AnswerExample[] examples =
[
    new("Where does Ani live?", "Ani lives in Bandung. Ani works as a teacher.", "Bandung"),
    new("What does Ani do?",    "Ani lives in Bandung. Ani works as a teacher.", "a teacher"),
    // ...
];

peft.TrainQuestionAnswering(examples, new TrainingOptions { Epochs = 10, BatchSize = 4, LearningRate = 2e-3 });

peft.Answer("What does Hendra do?", "Hendra lives in Semarang. Hendra works as a chef.");   // a chef
```

The head is `qa_outputs`, a start and an end score for every token, and the loss is Transformers'
own: the mean of a cross-entropy over start positions and one over end positions. The question
and the passage go in as a pair, with the passage as segment 1. `AnswerStart` is SQuAD's
`answer_start`; leave it out and the first occurrence of the answer is used. A pair longer than
`MaxLength` loses the end of its passage. An answer cut off that way is trained as pointing at
`[CLS]`, which is how Transformers marks "not in this window". Long passages are not split into
overlapping windows, so keep them within the limit.

> **In Python, merge before you predict.** PEFT 0.21's `PeftModelForQuestionAnswering.forward`
> accepts `token_type_ids` and does not pass them on, so through the wrapper every passage is read as
> segment 0. Transformers' own `BertForQuestionAnswering`, and HF.Net, read it as segment 1. Call
> `model.merge_and_unload()` and the predictions match HF.Net to 3e-10. The same bug means a QA
> adapter trained in Python with PEFT learned without segments, so in HF.Net it runs on an input
> representation slightly different from the one it was trained on.

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
  dropout on, and through each head: the BERT pooler head, the mean head, the token head, and the
  answer-span head on a sentence pair. A deliberately wrong
  backward pass (the attention scale left out, or the pooler's tanh derivative) fails it by a factor
  of two to three.
- **Forward.** With no adapters, the training forward pass matches the inference encoder to 1e-12.
  On `bert-base-uncased` after training, predictions from the adapters in the loop and from the
  merged weights agree to about 1e-10.
- **Round trip, in Python.** An adapter and classifier trained and saved here, loaded into PEFT 0.21
  with `AutoModelForSequenceClassification`, give the same class probabilities as HF.Net to
  **5e-9**, with no warning from PEFT. The hidden states agree with HF.Net's merged model to 7e-7.
- **Named entities, in Python.** An adapter trained on 16 tagged sentences, loaded into
  `AutoModelForTokenClassification` with PEFT, gives Transformers' own token-classification pipeline
  the same spans and labels as `FindEntities`, with scores that agree to 5e-11. That holds on names
  it was never shown.
- **Question answering, in Python.** An adapter trained on 24 SQuAD-style examples, loaded into
  `AutoModelForQuestionAnswering` and merged, gives the same answers as HF.Net with scores that
  agree to 3e-10, on people and places it was never shown.
- **Round trip, here.** Reloaded with `PEFT.LoadAdapter`, predictions match the model that was
  saved to 5e-9. That is the F32 the file stores, which is also what PEFT writes.

### What it costs, and where it stops

- **CPU, one sequence at a time, unpadded.** The batch is a unit of averaging, not of vectorisation.
  A step costs about three forward passes per example. Here, bert-base on 32 short sentences for
  8 epochs took 58 to 91 s across runs.
- **Memory.** Training holds a float32 copy of the encoder's weights beside the model of record,
  about 340 MB for bert-base.
- **Sequence classification, token classification and extractive question answering.** Passages
  longer than `MaxLength` are cut, not split into overlapping windows.
- **The encoder's own dropout is off.** Python trains BERT with dropout 0.1 inside every block. Here
  only `LoraConfig.Dropout` applies, on the adapters' input. The trained adapter is valid either
  way; the two training runs are just not step-for-step the same.
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
in Python. A classifier trained with `Train` goes with it:

| Head | Where it is written | Loads in |
|---|---|---|
| BERT pooler + `classifier` | `adapter_model.safetensors` as `base_model.model.classifier.*`, `task_type` `SEQ_CLS` | HF.Net and Python |
| mean-pooled `classifier` | `head.safetensors` | HF.Net |
| token `classifier`, any family | `adapter_model.safetensors` as `base_model.model.classifier.*`, `task_type` `TOKEN_CLS` | HF.Net and Python |
| `qa_outputs`, any family | `adapter_model.safetensors` as `base_model.model.qa_outputs.*`, `task_type` `QUESTION_ANS` | HF.Net, and Python after `merge_and_unload()` |

Label names and the truncation length go in `hfnet_head.json`, not in `adapter_config.json`, where
PEFT would answer an unknown key with advice to upgrade. A `FitHead` head is a logistic regression
and is not saved.

In Python:

```python
import json
from transformers import AutoModelForSequenceClassification
from peft import PeftModel

labels = {int(k): v for k, v in json.load(open("my-adapter/hfnet_head.json"))["id2label"].items()}
base = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(labels), id2label=labels)
model = PeftModel.from_pretrained(base, "my-adapter")
```

And back in .NET, from the directory or from a Hub repository:

```csharp
var peft = PEFT.LoadAdapter(model, "./my-adapter");    // merge: true by default
peft.Predict("The staff were not friendly at all.");
```

An adapter trained in Python with `task_type="SEQ_CLS"` on BERT, or with `task_type="TOKEN_CLS"`
on any family, or with `task_type="QUESTION_ANS"`, loads the same way, head and all. The two classifiers have the same shape, so
`task_type` is what tells them apart. PEFT does not save label names, so without an `hfnet_head.json` they come back as `LABEL_0`,
`LABEL_1` and so on.

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
