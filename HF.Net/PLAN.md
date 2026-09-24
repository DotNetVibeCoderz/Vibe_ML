# PLAN

Roadmap for HF.Net. [Progress.md](Progress.md) says what exists today; this says where it is going
and why. `requirements.md` remains the specification of record.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

---

## The thesis

A .NET developer who wants to use a Hugging Face model currently has three options, and all three
are bad: stand up a Python service, hand-roll ONNX plumbing, or give up. HF.Net's bet is that the
missing piece is not raw tensor maths — .NET has that — but **the ecosystem's file formats and
conventions**: safetensors, `tokenizer.json`, `config.json`, the PEFT adapter layout, the Hub's
naming conventions and the three separate spellings of a layer norm.

So the order of work is formats first, models second, training last.

---

## v0.2 — breadth of models

**The goal:** stop refusing so much.

1. **Vision encoders.** ViT and CLIP. Both are transformer encoders over patch embeddings rather
   than token embeddings, so the block is already there; what is missing is the patch embedding, the
   image preprocessing pipeline (resize, centre crop, normalise) and CLIP's dual-tower contrastive
   head. This is the largest single increase in what HF.Net can open.
2. ~~**Token classification and question answering.**~~ **Done.** `FindEntities` decodes BIO tags
   into whole entities and `Answer` searches a constrained start/end pair; both return real
   substrings of the input, taken from the tokenizer's offsets. The ordering trap worth recording:
   a token classifier's `classifier.weight` has the same shape a sequence classifier's does, so the
   token head has to be claimed first or every NER checkpoint loads as a sentence classifier with
   several hundred nonsense classes.
3. ~~**Sentence-pair fidelity.**~~ **Done.** The *difference*
   `token_type_embeddings[1] - token_type_embeddings[0]` is carried on `LoadReport.SegmentDelta` and
   added per position to the rows of the second sequence. Segment 0 stays folded into the word
   embeddings, so both segments are now exact and `CheckpointLoader.SupportsPairs` is `true`.
4. **Sharded checkpoints end to end.** The index reader exists; the loader should stream a model
   that does not fit in memory rather than requiring it to.

## v0.3 — training

**The goal:** make LoRA adapters trainable, not merely loadable.

The blocker is specific and worth stating. The foundation's autodiff encoder
(`TransformerTape.MultiHeadAttention`) takes no biases on its Q/K/V projections, while every
pretrained BERT has them. A gradient taken through it would therefore be the gradient of a
*slightly different model* — which trains, converges, and produces weights that are quietly wrong.

Two routes, to be decided by measurement:

- **(a)** Contribute biased attention to `GraviText`'s tape and train through it.
- **(b)** Write a tape encoder inside GraviPEFT from the foundation's `Tensor` primitives, matching
  the pretrained architecture exactly.

(a) is better for the ecosystem; (b) does not require a release of the foundation. Either way the
first milestone is **gradient agreement**: a numerically differentiated loss and the tape's gradient
agreeing to 1e-6 on a two-layer model. Nothing else is worth building until that passes.

Then: Adam with weight decay, gradient accumulation, a learning-rate schedule, and adapters that
round-trip through the PEFT format and give the same predictions in Python.

## v0.4 — performance

**The goal:** publishable numbers, and the honesty to report them properly.

1. **Fill out the benchmark suites.** Every library has a harness project; they need real cases:
   tokenizer throughput on a multilingual corpus, encoder latency by sequence length, memory-mapped
   versus full dataset loads, CPU versus ONNX Runtime versus an accelerator.
2. **Measure before optimising, and say which configuration a number came from.** The foundation's
   experience is instructive: its ILGPU path measured 5–8× *slower* than the CPU because everything
   is `double`. Any claim here needs the hardware and the configuration attached.
3. **A single-precision path.** The most likely real win, and the largest change. `NdArray` is
   `double`; an F32 encoder would halve the memory traffic and let SIMD do twice the work per
   instruction. This depends on the foundation's generic `NdArray<T>` work.
4. **KV caching** if and when decoder models arrive — generation is quadratic without it.

## v0.5 — diffusion in earnest

The schedulers are exact and tested. What is missing is a pipeline anyone can run without first
converting a model by hand: a curated list of ONNX diffusion repositories that work, image-to-image
and inpainting, LoRA applied to the UNet, and negative prompting with per-token weights.

## Continuous

These do not wait for a version.

- **Bilingual documentation.** Every page under `docs/` has a counterpart under `docs/id/`. A page
  that exists in only one language is an unfinished page.
- **Samples and notebooks** for each library, and screenshots in the docs. `samples/HFGallery` is
  where a new capability earns its demonstration: a case there runs against a real model, so a
  feature that cannot be shown working in it is not finished.
- **Tests pinned to something independently known** — a published figure, a closed-form answer, or
  the reference implementation's own output. A test that only checks HF.Net against itself passes
  just as happily when both sides are wrong.
- **Say no clearly.** Every refusal in the codebase names what is unsupported and what to do
  instead. A library that half-loads a model it does not understand is worse than one that declines.

## Publishing

`Gravicode.HFNet.*` is on nuget.org, `GraviHub` pushed first — everything depends on it, and a
dependency that is not yet indexed leaves the dependents unrestorable for a few minutes.
`hf-net-release.yml` does this from a `hf-net-v*` tag. Generated projects still use
`ProjectReference`, which is why HFAppGen has an `HFNetProjectReferences` tool: a generated app that
restores from nuget.org cannot be tested against uncommitted changes to the libraries.

## Deliberately out of scope

- **Training a transformer from scratch.** The foundation's `TransformerClassifier` already does
  this, and for a few hundred labelled documents TF-IDF plus a linear model remains the stronger and
  far cheaper baseline.
- **Reimplementing ONNX Runtime.** GraviOptimum binds to it; competing with it would be foolish.
- **A Python bridge.** The entire point is not needing one.
