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

1. ~~**Vision encoders.**~~ **ViT and DeiT done; CLIP not.** The patch embedding, the image
   processor and a pre-norm encoder block are in `GraviTransformers.Vision`, and
   `google/vit-base-patch16-224` agrees with torch to within 0.08 of a percentage point on the top
   five. Two things are worth carrying forward. The block had to be written rather than reused: ViT
   is pre-norm, BERT is post-norm, and each loads the other's parameters without a word of
   complaint. And CLIP still needs a **causal** text tower, which is the same gap that keeps
   decoder-only models out — so it belongs with that work, not with this.
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
4. **Sharded checkpoints that do not fit in memory.** Half of this is already true and worth being
   precise about: `WeightStore.Open` reads `model.safetensors.index.json`, memory-maps every shard
   and reads a tensor only when asked, so *inspecting* a model larger than RAM already works. What
   does not is running one — `CheckpointLoader.Load` materialises every parameter as `double`,
   which is 8 bytes per value whatever the file held. Streaming a forward pass layer by layer is the
   remaining work.
5. ~~**Position-embedding interpolation.**~~ **Done.** `VisionTransformer.Load(id, imageSize: 384)`,
   or any square or rectangle of whole patches passed to `Forward`. The resampler is torch's
   bicubic, pinned to 1e-12, and the whole model matches torch to ten decimal places at 160, 224
   and 384 px. Getting there turned up item 6.
6. ~~**Exact GELU in the text encoder.**~~ **Done, without a foundation release.** Text inference
   now runs on HF.Net's own kernels (`CompiledEncoder`), which take the activation from
   `hidden_act`. The foundation's tanh GELU had left `bert-base-uncased` hidden states 2.8e-2 away
   from torch; they now agree to about 1e-13 in float64, and fill-mask probabilities to ten digits.

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
3. **A single-precision path.** Half done, and the half that was free. Weights are now held as
   float32 inside HF.Net's kernels. That is lossless, because every checkpoint stores F32 or
   narrower, and it halves the bytes a short input streams. Activations and sums stay `double`,
   which is why the managed encoder still agrees with torch in float64 to about 1e-13. Going fully
   float32 would double SIMD width once more, at the cost of that agreement, and would move the
   comparison target to torch's own float32. Worth doing only as an opt-in mode.
4. **A better GEMM for long sequences.** The linear kernel (4 rows x 2 outputs, float32 weights
   widened on the fly) runs at roughly the foundation's packed `MatMul` rate. It is faster below
   about 200 rows and 1.3x slower at 577 (ViT at 384 px). A packed panel with a wider register tile
   is the known next step.
5. **KV caching** if and when decoder models arrive — generation is quadratic without it.

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
