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

1. ~~**Vision encoders.**~~ **Done: ViT, DeiT and CLIP.** The patch embedding, the image processor
   and a pre-norm encoder block are in `GraviTransformers.Vision`. Two things are worth carrying
   forward. The block had to be written rather than reused: ViT is pre-norm, BERT is post-norm, and
   each loads the other's parameters without a word of complaint. And CLIP needed a **causal** text
   tower - the same attention GPT-2 needed, so they came together. The image processor is PIL's
   resize reproduced to the byte, and from an image file ViT matches torch to twelve digits and CLIP's
   logits to 8e-14.
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
4. ~~**Sharded checkpoints that do not fit in memory.**~~ **Done.** `StreamingEncoder` keeps no
   parameter resident: each forward pass reads the embedding rows its tokens use and then each layer
   in turn, straight from the memory-mapped safetensors to float32, across shards. `bert-large-uncased`
   runs in 0.45 GB of private memory against 5.9 GB loaded whole, bit-identical. A pickle cannot be
   read in parts and is refused with the conversion to use.
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

**The milestone is met, by route (b).** The blocker was specific: the foundation's autodiff encoder
(`TransformerTape.MultiHeadAttention`) has no biases on its Q/K/V projections, while every pretrained
BERT has them. A gradient taken through it would have been the gradient of a *slightly different
model*. `GraviPEFT` now has its own backward pass (`LoraEncoder`), written by hand over the same
kernels inference runs on. Route (a), contributing biased attention to `GraviText`'s tape, is still
worth doing for the ecosystem, but nothing here waits on it.

1. ~~**Gradient agreement.**~~ **Done.** Every adapter entry, on all six projections of a two-layer
   model with every bias and norm perturbed, agrees with central differences to 1e-6. That holds
   for both GELUs and with dropout. Breaking the backward pass on purpose fails the check by 2x.
2. ~~**AdamW, gradient accumulation, a learning-rate schedule.**~~ **Done**, as `torch.optim.AdamW`,
   `get_linear_schedule_with_warmup` and `clip_grad_norm_` compute them, and pinned to those
   formulas. That includes the reference's zero learning rate on the first step of any warm-up.
3. ~~**Adapters that round-trip through the PEFT format.**~~ **Done.** Adapters now take the
   checkpoint's own module paths. An adapter trained here, loaded into PEFT 0.21 in Python, gives
   the same hidden states as HF.Net's merged model to 7e-7. Before this, `ApplyLoRA` named them
   `layer.0.query`, and Python loaded none of them without saying so.
4. ~~**Saving the head.**~~ **Done.** On BERT the head is now `BertForSequenceClassification`'s own
   (the frozen pretrained pooler, then `classifier`), saved as PEFT saves `SEQ_CLS`. Loaded in Python
   with `AutoModelForSequenceClassification`, it predicts the same probabilities to 5e-9. Other
   families keep a mean-pooled head that only HF.Net reads back, because their Transformers heads
   have layers PEFT does not save.
5. ~~**Token classification and question answering heads.**~~ **Done.** `TrainTokenClassifier`
   and `TrainQuestionAnswering`, each head saved in PEFT's layout and agreeing with Python: entity
   scores to 5e-11, answer scores to 3e-10. Question answering found a PEFT bug worth knowing:
   `PeftModelForQuestionAnswering.forward` drops `token_type_ids`, so in Python it has to be merged
   first.
6. ~~**Batched steps.**~~ **Done, by packing rather than padding.** A micro-batch's examples are
   laid end to end: the linear layers see all their rows at once, while attention runs block by
   block and positions restart per example. There is no padding and so no mask, and tests pin the
   packed forward pass and gradients to the examples run one at a time. With the register-blocked
   GEMM from v0.4, packing 8 sentences makes a step 1.8x faster than one at a time.
7. ~~**Long passages.**~~ **Done.** A passage too long for `MaxLength` is split into overlapping
   windows (`DocStride`), as Transformers' `truncation="only_second"` with `stride` does, both in
   training and in `Answer`.

v0.3 is complete. Route (a), contributing biased attention to the foundation's tape, remains open,
but nothing depends on it now.

## v0.4 — performance

**The goal:** publishable numbers, and the honesty to report them properly.

1. ~~**Fill out the benchmark suites.**~~ **Done.** `benchmarks/HFNet.Benchmarks` is one
   BenchmarkDotNet project covering every library: tokenizer throughput on a multilingual corpus,
   encoder latency by sequence length, memory-mapped against full dataset loads, GEMM against the
   foundation and ILGPU, LoRA steps, sharded gradients, managed against ONNX Runtime, diffusion steps,
   Hub transfer. Results are in [docs/benchmarks.md](docs/benchmarks.md).
2. **Measure before optimising, and say which configuration a number came from.** The foundation's
   experience is instructive: its ILGPU path measured 5–8× *slower* than the CPU because everything
   is `double`. Any claim here needs the hardware and the configuration attached.
3. ~~**A single-precision path.**~~ **Done, as the opt-in it was meant to be.**
   `ComputeOptions.LinearLayers = Precision.Single` runs the products in float32: eight rows are
   transposed into one vector per input and each of a panel's twelve weights broadcast against it, so
   every multiply-add covers eight values instead of four. The kernel is bit-identical to an in-order
   float32 sum. bert-base at 128 tokens: 434 ms to 326 ms; ViT-base 743 ms to 551 ms, its top
   probability moving by 3.5e-8. Training ignores the setting, since its backward pass is the
   derivative of the double forward pass. Profiling on the way found a GELU epilogue running on one
   thread and attention scoring one key at a time; fixing both took ViT from 1,034 ms to 774 ms in
   double.
4. ~~**A better GEMM.**~~ **Done.** Weights are packed into float32 panels of twelve outputs, and a
   4x12 register-blocked kernel runs the product, widening each panel slice once per call. From about
   12 GMAC/s to 35-54 at 80 rows. bert-base went from 111 ms to 48 ms (torch: 38 ms), ViT-base from
   1,964 ms to 948 ms, and a LoRA training step about 2.2x faster, all with unchanged results. What
   separates it from torch now is mostly float64 against float32, which is item 3.
5. ~~**KV caching**~~ **Done with GPT-2.** `CausalLanguageModel` keeps every layer's keys and values,
   so each new token costs one row: about 22 tokens a second for `gpt2` on a laptop CPU, 25 in single
   precision.

## v0.5 — diffusion in earnest

~~Done.~~ The pipeline now follows diffusers' ONNX pipelines step for step - NumPy's generator
reproduced bit for bit, CLIP's tokenizer, batched guidance, the geometry read from the repository's
configs - and on float32 exports gives diffusers' image to the pixel, on float16 ones to about 55 dB.
Image to image (with the VAE encoder's unseeded sampling turned into its mean), inpainting on 9-channel
and on ordinary UNets, AUTOMATIC1111 emphasis, and LoRA merged into the ONNX weights without rewriting
them. A curated list of repositories that work is in [docs/GraviDiffusers.md](docs/GraviDiffusers.md),
with `onnxruntime/sd-turbo` marked CUDA-only.

## Next

- **Rotary-position decoders** - Llama, Mistral, GPT-NeoX. The causal attention and the KV cache exist
  now; what is missing is the block (rotary positions, RMSNorm, gated feed-forward, grouped queries).
- **Compile the notebooks in CI**, not just validate their JSON.
- **Route (a) of v0.3**, biased attention in the foundation's autodiff tape, for the wider ecosystem.

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
