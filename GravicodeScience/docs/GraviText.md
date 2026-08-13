# GraviText

*[Bahasa Indonesia](id/GraviText.md)* · Modern NLP for .NET — tokenization, embeddings, transformers and task pipelines.

> **Read this first.** No pretrained transformer weights ship with this library. The encoder's
> forward pass is complete and correct, but a freshly constructed `TransformerModel` is randomly
> initialised, so its output vectors are structured noise. Two ways out: train the architecture on
> your own labelled text with [`TransformerClassifier`](#training-one), or use `Word2Vec` /
> `TfidfVectorizer`, which learn from your corpus and are the better choice for a small dataset.
> See [Transformers](#transformers) below.

Both English and Bahasa Indonesia are supported throughout — stop words, stemming and the
sentiment lexicon all ship in both.

## Tokenization

```csharp
using Gravicode.Science.GraviText.Tokenization;

new WhitespaceTokenizer().Tokenize(text);
new RegexTokenizer().Tokenize("Hello, World! It's 2024.");   // [hello, world, it's, 2024]
new CharacterTokenizer().Tokenize(text);
SentenceSplitter.Split("First. Second! Third?");

TextNormalizer.Normalize("  Café  RÉSUMÉ ");                 // "cafe resume"
TextNormalizer.StripAccents(text);
TextNormalizer.RemovePunctuation(text);
```

`StripAccents` uses an explicit folding table rather than Unicode normalisation, because the
libraries build with `InvariantGlobalization` where `String.Normalize` silently returns its input
unchanged.

### Vocabulary and sub-words

```csharp
var vocabulary = Vocabulary.Build(tokenizedDocuments, minFrequency: 2, maxSize: 30_000);
vocabulary["hello"];        // id, or UnknownId
vocabulary[42];             // token
vocabulary.Save("vocab.txt");

var learned = WordPieceTokenizer.Train(documents, vocabularySize: 5000, minFrequency: 2);
var tokenizer = new WordPieceTokenizer(learned);

tokenizer.Tokenize("playing");             // [play, ##ing]
var (ids, mask) = tokenizer.Encode(text, maxLength: 128);   // adds [CLS]/[SEP], pads
```

WordPiece matches each word greedily from the longest prefix down and re-matches the remainder
with a `##` marker. That is what lets a fixed 30k vocabulary cover an open-ended language: an
unseen word decomposes into known pieces instead of collapsing to `[UNK]`.

## Linguistics

```csharp
using Gravicode.Science.GraviText.Linguistics;

StopWords.English;  StopWords.Indonesian;  StopWords.Bilingual;
StopWords.Remove(tokens, StopWords.Indonesian);

PorterStemmer.Stem("connecting");        // connect
PorterStemmer.StemAll(tokens);

IndonesianStemmer.Stem("makanan");       // makan
IndonesianStemmer.Stem("berlari");       // lari
IndonesianStemmer.Stem("menyapu");       // sapu

Lemmatizer.Lemmatize("were");            // be
NGrams.Extract(tokens, 2);
NGrams.Range(tokens, 1, 3);

LanguageDetector.Detect("Solusi ini adalah ...");   // ("id", 0.83)
```

The Indonesian stemmer implements the Nazief-Adriani forbidden-affix rules. Without them
`berlari` would strip both `ber-` and a spurious `-i` suffix and yield `lar`; because `be-` never
combines with `-i`, the trailing letter is correctly kept as part of the root.

## Vectorization

```csharp
using Gravicode.Science.GraviText.Vectorization;

var options = new VectorizerOptions
{
    StopWords = StopWords.Bilingual,
    Stemmer = PorterStemmer.Stem,
    MinNGram = 1,
    MaxNGram = 2,
    MinDocumentFrequency = 2,
    MaxDocumentFrequencyRatio = 0.9,
    MaxFeatures = 20_000,
};

var counts = new CountVectorizer(options);
counts.FitTransform(documents);
counts.TransformSparse(documents);        // what real corpora need

var tfidf = new TfidfVectorizer(options, sublinearTf: true);
var matrix = tfidf.FitTransform(documents);
tfidf.TopTerms(matrix, row: 0, count: 10);
```

The document-frequency bounds are the useful part: the minimum drops typos and one-off terms that
only add dimensions, and the maximum drops terms so common they carry no signal — a corpus-specific
stop word list, learned rather than hand-written.

```csharp
Similarity.Cosine(a, b);
Similarity.Jaccard(tokensA, tokensB);
Similarity.Levenshtein("kitten", "sitting");    // 3
Similarity.MostSimilar(matrix, query, top: 5);
```

## Embeddings

```csharp
using Gravicode.Science.GraviText.Embeddings;

var embeddings = new Word2Vec(
        dimensions: 100, windowSize: 5, minCount: 5,
        negativeSamples: 5, epochs: 5, seed: 42)
    .Train(documents);

embeddings.Similarity("king", "queen");
embeddings.MostSimilar("king", top: 10);
embeddings.Analogy("man", "king", "woman");     // b - a + c
embeddings.Average(tokens);                     // a simple sentence vector
embeddings.Save("vectors.vec");                 // word2vec text format

new GloVe(dimensions: 50, windowSize: 5).Train(documents);
```

Word2Vec learns by prediction: for each (centre, context) pair it pushes their vectors together
and pushes a handful of randomly drawn negatives apart. Negatives come from the unigram
distribution raised to the 3/4 power, and frequent words are randomly dropped by subsampling —
both details matter, and both are implemented.

### Removing the common component

```csharp
var centred = embeddings.RemoveCommonComponent();
```

Trained vectors share a large common direction, because every word is pushed the same way by the
negative sampling it shares with the rest of the vocabulary. That direction carries no meaning but
dominates the cosine — on the 80-review demo corpus the mean pairwise cosine is **0.995** before
removal and **0.431** after. This is standard post-processing (Mu & Viswanath, "all-but-the-top"),
and on small corpora it is the difference between an informative similarity and a useless one.

**Embedding quality needs data.** Eighty short reviews is a demo. Real word vectors need millions
of tokens; the sample says so explicitly rather than presenting noisy neighbours as a result.

## Transformers

```csharp
using Gravicode.Science.GraviText.Transformers;

var config = TransformerConfig.Preset("bert-base", vocabularySize: 30522);
// or: new TransformerConfig(vocab, HiddenSize: 128, Layers: 4, Heads: 8, IntermediateSize: 512)

var model = new TransformerModel(config, seed: 42).WithTokenizer(wordPiece);

model.Forward(tokenIds, attentionMask);   // one vector per position
model.Encode("some text", maxLength: 128); // mean-pooled to a single vector
model.EncodeBatch(documents);              // parallel across documents
model.AttentionMaps;                       // per layer, per head

model.LoadWeights("weights.json");         // sets HasPretrainedWeights
model.HasPretrainedWeights;
```

Presets: `bert-base`, `bert-small`, `bert-mini`, `bert-tiny`.

The building blocks are exposed individually: `MultiHeadAttention`, `TransformerEncoderLayer`,
`LayerNorm`, `DenseLayer`, `Activations.Gelu`. The `sqrt(d)` divisor in attention is not
cosmetic — without it dot products grow with width, the softmax saturates and gradients vanish.

Pooling uses the **mean** of the final hidden states rather than the `[CLS]` vector, because
`[CLS]` only carries sentence meaning after a model has been fine-tuned to put it there.

### Training one

`TransformerModel` computes a forward pass and nothing else. That is why a fresh one stays random
for ever: there was no gradient, so there was no way to change a weight. `TransformerClassifier`
is the same architecture built on the
[autodiff tape](GraviNum.md#automatic-differentiation), which makes the weights trainable on your
own labelled text.

```csharp
var config = new TransformerConfig(vocabularySize: 5000, HiddenSize: 64, Layers: 2, Heads: 4,
                                   IntermediateSize: 128, MaxPositions: 64);

var model = new TransformerClassifier(config, seed: 42)
    .Fit(tokenisedDocuments, labels, epochs: 20, learningRate: 0.005);

model.Predict(tokenIds);
model.Score(heldOutDocuments, heldOutLabels);
model.LossHistory;                       // mean loss per epoch
```

The layers are available on their own for building something else — `TransformerTape.LayerNorm`,
`MultiHeadAttention`, `EncoderLayer`, `Gelu`, `SoftmaxRows`, `Embed`, `MeanPool`. Every one of them
composes from tape operations that already existed for the graph networks; none needed a bespoke
kernel, so none needed its own derivation. Check any layer you write with `GradientCheck`.

> **This is not a way to obtain BERT.** A model trained here learns only from the corpus you give
> it, and a few hundred documents will not produce general language understanding — what they can
> produce is a task-specific classifier. For a small labelled set, `TfidfVectorizer` feeding a
> linear model is the stronger and far cheaper baseline, and is worth beating before reaching for
> this. What changed is that the architecture is now trainable at all, not that it is now
> pretrained.

## Task pipelines

### Sentiment

```csharp
using Gravicode.Science.GraviText.Tasks;

var analyzer = new SentimentAnalyzer();
analyzer.Analyze("produknya bagus dan sangat memuaskan");   // positive
analyzer.Analyze("this is not good");                        // negative — negation is handled

analyzer.Train(documents, labels);      // now uses the supervised model instead
analyzer.Explain("positive", 15);       // the terms that drove it
```

The lexicon path exists because sentiment is the one task where a decent answer is available with
no training data: a polarity list plus negation and intensifier handling gets a long way. With
labels, the supervised classifier is materially better.

### Classification

```csharp
var classifier = new TextClassifier().Train(documents, labels);
classifier.Predict("some text");            // label, confidence, per-class scores
classifier.Evaluate(testDocs, testLabels);
classifier.TopFeatures("positive", 15);     // read straight off the coefficients
```

TF-IDF into logistic regression remains a strong baseline, trains in seconds, and — unlike a
transformer — will tell you exactly which words drove a decision.

### Entities, summarisation, keywords

```csharp
var ner = new NamedEntityRecognizer()
    .AddGazetteer("PRODUCT", ["GraviNum", "GraviFrame"]);
ner.Recognize("Kang Fadhil dari Gravicode Studios di Bandung sejak 2019");
// Kang Fadhil [PERSON], Gravicode Studios [ORGANIZATION], Bandung [LOCATION], 2019 [DATE]

new TextRankSummarizer().Summarize(article, sentenceCount: 3);
new KeywordExtractor().Fit(corpus).Extract(document, count: 10);
```

**The recogniser is rule-based, not a trained sequence model.** It reliably finds entities that
match its gazetteers or sit next to a trigger word (`PT`, `Dr.`, `di`, `in`), and it will miss
novel entities that a CRF or fine-tuned transformer would catch. It is here because it needs no
training data and is fully inspectable — extend it with `AddGazetteer`.

The summariser is **extractive**: sentences come verbatim from the source, ranked by PageRank over
a graph of content-word overlap. It cannot hallucinate.

## Common mistakes

| Symptom | Cause |
|---|---|
| Transformer vectors look meaningless | No pretrained weights — see the note at the top. `TransformerModel` cannot be trained; use `TransformerClassifier` |
| Every word similarity is ~1.0 | Call `RemoveCommonComponent()` |
| Indonesian stop words are not removed | Pass `StopWords.Indonesian` or `Bilingual` explicitly |
| Word2Vec vocabulary is empty | `minCount` is above the corpus frequencies |

## Trainable sub-word tokenizers

`WordPieceTokenizer` applies a vocabulary it is given. Two tokenizers learn one from a corpus, by
genuinely different routes.

### Byte-pair encoding

`BpeTokenizer` starts from characters and repeatedly merges the most frequent adjacent pair,
recording each merge in order. Applying it later means replaying those merges.

```csharp
var tokenizer = BpeTokenizer.Train(corpus, vocabularySize: 8000, minFrequency: 2);

tokenizer.Encode("lowest");        // ["low", "est</w>"] — unseen, but decomposes
tokenizer.Tokenize(document);
tokenizer.Decode(pieces);
tokenizer.Save("merges.txt");
```

The difference from WordPiece is which pair merges: BPE takes the most *frequent*, WordPiece the one
that most increases corpus likelihood. In practice the vocabularies look similar and BPE is simpler
to train.

**The merge order is the model.** The vocabulary alone cannot tokenize — the same token set applied
in a different order produces a different segmentation — which is why `Save` writes ranked merges
rather than a token list. For the same reason merges are applied by *rank*, not left to right: an
early low-rank merge can consume a symbol a higher-rank merge needed.

Words carry an end-of-word marker, so `"est"` ending a word is a different token from `"est"` inside
one. Without it the tokenizer learns merges that span word boundaries and produces segmentations that
do not survive re-spacing the text. Training breaks ties deterministically, so two runs over the same
corpus cannot diverge — irreproducibility here is invisible until a model trained on one tokenizer is
served with another.

### Unigram (SentencePiece)

`UnigramTokenizer` is a different idea, not a variant. It starts from a large candidate vocabulary,
assigns every piece a probability, and *prunes downwards* — repeatedly dropping the pieces whose
removal costs the corpus likelihood least.

```csharp
var tokenizer = UnigramTokenizer.Train(corpus, vocabularySize: 8000);

tokenizer.Encode(text);                                // Viterbi: the best segmentation
tokenizer.SampleEncoding(text, rng, alpha: 0.2);       // an alternative segmentation
tokenizer.Decode(pieces);                              // exactly reversible
```

Two consequences follow, and both are why it is worth having alongside BPE:

- **Segmentation is globally optimal**, found by Viterbi over the piece lattice rather than greedily,
  so it does not depend on the order rules happened to be learned.
- **Alternative segmentations can be sampled**, which is the basis of subword regularisation —
  training a model on several segmentations of each sentence makes it markedly more robust to the
  tokenizer's arbitrary choices. BPE cannot do this without extra machinery, having no probabilities
  to sample from.

**Whitespace is part of the input, not a delimiter.** Spaces become a visible marker and the text is
treated as one raw stream, which is what makes the scheme fully reversible and what lets it handle
languages that do not put spaces between words at all.

Single characters are never pruned: a vocabulary that cannot spell a character cannot segment text
containing it. The EM step accumulates each piece's expected count over *all* segmentations weighted
by probability, not just the best one — using the Viterbi path alone makes rare pieces look worse
than they are.

## CRF sequence labelling

Labelling each token independently produces sequences that are locally plausible and globally
impossible — an `I-PER` with no `B-PER` before it. `LinearChainCrf` adds transition scores between
adjacent labels and decodes the highest-scoring *sequence*.

```csharp
var crf = new LinearChainCrf(labels.Count);
crf.ApplyBioConstraints(labels);           // an I-X may only follow a B-X or I-X of the same type
crf.Fit(emissionMatrices, tagSequences);

crf.Decode(emissions);        // Viterbi: the best sequence
crf.Marginals(emissions);     // per-token confidence, by forward-backward
crf.LogLikelihood(emissions, labels);
```

**The best sequence is not the sequence of best tokens.** Taking each token's top label
independently ignores every transition score, which is the only thing the model added.

The partition function is computed exactly by the forward algorithm in O(n·k²), over what would
otherwise be kⁿ sequences. That is what makes this a probability model rather than a scoring
function, and why the training gradient — observed counts minus expected counts, the classic
exponential-family form — is exact rather than sampled.

`Forbid` gives a transition a score no path can recover from. Stating a constraint that way is better
than hoping the training data teaches it: a learned penalty can always be outvoted by a confident
emission, and then the output is a labelling the scheme says cannot exist. Forbidden transitions have
no gradient and never acquire one, so training cannot dissolve them.

**Emissions come from outside.** The CRF takes a matrix of per-token scores — from a linear model, a
transformer, anything — and learns only the transitions. That is what makes it composable, and it
matches how a CRF is used in practice, as the final layer of a tagger.

## Trained NER

`NamedEntityRecognizer` matches gazetteers and capitalisation patterns. `TrainedNer` learns from
annotated text, which is what lets it generalise to names it has never seen.

```csharp
var sentences = TaggedSentence.LoadConll("datasets/ner_conll.txt");
var ner = new TrainedNer().Fit(sentences.Take(315).ToList());

ner.Recognize("Kartika Wijaya bekerja di Gravicode .");
//   PER: 'Kartika Wijaya'
//   ORG: 'Gravicode'

ner.Evaluate(heldOut);       // P 98.1%  R 97.7%  F1 97.9%  (213 predicted, 214 actual)
```

The design is two-stage: a linear model scores each token against its features, and a
`LinearChainCrf` decodes the best sequence from those scores. That split is what makes the BIO
constraints enforceable — the per-token model cannot know that an `I-PER` may not follow an `O`, and
the CRF makes it structurally impossible rather than merely unlikely.

Features are deliberately shape-based rather than identity-based. The word itself is one feature
among many; the rest describe capitalisation, affixes, digit content and the neighbours. The
word-shape feature does most of the work — mapping capitals to `X`, lower-case to `x` and digits to
`d` turns "Jakarta" and "Bandung" into the same `Xxxxxxx`, so evidence about one transfers to the
other. That is the whole difference between this and a gazetteer.

**Evaluate on entities, not tokens.** Token accuracy is dominated by the `O` tag — a model predicting
`O` everywhere scores above 85% on most corpora — so it is close to meaningless. `Evaluate` scores
whole entities, requiring the boundaries and the type all to be right, which is what the CoNLL
measure means and what published figures refer to.

## Decoder stack

`TransformerDecoder` is structurally an encoder with two changes: the attention is causally masked,
and a language-model head projects each position back to the vocabulary.

```csharp
var decoder = new TransformerDecoder(config, vocabulary);

decoder.Generate(prompt, maxNewTokens: 50, options: SamplingOptions.Nucleus, rng: rng);
decoder.Generate("the cat", tokenizer, maxNewTokens: 20);
decoder.Perplexity(tokenIds);
```

**Causal masking is what makes generation-time training possible.** Every position predicts its
successor, and because no position can see its own answer, all n predictions come from one forward
pass rather than one per token. Leaving the mask off does not produce a worse model — it produces one
that appears to train beautifully and generates nothing, because at inference the future it learned
to rely on is not there.

Sampling options are applied in the usual order: penalise repeats, scale by temperature, cut by
top-k, then by top-p. Applying temperature after the cuts would change which tokens the cuts should
have selected. The repetition penalty's sign matters — dividing a *negative* logit by the penalty
raises it, which is the opposite of a penalty.

```csharp
SamplingOptions.Greedy;                                  // deterministic
SamplingOptions.Nucleus;                                 // top-p at 0.9
new SamplingOptions(Temperature: 0.8, TopK: 40, RepetitionPenalty: 1.1);
```

This is **forward-only**, like `TransformerModel`: it runs a model whose weights came from elsewhere.
A trainable decoder belongs on the autodiff tape alongside `TransformerTape`.

**Generation is quadratic here, knowingly.** Each new token re-runs the whole prefix rather than
caching the keys and values of tokens already processed. A KV cache makes this linear and is the
single most valuable optimisation for a real generator; it is left out because it doubles the state a
reader has to hold, and the shapes this runs at do not need it.

---

## Visualisations

Rendered by `samples/GraviText.Console`. `notebooks/GraviText.Notebook.ipynb` adds a causal
attention heatmap, where everything above the diagonal is exactly zero.

![Word embeddings projected to two dimensions](screenshots/gravitext_embeddings.png)

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
