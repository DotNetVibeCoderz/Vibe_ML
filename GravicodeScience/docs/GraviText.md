# GraviText

*[Bahasa Indonesia](id/GraviText.md)* · Modern NLP for .NET — tokenization, embeddings, transformers and task pipelines.

> **Read this first.** No pretrained transformer weights ship with this library. The encoder's
> forward pass is complete and correct, but a freshly constructed `TransformerModel` is randomly
> initialised, so its output vectors are structured noise. For semantics without a weight file,
> use `Word2Vec` or `TfidfVectorizer`, which learn from *your* corpus. See
> [Transformers](#transformers) below.

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
| Transformer vectors look meaningless | No pretrained weights — see the note at the top |
| Every word similarity is ~1.0 | Call `RemoveCommonComponent()` |
| Indonesian stop words are not removed | Pass `StopWords.Indonesian` or `Bilingual` explicitly |
| Word2Vec vocabulary is empty | `minCount` is above the corpus frequencies |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
