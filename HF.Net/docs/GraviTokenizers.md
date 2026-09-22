# GraviTokenizers

**Tokenizers that reproduce what a model was trained with.**

Mirrors `tokenizers`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTokenizers.Components;
```

## Correctness first

```csharp
var tokenizer = HfTokenizer.FromPretrained("bert-base-uncased");
var encoding  = tokenizer.Encode("Hello, world! Tokenizers are unbelievable.");

Console.WriteLine(string.Join(' ', encoding.Tokens));
Console.WriteLine(string.Join(' ', encoding.Ids));
```

```
[CLS] hello , world ! token ##izer ##s are unbelievable . [SEP]
101 7592 1010 2088 999 19204 17629 2015 2024 23653 1012 102
```

Those are the ids the Python implementation produces. **Use a model's own tokenizer**: a mismatched
one yields ids the model was never trained on, and nothing downstream will tell you.

## Loading

`FromPretrained` tries three layouts in turn, because all three are in active use on the Hub:

1. `tokenizer.json` - the fast format
2. `vocab.json` + `merges.txt` - GPT-2 style byte-level BPE
3. `vocab.txt` - original BERT WordPiece

```csharp
HfTokenizer.FromPretrained("bert-base-uncased");
HfTokenizer.Load("tokenizer.json");
HfTokenizer.FromBertVocabulary("vocab.txt", lowercase: true);
HfTokenizer.FromGpt2Files("vocab.json", "merges.txt");
```

A loader that only understands the first cannot open a large share of the older and smaller models.

## Encoding

```csharp
tokenizer.Encode(text);                                   // with special tokens
tokenizer.Encode(text, addSpecialTokens: false);
tokenizer.Encode(question, context);                      // a pair; sets TypeIds
tokenizer.EncodeBatch(texts, BatchOptions.Default);       // pad to the longest
tokenizer.EncodeBatch(texts, BatchOptions.Exactly(128));  // pad and truncate
tokenizer.Decode(ids);
```

`Encoding` carries `Ids`, `Tokens`, `AttentionMask`, `TypeIds`, `SpecialTokensMask` and `Offsets`.
`EncodingBatch` exposes `Ids`, `AttentionMask` and `TypeIds` as `[batch, width]` `NdArray`s.

> **`BatchOptions` is a record struct.** `new BatchOptions()` zeroes every field and does *not* run
> the primary constructor's defaults - so it means "no padding, truncate to zero". Use
> `BatchOptions.Default` or `BatchOptions.Exactly(n)`.

## Offsets

The part worth keeping. Offsets index the **original, untouched string**:

```csharp
const string Text = "Tokenizers are unbelievable.";
var encoding = tokenizer.Encode(Text, addSpecialTokens: false);

for (var i = 0; i < encoding.Length; i++)
    Console.WriteLine($"{encoding.Tokens[i],-14} '{encoding.Span(Text, i)}'");
```

```
token          'Token'
##izer         'izer'
##s            's'
are            'are'
unbelievable   'unbelievable'
.              '.'
```

Note the recovered capital: normalization runs **per pre-token**, not over the whole string, so case
folding and accent stripping cannot desynchronise the offsets after them.

`Span(text, first, last)` takes one range from the first token's start to the last token's end
rather than joining pieces - so the original whitespace and punctuation survive, which is what makes
an entity or answer span reportable as a real substring.

Per-subword offsets are exact where normalization preserved length (lowercasing). Where it did not -
accent folding, the byte alphabet - each piece honestly reports the whole word rather than a span
that would be off by a few characters in a way no caller could detect.

## The pipeline

Four stages, the same as the reference implementation:

```
normalize  ->  pre-tokenize  ->  subword model  ->  post-process
```

```csharp
var tokenizer = new HfTokenizer(
    new WordPieceModel(vocabulary),
    new BertPreTokenizer(),
    new BertNormalizer(lowercase: true),
    new WordPieceDecoder(),
    PostProcessor.Bert(clsId: 101, sepId: 102),
    addedTokens);
```

| Stage | Available |
|---|---|
| Normalizers | `BertNormalizer` `LowercaseNormalizer` `StripNormalizer` `ReplaceNormalizer` `SequenceNormalizer` |
| Pre-tokenizers | `BertPreTokenizer` `WhitespacePreTokenizer` `PunctuationPreTokenizer` `ByteLevelPreTokenizer` `MetaspacePreTokenizer` `SplitPreTokenizer` `SequencePreTokenizer` |
| Models | `WordPieceModel` `BpeModel` `UnigramModel` |
| Decoders | `WordPieceDecoder` `ByteLevelDecoder` `MetaspaceDecoder` `WhitespaceDecoder` |

Keeping the stages separate is what lets one class serve BERT, RoBERTa and SentencePiece models
rather than needing a tokenizer class per family.

## The three models

**WordPiece** is greedy longest-match from the left and never backtracks. That is the reference
behaviour, and it is why a single unknown character in the middle of a word makes the **whole word**
`[UNK]` rather than just that character. Emitting the pieces found so far would produce a token
sequence the model was never trained on, which is worse than admitting the word is out of
vocabulary.

**BPE** applies merges **by rank, not by position**. Scanning for the first mergeable pair lets an
early low-rank merge consume a symbol a higher-rank merge needed - producing a different
segmentation for a small fraction of words, small enough to pass a smoke test and large enough to
move a model's output.

**Unigram** finds the highest-probability segmentation by Viterbi, a global optimum rather than a
greedy walk. A locally attractive long piece that forces a bad split afterwards loses to the better
overall segmentation. A single character with no piece still has to be crossable, or a word
containing one unknown character has no path at all.

## Byte-level BPE

```csharp
var gpt2 = HfTokenizer.FromPretrained("gpt2");
gpt2.Encode("Hello world").Tokens;      // Hello  Ġworld
gpt2.Encode("Hello world").Ids;         // 15496 995
```

Every one of the 256 byte values maps to a distinct printable character, so any input - emoji,
invalid UTF-8, control characters - is representable and round-trips exactly. No byte maps to
whitespace or to a control character, either of which would be destroyed by a later normalization
step. `ByteAlphabet.Encode` and `.Decode` expose the mapping.

## Post-processing

The special tokens a model expects are stored as a template rather than hardcoded, because the
convention differs per family:

| Family | Single | Pair |
|---|---|---|
| BERT | `[CLS] $A [SEP]` | `[CLS] $A [SEP] $B [SEP]` |
| RoBERTa | `<s> $A </s>` | `<s> $A </s> </s> $B </s>` |
| GPT-2 | `$A` | `$A $B` |

RoBERTa doubling its separator between a pair is not decoration - the model was trained with it, and
a single separator shifts the second segment by one.

## Shortcuts and training

```csharp
Tokenizer.BPE(text);            // GPT-2 byte-level
Tokenizer.WordPiece(text);      // bert-base-uncased
Tokenizer.SentencePiece(text);  // xlm-roberta-base

Tokenizer.TrainWordPiece(corpus, vocabularySize: 5000);
Tokenizer.TrainBpe(corpus, vocabularySize: 1000);
Tokenizer.TrainUnigram(corpus, vocabularySize: 1000);
```

The shortcuts download a well-known tokenizer on first use and keep it for the process. They are for
exploring and for samples; anything with a model should use that model's own.

## A note on normalization

These libraries build with `InvariantGlobalization`, under which `String.Normalize` silently returns
its input unchanged. Accent stripping therefore goes through an explicit folding table - a
decomposition-based implementation would look correct, compile, and quietly do nothing. `NFC`,
`NFD`, `NFKC` and `NFKD` entries in a `tokenizer.json` are accepted and ignored for the same reason,
which is the honest behaviour: pretending to normalize would be worse, and refusing to load would
reject most real tokenizers.

## See also

[GraviTransformers](GraviTransformers.md) · [GraviHub](GraviHub.md)
