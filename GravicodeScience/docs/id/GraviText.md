# GraviText

*[English](../GraviText.md)* · NLP modern untuk .NET — tokenisasi, embedding, transformer, dan pipeline tugas.

> **Baca ini dulu.** Library ini tidak menyertakan bobot transformer terlatih. Forward pass
> encoder-nya lengkap dan benar, tetapi `TransformerModel` yang baru dibuat diinisialisasi secara
> acak, sehingga vektor keluarannya adalah derau berstruktur. Untuk makna semantik tanpa berkas
> bobot, gunakan `Word2Vec` atau `TfidfVectorizer`, yang belajar dari korpus *Anda*. Lihat bagian
> [Transformer](#transformer) di bawah.

Bahasa Inggris dan Bahasa Indonesia didukung sepenuhnya — daftar stop word, stemming, dan leksikon
sentimen tersedia untuk keduanya.

## Tokenisasi

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

`StripAccents` memakai tabel pelipatan eksplisit alih-alih normalisasi Unicode, karena library
dibangun dengan `InvariantGlobalization` di mana `String.Normalize` diam-diam mengembalikan
masukannya tanpa perubahan.

### Kosakata dan sub-kata

```csharp
var vocabulary = Vocabulary.Build(tokenizedDocuments, minFrequency: 2, maxSize: 30_000);
vocabulary["hello"];        // id, atau UnknownId
vocabulary[42];             // token
vocabulary.Save("vocab.txt");

var learned = WordPieceTokenizer.Train(documents, vocabularySize: 5000, minFrequency: 2);
var tokenizer = new WordPieceTokenizer(learned);

tokenizer.Tokenize("playing");             // [play, ##ing]
var (ids, mask) = tokenizer.Encode(text, maxLength: 128);   // menambah [CLS]/[SEP], memberi padding
```

WordPiece mencocokkan setiap kata secara rakus dari prefiks terpanjang ke bawah lalu mencocokkan
sisanya dengan penanda `##`. Itulah yang membuat kosakata tetap 30 ribu mampu menutup bahasa yang
terbuka: kata yang belum pernah dilihat terurai menjadi potongan yang dikenal, bukan runtuh menjadi
`[UNK]`.

## Linguistik

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

Stemmer Indonesia menerapkan aturan kombinasi imbuhan terlarang Nazief-Adriani. Tanpa aturan itu
`berlari` akan kehilangan `ber-` sekaligus akhiran `-i` palsu dan menghasilkan `lar`; karena `be-`
tidak pernah berpasangan dengan `-i`, huruf terakhirnya benar dipertahankan sebagai bagian akar.

## Vektorisasi

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
counts.TransformSparse(documents);        // yang dibutuhkan korpus nyata

var tfidf = new TfidfVectorizer(options, sublinearTf: true);
var matrix = tfidf.FitTransform(documents);
tfidf.TopTerms(matrix, row: 0, count: 10);
```

Batas frekuensi dokumen adalah bagian yang berguna: batas bawah membuang salah ketik dan istilah
sekali muncul yang hanya menambah dimensi, batas atas membuang istilah yang begitu umum sehingga
tidak membawa sinyal — sebuah daftar stop word khas korpus, dipelajari alih-alih ditulis tangan.

```csharp
Similarity.Cosine(a, b);
Similarity.Jaccard(tokensA, tokensB);
Similarity.Levenshtein("kitten", "sitting");    // 3
Similarity.MostSimilar(matrix, query, top: 5);
```

## Embedding

```csharp
using Gravicode.Science.GraviText.Embeddings;

var embeddings = new Word2Vec(
        dimensions: 100, windowSize: 5, minCount: 5,
        negativeSamples: 5, epochs: 5, seed: 42)
    .Train(documents);

embeddings.Similarity("king", "queen");
embeddings.MostSimilar("king", top: 10);
embeddings.Analogy("man", "king", "woman");     // b - a + c
embeddings.Average(tokens);                     // vektor kalimat sederhana
embeddings.Save("vectors.vec");                 // format teks word2vec

new GloVe(dimensions: 50, windowSize: 5).Train(documents);
```

Word2Vec belajar lewat prediksi: untuk setiap pasangan (pusat, konteks) ia mendekatkan vektor
keduanya dan menjauhkan segelintir kata negatif yang diambil acak. Kata negatif diambil dari
distribusi unigram dipangkatkan 3/4, dan kata yang sangat sering muncul dibuang secara acak lewat
subsampling — keduanya penting, dan keduanya diimplementasikan.

### Menghapus komponen bersama

```csharp
var centred = embeddings.RemoveCommonComponent();
```

Vektor hasil pelatihan berbagi satu arah bersama yang besar, karena setiap kata terdorong ke arah
yang sama oleh negative sampling yang sama-sama dialaminya. Arah itu tidak membawa makna tetapi
mendominasi kosinus — pada korpus demo 80 ulasan, rata-rata kosinus berpasangan **0,995** sebelum
penghapusan dan **0,431** sesudahnya. Ini pascapemrosesan standar (Mu & Viswanath,
"all-but-the-top"), dan pada korpus kecil inilah pembeda antara kemiripan yang informatif dan yang
tidak berguna.

**Mutu embedding membutuhkan data.** Delapan puluh ulasan pendek adalah demo. Vektor kata sungguhan
memerlukan jutaan token; contohnya menyatakan itu secara eksplisit alih-alih menyajikan tetangga
yang berisik sebagai hasil.

## Transformer

```csharp
using Gravicode.Science.GraviText.Transformers;

var config = TransformerConfig.Preset("bert-base", vocabularySize: 30522);
// atau: new TransformerConfig(vocab, HiddenSize: 128, Layers: 4, Heads: 8, IntermediateSize: 512)

var model = new TransformerModel(config, seed: 42).WithTokenizer(wordPiece);

model.Forward(tokenIds, attentionMask);    // satu vektor per posisi
model.Encode("teks contoh", maxLength: 128); // dipool menjadi satu vektor
model.EncodeBatch(documents);               // paralel antar dokumen
model.AttentionMaps;                        // per lapisan, per head

model.LoadWeights("weights.json");          // menyetel HasPretrainedWeights
model.HasPretrainedWeights;
```

Preset: `bert-base`, `bert-small`, `bert-mini`, `bert-tiny`.

Blok penyusunnya tersedia terpisah: `MultiHeadAttention`, `TransformerEncoderLayer`, `LayerNorm`,
`DenseLayer`, `Activations.Gelu`. Pembagi `sqrt(d)` pada attention bukan kosmetik — tanpanya hasil
perkalian titik tumbuh seiring lebar model, softmax jenuh, dan gradien lenyap.

Pooling memakai **rata-rata** hidden state akhir, bukan vektor `[CLS]`, karena `[CLS]` baru membawa
makna kalimat setelah model di-fine-tune untuk menaruhnya di sana.

## Pipeline tugas

### Sentimen

```csharp
using Gravicode.Science.GraviText.Tasks;

var analyzer = new SentimentAnalyzer();
analyzer.Analyze("produknya bagus dan sangat memuaskan");   // positive
analyzer.Analyze("this is not good");                        // negative — negasi ditangani

analyzer.Train(documents, labels);      // kini memakai model terlatih
analyzer.Explain("positive", 15);       // istilah yang mendorong keputusannya
```

Jalur leksikon ada karena sentimen adalah satu-satunya tugas yang bisa dijawab dengan layak tanpa
data latih: daftar polaritas ditambah penanganan negasi dan penguat sudah cukup jauh. Dengan label,
classifier terlatih jelas lebih baik.

### Klasifikasi

```csharp
var classifier = new TextClassifier().Train(documents, labels);
classifier.Predict("teks contoh");          // label, keyakinan, skor tiap kelas
classifier.Evaluate(testDocs, testLabels);
classifier.TopFeatures("positive", 15);     // dibaca langsung dari koefisien
```

TF-IDF ke regresi logistik tetap baseline yang kuat, dilatih dalam hitungan detik, dan — tidak
seperti transformer — akan memberi tahu persis kata mana yang mendorong sebuah keputusan.

### Entitas, ringkasan, kata kunci

```csharp
var ner = new NamedEntityRecognizer()
    .AddGazetteer("PRODUCT", ["GraviNum", "GraviFrame"]);
ner.Recognize("Kang Fadhil dari Gravicode Studios di Bandung sejak 2019");
// Kang Fadhil [PERSON], Gravicode Studios [ORGANIZATION], Bandung [LOCATION], 2019 [DATE]

new TextRankSummarizer().Summarize(article, sentenceCount: 3);
new KeywordExtractor().Fit(corpus).Extract(document, count: 10);
```

**Pengenal entitas ini berbasis aturan, bukan model urutan terlatih.** Ia andal menemukan entitas
yang cocok dengan gazetteer-nya atau berdampingan dengan kata pemicu (`PT`, `Dr.`, `di`, `in`), dan
akan melewatkan entitas baru yang bisa ditangkap CRF atau transformer yang di-fine-tune. Ia ada
karena tidak memerlukan data latih dan sepenuhnya dapat diperiksa — perluas dengan `AddGazetteer`.

Peringkasnya bersifat **ekstraktif**: kalimatnya diambil apa adanya dari sumber, diperingkat dengan
PageRank atas graf tumpang tindih kata isi. Ia tidak mungkin berhalusinasi.

## Kesalahan yang sering terjadi

| Gejala | Penyebab |
|---|---|
| Vektor transformer tampak tak bermakna | Tidak ada bobot terlatih — lihat catatan di awal |
| Semua kemiripan kata ≈ 1,0 | Panggil `RemoveCommonComponent()` |
| Stop word Indonesia tidak terbuang | Berikan `StopWords.Indonesian` atau `Bilingual` secara eksplisit |
| Kosakata Word2Vec kosong | `minCount` di atas frekuensi korpus |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
