# GraviTokenizers

**Tokenizer yang mereproduksi persis apa yang dipakai saat model dilatih.**

Padanan `tokenizers`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviTokenizers;
using Gravicode.HFNet.GraviTokenizers.Components;
```

## Ketepatan lebih dulu

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

Itu id yang dihasilkan implementasi Python. **Gunakan tokenizer milik model itu sendiri**: tokenizer
yang tidak cocok menghasilkan id yang tidak pernah dilihat model saat latihan, dan tidak ada satu pun
bagian di hilir yang akan memberi tahu Anda.

## Memuat

`FromPretrained` mencoba tiga tata letak berurutan, karena ketiganya masih dipakai aktif di Hub:

1. `tokenizer.json` — format cepat
2. `vocab.json` + `merges.txt` — BPE byte-level gaya GPT-2
3. `vocab.txt` — WordPiece BERT orisinal

```csharp
HfTokenizer.FromPretrained("bert-base-uncased");
HfTokenizer.Load("tokenizer.json");
HfTokenizer.FromBertVocabulary("vocab.txt", lowercase: true);
HfTokenizer.FromGpt2Files("vocab.json", "merges.txt");
```

Pemuat yang hanya memahami bentuk pertama tidak bisa membuka sebagian besar model lama dan kecil.

## Encoding

```csharp
tokenizer.Encode(text);                                   // dengan special token
tokenizer.Encode(text, addSpecialTokens: false);
tokenizer.Encode(question, context);                      // pasangan; mengisi TypeIds
tokenizer.EncodeBatch(texts, BatchOptions.Default);       // pad ke yang terpanjang
tokenizer.EncodeBatch(texts, BatchOptions.Exactly(128));  // pad dan potong
tokenizer.Decode(ids);
```

`Encoding` membawa `Ids`, `Tokens`, `AttentionMask`, `TypeIds`, `SpecialTokensMask` dan `Offsets`.
`EncodingBatch` menyediakan `Ids`, `AttentionMask` dan `TypeIds` sebagai `NdArray` `[batch, width]`.

> **`BatchOptions` adalah record struct.** `new BatchOptions()` menol-kan semua field dan **tidak**
> menjalankan nilai bawaan primary constructor — jadi artinya "tanpa padding, potong ke nol". Gunakan
> `BatchOptions.Default` atau `BatchOptions.Exactly(n)`.

## Offset

Bagian yang paling berharga. Offset menunjuk ke **string asli yang tidak diubah**:

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

Perhatikan huruf kapital yang kembali utuh: normalisasi berjalan **per pre-token**, bukan atas seluruh
string, sehingga pelipatan huruf besar-kecil dan pengupasan aksen tidak bisa menggeser offset
sesudahnya.

`Span(text, first, last)` mengambil satu rentang dari awal token pertama sampai akhir token terakhir,
bukan menyambung potongan — sehingga spasi dan tanda baca asli tetap utuh. Itulah yang membuat span
entitas atau jawaban bisa dilaporkan sebagai substring sungguhan.

Offset per-subword tepat ketika normalisasi mempertahankan panjang (huruf kecil). Ketika tidak —
pelipatan aksen, alfabet byte — tiap potongan secara jujur melaporkan seluruh kata alih-alih rentang
yang meleset beberapa karakter dengan cara yang tidak bisa dideteksi pemanggil.

## Pipeline

Empat tahap, sama seperti implementasi rujukan:

```
normalize  ->  pre-tokenize  ->  model subword  ->  post-process
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

| Tahap | Tersedia |
|---|---|
| Normalizer | `BertNormalizer` `LowercaseNormalizer` `StripNormalizer` `ReplaceNormalizer` `SequenceNormalizer` |
| Pre-tokenizer | `BertPreTokenizer` `WhitespacePreTokenizer` `PunctuationPreTokenizer` `ByteLevelPreTokenizer` `MetaspacePreTokenizer` `SplitPreTokenizer` `SequencePreTokenizer` |
| Model | `WordPieceModel` `BpeModel` `UnigramModel` |
| Decoder | `WordPieceDecoder` `ByteLevelDecoder` `MetaspaceDecoder` `WhitespaceDecoder` |

Memisahkan tahap-tahap inilah yang membuat satu kelas bisa melayani model BERT, RoBERTa dan
SentencePiece sekaligus, tanpa perlu satu kelas tokenizer per keluarga.

## Tiga model

**WordPiece** memakai pencocokan terpanjang serakah dari kiri dan tidak pernah mundur. Itu perilaku
rujukannya, dan itulah sebabnya satu karakter tak dikenal di tengah kata membuat **seluruh kata**
menjadi `[UNK]`, bukan hanya karakter itu. Mengeluarkan potongan yang sudah ditemukan akan menghasilkan
urutan token yang tidak pernah dilatihkan ke model — lebih buruk daripada mengakui kata itu di luar
kosakata.

**BPE** menerapkan merge **berdasarkan peringkat, bukan posisi**. Memindai pasangan pertama yang bisa
digabung memungkinkan merge berperingkat rendah memakan simbol yang dibutuhkan merge berperingkat lebih
tinggi — menghasilkan segmentasi berbeda pada sebagian kecil kata; cukup kecil untuk lolos uji
sekilas, dan cukup besar untuk menggeser keluaran model.

**Unigram** mencari segmentasi berprobabilitas tertinggi lewat Viterbi, optimum global alih-alih
langkah serakah. Potongan panjang yang menggiurkan secara lokal tetapi memaksa pembelahan buruk
sesudahnya akan kalah oleh segmentasi keseluruhan yang lebih baik. Satu karakter tanpa potongan
padanan tetap harus bisa dilintasi, kalau tidak kata yang memuat satu karakter tak dikenal tidak punya
jalur sama sekali.

## BPE byte-level

```csharp
var gpt2 = HfTokenizer.FromPretrained("gpt2");
gpt2.Encode("Hello world").Tokens;      // Hello  Ġworld
gpt2.Encode("Hello world").Ids;         // 15496 995
```

Setiap dari 256 nilai byte dipetakan ke karakter tercetak yang berbeda, sehingga masukan apa pun —
emoji, UTF-8 tak sah, karakter kendali — bisa diwakili dan pulang-pergi persis. Tidak ada byte yang
dipetakan ke spasi atau karakter kendali, yang keduanya akan dihancurkan tahap normalisasi berikutnya.
`ByteAlphabet.Encode` dan `.Decode` membuka pemetaan itu.

## Post-processing

Special token yang diharapkan model disimpan sebagai templat, bukan ditanam di kode, karena
konvensinya berbeda per keluarga:

| Keluarga | Tunggal | Pasangan |
|---|---|---|
| BERT | `[CLS] $A [SEP]` | `[CLS] $A [SEP] $B [SEP]` |
| RoBERTa | `<s> $A </s>` | `<s> $A </s> </s> $B </s>` |
| GPT-2 | `$A` | `$A $B` |

RoBERTa menggandakan separator di antara pasangan bukan sebagai hiasan — model dilatih dengan itu, dan
separator tunggal menggeser segmen kedua sebanyak satu posisi.

## Pintasan dan pelatihan

```csharp
Tokenizer.BPE(text);            // GPT-2 byte-level
Tokenizer.WordPiece(text);      // bert-base-uncased
Tokenizer.SentencePiece(text);  // xlm-roberta-base

Tokenizer.TrainWordPiece(corpus, vocabularySize: 5000);
Tokenizer.TrainBpe(corpus, vocabularySize: 1000);
Tokenizer.TrainUnigram(corpus, vocabularySize: 1000);
```

Pintasan mengunduh tokenizer terkenal saat pertama dipakai lalu menyimpannya selama proses berjalan.
Itu untuk eksplorasi dan contoh; apa pun yang sudah punya model harus memakai tokenizer milik model
tersebut.

## Catatan tentang normalisasi

Pustaka ini dibangun dengan `InvariantGlobalization`, yang membuat `String.Normalize` diam-diam
mengembalikan masukannya apa adanya. Karena itu pengupasan aksen memakai tabel pelipatan eksplisit —
implementasi berbasis dekomposisi akan tampak benar, ter-compile, dan tidak melakukan apa-apa. Entri
`NFC`, `NFD`, `NFKC` dan `NFKD` di sebuah `tokenizer.json` diterima lalu diabaikan dengan alasan yang
sama, dan itulah perilaku yang jujur: berpura-pura menormalkan lebih buruk, sementara menolak memuat
akan menolak hampir semua tokenizer sungguhan.

## Lihat juga

[GraviTransformers](GraviTransformers.md) · [GraviHub](GraviHub.md)
