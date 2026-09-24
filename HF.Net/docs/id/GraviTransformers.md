# GraviTransformers

**Memuat encoder Hugging Face terlatih dan menjalankannya.**

Padanan `transformers`. Pustaka andalan.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviTransformers;
```

## Seluruhnya dalam lima baris

```csharp
using var model = TransformerModel.Load("bert-base-uncased");

foreach (var fill in model.FillMask("The capital of France is [MASK].", topK: 3))
    Console.WriteLine($"{fill.Token,-10} {fill.Score:P2}");
```

```
paris      41.53 %
lille       7.16 %
lyon        6.31 %
```

`Load` menyatukan empat hal yang semuanya harus sepakat: `config.json`, bobot, pemetaan nama parameter
checkpoint ke encoder, dan tokenizer yang cocok. Buang objeknya setelah selesai — ia memegang berkas
bobot dalam keadaan terbuka.

## Apa yang dijalankannya

```csharp
TransformerModel.SupportedModelTypes
// bert  roberta  xlm-roberta  distilbert  electra  camembert  mpnet  deberta
```

Selain itu **ditolak**, bukan dimuat setengah jalan:

```csharp
TransformerModel.Load("gpt2");
// NotSupportedException: 'gpt2' adalah model 'gpt2'. GraviTransformers menjalankan encoder
// keluarga BERT (...); arsitektur decoder-only dan encoder-decoder memerlukan causal masking
// dan cross-attention yang tidak dimiliki encoder ini.
```

Mengisi encoder dari bobot decoder menghasilkan model yang berjalan mulus dan mengembalikan omong
kosong — hasil terburuk yang mungkin.

## Tugas

### Klasifikasi

```csharp
using var model = TransformerModel.Load("distilbert-base-uncased-finetuned-sst-2-english");

Console.WriteLine(string.Join(", ", model.Labels));     // NEGATIVE, POSITIVE

foreach (var prediction in model.Predict("I absolutely loved this film.", topK: 2))
    Console.WriteLine(prediction);                      // POSITIVE: 99.99 %
```

Memerlukan checkpoint dengan head klasifikasi; checkpoint dasar tidak punya, dan `Predict`
mengatakannya alih-alih mengarang. Periksa dengan `model.HasClassificationHead`.

### Fill-mask

```csharp
model.FillMask("He was a [MASK] player in the national team.", topK: 5);
// regular 55.0 %, key 19.4 %, former 5.8 %, capped 2.1 %, prominent 1.4 %
```

Ini sekaligus uji paling tajam bahwa sebuah checkpoint termuat dengan benar. Model dengan bobot
ter-transpose atau position embedding yang bergeser tetap menghasilkan vektor yang tampak masuk akal —
tetapi tidak akan menjawab pertanyaan tentang Prancis dengan *paris*.

### Named entity

```csharp
using var model = TransformerModel.Load("dslim/bert-base-NER");

foreach (var entity in model.FindEntities(text))
    Console.WriteLine($"{entity.Label,-5} {entity.Text}  [{entity.Start}..{entity.End})  {entity.Score:P1}");

// PER   Kang Fadhil        [0..11)     98,8 %
// ORG   Gravicode Studios  [20..37)    99,5 %
// LOC   Bandung            [41..48)    99,7 %
```

Tag BIO didekode menjadi entitas utuh, dan tiap entitas dikembalikan sebagai **rentang dari string
asli**, bukan sebagai potongan subword yang disambung kembali. Dengan begitu kapitalisasi dan tanda
baca di dalam entitas tetap utuh, dan pemanggilnya tidak perlu membersihkan awalan `##` dengan
menebak-nebak. Pemeriksaannya: `model.HasTokenClassificationHead`.

Token classifier menyimpan `classifier.weight` dengan bentuk yang sama seperti sequence classifier,
jadi pemuat mengklaim token head lebih dulu; jika terbalik, setiap checkpoint NER akan termuat
sebagai pengklasifikasi kalimat dengan ratusan kelas tak bermakna.

### Question answering

```csharp
using var model = TransformerModel.Load("distilbert-base-cased-distilled-squad");

foreach (var answer in model.Answer(question, passage, topK: 3))
    Console.WriteLine($"{answer.Text}  ({answer.Score:P1})");

// safetensors and PyTorch checkpoints  (52,4 %)
// both safetensors and PyTorch checkpoints  (37,8 %)
```

Pertanyaan dan paragraf masuk sebagai pasangan, dan itulah yang menguji `SegmentDelta`. Pencarian
rentangnya dibatasi, bukan sepasang argmax yang berdiri sendiri: hanya posisi di segmen 1 yang
memenuhi syarat, akhir tidak boleh mendahului awal, dan panjangnya dibatasi. Pencarian tanpa batasan
dengan senang hati menjawab dengan rentang yang mulai di pertanyaan dan berakhir di paragraf.
Pemeriksaannya: `model.HasQuestionAnsweringHead`.

### Embedding

```csharp
var vector  = model.Embed("HF.Net brings Hugging Face models to .NET.");   // [hidden]
var matrix  = model.EmbedBatch(texts);                                     // [batch, hidden]
var hidden  = model.Hidden(text);                                          // [tokens, hidden]

model.Similarity("the cat sat on the mat", "the dog sat on the rug");      // 0.8949
model.Similarity("the cat sat on the mat", "quarterly earnings beat");     // 0.4936
```

`Embed` melakukan **mean-pooling** atas token, bukan mengambil `[CLS]`. Pada model yang belum
di-fine-tune untuk kemiripan kalimat, `[CLS]` nyaris konstan sehingga setiap pasangan kalimat tampak
mirip. `EmbedBatch` memparalelkan antar-masukan, dan di situlah throughput-nya berada.

## Konfigurasi

```csharp
var config = PretrainedConfig.FromPretrained("bert-base-uncased");

config.ModelType; config.HiddenSize; config.Layers; config.Heads;
config.IntermediateSize; config.VocabularySize; config.MaxPositions;
config.IdToLabel; config.LabelCount; config.HeadSize; config.PositionOffset;
```

Setiap dimensi dibaca dari **daftar alias**, karena tiap keluarga menamai besaran yang sama dengan
berbeda: BERT menulis `num_hidden_layers`, DistilBERT `n_layers`, GPT-2 `n_layer`. Dimensi yang hilang
**melempar exception** alih-alih memakai nilai bawaan — model dengan dua belas lapis sementara
checkpoint-nya enam akan termuat "berhasil" dan mengembalikan derau.

`PositionOffset` bernilai 2 untuk RoBERTa dan kerabatnya, yang mencadangkan dua slot posisi pertama
untuk padding — itulah sebabnya `max_position_embeddings` mereka 514, bukan 512. Mengabaikannya
menggeser setiap position embedding sebesar dua, tanpa ketidakcocokan bentuk yang bisa menangkapnya.

## Bobot

```csharp
using var weights = WeightStore.FromPretrained("bert-base-uncased");

weights.Names;
weights.Contains("bert.embeddings.word_embeddings.weight");
weights.Read(name);
weights.TryReadAny(out var tensor, "classifier.weight", "classifier.out_proj.weight");
```

Menyelesaikan tata letaknya sekali — satu berkas safetensors, safetensors ber-shard dengan indeks,
atau pickle PyTorch — lalu menjawab berdasarkan nama parameter. safetensors lebih diutamakan bila
keduanya ada: ia tidak memerlukan penafsiran, jadi lebih cepat dibuka dan aman pada berkas yang tidak
tepercaya.

Tensor dibaca sesuai permintaan dan tidak di-cache. Memuat model menyentuh setiap parameter satu kali,
dan caching akan melipatgandakan memori puncak tanpa manfaat.

## Memuat checkpoint secara manual

```csharp
var (encoder, report) = CheckpointLoader.Load(config, weights, strict: true);

report.Prefix;       // "bert." — dideteksi, bukan diasumsikan
report.Loaded;       // 196
report.Missing;      // []
report.IsComplete;
```

Tiga hal berbeda antar-keluarga dan ketiganya **dideteksi**:

1. **Awalan parameter** — `bert.`, `roberta.`, `distilbert.`, atau tidak ada sama sekali. Checkpoint
   hasil fine-tune rutin diekspor ulang dengan awalan berbeda dari yang tersirat arsitekturnya.
2. **Tata letak blok** — DistilBERT menamai bagiannya `q_lin`, `k_lin`, `ffn.lin1` dan seterusnya.
3. **Konvensi transpose** — diperiksa terhadap **bobot feed-forward**, satu-satunya matriks tak-persegi
   dalam satu blok. Proyeksi attention berbentuk persegi dan menerima kedua pembacaan tanpa keluhan.

### Dua konvensi yang perlu diketahui

**Layer norm punya dua ejaan.** Checkpoint BERT orisinal berasal dari TensorFlow dan menamainya
`gamma` dan `beta`; ekspor PyTorch modern menamainya `weight` dan `bias`. `bert-base-uncased` sendiri
masih diterbitkan dengan ejaan lama, sehingga pemuat yang hanya tahu ejaan baru gagal pada model
paling banyak diunduh di Hub. Keduanya diterima.

**Token type embedding dilipat ke dalam word embedding.** Untuk masukan satu-urutan setiap posisi
bersegmen 0, jadi menambahkan `token_type_embeddings[0]` ke setiap baris matriks word embedding adalah
*persis* setara. Membuang sukunya justru akan menggeser setiap hidden state sebesar vektor konstan
yang tidak dihilangkan layer norm pertama, karena ia ditambahkan sebelum norm, bukan sesudah.

Pasangan kalimat juga memerlukan segmen 1, dan pelipatan tidak bisa menyediakannya. **Selisih**
`token_type_embeddings[1] - token_type_embeddings[0]` dibawa pada `LoadReport.SegmentDelta` dan
ditambahkan per posisi untuk baris yang termasuk urutan kedua, sehingga kedua segmen tereproduksi
persis. `CheckpointLoader.SupportsPairs` bernilai `true`.

## Task head

Tiga tata letak mencakup hampir semua encoder hasil fine-tune di Hub, dan ketiganya berbeda baik pada
nama parameter maupun aktivasinya:

| Keluarga | Lapisan | Aktivasi |
|---|---|---|
| BERT | pooler dense, lalu `classifier` | tanh |
| DistilBERT | `pre_classifier`, lalu `classifier` | ReLU |
| RoBERTa | `classifier.dense`, lalu `classifier.out_proj` | tanh |

Memakai aktivasi yang salah bukan soal kosmetik: tanh menjenuh sedangkan ReLU tidak, sehingga logit
yang dihasilkan keliru dengan cara yang tetap memeringkat kelas secara masuk akal.

Proyeksi keluaran head masked-language biasanya **diikat** ke word embedding masukan sehingga tidak
ada di checkpoint. Kembali memakai matriks embedding bukan jalan pintas, melainkan memang modelnya —
menganggap tensor yang hilang sebagai "tidak ada head" akan membuat fill-mask tidak tersedia pada
sebagian besar checkpoint dasar, yang justru merekalah yang memilikinya.

## Kinerja

Inferensi berjalan dengan `double` di CPU. Pustaka ini ada supaya sebuah model bisa **dimuat,
diperiksa dan dipahami** dalam .NET murni — bukan untuk melayani permintaan. Untuk throughput,
ekspor ke ONNX dan pakai [GraviOptimum](GraviOptimum.md).

Tidak ada KV cache karena ini encoder: setiap posisi toh memperhatikan semua posisi lain.

## Batasan

- Arsitektur decoder-only dan encoder-decoder ditolak.
- Model vision (ViT, CLIP) ada di peta jalan; lihat
  [PLAN.md](../../PLAN.md).

## Lihat juga

[GraviTokenizers](GraviTokenizers.md) · [GraviPEFT](GraviPEFT.md) · [GraviOptimum](GraviOptimum.md)
