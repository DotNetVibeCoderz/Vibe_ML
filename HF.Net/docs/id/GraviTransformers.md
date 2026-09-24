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

## Vision

Vision Transformer adalah blok encoder yang sama di atas embedding yang berbeda, jadi ia tinggal di
sini alih-alih di pustaka tersendiri.

```csharp
using Gravicode.HFNet.GraviTransformers.Vision;

using var model = VisionTransformer.Load("google/vit-base-patch16-224");

foreach (var prediction in model.Classify("bee.jpg", topK: 5))
    Console.WriteLine($"{prediction.Label,-24} {prediction.Score:P2}");
```

```
bee                       94,46 %
pot, flowerpot             1,32 %
daisy                      0,86 %
ant, emmet, pismire        0,30 %
fly                        0,29 %
```

Gambar dipotong menjadi kisi persegi 16 piksel, tiap persegi diproyeksikan menjadi satu vektor,
sebuah vektor `[CLS]` terlatih ditaruh di depan, lalu position embedding ditambahkan. Setelah itu ia
hanyalah urutan 197 token seperti urutan mana pun.

`Embed` mengembalikan vektor `[CLS]`, bukan rata-rata seluruh patch: tidak seperti encoder teks
biasa, ViT dilatih awal dengan tujuan klasifikasi tepat pada posisi itu, jadi ia sudah memuat
ringkasan seluruh gambar yang justru harus disusun ulang oleh mean pooling. `Similarity`
membandingkan dua gambar dengannya.

```csharp
var vector = model.Embed("bee.jpg");             // [hidden]
model.Similarity("bee.jpg", "wasp.jpg");
```

`VisionTransformer.Open(directory)` memuat model yang tidak pernah ada di Hub.

### Pre-norm, dan mengapa ini tipe tersendiri

ViT melakukan normalisasi **sebelum** setiap sublapisan dan menambahkan residual sesudahnya. BERT
menambahkan residual lebih dulu lalu menormalisasi jumlahnya. Setiap parameter berbentuk sama pada
kedua cara itu, sehingga checkpoint ViT yang dimuat ke encoder post-norm akan termuat tanpa satu pun
keluhan, lalu mengembalikan omong kosong yang terdengar yakin.

Ada satu uji yang bersih untuk itu, dan ia ada di dalam suite: nolkan semua proyeksi dalam satu
blok. Blok pre-norm lalu menjadi **identitas** — residual ditambahkan ke nol. Blok post-norm
mengembalikan `LayerNorm(input)`, yang bukan input.

### Prapemrosesan adalah bagian dari model

```csharp
model.Processor
// ImageProcessor(224x224, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5])
```

Angkanya berasal dari `preprocessor_config.json` milik repositori itu sendiri, tidak pernah dari
nilai bawaan yang ditulis di dalam HF.Net. Model yang dilatih pada masukan di `[-1, 1]` lalu diberi
masukan di `[0, 1]` tetap menjawab, tetap menjawab dengan yakin, dan tidak ada apa pun dalam
keluarannya yang memberi tahu bahwa masukannya salah.

Bila prosesor dan model berselisih soal panjang sisi — atau bila repositori tidak menyertakan konfig
prosesor sama sekali — `image_size` **milik model** yang menang, kecuali Anda meminta ukuran lain.
Itulah kisi tempat position embedding dipelajari, jadi itulah satu-satunya resolusi yang tidak perlu
diinterpolasi.

### Resolusi lain

```csharp
using var model = VisionTransformer.Load("google/vit-base-patch16-224", imageSize: 384);

model.Processor.Size;                            // 384
model.Classify("bee.jpg");                       // kisi patch 24x24, bukan 14x14
```

Kelipatan berapa pun dari ukuran patch bisa dipakai, begitu pula persegi panjang yang diberikan
langsung ke `Forward(pixels)` atau `Classify(pixels)`. Checkpoint hanya menyimpan posisi untuk kisi
14x14; untuk kisi lain, posisi itu diubah ukurannya sebagai gambar 768 kanal, persis seperti yang
dilakukan transformers dengan `interpolate_pos_encoding=True`. Posisi token kelas bukan bagian dari
kisi dan dibiarkan apa adanya. Tabel untuk tiap kisi dihitung sekali lalu disimpan.

Resampler-nya harus sama persis dengan milik torch, karena resampler lain menghasilkan model yang
tetap berjalan namun diam-diam menggeser setiap patch. Tiga detail menentukannya, dan masing-masing
menggeser hasil beberapa perseratus: kernelnya kubik Keys dengan `a = -0.75`, bukan `-0.5` yang
dipakai sebagian besar pustaka gambar; koordinat sumbernya setengah piksel,
`(i + 0.5) * in / out - 0.5`; dan tap yang jatuh di luar kisi mengulang tepinya. Uji-ujinya
mengunci ketiganya terhadap `torch.nn.functional.interpolate` hingga 1e-12.

Dengan piksel yang identik, `google/vit-base-patch16-224` mengembalikan probabilitas lima teratas
yang sama dengan torch hingga sepuluh angka desimal pada 160, 224, dan 384 piksel. Ukuran yang
bukan kelipatan utuh dari patch ditolak, bukan dipotong.

Resolusi yang lebih tinggi ada harganya: 384 piksel (577 token) memakan waktu sekitar 3,5 kali 224
piksel (197 token) — 6,8 detik melawan 2,0 detik. Sebagian besar ada di lapisan linear, yang tumbuh
sebanding jumlah patch; attention tumbuh kuadratik tetapi bukan lagi bagian yang terbesar.

### Apa yang bisa dijalankan

`vit` dan `deit`. Backbone berjendela atau konvolusional — Swin, ConvNeXt — ditolak dengan menyebut
namanya, disertai arahan ke ONNX. ViT dengan head yang belum dikenal tetap termuat: encoder-nya sama
dan fiturnya tetap terbaca, `HasClassificationHead` bernilai `false`, dan `Classify` mengatakannya
alih-alih mengarang kelas.

Aktivasi feed-forward mengikuti `hidden_act` di konfig. `gelu` di sana berarti GELU **eksak**
berbasis erf, sedangkan `gelu_new` berarti aproksimasi tanh. Keduanya berbeda hingga 4e-4 per nilai,
cukup untuk menggeser probabilitas pada angka desimal ketiga. Aktivasi yang tidak dikenal encoder
ini ditolak dengan menyebut namanya.

### Kecepatan

`google/vit-base-patch16-224` memakan sekitar **2 detik** per gambar pada 224 piksel melawan 226 ms
milik torch. Dengan piksel yang sama, probabilitas lima teratasnya sepakat dengan torch dalam float64
hingga 1,3e-15. Dulu ia memakan 12 detik; lihat [benchmark](benchmarks.md) untuk apa yang berubah.

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

Inferensi berjalan di atas kernel milik HF.Net sendiri, yang dipakai bersama oleh encoder teks dan
vision. Bobot disimpan sebagai float32 — eksak, karena setiap checkpoint menyimpan F32 atau lebih
sempit — sedangkan aktivasi dan penjumlahan dalam `double`. Untuk satu kalimat 12 token,
`bert-base-uncased` memakan sekitar **111 ms** melawan 36 ms milik torch, dan hidden state-nya
sepakat dengan torch dalam float64 hingga sekitar **1e-13**. Untuk throughput, ekspor ke ONNX dan
pakai [GraviOptimum](GraviOptimum.md): model yang sama di sana memakan 24 ms, lebih cepat daripada
torch.

`Encoder` adalah model acuan, tetapi inferensi berjalan pada salinan terkompilasi dari parameternya.
Bila Anda mengubahnya — secara manual, atau lewat apa pun selain `PeftModel.Merge` yang sudah
melakukannya sendiri — panggil `WeightsChanged()`, atau prediksi berikutnya masih memakai nilai lama:

```csharp
model.Encoder.Layers[0].Intermediate.Weights[0, 0] = 0.5;
model.WeightsChanged();
```

Tidak ada KV cache karena ini encoder: setiap posisi toh memperhatikan semua posisi lain.

## Batasan

- Arsitektur decoder-only dan encoder-decoder ditolak.
- CLIP belum tersedia: menara teksnya kausal, sedangkan encoder ini tidak. ViT dan DeiT tersedia.
- Aktivasi selain `gelu`, `gelu_new`, `gelu_pytorch_tanh`, `gelu_fast`, `gelu_python` dan `relu`
  ditolak saat memuat, dengan menyebut namanya.

## Lihat juga

[GraviTokenizers](GraviTokenizers.md) · [GraviPEFT](GraviPEFT.md) · [GraviOptimum](GraviOptimum.md)
