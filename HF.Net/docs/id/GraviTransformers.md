# GraviTransformers

**Model Hugging Face terlatih: encoder keluarga BERT; decoder GPT-2, Llama, Mistral, Qwen, dan Pythia; ViT dan CLIP.**

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
// NotSupportedException: 'gpt2' is a 'gpt2' model, a decoder that continues text.
// Load it with CausalLanguageModel.Load, not as an encoder.
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
// ImageProcessor(224x224, Bilinear, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5])
```

Angkanya berasal dari `preprocessor_config.json` milik repositori itu sendiri, tidak pernah dari
nilai bawaan yang ditulis di dalam HF.Net. Model yang dilatih pada masukan di `[-1, 1]` lalu diberi
masukan di `[0, 1]` tetap menjawab, tetap menjawab dengan yakin, dan tidak ada apa pun dalam
keluarannya yang memberi tahu bahwa masukannya salah.

Bila prosesor dan model berselisih soal panjang sisi — atau bila repositori tidak menyertakan konfig
prosesor sama sekali — `image_size` **milik model** yang menang, kecuali Anda meminta ukuran lain.
Itulah kisi tempat position embedding dipelajari, jadi itulah satu-satunya resolusi yang tidak perlu
diinterpolasi.

Pengubahan ukurannya adalah **algoritme PIL sendiri, direproduksi sampai ke byte**: image processor
transformers mengubah ukuran lewat PIL, dan model melihat apa pun yang dihasilkan PIL. "Bilinear" dan
"bicubic" pustaka lain berbeda di tiga tempat yang masing-masing menggeser piksel satu tingkat atau
lebih - PIL melebarkan kernel saat mengecilkan, memakai `a = -0.5` untuk kubiknya, dan bekerja dalam
fixed point 22 bit dengan gambar yang dibulatkan ke byte di antara lintasan horizontal dan vertikal.
Penskalaan ulang dan normalisasi berjalan dalam float32, seperti milik rujukannya. Dari PNG tensornya
identik dengan milik `ViTImageProcessor` dan `CLIPImageProcessor`, bit demi bit. JPEG bisa berbeda satu
tingkat pada sebagian piksel, karena libjpeg milik PIL dan ImageSharp mendekode JPEG sedikit berbeda.
Pengubahan ukuran sisi terpendek dan potong-tengah (CLIP, DeiT) mengikuti konfigurasi prosesor.

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

`google/vit-base-patch16-224` memakan sekitar **0,74 detik** per gambar pada 224 piksel, 0,55 dalam
presisi tunggal, melawan 226 ms milik torch. Dari berkas gambar yang sama, probabilitas lima teratasnya
sepakat dengan torch dalam float64 sampai dua belas digit. Dulu ia memakan 12 detik; lihat
[benchmark](benchmarks.md) untuk apa yang berubah.

## CLIP

```csharp
using var clip = ClipModel.Load("openai/clip-vit-base-patch32");

foreach (var label in clip.ZeroShot("bee.jpg", ["a bee", "a flower", "a butterfly", "a bird"]))
    Console.WriteLine($"{label.Label,-12} {label.Score:P2}");
```

```
a bee        76.65 %
a flower     21.03 %
a butterfly   2.21 %
a bird        0.09 %
```

Dua encoder yang dilatih agar sebuah gambar dan kalimat yang menggambarkannya berdekatan. Setiap label
dimasukkan ke `This is a photo of {label}.` - templat yang dipakai pipeline transformers - lalu
softmax dijalankan atas kemiripan gambar dengan setiap kalimat, diskalakan dengan suhu yang dipelajari
checkpoint. Tidak ada yang dilatih. `EmbedImage`, `EmbedText`, `Logits`, dan `Similarity` memberi
bagian-bagiannya.

Menara visinya adalah ViT dengan satu layer norm tambahan sebelum blok pertama. Menara teksnya blok
pre-norm yang sama, dijalankan **kausal** - setiap token hanya memperhatikan dirinya dan token
sebelumnya - dan vektor kalimat dibaca pada token akhir teks. Posisi mana itu punya dua jawaban:
checkpoint OpenAI menulis `eos_token_id: 2` di konfigurasinya, yang lebih tua dari kolom itu, dan
dibaca pada id token tertinggi; yang lebih baru pada token akhir pertama. Keduanya mengikuti
transformers. Checkpoint OpenAI juga memakai `quick_gelu`, `x * sigmoid(1.702 x)`, bentuk GELU ketiga.

Pada `openai/clip-vit-base-patch32`, logit-nya sepakat dengan torch dalam float64 sampai **8e-14** dari
berkas gambar yang sama.

## Pembangkitan teks

```csharp
using var model = CausalLanguageModel.Load("HuggingFaceTB/SmolLM2-135M");   // atau gpt2, Qwen/Qwen2.5-0.5B, ...

model.Generate("The lighthouse keeper opened the door and", new GenerationSettings(MaxNewTokens: 40));

foreach (var piece in model.Stream(prompt, new GenerationSettings(
             MaxNewTokens: 60, Sample: true, Temperature: 0.8, TopP: 0.95, RepetitionPenalty: 1.2, Seed: 1)))
    Console.Write(piece);
```

`CausalLanguageModel` menjalankan empat keluarga blok, dipilih dari `model_type` di konfigurasi:

| `model_type` | Checkpoint | Apa yang dilakukan bloknya |
|---|---|---|
| `gpt2` | gpt2, distilgpt2, gpt2-medium | posisi yang dipelajari, LayerNorm, GELU, bobot `Conv1D` |
| `llama` | Llama 2/3, TinyLlama, SmolLM2 | posisi rotary, RMSNorm, feed-forward SiLU bergerbang, grouped-query attention |
| `mistral` | Mistral | seperti Llama, dengan jendela attention geser |
| `qwen2`, `qwen3` | Qwen2.5, Qwen3 | seperti Llama, dengan bias query/key/value (Qwen2) atau norm per head (Qwen3) |
| `gpt_neox` | Pythia | posisi rotary pada sebagian tiap head, LayerNorm, residual paralel |

Penskalaan rotary `linear` dan `llama3` (milik Llama 3.1/3.2) diterapkan; `yarn`, `dynamic`, dan
`longrope`, yang mengubah posisi sesuai panjang teks, ditolak dengan menyebut namanya. Begitu pula
decoder lain - Gemma, Phi-3, Falcon, mixture of experts - disertai arahan ke ONNX.

**Cara pemeriksaannya.** Untuk setiap keluarga, checkpoint acak kecil dijalankan di transformers
dalam float64, dengan tiga tempat yang tetap dihitung transformers dalam float32 - RMSNorm, sudut
rotary, dan softmax eager - dialihkan ke float64, sehingga yang dibandingkan adalah aritmetika yang
sama. Logit di setiap posisi sepakat sampai 1e-10, dan generasi greedy melalui cache sama dengan
`generate` milik transformers token demi token: grouped-query attention, kedua penskalaan rotary,
jendela geser yang lebih pendek dari masukan, bias Qwen2, norm per head Qwen3 dan lebar head yang
bukan hidden / heads, rotasi sebagian Pythia dengan kedua susunan residual. `gpt2` sendiri memberi
teks greedy transformers kata demi kata, dan SmolLM2-135M, Pythia-160m, Qwen2.5-0.5B, serta
Qwen3-0.6B memberi 30 token greedy pertama transformers secara identik. Dibanding eksekusi float32
biasa transformers, logit model sungguhan berbeda di digit keenam atau ketujuh, yang bisa mengubah
pilihan greedy bila dua token nyaris seri.

**Prompt** dienkode seperti `tokenizer(prompt)` milik transformers mengenkodenya, termasuk special
token - token awal teks Llama di depan, tidak ada pada GPT-2. Model chat mengharapkan templat chat-nya;
terapkan dulu ke teksnya. Generasi berhenti di token akhir mana pun milik model, termasuk yang
tercantum di `generation_config.json` (`<|eot_id|>` Llama 3).

Generasi menyimpan **cache key/value** yang tumbuh seiring teks, bukan disiapkan untuk seluruh konteks
model: key dan value sebuah token tidak pernah berubah setelah dihitung, jadi setiap token baru hanya
memakan satu baris melalui jaringan. Sekitar 22 token per detik untuk `gpt2` di CPU laptop, 25 dalam
presisi tunggal. Dengan grouped-query attention, cache hanya menyimpan head key/value.

Dua detail checkpoint yang perlu diketahui. GPT-2 menyimpan proyeksinya sebagai `Conv1D`, lapisan
linear yang disimpan `[masukan, keluaran]` - kebalikan dari semua lapisan lain di Hub - dan dibalik
saat dimuat. NeoX menggabungkan query, key, dan value per head, `[q_h | k_h | v_h]` untuk setiap head
bergiliran, dan dikelompokkan ulang saat dimuat. Kesalahan pada salah satunya menghasilkan model yang
berjalan dan mengeluarkan omong kosong yang terdengar lancar.

**Memori.** Bobot disimpan sebagai float32. Checkpoint dengan embedding terikat saat ini menyimpan
tabel tokennya dua kali - sekali untuk pencarian, sekali dikemas untuk lapisan keluaran - jadi model 1B
dengan kosakata 128 ribu token butuh sekitar 1 GB lebih dari bobotnya.

## Presisi

```csharp
ComputeOptions.LinearLayers = Precision.Single;
```

Semuanya berjalan dalam presisi ganda secara bawaan, itulah sebabnya hidden state sepakat dengan
float64 milik torch sampai sekitar 1e-13. Mengalihkan lapisan linear ke float32 - presisi yang dipakai
torch sendiri secara bawaan - kira-kira menggandakan kecepatannya, karena satu instruksi vektor memuat
delapan float sementara hanya empat double. Attention, norm, dan aktivasi tetap double.

| | double | single |
|---|---|---|
| `bert-base-uncased`, 128 token | 434 ms | 326 ms |
| `google/vit-base-patch16-224` | 743 ms | 551 ms |
| `openai/clip-vit-base-patch32`, gambar dan 5 label | 231 ms | 159 ms |
| `gpt2`, token per detik | 22 | 25 |

Probabilitas teratas ViT bergeser 3,5e-8. Pelatihan mengabaikan setelan ini: satu langkah LoRA atau
prefix tuning selalu menjalankan forward pass-nya dalam double, karena backward pass yang ditulis
tangan adalah turunan dari fungsi itu.

## Model yang lebih besar dari memori

```csharp
using var encoder = StreamingEncoder.Load("bert-large-uncased");
var vector = encoder.Embed("a sentence");
```

`TransformerModel` menyimpan setiap parameter di memori. `StreamingEncoder` tidak menyimpan apa pun:
setiap forward pass membaca baris embedding yang dipakai token-tokennya, lalu bobot setiap lapisan
secara bergiliran, langsung dari safetensors yang dipetakan ke memori, menjalankan lapisan itu, lalu
melepaskannya. Checkpoint yang dipecah (sharded) dibaca lintas pecahannya.

Pada `bert-large-uncased` embedding-nya sama sampai bit terakhir, dan puncak memori privat turun dari
**5,9 GB menjadi 0,45 GB**. Setiap pass membaca bobot lagi, jadi ini cocok untuk model yang kalau tidak
begitu sama sekali tidak bisa dijalankan. Pickle PyTorch tidak bisa dibaca per bagian dan ditolak,
disertai cara konversinya.

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
sempit — sedangkan aktivasi dan penjumlahan dalam `double`. Lapisan linear berjalan sebagai GEMM
dengan blok register di atas panel bobot float32. Untuk satu kalimat 12 token, `bert-base-uncased`
memakan sekitar **48 ms** melawan 38 ms milik torch, dan hidden state-nya sepakat dengan torch dalam
float64 hingga sekitar **1e-13**. Untuk throughput, ekspor ke ONNX dan pakai
[GraviOptimum](GraviOptimum.md): model yang sama di sana memakan sekitar 31 ms, lebih cepat daripada
torch.

`Encoder` adalah model acuan, tetapi inferensi berjalan pada salinan terkompilasi dari parameternya.
Bila Anda mengubahnya — secara manual, atau lewat apa pun selain `PeftModel.Merge` yang sudah
melakukannya sendiri — panggil `WeightsChanged()`, atau prediksi berikutnya masih memakai nilai lama:

```csharp
model.Encoder.Layers[0].Intermediate.Weights[0, 0] = 0.5;
model.WeightsChanged();
```

Encoder tidak butuh KV cache - setiap posisi toh memperhatikan semua posisi lain. Decoder-nya, `CausalLanguageModel`, punya.

## Batasan

- Decoder selain GPT-2, Llama, Mistral, Qwen2/3, dan GPT-NeoX, serta arsitektur encoder-decoder, ditolak.
- `ClipModel` membaca checkpoint CLIP lengkap; SigLIP dan ALIGN ditolak.
- Aktivasi selain `gelu`, `gelu_new`, `gelu_pytorch_tanh`, `gelu_fast`, `gelu_python`, `quick_gelu`
  dan `relu` ditolak saat memuat, dengan menyebut namanya.

## Lihat juga

[GraviTokenizers](GraviTokenizers.md) · [GraviPEFT](GraviPEFT.md) · [GraviOptimum](GraviOptimum.md)
