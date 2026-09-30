# GraviPEFT

**LoRA dan prefix tuning, dalam format Hugging Face PEFT.**

Padanan `peft`.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviPEFT;
```

## Apa yang dilakukannya

Ia **melatih** adapter LoRA bersama sebuah classification head, dengan backpropagation melewati
encoder terlatih yang bobotnya sendiri tetap dibekukan. Ia **memasang** adapter, termasuk yang
dilatih dengan PEFT di Python, dan **melipatnya** ke dalam bobot secara persis. Ia **menyimpan dan
memuat** tata letak Hugging Face PEFT. Pada BERT, adapter yang dilatih di sini, termasuk
classifier-nya, bisa dimuat di Python dan memberi prediksi yang sama di sana. Ia juga bisa melatih
head saja di atas encoder yang dibekukan, baseline murah yang layak dicoba lebih dulu.

```csharp
PeftModel.SupportsAdapterTraining   // true, sejak 0.3
```

## Memasang adapter

```csharp
using var model = TransformerModel.Load("bert-base-uncased");

var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

var (adapter, encoder, fraction) = peft.ParameterEfficiency();
Console.WriteLine($"{adapter:N0} dari {encoder:N0} = {fraction:P3}");
```

Model yang dikembalikan berperilaku **identik** dengan masukannya, karena matriks `B` setiap adapter
bernilai nol. Itulah sifat penentu LoRA, bukan detail implementasi: pelatihan bermula dari perilaku
terlatih, bukan dari gangguan atasnya. Menginisialisasi kedua matriks secara acak tetap berlatih,
tetap konvergen, dan berakhir di tempat yang terukur lebih buruk.

`A` diinisialisasi Kaiming-uniform dan bukan nol, kalau tidak gradien tidak akan pernah mengalir.

## Konfigurasi

```csharp
new LoraConfig(Rank: 8, Alpha: 16, TargetModules: ["query", "value"], Dropout: 0);
```

`Scaling` adalah `Alpha / Rank`. Normalisasi itulah yang menjadikan rank sebagai tombol **kapasitas**,
bukan tombol laju belajar: tanpanya, menggandakan rank menggandakan besar pembaruan saat inisialisasi.

Hanya mengadaptasi proyeksi query dan value adalah rekomendasi makalah aslinya dan menjadi bawaan di
sini. Menambahkan key dan output kira-kira menggandakan parameter yang dilatih demi perubahan yang
biasanya masih di dalam derau.

## Memuat adapter terbitan

```csharp
var peft = PEFT.LoadAdapter(model, "some-user/some-lora", merge: true);
```

Membaca `adapter_model.safetensors` (atau `.bin`) berdampingan dengan `adapter_config.json`, persis
dalam tata letak yang ditulis PEFT:

```
base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.weight
base_model.model.bert.encoder.layer.0.attention.self.query.lora_B.weight
```

Adapter yang hanya punya separuh pasangan **ditolak**: menerapkannya akan menambahkan pembaruan
berperingkat nol yang tampak seperti adapter yang bekerja padahal tidak melakukan apa-apa.

## Melipat (merge)

```csharp
peft.Merge();
```

Melipat setiap adapter ke dalam bobot dasar, di tempat dan secara persis. Setelah dilipat, inferensi
memakan biaya persis sama dengan model dasar — tidak ada adapter tersisa untuk dievaluasi.

Ini tepat dilakukan sebelum melayani dan keliru dilakukan sebelum menukar adapter, karena lipatan itu
**tidak bisa dibatalkan** hanya dari bobot hasilnya.

Path adapter dicocokkan berdasarkan indeks lapisan ditambah akhiran, bukan path penuh, karena adapter
yang sama diterbitkan dengan beberapa awalan. Bila tidak ada yang cocok, `Merge` melempar exception
alih-alih diam-diam tidak menerapkan apa pun.

## Melatih head

```csharp
peft.FitHead(texts, labels);

peft.Predict("this is excellent");      // -> Prediction[]
peft.Score(heldOutTexts, heldOutLabels);
peft.HeadLabels;
```

Melatih regresi logistik pada embedding beku dari encoder yang sudah diadaptasi. Ini murah karena
encoder dievaluasi **sekali per contoh** lalu dipakai ulang: forward pass transformer mendominasi
biaya, dan melatih head kemudian hanyalah regresi atas beberapa ratus dimensi.

Inilah baseline yang layak dijalankan sebelum `Train`. Bila fitur beku sudah memisahkan kelas-kelasnya,
hanya ini yang Anda perlukan. Bila belum, seperti pada negasi, di mana "good" dan "not good" berbagi
hampir semua token, adapter punya sesuatu untuk ditambahkan.

## Melatih adapter

```csharp
using var model = TransformerModel.Load("bert-base-uncased");
var peft = PEFT.ApplyLoRA(model, new LoraConfig(Rank: 8, Alpha: 16));

var report = peft.Train(texts, labels, new TrainingOptions
{
    Epochs = 8,
    BatchSize = 8,
    LearningRate = 1e-3,
    Progress = new Progress<TrainingProgress>(p => Console.WriteLine($"{p.Step}/{p.TotalSteps} {p.Loss:F4}")),
});

Console.WriteLine(report);                 // 32 steps in a minute or so, loss 0.74 -> ... -> 0.008
peft.Predict("The staff were not friendly at all.");
```

Setiap langkah menjalankan tiap contoh melewati encoder dengan adapter di dalam jalurnya, memasang
classification head di atas hidden state terakhir, lalu melakukan backpropagation atas loss
cross-entropy ke `A` dan `B` setiap adapter serta ke head. Setelah itu ia mengambil satu langkah
AdamW. Bobot dasar tidak pernah berubah. Adapter diperbarui di tempat, sehingga `SaveAdapter` dan
`Merge` sesudahnya melihat nilai hasil pelatihan.

Bentuk head-nya bergantung pada checkpoint:

- **Pada BERT**, head-nya persis milik `BertForSequenceClassification`: baris `[CLS]` melewati pooler
  pralatih dari checkpoint (dense dan tanh, tetap dibekukan seperti yang dilakukan PEFT), lalu
  `classifier` yang dilatih. `peft.HeadLoadsInPython` bernilai `true`, dan adapter yang disimpan
  memberi prediksi yang sama di Python.
- **Pada model lain**, head-nya `classifier` di atas rata-rata baris. Head bawaan RoBERTa dan
  DistilBERT punya lapisan yang tidak disimpan PEFT, sehingga head tiruannya pun tidak akan
  terulang di Python. Head ini disimpan dan dimuat ulang hanya oleh HF.Net.

| Opsi | Bawaan | |
|---|---|---|
| `Epochs` | 3 | |
| `BatchSize` | 8 | contoh yang rata-rata loss-nya membentuk satu micro-batch |
| `GradientAccumulation` | 1 | micro-batch per langkah optimizer |
| `LearningRate` | 5e-4 | puncak; LoRA butuh kira-kira sepuluh kali laju fine-tuning penuh |
| `WeightDecay` | 0 | terpisah, gaya AdamW; bias dikecualikan |
| `WarmupFraction` | 0 | warm-up linear, lalu turun linear sampai nol |
| `MaxGradientNorm` | 1.0 | clipping norma global; 0 mematikannya |
| `MaxLength` | 128 | pemotongan, dalam token; untuk tanya-jawab, ukuran jendela |
| `DocStride` | `MaxLength / 3` | hanya tanya-jawab: token yang dibagi oleh jendela passage yang bertumpuk |
| `Seed` | 42 | pengacakan, inisialisasi head, dropout adapter |

Optimizer, jadwal dan clipping-nya adalah milik `Trainer` Hugging Face, ditulis ulang dari
`torch.optim.AdamW`, `get_linear_schedule_with_warmup` dan `clip_grad_norm_`, dan pengujiannya
mengunci ketiganya pada rumus-rumus itu. Satu akibatnya kerap mengejutkan: bila ada warm-up
sedikit pun, langkah pertama diambil dengan laju belajar tepat nol, sama seperti di Python.
`LoraConfig.Dropout` hanya berlaku saat pelatihan.

### Melatih token classifier

Untuk named entity, beri tag pada kata lalu latih classifier di atas setiap token:

```csharp
IReadOnlyList<string>[] words = [["Ani", "lives", "in", "Bandung", "."], /* ... */];
IReadOnlyList<string>[] tags  = [["B-PER", "O", "O", "B-LOC", "O"], /* ... */];

peft.TrainTokenClassifier(words, tags, new TrainingOptions { Epochs = 8, BatchSize = 4, LearningRate = 2e-3 });

peft.FindEntities("Kartini moved to Yogyakarta.");   // PER: Kartini, LOC: Yogyakarta
```

Kata beserta satu tag per kata adalah bentuk dataset gaya CoNLL. Kata-katanya digabung dengan spasi
lalu di-tokenisasi, dan hanya potongan pertama setiap kata yang dilatih. Itulah
`label_all_tokens=False` milik Transformers, dan loss-nya adalah rata-rata atas potongan-potongan itu
dalam satu langkah, seperti cara cross-entropy PyTorch menghitungnya pada satu batch.
`FindEntities` membaca label setiap kata dari potongan pertamanya, seperti
`aggregation_strategy="first"`, dan mengembalikan rentang teks asli dengan huruf besar-kecilnya utuh.

Head-nya satu `classifier` di atas setiap token. Hanya itu yang dimiliki model
`ForTokenClassification` setiap keluarga, jadi token classifier yang disimpan bisa dimuat di Python
pada RoBERTa dan DistilBERT juga, bukan hanya BERT. Melatih satu jenis head menggantikan jenis
lainnya, karena keduanya bergantung pada adapter yang sama.

### Melatih tanya-jawab

Untuk tanya-jawab ekstraktif, berikan pertanyaan, bagian teks (passage) yang menjawabnya, dan
jawabannya persis seperti yang tertulis di sana, dalam bentuk SQuAD:

```csharp
AnswerExample[] examples =
[
    new("Where does Ani live?", "Ani lives in Bandung. Ani works as a teacher.", "Bandung"),
    new("What does Ani do?",    "Ani lives in Bandung. Ani works as a teacher.", "a teacher"),
    // ...
];

peft.TrainQuestionAnswering(examples, new TrainingOptions { Epochs = 10, BatchSize = 4, LearningRate = 2e-3 });

peft.Answer("What does Hendra do?", "Hendra lives in Semarang. Hendra works as a chef.");   // a chef
```

Head-nya `qa_outputs`, skor awal dan akhir untuk setiap token, dan loss-nya milik Transformers
sendiri: rata-rata cross-entropy atas posisi awal dan cross-entropy atas posisi akhir. Pertanyaan dan
passage masuk sebagai pasangan, dengan passage sebagai segmen 1. `AnswerStart` adalah
`answer_start` milik SQuAD; bila dikosongkan, kemunculan pertama jawaban yang dipakai.

Passage yang terlalu panjang untuk `MaxLength` dipecah menjadi jendela-jendela yang bertumpuk, masing-
masing memuat seluruh pertanyaan dan sepotong passage yang berbagi `DocStride` token dengan jendela
berikutnya. Itulah `truncation="only_second"` dengan `stride` milik Transformers. Setiap jendela
menjadi satu contoh pelatihan. Jendela yang tidak memuat jawaban secara utuh dilatih sebagai menunjuk
`[CLS]`, cara Transformers menandai "tidak ada di jendela ini". `Answer` membaca passage panjang
dengan jendela yang sama dan mengembalikan rentang terbaik dari semuanya. Pada bert-base dengan
jendela 48 token, ia menjawab dari kalimat keempat sebuah pasangan sepanjang 84 token.

> **Di Python, lipat dulu sebelum memprediksi.** `PeftModelForQuestionAnswering.forward` milik PEFT
> 0.21 menerima `token_type_ids` tetapi tidak meneruskannya, sehingga lewat wrapper itu setiap passage
> dibaca sebagai segmen 0. `BertForQuestionAnswering` milik Transformers sendiri, dan HF.Net,
> membacanya sebagai segmen 1. Panggil `model.merge_and_unload()` dan prediksinya sama dengan HF.Net
> hingga 3e-10. Bug yang sama berarti adapter QA yang dilatih di Python dengan PEFT belajar tanpa
> segmen, sehingga di HF.Net ia berjalan di atas representasi masukan yang sedikit berbeda dari yang
> dipelajarinya.

### Mengapa ia punya backward pass sendiri

Fondasi sudah memiliki encoder autodiff, dan encoder itu tidak dipakai karena satu alasan spesifik:
ia tidak punya bias pada proyeksi query, key dan value, padahal setiap BERT terlatih memilikinya.
Gradien yang diambil melewatinya adalah gradien dari *model yang sedikit berbeda*. Ia akan berlatih,
konvergen, dan menghasilkan adapter yang diam-diam salah untuk model tempat adapter itu dipasang.

Backward pass di sini ditulis tangan di atas kernel yang sama dengan yang dipakai inferensi. Karena
bobot dasar dibekukan, satu-satunya gradien yang dibutuhkan adalah milik adapter. Selebihnya
dipropagasikan lalu dibuang, dan tidak ada yang dipropagasikan di bawah lapisan teradaptasi yang
paling rendah.

### Bagaimana ia diperiksa

- **Gradien.** Setiap entri setiap adapter, pada keenam proyeksi sebuah model dua lapis yang setiap
  bias dan norm-nya digeser dari nilai awalnya, dicocokkan dengan beda-hingga pusat. Selisihnya
  di bawah 1e-6 untuk GELU eksak maupun GELU tanh, tetap begitu saat dropout aktif, dan berlaku
  lewat setiap head: head pooler BERT, head rata-rata, head token, dan head rentang jawaban pada
  pasangan kalimat. Backward pass yang sengaja dibuat salah (skala
  atensinya dihilangkan, atau turunan tanh pada pooler) gagal dengan selisih dua sampai tiga kali lipat.
- **Forward.** Tanpa adapter, forward pass pelatihan sama dengan encoder inferensi hingga 1e-12. Pada
  `bert-base-uncased` sesudah pelatihan, prediksi dari adapter di dalam jalur dan dari bobot yang
  sudah dilipat sama hingga sekitar 1e-10.
- **Pulang-pergi, di Python.** Adapter dan classifier yang dilatih dan disimpan di sini, lalu dimuat
  ke PEFT 0.21 bersama `AutoModelForSequenceClassification`, memberi probabilitas kelas yang sama
  dengan HF.Net hingga **5e-9**, tanpa peringatan dari PEFT. Hidden state-nya sama dengan model
  HF.Net yang sudah dilipat hingga 7e-7.
- **Named entity, di Python.** Adapter yang dilatih pada 16 kalimat bertag, dimuat ke
  `AutoModelForTokenClassification` dengan PEFT, membuat pipeline token-classification milik
  Transformers sendiri memberi rentang dan label yang sama dengan `FindEntities`, dengan skor yang
  sama hingga 5e-11. Itu berlaku juga pada nama yang tidak pernah ditunjukkan kepadanya.
- **Tanya-jawab, di Python.** Adapter yang dilatih pada 24 contoh gaya SQuAD, dimuat ke
  `AutoModelForQuestionAnswering` lalu dilipat, memberi jawaban yang sama dengan HF.Net dengan skor
  yang sama hingga 3e-10, pada orang dan tempat yang tidak pernah ditunjukkan kepadanya.
- **Pulang-pergi, di sini.** Dimuat ulang dengan `PEFT.LoadAdapter`, prediksinya sama dengan model
  yang disimpan hingga 5e-9. Itulah F32 yang disimpan file-nya, yang juga ditulis PEFT.

### Biayanya, dan batasnya

- **CPU, satu micro-batch per pass, dipadatkan (packed) alih-alih diberi padding.** Contoh-contoh
  dalam satu micro-batch disusun berurutan dan melewati lapisan linear sebagai satu matriks,
  sementara atensi tetap di dalam masing-masing contoh dan position embedding dimulai ulang pada
  setiap contoh. Tidak ada padding, jadi tidak ada yang terbuang atau perlu di-mask, dan sebuah test
  mengunci pass yang dipadatkan agar sama dengan menjalankan contoh satu per satu. Pada bert-base,
  32 kalimat pendek selama 8 epoch memakan 20 detik bila dipadatkan 8 sekaligus, dibanding 37 detik
  satu per satu. GEMM dengan blok register paling diuntungkan oleh baris yang lebih banyak, jadi
  pemadatan inilah yang membuatnya bisa berjalan pada kecepatan penuh.
- **Memori.** Pelatihan menyimpan salinan float32 dari bobot encoder di samping model rujukan,
  sekitar 340 MB untuk bert-base.
- **Klasifikasi sekuens, klasifikasi token, dan tanya-jawab ekstraktif.** Passage yang panjang
  memakan satu pass per jendela, baik saat pelatihan maupun saat menjawab.
- **Dropout milik encoder dimatikan.** Python melatih BERT dengan dropout 0,1 di setiap blok. Di sini
  hanya `LoraConfig.Dropout` yang berlaku, pada masukan adapter. Adapter hasilnya tetap sah; hanya
  saja kedua proses pelatihan itu tidak sama langkah demi langkah.
- **`Train` sesudah `Merge` ditolak.** Pembaruannya akan terhitung dua kali. Untuk melanjutkan
  pelatihan adapter yang sudah dilipat, muat ulang model dasar lalu pasang adapter tanpa dilipat
  dengan `PeftModel.WithAdapters(model, adapters, merge: false)`.

Untuk lebih dari beberapa ribu contoh, latih dengan PEFT di Python lalu muat hasilnya di sini. Jalur
itu juga persis.

## Menyimpan

```csharp
peft.SaveAdapter("./my-adapter");
```

Menulis `adapter_model.safetensors` dan `adapter_config.json` dalam tata letak PEFT, sehingga hasilnya
bisa dimuat di Python. Classifier yang dilatih dengan `Train` ikut tersimpan:

| Head | Ditulis ke | Bisa dimuat di |
|---|---|---|
| pooler BERT + `classifier` | `adapter_model.safetensors` sebagai `base_model.model.classifier.*`, `task_type` `SEQ_CLS` | HF.Net dan Python |
| `classifier` rata-rata | `head.safetensors` | HF.Net |
| `classifier` token, keluarga apa pun | `adapter_model.safetensors` sebagai `base_model.model.classifier.*`, `task_type` `TOKEN_CLS` | HF.Net dan Python |
| `qa_outputs`, keluarga apa pun | `adapter_model.safetensors` sebagai `base_model.model.qa_outputs.*`, `task_type` `QUESTION_ANS` | HF.Net, dan Python setelah `merge_and_unload()` |

Nama label dan panjang pemotongan disimpan di `hfnet_head.json`, bukan di `adapter_config.json`,
tempat PEFT akan menanggapi kunci yang tidak dikenalnya dengan saran untuk upgrade. Head dari
`FitHead` adalah regresi logistik dan tidak disimpan.

Di Python:

```python
import json
from transformers import AutoModelForSequenceClassification
from peft import PeftModel

labels = {int(k): v for k, v in json.load(open("my-adapter/hfnet_head.json"))["id2label"].items()}
base = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(labels), id2label=labels)
model = PeftModel.from_pretrained(base, "my-adapter")
```

Dan kembali di .NET, dari direktori itu atau dari repositori Hub:

```csharp
var peft = PEFT.LoadAdapter(model, "./my-adapter");    // merge: true secara bawaan
peft.Predict("The staff were not friendly at all.");
```

Adapter yang dilatih di Python dengan `task_type="SEQ_CLS"` pada BERT, atau dengan
`task_type="TOKEN_CLS"` pada keluarga apa pun, atau dengan `task_type="QUESTION_ANS"`, dimuat dengan
cara yang sama, beserta head-nya.
Kedua classifier itu berbentuk sama, jadi `task_type`-lah yang membedakannya. PEFT tidak menyimpan nama label, jadi tanpa `hfnet_head.json` nama-namanya
kembali sebagai `LABEL_0`, `LABEL_1`, dan seterusnya.

Adapter buatan `ApplyLoRA` diberi kunci berupa **path modul checkpoint itu sendiri**, misalnya
`bert.encoder.layer.0.attention.self.query`, karena PEFT mencocokkan tensor ke modul berdasarkan
nama dan melewatkan, tanpa galat, tensor yang tidak bisa ditempatkannya. Muat adapter ke kelas model
yang memakai nama yang sama. Untuk `bert-base-uncased`, yang nama-nama checkpoint-nya diawali
`bert.`, artinya `AutoModelForSequenceClassification` atau `BertForMaskedLM`, bukan `BertModel`
polos. Sebelum 0.3 kuncinya `layer.0.query`, yang oleh PEFT di Python dimuat sebagai tidak ada adapter
sama sekali.

## Prefix tuning

```csharp
var tuned = PEFT.ApplyPrefixTuning(model, new PrefixTuningConfig(VirtualTokens: 10));
Console.WriteLine(tuned);    // PrefixTuningModel(..., 10 virtual tokens, 5,120/4,367,104 params = 0.117%)

tuned.Train(texts, labels, new TrainingOptions { Epochs = 3, BatchSize = 16, LearningRate = 2e-2 });
tuned.Predict("an absolute joy to watch");
tuned.Save("prefix-adapter");

var again = PrefixTuningModel.Load(model, "prefix-adapter");   // atau yang disimpan PEFT
```

Bila LoRA mengubah setiap matriks bobot sedikit, prefix tuning tidak mengubah satu pun. Setiap lapisan
attention mendapat beberapa posisi tambahan di depan teks - bukan token, yang akan terbatas pada
embedding kosakata, melainkan key dan value per lapisan dan head yang dilatih langsung. Hanya prefix
dan classifier yang dilatih: 0,12% parameter pada contoh di atas.

Pelatihannya memakai loop yang sama dengan LoRA dan backward pass yang sama, yang membawa bagian
gradien setiap lapisan kembali ke irisan prefix miliknya. Gradien prefix diperiksa terhadap
diferensiasi numerik sampai 1e-6, sendirian, dikemas (packed), dan bersama adapter LoRA. Ia butuh laju
pembelajaran lebih besar daripada LoRA, sekitar 1e-2.

**Formatnya adalah `PREFIX_TUNING` milik PEFT**: `prompt_embeddings` berbentuk `[virtual tokens,
layers * 2 * hidden]`, setiap barisnya key lalu value untuk setiap lapisan, dan pada BERT classifier
disimpan sebagai `base_model.classifier.*` di atas pooler yang dibekukan. Adapter yang disimpan PEFT
0.21 dimuat di sini dan memberi logit-nya sampai 1e-10; yang dilatih di sini dimuat di Python dengan
`PeftModel.from_pretrained` dan memberi logit yang sama sampai 1e-9.

Satu detail menentukan kesepakatan itu. PEFT memberikan prefix ke BERT sebagai `past_key_values`, dan
BERT menomori posisinya mulai dari panjangnya - jadi dengan sepuluh virtual token, token nyata pertama
berada di posisi 10, bukan 0. Tanpa pergeseran itu logit-nya berselisih 3,4. HF.Net ikut menggeser,
yang juga berarti sebuah teks paling panjang `max_position_embeddings - virtual tokens`.

## Mengapa jumlah parameter itu penting

```
bobot 768 x 768        589.824 nilai
adapter rank 8          12.288 nilai     ~2 %
```

Adapter rank-8 atas satu proyeksi kira-kira dua persen dari bobot yang diadaptasinya. Itulah seluruh
argumennya: adapter khusus-tugas berukuran beberapa megabyte, dikirim bersama satu model dasar
bersama, dan bisa ditukar tanpa memuat ulang model itu.

## Lihat juga

[GraviTransformers](GraviTransformers.md) · [GraviHub](GraviHub.md)
