# GraviHub

**Hugging Face Hub dari .NET, beserta pembaca format yang disajikannya.**

Padanan `huggingface_hub`. Seluruh bagian HF.Net lain bergantung padanya.

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*

```csharp
using Gravicode.HFNet.GraviHub;
using Gravicode.HFNet.GraviHub.Io;
```

## Pemakaian satu baris

```csharp
var directory = Hub.DownloadModel("bert-base-uncased");
var info      = Hub.ModelInfo("bert-base-uncased");
var path      = Hub.DownloadFile("bert-base-uncased", "config.json");

foreach (var hit in Hub.SearchModels("sentiment", limit: 5, task: "text-classification"))
    Console.WriteLine($"{hit.Id} - {hit.Downloads:N0} unduhan");
```

`Hub` adalah facade atas `HubClient` tingkat proses dan sifatnya **blocking**. Itu wajar di aplikasi
konsol atau notebook, dan keliru di UI atau server — di sana pakailah `HubClient`.

## HubClient

Satu instance memegang satu `HttpClient` dan aman dipakai lintas thread. **Buat sekali saja untuk
seumur hidup proses.** Membuatnya setiap kali mengunduh akan menghabiskan socket saat beban tinggi,
yang muncul sebagai `SocketException` sesekali, jauh setelah kode penyebabnya dijalankan.

```csharp
using var client = new HubClient(new HubOptions
{
    Token      = null,                  // jatuh ke HF_TOKEN
    CacheRoot  = null,                  // jatuh ke HF_HUB_CACHE, lalu HF_HOME
    Timeout    = TimeSpan.FromMinutes(30),
    MaxRetries = 3,
});

var progress = new Progress<TransferProgress>(p => Console.WriteLine(p));

await client.SnapshotAsync("bert-base-uncased", RepoKind.Model, "main",
    allowPatterns: ["*.safetensors", "*.json"],
    ignorePatterns: null,
    progress);
```

### Mengapa pola berkas itu penting

Repositori model biasa membawa bobot yang sama dua sampai tiga kali: safetensors berdampingan dengan
pickle PyTorch dan ekspor ONNX, ditambah varian TensorFlow dan Flax pada model-model lama. Mengunduh
snapshot tanpa pola rutin memindahkan tiga kali lipat dari yang dibutuhkan — sering kali ini selisih
antara 400 MB dan 1,6 GB.

`Hub.DownloadModel(id, weightsOnly: true)` — nilai bawaannya — sudah menerapkan daftar izin yang
masuk akal.

## Metadata repositori

```csharp
var info = Hub.ModelInfo("bert-base-uncased");

info.Id; info.Sha; info.LastModified; info.Downloads; info.Likes;
info.Tags; info.PipelineTag; info.Private; info.Gated;
info.HasSafeTensors;
info.FilesWithExtension(".safetensors");
info.File("config.json");
```

Daftar berkas diambil dari tree API, bukan dari array `siblings`, karena `siblings` hanya melaporkan
nama — pemanggil yang ingin memilih berkas bobot yang lebih kecil, atau melewati shard 10 GB, tidak
punya dasar untuk memutuskan. Karena itu `RepoFile` membawa `Size`, `Sha` dan `IsLfs`.

## Cache

```csharp
Hub.Cache.Root;
Hub.Cache.Contains("bert-base-uncased", RepoKind.Model, "main", "config.json");
Hub.Cache.SizeInBytes();
Hub.Cache.Evict("bert-base-uncased", RepoKind.Model);
```

Berkas yang sudah di-cache **divalidasi ulang, bukan diunduh ulang**: satu permintaan HEAD
membandingkan ETag-nya, yang untuk objek LFS adalah hash isinya. Biaya `DownloadFile` pada checkpoint
400 MB dalam keadaan mapan karenanya hanya satu perjalanan bolak-balik.

Unduhan parsial ditulis ke berkas `.part` dan baru dipindahkan setelah lengkap, sehingga transfer yang
terputus tidak mungkin dikira selesai. Berkas `.part` sisa jalan sebelumnya dilanjutkan dengan range
request bila server mengizinkan.

## safetensors

Format yang disajikan Hub. Panjang header `u64` little-endian, header JSON yang memetakan tiap nama ke
dtype, bentuk dan rentang byte-nya, lalu satu blok data kontigu.

```csharp
// Periksa tanpa membaca data tensor apa pun.
foreach (var tensor in SafeTensors.Inspect(path))
    Console.WriteLine(tensor);          // bert.encoder.layer.0... F32 [768x768]

// Memory-mapped; ambil hanya yang diperlukan.
using var reader = SafeTensors.Open(path);
var embeddings = reader.Read("bert.embeddings.word_embeddings.weight");

// Checkpoint ber-shard.
var shards = SafeTensors.ReadShardIndex("model.safetensors.index.json");

// Menulis.
SafeTensors.Write(path, tensors, SafeTensorDType.F32, metadata);
SafeTensors.Write(path, tensors, name => name.Contains("position") ? SafeTensorDType.F32
                                                                   : SafeTensorDType.BF16);
```

Dtype yang didukung saat membaca: `Bool U8 I8 F8E4M3 F8E5M2 U16 I16 F16 BF16 U32 I32 F32 U64 I64 F64`.
Penulisan terbatas pada `F16 BF16 F32 F64`, karena `NdArray` menyimpan `double`.

**Offset di header relatif terhadap awal blok data, bukan awal berkas.** Membacanya sebagai absolut
menghasilkan tensor yang bergeser sebesar panjang header dan penuh angka yang tampak masuk akal.

Berkas di-memory-map, bukan dibaca ke dalam array byte. Sebuah checkpoint rutin lebih besar daripada
RAM, dan kasus lazim — mengambil dua puluh dari empat ratus tensor — tidak seharusnya membayar
sisanya. `ReadAll` adalah kemudahan yang mahal: checkpoint 7B dalam F16 berukuran 14 GB di disk dan
56 GB setelah dilebarkan ke `double`.

### float8

`F8_E4M3` dibaca sebagai `float8_e4m3fn`: akhiran `fn` berarti *finite*, sehingga eksponen serba-satu
menyimpan nilai normal dan hanya mantissa serba-satu di sebelahnya yang berarti NaN. Membacanya sebagai
IEEE menghasilkan infinity di tempat format itu menyimpan 448, yang kemudian meracuni setiap
penjumlahan di hilir. `F8_E5M2` berbentuk IEEE dan memang punya infinity sungguhan.

### Menulis bfloat16

Penyempitan membulatkan ke genap terdekat, bukan memotong. Pemotongan adalah implementasi yang paling
kentara dan ia membiaskan setiap bobot ke arah nol — kecil per nilai, sistematis di seluruh checkpoint,
dan persis jenis kesalahan yang lolos dari uji round-trip bertoleransi longgar.

## Checkpoint PyTorch

Sebagian besar Hub masih mengirim `pytorch_model.bin`, sehingga pemuat yang hanya memahami safetensors
tidak bisa membuka sebagian besar model — termasuk banyak model kecil yang banyak dipakai.

```csharp
using var checkpoint = PyTorchCheckpoint.Open("pytorch_model.bin");

foreach (var tensor in checkpoint.Tensors) Console.WriteLine(tensor);
var weights = checkpoint.Read("bert.embeddings.word_embeddings.weight");
```

Berkas `.bin` adalah ZIP berisi state dict ter-pickle berdampingan dengan penyimpanan tensor mentah.
Pickle adalah mesin tumpukan yang opcode `REDUCE`-nya memanggil callable bernama sembarang — dan
justru itulah alasan safetensors ada. **Tidak ada isi berkas yang dieksekusi di sini:** `GLOBAL`
diselesaikan terhadap daftar izin tetap berisi konstruktor tensor torch dan tidak lebih; apa pun di
luar itu melempar exception.

Bila kedua format tersedia, tetap pilih safetensors: ia tidak memerlukan penafsiran sama sekali.

Format bare-pickle pra-1.6 dideteksi dan ditolak dengan pesan yang menjelaskannya, bukan gagal di
kedalaman loop opcode.

## Mengunggah

```csharp
Hub.Upload("my-user/my-model", "adapter_config.json", "adapter_config.json");
```

**Dibatasi 10 MB.** Commit API menerima isi berkas inline sebagai base64, jadi seluruh berkas tersimpan
di memori dua kali dan ikut dalam body permintaan. Apa pun yang akan disimpan Hub di LFS — yakni setiap
berkas bobot sungguhan — memerlukan protokol batch LFS, yang tidak diimplementasikan GraviHub. Batas
itu ditegakkan alih-alih membiarkan unggahan 5 GB gagal perlahan tanpa penjelasan.

## Kesalahan

`HubException` membawa status HTTP dan nama repositori, dan pesannya menyebut kemungkinan penyebabnya:

| Status | Arti |
|---|---|
| 401 / 403 | Privat atau gated. Set `HF_TOKEN`, dan setujui lisensinya di Hub bila gated. |
| 404 | Periksa id, revisi, dan apakah itu dataset alih-alih model. |
| 429 | Kena rate limit. Klien terautentikasi mendapat batas jauh lebih tinggi. |

Percobaan ulang memakai exponential backoff bertopi dan hanya berlaku untuk 429, 408 dan 5xx. Status
4xx tidak akan menjadi benar hanya karena ditanya lagi.

## Lihat juga

[GraviTransformers](GraviTransformers.md) · [GraviDatasets](GraviDatasets.md) · [GraviOptimum](GraviOptimum.md)
