# HF Gallery

*[English](../hf-gallery.md)*

`samples/HFGallery` adalah aplikasi desktop yang menjalankan sepuluh use case HF.Net terhadap model
sungguhan, dan menampilkan hasilnya bersama kode yang menghasilkannya. Tidak ada yang dipalsukan:
setiap panel yang terlihat adalah keluaran checkpoint yang diunduh dari Hub saat itu juga.

```bash
dotnet run --project samples/HFGallery
```

| Flag | Fungsinya |
|---|---|
| *(tanpa flag)* | Membuka jendela pada use case pertama. |
| `--list` | Mencetak daftar use case lalu keluar. |
| `--run <case>` | Menjalankan satu use case di terminal dan mencetak hasilnya. |
| `--open <case>` | Membuka jendela dan langsung menjalankan use case itu. |
| `--light` | Memakai tema terang. |

`<case>` cukup awalan judulnya, jadi `--run Named` sudah memadai.

---

## Sepuluh use case

| Use case | Library | Model | Yang ditunjukkan |
|---|---|---|---|
| What is in this picture | GraviTransformers | `google/vit-base-patch16-224` | Vision Transformer di atas patch gambar |
| Sentiment | GraviTransformers | `distilbert-base-uncased-finetuned-sst-2-english` | Head hasil fine-tune beserta nama labelnya sendiri |
| Fill in the blank | GraviTransformers | `bert-base-uncased` | Head masked language model |
| Named entities | GraviTransformers | `dslim/bert-base-NER` | Klasifikasi token, sebagai rentang teks asli |
| Question answering | GraviTransformers | `distilbert-base-cased-distilled-squad` | Ekstraksi rentang atas pasangan kalimat |
| Semantic search | GraviTransformers | `bert-base-uncased` | Sekali embed korpus, lalu diperingkat per kueri |
| Embedding map | GraviTransformers | `bert-base-uncased` | 768 dimensi diproyeksikan ke dua dimensi |
| Tokenizer | GraviTokenizers | `bert-base-uncased` | Karakter mana menjadi potongan yang mana |
| Inside a checkpoint | GraviHub | repo apa pun yang punya safetensors | Ke mana sebenarnya 420 MB sebuah model pergi |
| Diffusion schedules | GraviDiffusers | *tidak ada* | Sisa sinyal pada setiap timestep pelatihan |

---

## What is in this picture

Vision Transformer adalah blok encoder yang sama seperti BERT di atas embedding yang berbeda:
gambar menjadi kisi 14x14 persegi berukuran 16 piksel, tiap persegi menjadi satu vektor, dan sebuah
vektor `[CLS]` terlatih ditaruh di depan. Gambarnya diambil dari Hub, jadi tidak ada berkas yang
perlu ikut disimpan di repositori.

![What is in this picture](../screenshots/hfgallery-image.png)

## Named entities

Rentang digambar pada posisinya di teks asli, diambil dari offset karakter milik tokenizer. Di
situlah bedanya daftar string hasil ekstraksi dengan jawaban yang bisa diperiksa: kapitalisasi,
tanda baca, dan posisinya tetap utuh, dan kesalahan satu indeks langsung terlihat alih-alih
tampak masuk akal.

![Named entities](../screenshots/hfgallery-entities.png)

## Question answering

Pertanyaan dan paragraf masuk sebagai pasangan, dan itu pula yang menguji segment embedding. Hanya
posisi di dalam paragraf yang memenuhi syarat dan akhir rentang tidak pernah mendahului awalnya,
sehingga jawabannya selalu rentang nyata dari teks, bukan rakitan dua argmax yang tak berhubungan.

![Question answering](../screenshots/hfgallery-qa.png)

## Embedding map

Delapan kalimat dari tiga topik, di-embed dengan `bert-base-uncased` lalu diproyeksikan ke dua
komponen utama pertamanya. Kelompoknya memisah tanpa diberi tahu — pemisahan itulah klaim di balik
semantic search, ditunjukkan alih-alih sekadar dinyatakan.

![Embedding map](../screenshots/hfgallery-embedding-map.png)

## Inside a checkpoint

`bert-base-uncased` berukuran 420 MB. Treemap-nya menunjukkan ke mana byte itu pergi: feed-forward
216 MB, attention 108 MB, embeddings 91 MB. Membacanya hanya memerlukan parse header, bukan
memuat 420 MB — berkasnya di-memory-map dan hanya daftarnya yang disentuh.

![Inside a checkpoint](../screenshots/hfgallery-checkpoint.png)

## Diffusion schedules

Tiga noise schedule sepanjang 1.000 timestep pelatihan. Ketiganya berakhir mendekati nol dan sama
sekali berbeda dalam cara mencapainya — itulah sebabnya checkpoint Stable Diffusion yang dijalankan
dengan beta `Linear` menghasilkan gambar yang terlihat seperti prompt buruk, bukan seperti bug.

![Diffusion schedules](../screenshots/hfgallery-schedules.png)

## Tokenizer, dalam tema terang

Palet terang punya langkah warnanya sendiri, bukan pembalikan dari yang gelap, dan `--light` ada
supaya palet itu benar-benar bisa dilihat.

![Tokenizer](../screenshots/hfgallery-tokenizer-light.png)

---

## Bagaimana grafiknya dibuat

Tidak ada library charting. `Controls/Charts.cs` menggambar empat bentuk — bar, scatter, garis,
treemap — langsung ke `DrawingContext`, dan `Controls/SpanView.cs` menyusun teks bertanda dari
inline teks supaya pembungkusan baris dan seleksi datang dari text stack.

Palet di `Themes/Tokens.axaml` dicari di ruang OKLCH dan diperiksa dengan validator, bukan dipilih
dengan mata. Justru itu yang memunculkan temuan yang layak dicatat: **ramp delapan pita milik
HFAppGen, yang dipakai pada offset rail-nya, gagal sebagai palet kategorikal.** Pasangan yang
bersebelahan hanya terpaut ΔE 6,8 pada penglihatan normal, jauh di bawah batas 15 — tepat untuk rail
yang kedekatannya bermakna, keliru begitu warna yang sama harus mengatakan *ini benda yang mana*.
Palet kategorikal yang dipakai galeri terpaut ΔE 9,4 pada deuteranopia dan ΔE 15,8 pada penglihatan
normal, untuk semua pasangan, pada kedua latar.

Dua akibatnya terlihat langsung di aplikasi:

- **Warna diberikan dalam urutan tetap dan tidak pernah didaur ulang.** Kategori ketujuh memakai
  tinta redup dan terbaca sebagai "lainnya", bukan mengulang warna kategori pertama, karena warna
  yang berulang adalah kebohongan tentang identitas.
- **Warna mengikuti entitasnya, bukan peringkatnya.** Pada grafik sentiment, POSITIVE tetap pada
  warnanya baik ia urutan pertama maupun kedua, jadi menjalankan ulang pada teks lain tidak pernah
  mengecat ulang bar-nya.

Warna sekuensial dipakai ketika yang dikodekan memang besaran — kandidat fill-mask, sel treemap —
dan tema gelap serta terang punya ramp terpisah, karena ramp yang cukup lebar untuk terbaca sebagai
ramp selalu menaruh salah satu ujungnya terlalu dekat dengan latar tempat ia digambar.

---

## Menambahkan use case

Turunkan dari `GalleryCase`, isi metadatanya, dan kembalikan sebuah `CaseResult`:

```csharp
internal sealed class MyCase : GalleryCase
{
    public override string Title => "My case";
    public override string Blurb => "Satu baris yang menjelaskan apa yang ditunjukkan.";
    public override string Library => "GraviTransformers";
    public override string Model => "bert-base-uncased";
    public override string? DefaultInput => "Sesuatu untuk titik mulai.";

    public override string Code => """
        // C# yang tampil di panel kode.
        """;

    public override async Task<CaseResult> RunAsync(
        string input, string second, IProgress<string> progress, CancellationToken token)
    {
        var model = await ModelCache.ModelAsync(Model, progress, token);

        return new CaseResult
        {
            Summary = "Satu kalimat yang menjelaskan hasilnya.",
            Bars = [new Datum("pertama", 0.9), new Datum("kedua", 0.1)],
        };
    }
}
```

Lalu tambahkan ke `Catalog.All`. Field `CaseResult` mana pun yang diisi akan digambar; sisanya
tetap tersembunyi. Model diambil lewat `ModelCache` supaya dua use case pada checkpoint yang sama
berbagi satu proses pemuatan.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.*
