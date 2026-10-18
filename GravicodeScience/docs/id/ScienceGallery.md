# Gravicode Science Gallery

*[English](../ScienceGallery.md)* · 45 use case dunia nyata untuk ketujuh library
Gravicode.Science — masing-masing langsung berjalan di mesin Anda saat dibuka, menggambar chart-nya
dengan [Visual.Net](https://github.com/DotNetVibeCoderz/Vibe_Charting/tree/main/Visual.Net), dan
menampilkan berkas sumber persis yang dijalankannya.

![Galeri](../screenshots/gallery-home.png)

```bash
dotnet run --project tools/ScienceGallery
```

Galeri ini dibuat untuk menemukan use case yang mirip kebutuhan Anda, melihatnya bekerja, lalu
mengambil kodenya. Cakupannya pendidikan, sains dan riset, perkantoran, industri, kesehatan,
keuangan, utilitas sehari-hari, dan game, dan setiap library muncul di beberapa di antaranya.

## Menemukan use case

- **Domain** di sidebar mempersempit grid ke satu bidang.
- **Library** menyaring berdasarkan apa yang dipakai sebuah use case. Pilih lebih dari satu dan grid
  hanya menampilkan use case yang memakai *semuanya* — misalnya GraviFrame + SciNet untuk olah data
  lalu statistik.
- **Pencarian** mencocokkan judul, ringkasan, skenario, domain, dan nama library: coba *Kalman*,
  *invoice*, *ODE*, atau *grading*.

![Disaring dengan GraviFrame dan SciNet](../screenshots/gallery-filter.png)

### Strip spektrum

Setiap kartu membawa strip yang terbaca seperti spektroskop. Setiap library Gravi adalah garis
emisi nyata — warna dan panjang gelombang yang sama dengan ikon NuGet-nya — dan garis yang
dibutuhkan sebuah use case menyala, sisanya tinggal tanda samar. SciNet, sang instrumen alih-alih
satu garis, menyalakan seluruh pita di belakangnya. Sekilas pandang, strip itu memberi tahu apa yang
dibutuhkan sebuah use case sebelum Anda membukanya.

## Membuka use case

Membuka sebuah kartu langsung menjalankannya, di luar thread UI, dan menampilkan tiga hal
berdampingan:

- **Hasil** — chart (digambar oleh Matplotlib.Net), angka-angka kunci beserta artinya, dan satu dua
  kalimat yang menafsirkannya. Angkanya dihitung saat run, bukan ditulis ke dalam teks.
- **Kode** — berkas sumber milik showcase itu sendiri, ditanam ke aplikasi saat build. Yang Anda
  baca adalah yang dijalankan: tidak ada salinan terpisah yang dirapikan dan bisa melenceng.
- **Aksi** — *Run again*, *Copy code*, dan *Save chart as PNG*. Baris status menyebut berapa lama
  run itu berlangsung di mesin ini.

![Mendeteksi inti sel, termasuk yang bersentuhan](../screenshots/gallery-detail-2.png)

Tema terang adalah bawaan; sakelar di bagian bawah sidebar menyalakan tema gelap.

![Tema gelap](../screenshots/gallery-detail-dark.png)

## Daftar use case

| Domain | Use case | Library |
|---|---|---|
| Pendidikan | Apakah metode mengajar baru menaikkan nilai ujian? | SciNet, GraviNum |
| | Membentuk kelompok belajar dari siapa duduk dengan siapa | GraviGraph, GraviNum |
| | Seberapa jauh bandul bisa berayun sebelum T = 2π√(L/g) meleset? | SciNet, GraviNum |
| | Di mana bola yang dilempar akan mendarat? | SciNet, GraviNum |
| | Esai mana yang saling menyalin? | GraviText, GraviNum |
| | Siswa mana yang perlu konselor semester ini? | GraviLearn, GraviFrame, GraviNum |
| Sains & riset | Kekacauan dalam model cuaca | SciNet, GraviNum |
| | Seberapa jauh bintang yang berdenyut ini? | SciNet, GraviNum |
| | Seberapa cepat enzim ini, dan seberapa kuat ia mengikat? | SciNet, GraviFrame, GraviNum |
| | Berapa banyak kalsit dalam sampel batuan ini? | SciNet, GraviNum |
| | Kapan wabah memuncak, dan seberapa besar? | SciNet, GraviNum |
| | Makalah mana yang benar-benar menjadi pijakan bidangnya? (Cora) | GraviGraph, SciNet, GraviNum |
| Perkantoran | Mengubah pesanan setahun menjadi laporan penjualan bulanan | GraviFrame, GraviNum |
| | Apakah model sentimen terlatih sepadan dengan pelabelannya? (ulasan IMDB) | GraviText, GraviFrame, GraviLearn |
| | Mengambil nomor invoice, vendor, tanggal, dan jumlah dari email | GraviText, GraviNum |
| | Memilah kotak masuk bersama ke topik yang tak pernah dilabeli | GraviText, GraviLearn, GraviNum |
| | Menempatkan sepuluh rapat di sepuluh ruangan tanpa ada yang berdesakan | SciNet |
| | Menyusun jadwal minggu depan yang menutup semua shift dan menghormati cuti | SciNet, GraviNum |
| Industri | Menangkap mesin yang melenceng sebelum menghasilkan barang reject | SciNet, GraviFrame, GraviNum |
| | Lintasan bearing mana yang retak di motor ini? | SciNet, GraviNum |
| | Berjalan lebih sedikit di setiap daftar pengambilan gudang | SciNet, GraviNum |
| | Pemasok mana yang bisa menghentikan ketiga pabrik? | GraviGraph, GraviNum |
| | Berapa daya yang akan ditarik gedung dalam 48 jam ke depan? | GraviProb, GraviFrame, GraviNum |
| | Seberapa hangat pendingin sebelum reaktor lepas kendali? | SciNet, GraviNum |
| Kesehatan | Detak jantung dan variabilitasnya dari EKG yang bising | SciNet, GraviNum |
| | Dua atau tiga kali sehari: jadwal mana yang menjaga kadar obat? | SciNet, GraviNum |
| | Seberapa yakin kita bahwa pengobatan baru lebih baik? | GraviProb, SciNet, GraviNum |
| | Apakah pasien bebas kambuh lebih lama dengan terapi baru? | SciNet, GraviNum |
| | Menghitung sel di gambar mikroskop, termasuk yang bersentuhan | SciNet, GraviNum |
| Keuangan | Campuran terbaik lima aset untuk setiap tingkat risiko | SciNet, GraviNum |
| | Berapa nilai opsi saham ini hari ini? | SciNet, GraviNum |
| | Berapa yang bisa hilang dari posisi ini di hari buruk? | SciNet, GraviFrame |
| | Menandai pembayaran kartu yang tidak mirip pemiliknya | GraviLearn, GraviNum |
| | Berapa biaya KPR ini sebenarnya per tahun? | SciNet, GraviFrame |
| Utilitas | Membersihkan dengung dan desis dari rekaman | SciNet, GraviNum |
| | Menyelamatkan foto berbintik dengan piksel mati | SciNet, GraviNum |
| | Kenapa aplikasi lari saya bilang saya berlari lebih jauh? | GraviProb, SciNet, GraviNum |
| | Toko mana yang terdekat, dan di mana toko berikutnya sebaiknya dibuka? | SciNet, GraviNum |
| | Bertukar data dengan rekan pengguna MATLAB tanpa kehilangan apa pun | SciNet, GraviFrame, GraviNum |
| Game | Membuat pulau yang meyakinkan untuk peta game | SciNet, GraviNum |
| | Membangun labirin, memecahkannya, dan membiarkan setiap monster menemukan pemain | GraviGraph, GraviNum |
| | Apakah bola memantul di game kehilangan energi sebesar yang semestinya? | SciNet, GraviNum |
| | Apakah peluang menang dari matchmaker jujur? | GraviLearn, GraviFrame, GraviNum |
| | Berapa peluang pasti dari lemparan dadu ini? | SciNet, GraviNum |
| | Berapa lama pemain bertahan, dan berapa nilai seorang pemain baru? | GraviProb, SciNet, GraviNum |

Judul di aplikasi berbahasa Inggris; tabel di atas menerjemahkannya. Beberapa use case terang-terangan
menyebut di mana sebuah metode kurang, karena memang itu yang ditunjukkan hasilnya: pada 80 ulasan
IMDB classifier sentimen terlatih kalah dari daftar kata, kedua ticker di `finance_timeseries.csv`
ternyata mendekati normal (sehingga teks Value-at-Risk menyebutnya alih-alih mengklaim ekor tebal),
dan ODE reaktor tidak kaku, sehingga BDF bukan solver yang lebih cepat di sana.

## Cara dibangunnya

```
tools/ScienceGallery/
  Models/        Showcase (kontrak yang diimplementasikan setiap use case), library, domain
  Showcases/     satu berkas per use case, per domain - inilah yang ditampilkan panel kode
  Services/      registri, gaya chart, pencarian dataset, tur screenshot
  Controls/      SpectrumStrip
  ViewModels/    penyaringan, menjalankan, hasil
  Views/         jendela
  Themes/        token (terang dan gelap) dan gaya kontrol
  Assets/Fonts/  Bricolage Grotesque dan JetBrains Mono (SIL Open Font Licence)
```

### Menambah use case

Buat berkas di `Showcases/<Domain>/`, turunkan dari `Showcase`, lalu isi judul, domain, library,
ringkasan, skenario, dan `Run`. Registri menemukannya lewat reflection dan build menanam sumbernya,
jadi tidak ada yang perlu didaftarkan:

```csharp
public sealed class MyUseCase : Showcase
{
    public override string Title => "Pertanyaan konkret dalam kata-kata biasa";
    public override Domain Domain => Domain.Research;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviNum];
    public override string Summary => "Satu kalimat untuk kartu.";
    public override string Scenario => "Siapa yang melakukan ini, kenapa, dan apa arti hasilnya.";

    public override void Run(ShowcaseOutput output)
    {
        var figure = GalleryCharts.Figure();
        var ax = figure.AddSubplot();
        // ... hitung, gambar ...
        output.Chart(figure);
        output.Metric("Angka kunci", "42", "artinya");
        output.Line("Satu kalimat yang menafsirkan hasilnya.");
    }
}
```

Berkas itu ditampilkan kepada pengguna sebagai contoh untuk disalin, jadi tulis sebagai kode
pengajaran: mandiri, deterministik (ber-seed), diberi komentar di bagian yang alasannya tidak
jelas, dan cepat (jauh di bawah 2 detik).

### Visual.Net berasal dari NuGet

Chart memakai paket `Gravicode.Matplotlib.Net` dan `Gravicode.Seaborn.Net` (Visual.Net 1.1.0, dibangun
dari [Vibe_Charting](https://github.com/DotNetVibeCoderz/Vibe_Charting)). Paket itu bergantung pada
`Gravicode.Science.GraviNum`, `GraviFrame` dan `GraviLearn`; gallery mereferensikan library tersebut
sebagai proyek dengan package id yang sama, sehingga NuGet memakai build lokal dan hanya ada satu
salinan untuk tiap library. Visual.Net merender dengan SkiaSharp 4, yang datang secara transitif dan
sengaja tidak di-pin di `Directory.Packages.props` — pinning transitif akan memindahkan sample
ScottPlot dari SkiaSharp 3.

### Mode baris perintah

| Perintah | Fungsinya |
|---|---|
| `ScienceGallery --run-all` | Menjalankan setiap use case tanpa jendela dan merender chart-nya; keluar dengan kode bukan nol bila ada yang gagal. CI menjalankan ini di Linux dan Windows. |
| `ScienceGallery --capture <folder>` | Menggerakkan jendela asli melewati beranda, satu domain, satu filter, empat use case, dan tema gelap, lalu menyimpan masing-masing sebagai PNG 1440×900. Screenshot di halaman ini berasal darinya. |

---

Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
