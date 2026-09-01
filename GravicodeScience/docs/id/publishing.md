# Penerbitan dan CI

Enam paket dikirim dari `src/`: `Gravicode.Science.GraviNum`, `.GraviFrame`, `.GraviLearn`,
`.GraviText`, `.GraviGraph`, dan `.GraviProb`. Selebihnya di repositori ini — sample, benchmark,
notebook, tes, ScienceAppGen — ditandai `IsPackable=false` dan tidak pernah ikut terbit.

## Di mana metadatanya berada

Semuanya di `Directory.Build.props`, dalam satu grup properti yang berlaku untuk setiap proyek di
bawah `src/`. Meletakkannya di sana, bukan di enam berkas `.csproj`, adalah yang menjaga
paket-paketnya tidak saling melenceng: satu versi, satu lisensi, satu URL repositori.

```xml
<PackageProjectUrl>https://github.com/DotNetVibeCoderz/Vibe_ML/tree/main/GravicodeScience</PackageProjectUrl>
<RepositoryUrl>https://github.com/DotNetVibeCoderz/Vibe_ML/tree/main/GravicodeScience</RepositoryUrl>
<RepositoryType>git</RepositoryType>
```

Keduanya mengarah ke subdirektorinya, bukan ke akar repositori, karena di situlah proyeknya
sebenarnya berada — GravicodeScience adalah satu proyek di dalam monorepo Vibe_ML, dan pembaca yang
mendarat di akar harus mencari-cari. Saran umumnya adalah mengisi `RepositoryUrl` dengan URL clone;
di sini tautan dalam itu lebih berguna bagi orang yang mengikutinya dari nuget.org.

Perhatikan bahwa `PublishRepositoryUrl` sengaja **tidak** disetel. Properti itu akan membiarkan
SourceLink menimpa `RepositoryUrl` dengan git origin, sehingga tautan dalam tadi tergantikan oleh
akar repositori.

Dua hal ikut serta di setiap paket:

- **Dokumentasi XML.** `GenerateDocumentationFile` aktif untuk proyek yang dipaketkan, sehingga
  komentar di repositori ini sampai ke pengguna lewat IntelliSense, bukan hanya lewat sumbernya.
  `CS1591` (komentar hilang pada anggota publik) dibungkam, bukan diperbaiki: celah pada prosa
  adalah celah pada dokumentasi, bukan build yang rusak, dan menjadikannya kesalahan berarti
  menggantungkan proses paket pada urusan tulis-menulis.
- **Paket simbol `.snupkg`**, sehingga debugger bisa masuk ke dalam kode pustakanya.

## Merilis

Versinya berasal dari tag, sehingga keduanya tidak mungkin berbeda:

```bash
git tag gravicode-science-v1.0.0
git push origin gravicode-science-v1.0.0
```

Tag-nya diberi awalan karena `v1.0.0` polos akan ambigu di sebuah monorepo. `release.yml`
menangkapnya, memeriksa bahwa ia terbaca sebagai versi semantik, mem-build, **menjalankan seluruh
rangkaian tes**, memaketkan pada versi itu, lalu mendorongnya ke nuget.org.

Menerbitkan ke nuget.org tidak dapat dibatalkan — sebuah versi bisa dinyahdaftarkan tetapi tidak
pernah bisa diganti. Itulah sebabnya hanya alur kerja rilis yang mendorong, sebabnya ia dipicu oleh
tag alih-alih oleh branch, dan sebabnya rangkaian tes menjadi gerbangnya. Entri `workflow_dispatch`
menyetel `dry_run` ke true secara bawaan dengan alasan yang sama.

`GraviNum` didorong lebih dulu karena semua paket lain bergantung padanya; dependensi yang belum
terindeks membuat paket-paket turunannya sesaat tidak bisa di-restore. `--skip-duplicate` berarti
menjalankan ulang setelah kegagalan sebagian akan menyelesaikan sisanya, bukan gagal pada yang
sudah terkirim.

### Satu secret

`NUGET_API_KEY`, di bawah **Settings → Secrets and variables → Actions**, dari
[nuget.org/account/apikeys](https://www.nuget.org/account/apikeys), dibatasi ke
`Gravicode.Science.*`. Tidak ada lagi yang diperlukan; GitHub Release dibuat dengan `GITHUB_TOKEN`
bawaan.

### Menerbitkan secara manual

```bash
dotnet pack Gravicode.Science.sln -c Release -p:Version=1.0.0

dotnet nuget push "artifacts/packages/Gravicode.Science.GraviNum.1.0.0.nupkg" \
  --api-key "$NUGET_API_KEY" --source https://api.nuget.org/v3/index.json

# lalu lima sisanya, dalam urutan apa pun
```

Paketnya jatuh ke `artifacts/packages/`, yang diabaikan git.

## Integrasi berkelanjutan

`ci.yml` berjalan pada setiap push ke `main` dan setiap pull request yang menyentuh
`GravicodeScience/**`.

| Job | Berjalan di | Yang dikerjakan |
|---|---|---|
| `build` | ubuntu-latest, windows-latest | Restore, build Release, jalankan 1.049 tes, unggah `.trx` |
| `notebooks` | ubuntu-latest | Mengompilasi setiap sel kode dari keenam notebook |
| `pack` | ubuntu-latest | Memaketkan keenam pustaka dan mengunggahnya sebagai artifact |

Matriksnya `fail-fast: false` — satu platform yang gagal tetap layak dilihat pada platform lainnya.

**Job notebook itu layak ada.** Notebook adalah JSON, jadi ia lolos validasi terlepas dari apakah
C# di dalamnya bisa dikompilasi. Keenam notebook pernah memanggil `plot.GetImageHtml(...)`, yang
ditandai ScottPlot 5.1.59 sebagai `[Obsolete(error: true)]`; setiap sel grafik akan gagal saat
dijalankan, dan baik build maupun rangkaian tes tidak bisa melihatnya.
`tools/verify/notebook_cells.py` menggabungkan sel kode tiap notebook menjadi satu berkas dan
mengompilasinya terhadap pustakanya — yang justru *lebih ketat* daripada notebook-nya, karena .NET
Interactive mengizinkan sebuah variabel dideklarasikan ulang antar sel dan pemeriksaan ini tidak.

Peringatan tidak diperlakukan sebagai kesalahan. Keenam pustaka yang dikirim build bersih;
ScienceAppGen punya tiga peringatan yang sudah diketahui, dan gagal karenanya berarti
menggantungkan paketnya pada IDE.

### Berkas alur kerja berada di akar repositori

GitHub hanya membaca alur kerja dari `.github/workflows/` di akar repositori. Berkas-berkas ini
disimpan di sini, berdampingan dengan proyek yang dijelaskannya, jadi setelah meng-clone Vibe_ML
keduanya perlu ditempatkan di akar:

```bash
mkdir -p .github/workflows
cp GravicodeScience/.github/workflows/*.yml .github/workflows/
```

Keduanya memang sudah ditulis untuk tata letak itu: setiap langkah menyetel
`working-directory: GravicodeScience` dan pemicunya disaring pada `GravicodeScience/**`, sehingga
perubahan di bagian lain monorepo diabaikan.

## Sebelum penerbitan pertama

- `testkey.txt` memuat kunci Azure OpenAI dan kunci Tavily yang aktif. Berkas itu ada di
  `.gitignore` — periksa bahwa ia tidak ada di riwayat repositori sebelum apa pun dipublikasikan.
  ScienceAppGen membaca nilai-nilai itu dari `SCIENCEAPPGEN_APIKEY`, `SCIENCEAPPGEN_ENDPOINT`, dan
  `TAVILY_API_KEY`, dan `app.config` dikirim dengan kedua ruas kunci kosong, jadi tidak ada yang
  memerlukan berkas itu untuk di-commit.
- Pesan awalan `Gravicode.Science.*` di nuget.org bila akunnya memilikinya, supaya tidak ada orang
  lain yang bisa menerbitkan dengan awalan itu.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
