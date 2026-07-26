# Deployment

---

## Menjalankan

```bash
dotnet publish src/BlazorML.Web -c Release -o ./publish
cd publish
dotnet BlazorML.Web.dll
```

Aplikasi membuat skema database saat pertama menyala, lalu mengisi data contoh bila database
masih kosong.

---

## Sebelum dibuka ke orang lain

1. **Ganti kata sandi akun contoh**, atau hapus akunnya. Kata sandi seed
   (`StudioML#2026`) tertulis di README dan di log startup.
2. **Matikan pendaftaran mandiri** bila tidak diperlukan: Pengaturan → Workspace →
   *Izinkan orang mendaftar sendiri*.
3. **Jalankan di belakang HTTPS.** Kunci API dikirim sebagai header.
4. **Pindahkan key ring data protection** ke lokasi yang persisten kalau aplikasi di-deploy
   sebagai container. Tanpa itu, setiap restart membuat pengaturan berkredensial yang tersimpan
   tidak bisa dibaca lagi — aplikasi tetap menyala dan jatuh kembali ke `appsettings.json`,
   tapi kunci API penyedia LLM perlu diisi ulang.

---

## Memilih database

Sunting `Database:Provider` di `appsettings.json`, lalu jalankan ulang.

| Provider | Cocok untuk |
|---|---|
| `Sqlite` | Bawaan. Satu berkas, tanpa server. Cukup untuk satu tim. |
| `SqlServer` | Lingkungan Windows/Azure yang sudah punya SQL Server |
| `PostgreSql` | Pilihan umum untuk deployment Linux |
| `MySql` | Kalau itu yang sudah tersedia |

Connection string tiap provider disimpan terpisah, jadi berpindah tidak menghapus yang lain.

> Provider database adalah satu-satunya pengaturan yang **tidak** bisa diubah dari dalam
> aplikasi. Alasannya jelas: ia tidak bisa disimpan di dalam database yang sedang ia
> konfigurasikan.

### Catatan SQLite

Kolom waktu disimpan sebagai UTC ticks saat provider-nya SQLite. SQLite tidak punya tipe
tanggal asli dan EF Core menolak menerjemahkan `ORDER BY` atas `DateTimeOffset` terhadapnya —
padahal hampir semua daftar di aplikasi ini diurutkan berdasarkan waktu. Konversi ini otomatis
dan tidak terlihat dari kode aplikasi; tiga provider lain memakai tipe tanggal aslinya.

---

## Memilih storage

Pengaturan → Penyimpanan. Berlaku langsung tanpa restart.

| Provider | Yang perlu diisi |
|---|---|
| `FileSystem` | Folder. Path relatif dihitung dari folder aplikasi. |
| `AzureBlob` | Connection string, nama container |
| `S3` | Access key, secret key, bucket, region |
| `MinIO` | Sama seperti S3, ditambah alamat server |

Container atau bucket dibuat otomatis kalau belum ada.

Berpindah provider **tidak memindahkan berkas yang sudah ada**. Dataset dan model lama akan
gagal dibaca sampai berkasnya dipindahkan sendiri ke lokasi baru dengan kunci yang sama.

---

## Runtime script

Modul script C# dan JavaScript ikut terpasang bersama aplikasi dan selalu tersedia.

Python dan R dijalankan sebagai proses anak. Pasang runtime-nya di mesin yang sama, lalu
pastikan bisa dipanggil:

```bash
python --version    # atau python3 di Linux/macOS
Rscript --version
```

Kalau tidak ditemukan, node script-nya gagal dengan pesan yang menyebutkan apa yang kurang —
bukan diam-diam melewatkan langkahnya. Path interpreter bisa diatur di Pengaturan → Scripting.

---

## Sumber daya

Pelatihan berjalan di dalam proses web. Untuk beban yang serius:

- Naikkan **batas run bersamaan** hanya sesuai jumlah core yang tersedia. Pelatihan ML.NET
  bersifat CPU-bound; menaikkannya melebihi core justru memperlambat semuanya.
- Turunkan **batas baris di memori** kalau server RAM-nya terbatas. Ini pengaman supaya satu
  berkas besar tidak menjatuhkan proses.
- **Batas waktu run** menghentikan eksperimen yang macet. Bawaannya 30 menit.

Semuanya di Pengaturan → Training.

---

## Email

Hanya dipakai untuk tautan pengaturan ulang kata sandi. Atur di **Pengaturan → Email**: host,
port, STARTTLS, kredensial, dan alamat pengirim.

Tanpa itu, alur reset tetap berfungsi — tautannya ditulis ke log aplikasi dengan level Warning
dan kalimat yang menjelaskan kenapa. Itu pilihan sadar: aplikasi ini di-deploy sendiri oleh
penggunanya, sering tanpa server email di tangan, dan reset yang sekadar gagal akan mengunci
administrator dari instalasinya sendiri.

Untuk produksi, isi SMTP-nya. Tautan di log berarti siapa pun yang bisa membaca log bisa
mengganti kata sandi akun mana pun.
