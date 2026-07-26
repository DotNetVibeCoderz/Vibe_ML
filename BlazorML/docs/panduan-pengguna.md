# Panduan pengguna

Dari data mentah sampai endpoint yang bisa dipanggil aplikasi lain.

---

## 1. Masukkan datanya

**Dataset → Impor berkas.** Terima CSV, TSV, dan JSON. Saat impor, aplikasi membaca seluruh
berkas sekali untuk menghitung profil tiap kolom: tipe, jumlah nilai kosong, jumlah nilai unik,
rentang, rata-rata, dan nilai yang paling sering muncul.

Tekan **Lihat kolom** untuk memeriksa hasilnya. Ini langkah yang sering dilewati dan sering
disesali — kolom yang terbaca sebagai teks padahal seharusnya angka akan membuat trainer
mengabaikannya diam-diam.

Kalau tipe kolomnya salah, perbaiki di kanvas dengan modul **Edit Metadata**.

---

## 2. Susun alurnya

**Eksperimen → Eksperimen baru**, atau salin contoh dari **Galeri**.

Di kanvas:

- **Seret** modul dari palette kiri, atau **klik** untuk menaruhnya di tengah.
- **Sambungkan** dengan menarik dari port bawah satu modul ke port atas modul lain.
- **Putus** sambungan dengan mengklik garisnya.
- **Klik modul** untuk membuka pengaturannya di panel kanan.

Warna pena menandakan kategori dan konsisten di mana-mana: biru untuk data masuk, teal untuk
transformasi, ungu untuk aksi LLM, kuning untuk algoritma, merah muda untuk pelatihan, hijau
untuk evaluasi.

Menyambungkan port yang tipenya tidak cocok akan ditolak dengan penjelasan. Sambungan yang
membentuk lingkaran juga ditolak, dengan menyebut node mana yang terlibat.

### Bentuk alur yang paling umum

```
Import Dataset
      ↓
Clean Missing Data
      ↓
Split Data ──────────────┐
      ↓ (train)          ↓ (test)
Train Model ← Algoritma  │
      ↓                  │
Score Model ←────────────┘
      ↓
Evaluate Model
```

Perhatikan **Split Data** punya dua keluaran. Keluaran kiri (baris latih) masuk ke Train Model;
keluaran kanan (baris uji) masuk ke Score Model. Menilai model dengan baris yang sama seperti
yang dipakai melatihnya akan memberi angka yang terlihat hebat dan tidak berarti apa-apa.

---

## 3. Jalankan

Tekan **Jalankan**. Node menyala satu per satu mengikuti urutan dependensinya, dan garis
sambungannya bergerak selagi data mengalir.

Setiap node yang selesai menunjukkan jumlah baris dan kolom keluarannya. Klik node itu untuk
melihat pratinjau isinya di panel kanan — inilah cara tercepat menemukan transformasi yang
tidak melakukan apa yang kamu kira.

Kalau sebuah node gagal, node di hilirnya ditandai **dilewati**, bukan gagal. Panel log di
bawah kanvas menjelaskan apa yang terjadi.

---

## 4. Baca hasilnya

Klik node **Evaluate Model**. Metrik yang ditampilkan menyesuaikan tugasnya:

| Tugas | Yang perlu diperhatikan |
|---|---|
| Klasifikasi biner | **AUC** untuk kualitas peringkat, **F1** kalau kelasnya timpang. Akurasi sendirian menyesatkan pada data timpang. |
| Klasifikasi multikelas | **Akurasi makro** memperlakukan tiap kelas sama rata; akurasi mikro didominasi kelas mayoritas. |
| Regresi | **RMSE** dalam satuan asli targetmu. **R²** menunjukkan seberapa banyak variasi yang tertangkap. |
| Clustering | **Keseimbangan** mendekati 1 berarti klasternya seukuran. |

Kalau angkanya terlalu bagus, curigai kebocoran label: kolom yang secara diam-diam berisi
jawabannya. Modul **Feature Importance** biasanya langsung membongkarnya — satu fitur akan
mendominasi total.

### Kalau hasilnya mengecewakan

Coba **Cross Validate** dulu, bukan langsung ganti algoritma. Cross-validation melaporkan
sebaran antar-fold; kalau sebarannya lebar, masalahmu adalah data yang sedikit atau tidak
konsisten, bukan pilihan algoritma.

Setelah itu baru **Tune Hyperparameters**, atau serahkan ke **AutoML** dengan anggaran waktu.

---

## 5. Terbitkan

Tambahkan modul **Register Model** di ujung alur, lalu jalankan. Modelnya muncul di halaman
**Model** lengkap dengan metriknya.

Dari sana tekan **Terbitkan** untuk membuat endpoint REST. Kunci API ditampilkan sekali —
salin saat itu juga. Detail pemanggilannya ada di [api.md](api.md).

Menerbitkan model dengan nama yang sama membuat versi berikutnya, tidak menimpa. Endpoint yang
sudah jalan tetap melayani versi yang ia tunjuk.

---

## 6. Minta bantuan Profesor Wicak

Tekan ikon chat di kanan atas halaman designer.

Dia bukan chatbot yang hanya bisa bicara. Dia bisa membaca datamu dan menyusun alurnya. Coba:

> "Dataset apa saja yang ada?"
>
> "Periksa keseimbangan kelas kolom Churn."
>
> "Susunkan alur klasifikasi churn lengkap dengan train/test split dan evaluasinya."
>
> "Kenapa AUC-nya cuma 0.6?"

Perubahan yang dia lakukan pada kanvas tersimpan sebagai versi baru, sama seperti perubahanmu
sendiri — jadi selalu bisa ditinjau dan dikembalikan.

Panelnya perlu kunci API salah satu penyedia model. Isi di **Pengaturan → Profesor Wicak**.

---

## 7. Bandingkan hasil dan kembalikan versi

Dari kanvas, tekan ikon riwayat di kanan atas — atau **Riwayat** di daftar eksperimen.

**Tab Run** menampilkan setiap kali eksperimen ini dijalankan, beserta metriknya. Pilih satu
metrik dan grafiknya membandingkan semua run: run terbaik ditandai dengan warna penuh, sisanya
meredup. Baseline-nya selalu di nol, jadi selisih kecil tampak kecil — batang yang dipotong
baseline-nya adalah cara klasik menyesatkan diri sendiri.

Ini cara tercepat menjawab "apakah perubahan tadi benar-benar membantu?". Kalau selisihnya tipis
dan berubah-ubah antar run, jawabannya kemungkinan tidak — jalankan **Cross Validate** sebelum
menyimpulkan.

**Tab Versi** menampilkan setiap penyimpanan graph, termasuk yang dilakukan Profesor Wicak,
dengan catatan singkat apa yang berubah. Tombol **Kembalikan** memuat versi itu ke kanvas.

Mengembalikan tidak menghapus apa pun: hasilnya menjadi versi baru di puncak riwayat. Jadi
mengembalikan pun bisa dibatalkan.

---

## 8. Kalau lupa kata sandi

Di halaman masuk, tekan **Lupa kata sandi?**, isi email, dan tautan pengaturan ulang dikirim.

Kalau workspace ini belum diatur mengirim email, alurnya tetap berjalan — tautannya ditulis ke
**log aplikasi**, dan administrator bisa meneruskannya. Atur SMTP di **Pengaturan → Email**
supaya terkirim otomatis.

Halaman itu menjawab hal yang sama apakah alamatmu terdaftar atau tidak. Itu disengaja: kalau
jawabannya berbeda, siapa pun bisa memakainya untuk mengetahui siapa saja yang punya akun di sini.
