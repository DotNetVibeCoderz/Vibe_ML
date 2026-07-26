# API scoring

Model yang diterbitkan menjadi endpoint REST berkunci API. Dokumentasi interaktif tersedia di
`/api-docs` setelah aplikasi menyala.

---

## Menerbitkan model

1. Jalankan eksperimen yang berujung pada modul **Register Model**.
2. Buka halaman **Model**, pilih modelnya, tekan **Terbitkan**.
3. Kunci API ditampilkan **satu kali**. Salin sekarang.

Hanya hash SHA-256 dari kunci yang disimpan, jadi bocornya database tidak berarti bocornya
kunci yang bisa dipakai.

---

## Autentikasi

Setiap permintaan memerlukan header:

```
X-Api-Key: bml_xxxxxxxxxxxxxxxxxxxxxxxx
```

Kunci yang tidak cocok, sudah dicabut, kedaluwarsa, atau milik endpoint lain sama-sama
menghasilkan **401** dengan pesan yang identik. Ini disengaja: pemanggil tanpa kunci yang sah
tidak boleh bisa menyimpulkan endpoint mana yang ada.

---

## Endpoint

### `GET /api/v1/endpoints`

Mendaftar endpoint yang sedang aktif.

```json
[
  {
    "name": "Prediksi churn",
    "slug": "prediksi-churn",
    "description": "Menebak pelanggan yang akan berhenti berlangganan",
    "model": "churn-fasttree",
    "task": "BinaryClassification",
    "scoreUrl": "/api/v1/score/prediksi-churn"
  }
]
```

### `POST /api/v1/score/{slug}`

Menjalankan model atas baris yang dikirim.

```bash
curl -X POST https://host/api/v1/score/prediksi-churn \
  -H "X-Api-Key: bml_xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
        "rows": [
          { "LamaBerlanggananBulan": 4, "TagihanBulanan": 245.0,
            "JumlahKomplain": 3, "Paket": "Prabayar", "PakaiInternet": "ya" }
        ]
      }'
```

Balasan:

```json
{
  "model": "churn-fasttree",
  "version": 1,
  "task": "BinaryClassification",
  "predictions": [
    { "PredictedLabel": true, "Probability": 0.87, "Score": 1.92 }
  ]
}
```

Kolom prediksi berbeda per task:

| Task | Kolom yang muncul |
|---|---|
| Klasifikasi biner | `PredictedLabel`, `Probability`, `Score` |
| Klasifikasi multikelas | `PredictedLabel`, `Score` (vektor probabilitas per kelas) |
| Regresi | `Score` |
| Clustering | `PredictedClusterId`, `Score` |
| Rekomendasi | `Score` (rating yang diperkirakan) |

### `GET /api/v1/schema/{slug}`

Menjelaskan kolom yang diharapkan model. Berguna untuk membangun form atau memvalidasi
payload sebelum mengirim.

---

## Kode status

| Kode | Arti |
|---|---|
| `200` | Berhasil |
| `400` | Array `rows` kosong atau tidak ada |
| `401` | Kunci tidak sah, dicabut, kedaluwarsa, atau endpoint tidak aktif |
| `422` | Baris tidak cocok dengan skema yang diharapkan model — pesan galat menjelaskan apa |

---

## Mengelola kunci

Di halaman **Endpoint**:

- **Kunci baru** membuat kunci tambahan; kunci lama tetap berlaku. Ini yang dipakai untuk
  rotasi tanpa memutus pemanggil yang sedang jalan.
- **Hentikan** membuat endpoint menolak semua permintaan tanpa menghapus kunci maupun model.

Kunci bisa diberi tanggal kedaluwarsa saat dibuat.
