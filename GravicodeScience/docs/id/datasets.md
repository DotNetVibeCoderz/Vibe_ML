# Dataset

*[English](../datasets.md)*

Semua isi [`datasets/`](../../datasets) disertakan dalam repositori, sehingga contoh, notebook, dan
tes berjalan tanpa langkah unduh. Total ukurannya sekitar 750 KB.

| Berkas | Baris | Sumber | Dipakai oleh |
|---|---:|---|---|
| `iris.csv` | 150 | Nyata — Fisher (1936) | GraviLearn |
| `titanic.csv` | 891 | Nyata — manifes penumpang Titanic | GraviFrame, GraviLearn |
| `mnist_subset/digits.csv` | 1.797 | Nyata — UCI optical digits | GraviLearn |
| `cora_graph.json` | 2.708 node | Nyata — jaringan sitasi Cora | GraviGraph |
| `imdb_reviews.csv` | 80 | **Sintetis** — ditulis untuk repositori ini | GraviText |
| `bayesian_coin.csv` | 200 | **Sintetis** — dibangkitkan, bias 0,62 | GraviProb |
| `finance_timeseries.csv` | 1.116 | **Sintetis** — geometric random walk | GraviFrame |
| `ner_conll.txt` | 420 kalimat | **Sintetis** — kalimat bahasa Indonesia yang dibangkitkan | GraviText |

Berkas sintetis ditandai dengan sengaja. Tiga di antaranya menggantikan dataset yang terlalu besar
untuk disertakan atau terkendala lisensi; penggantian itu dicatat di bawah pada masing-masing kasus.

---

## iris.csv

Pengukuran bunga iris Fisher: 150 bunga, empat pengukuran, tiga spesies yang terbagi rata.

```
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
```

Dimuat oleh `Datasets.LoadIris()`. Dua dari tiga spesiesnya tidak terpisah secara linear, itulah
sebabnya dataset ini tetap menjadi uji klasifikasi yang berguna, bukan sekadar sepele. Hasil acuan
yang dipatok rangkaian tes: PCA menjelaskan **92,46%** varians pada komponen pertama dan **5,31%**
pada komponen kedua; random forest mencapai sekitar 95% akurasi uji.

**Sumber**: [seaborn-data](https://github.com/mwaskom/seaborn-data). Domain publik.

---

## titanic.csv

Manifes penumpang Titanic: 891 penumpang dengan data keselamatan, kelas, jenis kelamin, usia,
tarif, dan pelabuhan keberangkatan.

Dimuat oleh `Datasets.LoadTitanic()`, yang mengkodekan `sex` dan `embarked` serta mengisi usia dan
tarif yang kosong dengan median kolom.

Nilai sesungguhnya di sini adalah bahwa data ini **berantakan**: `age` kosong untuk 177 penumpang
(19,9%) dan `deck` untuk 688 (77,2%). Itu menjadikannya dataset yang tepat untuk mendemonstrasikan
penanganan nilai kosong, bukan mainan yang sudah dibersihkan. Akurasi acuan yang dipublikasikan
0,78–0,83; rangkaian tes mensyaratkan di atas 0,75.

**Sumber**: [seaborn-data](https://github.com/mwaskom/seaborn-data). Domain publik.

---

## mnist_subset/digits.csv

Dataset optical-digits UCI: 1.797 angka tulisan tangan sebagai citra grayscale 8×8, diratakan
menjadi 64 kolom piksel ditambah label.

```
pixel0,pixel1,...,pixel63,label
```

Dimuat oleh `Datasets.LoadDigits()`. Ini subset yang sama dengan `load_digits` milik scikit-learn,
bukan MNIST penuh berisi 70.000 citra — cukup kecil untuk disertakan dan dijalankan dalam tes,
namun tetap merupakan masalah sepuluh kelas yang nyata. Pipeline
`StandardScaler → PCA(30) → kNN(3)` mencapai akurasi uji **97,96%**, sesuai implementasi acuan.

**Sumber**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits),
melalui scikit-learn. Domain publik (CC0).

---

## cora_graph.json

Jaringan sitasi Cora: 2.708 paper machine learning, 5.429 sitasi, tiap paper dideskripsikan oleh
bag-of-words biner 1.433 kata dan diberi salah satu dari tujuh label topik.

```json
{
  "directed": true,
  "featureDimension": 1433,
  "classes": ["Neural_Networks", "Rule_Learning", ...],
  "nodes": [{"id": 0, "paperId": "31336", "label": 0, "features": [8, 14, 251, ...]}],
  "edges": [[163, 0], [402, 0], ...]
}
```

Fitur node disimpan sebagai **indeks elemen tak-nol**, bukan vektor penuh. Fitur Cora rata-rata
memiliki sekitar delapan belas elemen tak-nol dari 1.433, sehingga bentuk sparse kira-kira delapan
puluh kali lebih kecil dan proporsional lebih cepat dimuat.

Dimuat oleh `Graph.Load()`. Angka acuan yang dipatok rangkaian tes:

- komponen terhubung lemah terbesar: **2.485 node** (91,8%)
- baseline kelas mayoritas: **30,2%**
- akurasi GCN yang dipublikasikan: ~81%; implementasi ini mencapai ~71% pada 60 epoch

Edge disimpan sebagai `mengutip → dikutip`. Karena sitasi hanya menunjuk satu arah,
`ConnectedComponents` melaporkan keterhubungan *lemah* — lihat [GraviGraph.md](GraviGraph.md).

**Sumber**: [LINQS](https://linqs.soe.ucsc.edu/data), dikonversi dari `cora.content` dan
`cora.cites`. Bebas dipakai untuk penelitian.

---

## imdb_reviews.csv

**Sintetis.** 80 ulasan film pendek berlabel positif atau negatif, 50 berbahasa Inggris dan 30
berbahasa Indonesia.

```
review,sentiment,language
"An absolute masterpiece from start to finish; ...",positive,en
"Film ini bagus sekali, ceritanya menarik ...",positive,id
```

Dataset IMDB asli berisi 50.000 ulasan dan berukuran sekitar 80 MB — terlalu besar untuk
disertakan dan terikat lisensinya sendiri. Ulasan ini ditulis khusus untuk repositori ini agar
contoh sentimen memiliki korpus dwibahasa yang cukup kecil untuk dilatih dalam waktu di bawah satu
detik.

**Arti hal ini bagi hasil.** 80 dokumen pendek adalah korpus berskala demo. Classifier terlatihnya
benar-benar memisahkan kedua kelas, tetapi tetangga Word2Vec pada
`samples/GraviText.Console` berisik, dan contohnya menyatakan itu. Embedding sungguhan membutuhkan
jutaan token. Ganti dengan korpus Anda sendiri untuk melihat perbedaannya:

```csharp
var reviews = DataFrame.ReadCsv("korpus-anda.csv");
```

---

## bayesian_coin.csv

**Sintetis.** 200 lemparan koin yang dibangkitkan dengan bias sebenarnya 0,62, terbagi dalam empat
sesi berisi 50 lemparan.

```
flip_id,outcome,session
1,1,session1
```

Berkas ini berisi **125 sisi angka dari 200 lemparan**. Gunanya berkas yang dibangkitkan adalah
parameter sebenarnya diketahui, sehingga `samples/GraviProb.Console` dapat memeriksa posterior
hasil sampling terhadap jawaban konjugat eksak — prior Beta(1,1) dengan likelihood binomial
menghasilkan Beta(126, 76), rata-rata 0,623762. Estimasi MCMC sesuai dalam selisih sekitar 0,002.

---

## finance_timeseries.csv

**Sintetis.** Dua ticker selama tiga tahun hari perdagangan sebagai geometric random walk dengan
komponen musiman mingguan.

```
date,ticker,close,volume
2022-01-03,GRVC,120.44,847231
```

Data pasar nyata tidak dapat didistribusikan ulang. Berkas ini dibangkitkan agar fitur deret waktu
memiliki struktur nyata untuk ditemukan — tren, musiman mingguan, dan pengelompokan volatilitas —
tanpa masalah lisensi. Ini adalah masukan untuk rolling window, persentase perubahan, resampling
kalender, dan perbandingan memory-mapped versus streaming.

---

## Membangkitkan ulang berkas sintetis

Generatornya ada dalam riwayat repositori; parameternya didokumentasikan di atas dan seed-nya
tetap, jadi berkasnya reprodusibel. Kelas `Datasets` juga menyediakan generator yang sama sekali
tidak memerlukan berkas, dan itulah yang dipakai unit test:

```csharp
Datasets.MakeBlobs(samples: 300, features: 2, centers: 3, spread: 1.0, seed: 42);
Datasets.MakeMoons(samples: 200, noise: 0.1, seed: 42);      // tidak terpisah linear
Datasets.MakeRegression(samples: 200, features: 5, noise: 0.5, seed: 42);

Graph.Random(nodes: 100, p: 0.05, seed: 42);
Graph.ScaleFree(nodes: 1000, edgesPerNode: 3, seed: 42);     // derajat power-law
Graph.Communities(communities: 3, sizePerCommunity: 50, seed: 42);
```

## Menambahkan data Anda sendiri

`Datasets.FindDatasetDirectory()` menelusuri ke atas dari assembly yang berjalan untuk mencari
folder `datasets`, sehingga contoh, notebook, dan tes menemukan data tanpa path yang ditulis keras.
Letakkan berkas di `datasets/` lalu baca langsung:

```csharp
var frame = DataFrame.ReadCsv(Datasets.ResolvePath("data-saya.csv"));
```

## ner_conll.txt

420 kalimat bahasa Indonesia beranotasi dalam format kolom CoNLL: satu token dan tag BIO-nya per
baris, dengan baris kosong antar kalimat. Tiga tipe entitas — `PER`, `LOC`, dan `ORG`.

```
Tim	O
dari	O
Bogor	B-LOC
mengunjungi	O
Bukit	B-ORG
Asam	I-ORG
.	O
```

**Sintetis**, dibangkitkan dari templat kalimat atas kumpulan tetap berisi nama, kota, dan
perusahaan. Ia menggantikan korpus seperti CoNLL-2003, yang terbebani lisensi dan tidak dapat
disertakan di sini.

Templatnya itulah yang membuatnya berguna alih-alih sirkular. Tipe entitas tidak dapat dibedakan
dari katanya saja — `Surabaya` dan `Tokopedia` sama-sama token tunggal berhuruf kapital — sehingga
satu-satunya cara membedakannya adalah konteks di sekitarnya, dan itu persis yang harus dipelajari
tagger terlatih. Entitas multi-token muncul pada kelas `PER` maupun `ORG`, sehingga kelanjutan BIO
benar-benar diuji, bukan diandaikan.

Karena dibangkitkan, kinerjanya pada data uji tinggi: `TrainedNer` mencapai sekitar 98% F1 entitas
pada pemisahan uji 25%. Angka itu menyatakan bahwa model mempelajari proses pembangkitnya, **bukan**
bahwa ia akan mencapai 98% pada teks berita. Perlakukan sebagai demonstrasi arsitektur yang
berfungsi, bukan sebagai tolok ukur.

Bacalah dengan `TaggedSentence.LoadConll`.

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
