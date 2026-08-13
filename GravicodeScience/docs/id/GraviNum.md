# GraviNum

*[English](../GraviNum.md)* · NumPy untuk .NET — fondasi numerik yang menopang semua library lain.

## NdArray

Sebuah `NdArray` adalah buffer `double[]` bersama ditambah bentuk (shape), stride, dan offset.
Reshape, transpose, dan slice menghasilkan **view**, sehingga biayanya hanya alokasi header kecil
dan tidak pernah memindahkan data. Hanya `Copy()` dan `AsContiguous()` yang mengalokasikan.

```csharp
var a = NdArray.Arange(12).Reshape(3, 4);

a[1, 2];              // 6
a.T[2, 1];            // 6 — elemen yang sama, lewat view
a.Shape;              // [3, 4]
a.IsContiguous;       // true
```

Karena view berbagi buffer, menulis melalui salah satunya terlihat pada yang lain:

```csharp
a.T[0, 1] = 99.0;
Console.WriteLine(a[1, 0]);   // 99
```

### Pembuatan

```csharp
NdArray.Zeros(3, 4);                    NdArray.Ones(2, 2);
NdArray.Full(7.5, 10);                  NdArray.Eye(4);
NdArray.Arange(0, 10, 2);               NdArray.Linspace(0, 1, 11);
NdArray.FromValues([1.0, 2.0, 3.0]);    NdArray.FromArray(new double[,] { { 1, 2 }, { 3, 4 } });
```

### Slicing

`Slice.At` memilih satu indeks dan **menghapus** sumbunya; `Slice.Range` mempertahankannya.

```csharp
a.Slice(Slice.All, Slice.Range(1, 3));   // view 3x2
a.Slice(Slice.At(1), Slice.All);         // rank 1 — sumbu 0 hilang
a.Slice(Slice.Range(0, 10, 2));          // setiap elemen kedua
a.Slice(Slice.Reversed);                 // view terbalik
a.Row(1);  a.Column(2);                  // baris dan kolom tanpa salinan
```

### Reshape dan penggabungan

```csharp
a.Reshape(2, -1);        // satu dimensi disimpulkan
a.Ravel();               // view datar (atau salinan bila strided)
a.Transpose(1, 0);       // permutasi sumbu
a.ExpandDims(0);         // sisipkan sumbu berukuran 1
a.Squeeze();             // buang semua sumbu berukuran 1
a.BroadcastTo(5, 3, 4);  // rentangkan tanpa menyalin

NdArray.Concatenate([x, y], axis: 0);
NdArray.Stack([x, y]);           // sumbu baru di depan
NdArray.VStack(x, y);  NdArray.HStack(x, y);
```

## Broadcasting

Bentuk disejajarkan dari kanan; dimensi bernilai 1 direntangkan agar cocok.

```csharp
NdArray.Ones(3, 4) + NdArray.Arange(4);    // (3,4) + (4,) -> (3,4)
NdArray.Ones(3, 1) * NdArray.Ones(1, 5);   // (3,1) * (1,5) -> (3,5)
```

Bentuk yang tidak kompatibel melempar `InvalidOperationException` yang menyebutkan kedua bentuk,
alih-alih menghasilkan nilai salah secara diam-diam.

## Operasi elemen demi elemen

`UFunc` memilih salah satu dari tiga jalur secara otomatis: loop SIMD `Vector<double>` bila kedua
operand kontigu dan berbentuk sama, versi ber-thread dari loop itu di atas ~24.000 elemen, dan
penelusuran broadcast strided untuk sisanya.

```csharp
a + b;   a - b;   a * b;   a / b;      // elemen demi elemen, dengan broadcasting
a * 2.0; 1.0 / a;                      // skalar
a.Sqrt(); a.Exp(); a.Log(); a.Abs(); a.Pow(3);
a.Sigmoid(); a.Relu(); a.Tanh();
a.Clip(0, 1);
a.Map(x => x * x + 1);                 // fungsi bebas

UFunc.AllClose(x, y, tolerance: 1e-9);
UFunc.Minimum(x, y);  UFunc.Maximum(x, y);
```

> `*` adalah perkalian **Hadamard** (elemen demi elemen), sesuai NumPy. Gunakan `LinAlg.Dot` atau
> `a.Dot(b)` untuk perkalian matriks.

## Aljabar linear

```csharp
LinAlg.Dot(a, b);           // matriks·matriks, matriks·vektor, vektor·vektor
LinAlg.Inner(x, y);         // hasil kali titik skalar
LinAlg.Outer(x, y);

LinAlg.Determinant(m);      LinAlg.Trace(m);      LinAlg.Diagonal(m);
LinAlg.Inverse(m);          LinAlg.Solve(a, b);
LinAlg.PseudoInverse(m);    LinAlg.LeastSquares(a, b);
LinAlg.MatrixRank(m);       LinAlg.ConditionNumber(m);
LinAlg.Norm(v);             LinAlg.Norm(v, p: 1);
LinAlg.MatrixPower(m, 5);   LinAlg.Kron(a, b);
```

`Solve` dan `Inverse` memakai LU dengan pivoting parsial. `LeastSquares` lewat pseudo-inverse
berbasis SVD, yang tetap berperilaku baik saat fitur berkolinear — kasus di mana persamaan normal
diam-diam mengembalikan hasil kacau.

### Dekomposisi

```csharp
var lu    = Decomposition.Lu(m);              // P A = L U
var qr    = Decomposition.Qr(m);              // A = Q R, Householder
var chol  = Decomposition.Cholesky(spd);      // A = L L'
var svd   = Decomposition.Svd(m);             // A = U diag(S) V'
var eigen = Decomposition.SymmetricEigen(s);  // nilai eigen menurun
var (re, im) = Decomposition.Eigenvalues(m);  // umum, bisa kompleks

lu.Solve(b);  qr.Solve(b);  svd.Reconstruct();
```

`Cholesky` melempar exception bila masukannya bukan definit positif alih-alih mengembalikan NaN,
sehingga sekaligus berfungsi sebagai pemeriksa definit positif.

**Mintalah hanya faktor yang Anda perlukan.** SVD penuh menghabiskan sebagian besar waktunya
mengakumulasi `U`, satu baris per baris masukan. Bila bukan itu yang Anda cari, katakan saja:

```csharp
var (values, v) = Decomposition.SvdRightVectors(m);  // melewati U — yang dibutuhkan PCA
var s           = Decomposition.SingularValues(m);   // melewati keduanya — rank, condition number
```

Nilai singularnya identik sampai bit terakhir, apa pun pilihannya; rotasi yang menghasilkannya
tetap berjalan. Pada matriks 20.000×20, melewati `U` berarti melewati sebagian besar waktu proses.

**Tentang algoritmanya.** `Svd` mereduksi ke bentuk bidiagonal dengan refleksi Householder lalu
menjalankan iterasi QR bergeser implisit (Golub–Kahan–Reinsch); `SymmetricEigen` melakukan
tridiagonalisasi lalu iterasi QL implisit dengan pergeseran Wilkinson. `SvdJacobi` dan
`SymmetricEigenJacobi` menghitung faktorisasi yang sama dengan merotasi pasangan kolom sampai tidak
ada yang berubah — jauh lebih lambat, tetapi lewat jalan yang sama sekali berbeda menuju jawaban
yang sama, dan justru itulah yang membuatnya berguna sebagai rujukan pembanding dalam pengujian.

## Mempercepat

Tiga jalur opsional berada di balik `LinAlg.Dot`, dipilih otomatis. Semuanya diperiksa terhadap
kernel terkelola, yang tetap ada sebagai rujukan.

### BLAS native, bila mesin punya

```csharp
using Gravicode.Science.GraviNum.Compute;

NativeBlas.Describe();      // apa yang ditemukan, atau "none found"
NativeBlas.IsAvailable;
NativeBlas.Enabled = false; // paksa jalur terkelola, untuk pembanding
```

**Tidak ada yang dibundel.** BLAS tersetel adalah binari besar dan spesifik platform; mengirim satu
per runtime identifier akan membebani setiap pengguna dengan megabyte yang tidak mereka minta. Jadi
ini mencari yang sudah ada di mesin — OpenBLAS, MKL, Accelerate — dan diam-diam tetap memakai kernel
terkelola bila tidak menemukannya. Arahkan ke build tertentu lewat variabel lingkungan
`GRAVICODE_BLAS`.

Diukur dengan OpenBLAS yang terbundel di dalam numpy:

| Ukuran | Terkelola | Native | |
|---:|---:|---:|---|
| 256 | 1,28 ms | 0,36 ms | **3,6×** |
| 512 | 9,05 ms | 1,90 ms | **4,8×** |
| 1024 | 74,2 ms | 18,7 ms | **4,0×** |

Kedua lebar integer ditangani. OpenBLAS standar mengekspor `cblas_dgemm` dengan indeks 32-bit;
build ILP64 mengekspor `cblas_dgemm64_` dengan 64-bit. Keduanya tidak bisa dipertukarkan — memanggil
yang satu lewat signature yang lain membaca byte yang salah sebagai dimensi — sehingga lebarnya
dideteksi dari simbol mana yang berhasil di-resolve. numpy dan scipy juga mengganti nama setiap
simbol dengan prefiks `scipy_` agar tidak bentrok; itu ikut ditangani, dan biasanya itulah satu-satunya
BLAS di mesin data science.

### Faktorisasi, bila ada LAPACK

`NativeLapack` mengikat `dgesv`, `dgeqrf`/`dorgqr`, `dgesvd`, `dsyev`, `dgetrf`, dan `dpotrf` lewat
probe yang sama, sehingga `LinAlg.Solve`, `LinAlg.Inverse`, `Decomposition.Lu`, `Cholesky`, `Qr`,
`Svd`, `SingularValues`, dan `SymmetricEigen` semuanya memakainya.

| | Terkelola | Native | |
|---|---:|---:|---|
| solve, 512×512, banyak RHS | 887 ms | 17,4 ms | **51×** |
| QR, 256×256 | 151 ms | 10,7 ms | **14×** |
| LU, 256×256 | 77,9 ms | 4,5 ms | **17×** |
| Cholesky, 256×256 | 14,4 ms | 1,6 ms | **9×** |
| Eigen simetris, 256×256 | 120 ms | 45,6 ms | 2,6× |
| SVD, 256×256 | 331 ms | 176 ms | 1,9× |

Yang diikat adalah antarmuka **LAPACKE**, bukan Fortran: LAPACKE menerima argumen layout sehingga
matriks row-major bisa langsung dilewatkan, sedangkan entry point Fortran hanya column-major dan
setiap panggilan akan butuh transpose masuk dan transpose keluar.

Empat konvensi berbeda dan dikonversi, bukan diasumsikan:

- Nilai eigen kembali **menaik**; library ini menjanjikan menurun.
- `dgesvd` mengembalikan `V^T`; `SvdResult` membawa `V`.
- Dengan layout row-major, LAPACKE sudah menaruh vektor eigen ke-*j* di kolom ke-*j*.
  Mentransposnya — seperti disarankan kebiasaan Fortran — merusak `A V = V Λ` sementara nilai
  eigennya tetap benar sempurna, persis jenis hasil setengah-benar yang lolos dari uji yang lemah.
- `dgetrf` melaporkan **urutan tukar baris**, bukan permutasi jadi: pada langkah *i*, baris *i*
  ditukar dengan baris `ipiv[i]`. `LuResult.Pivot` adalah permutasinya sendiri, jadi tukarannya
  diputar ulang. Membaca yang satu sebagai yang lain menghasilkan L dan U yang tampak sah tetapi
  merekonstruksi matriks yang salah.

`dpotrf` juga hanya menulis segitiga yang diminta dan meninggalkan sisanya berisi masukan, sehingga
pemanggil harus membersihkannya; matriks yang bukan definit positif kembali sebagai `info` positif
dan diubah menjadi exception yang sama dengan yang dilempar rutin terkelola.

> Setiap jalur native diperiksa terhadap *sifat pendefinisi* faktorisasinya — `A = QR`,
> `A V = V Λ`, `A x = b` yang memulihkan `x` yang diketahui — bukan terhadap rutin terkelola. Dua
> implementasi yang sepakat hanya menunjukkan keduanya berbagi asumsi yang sama.

### Kernel packed, bila tidak ada BLAS

`LinAlg.Dot` mengemas kedua operand ke buffer kontigu per petak di atas kira-kira delapan juta
multiply-add. Penyalinannya berbiaya satu lintasan dan terbayar berkali-kali, karena panel yang
sudah dikemas lalu dibaca oleh setiap blok baris alih-alih diambil ulang secara strided dari memori
utama.

| Ukuran | Sederhana | Packed | |
|---:|---:|---:|---|
| 256 | 1,45 ms | 0,88 ms | 1,65× |
| 1024 | 83,9 ms | 42,7 ms | 1,96× |
| 2048 | 759 ms | 365 ms | **2,08×** |

Di bawah ambang ia kalah — packing adalah biaya tetap — sehingga ambangnya disetel konservatif di
atas wilayah yang berisik. `PackedMatMul.Enabled = false` memaksa kernel sederhana.

> Mengukur ini memberi pelajaran yang layak diulang: satu run awal menunjukkan kubus 192 memakan
> waktu *delapan kali lebih lama* daripada kubus 224 — mustahil secara fisik. Penyebabnya tiered
> JIT: ukuran-ukuran awal masih berjalan tanpa optimasi. Dua kali pemanasan tidak cukup; metode
> panas baru dikompilasi ulang setelah sekitar tiga puluh panggilan.

### Presisi tunggal — sebuah prototipe

`Single.SingleKernels` berisi versi `float` dari dua kernel yang mendominasi waktu jalan. Ini
**alat ukur, bukan API kedua**: membuat `NdArray` generik atas `INumber<T>` akan menyentuh keenam
library, dan itu bukan perubahan yang layak dimulai tanpa tahu imbalannya.

| | double | float | |
|---|---:|---:|---|
| penjumlahan elemen, 1 juta | 2,24 ms | 1,07 ms | **2,09×** |
| penjumlahan elemen, 10 juta | 24,9 ms | 11,7 ms | **2,13×** |
| perkalian kubus 1024 | 94,9 ms | 52,2 ms | 1,82× |

Kedua peningkatan itu berbeda penyebabnya, dan karena itu berbeda besarnya. `Vector<float>` menampung
delapan lane melawan empat milik `Vector<double>`, yang menolong kerja compute-bound; dan setiap
nilai berukuran separuh, yang menolong kerja bandwidth-bound. Aritmetika elemen bersifat
bandwidth-bound dan mendarat rapi di 2,1×.

Biayanya: perkalian kubus 1000 menghasilkan galat relatif **1,3e-6** — float32 melakukan apa yang
memang dilakukan float32, sekitar tujuh digit desimal melawan enam belas milik double.

## Membaca bobot ONNX

```csharp
using Gravicode.Science.GraviNum.Io;

var weights = OnnxReader.ReadWeightsByName("model.onnx");
weights["encoder.weight"].ToNdArray();
```

Ini pembaca bobot, bukan runtime: ia mengekstrak *initializer* graf — tensor konstan bernama yang
menyimpan parameter terlatih — dan berhenti di situ. Itulah bagian yang penting di sini, karena
library ini punya lapisannya dan justru kekurangan angkanya.

Bebas dependensi secara sengaja. ONNX Runtime adalah paket native berukuran besar, dan menariknya
hanya untuk membaca beberapa array bukan pertukaran yang baik; yang dibutuhkan hanyalah format kawat
protobuf, dan hanya segelintir field-nya. Float32, float64, float16, int8/16/32/64 semuanya
di-decode dan dilebarkan ke `double`. Tensor yang tipenya tak bisa di-decode, atau yang datanya tidak
cocok dengan bentuk yang dideklarasikan, **dilewati alih-alih ditebak** — mengembalikan angka salah
secara diam-diam jauh lebih buruk daripada mengembalikan lebih sedikit angka.

Diverifikasi terhadap berkas yang ditulis library `onnx` resmi Python, bukan terhadap fixture yang
dibuat agar cocok dengan pembacanya.

## Diferensiasi otomatis

`Gravicode.Science.GraviNum.Autodiff` adalah tape mode-mundur (reverse-mode). Tulis komputasi
majunya sekali, dan turunannya didapat dari satu kali backward pass — tanpa menurunkan backward
pass dengan tangan.

```csharp
using Gravicode.Science.GraviNum.Autodiff;

var w = Tensor.Parameter(NdArray.Zeros(2, 1));
var b = Tensor.Parameter(NdArray.Zeros(1));

var error = Tensor.Constant(x).MatMul(w) + b - Tensor.Constant(y);
var loss  = (error * error).Mean();

loss.Backward();
w.Gradient;   // dLoss/dw, bentuknya sama dengan w
b.Gradient;   // dijumlahkan atas batch, karena b tadi di-broadcast
```

`Parameter` mengakumulasi gradien; `Constant` menghentikannya. Operasi yang tersedia meliputi
aritmetika, `Exp`, `Log`, `Sqrt`, `Pow`, `Tanh`, `Sigmoid`, `Relu`, `Abs`, `Softplus`, `Sum`,
`Mean`, `LogSumExp`, `MatMul`, `Transpose`, dan `Reshape`. `LogSumExp` menggeser keluar nilai
maksimumnya, sehingga tetap hidup pada masukan sekitar 1000 — tempat bentuk naif
`log(sum(exp(x)))` mengembalikan tak hingga.

**Periksa setiap gradien baru.** `GradientCheck` membandingkan tape dengan beda hingga terpusat,
yang tidak berbagi kode apa pun dengannya:

```csharp
var result = GradientCheck.Check(t => (t.Sigmoid() * t.Tanh()).Log().Sum(), NdArray.FromValues([0.8, 1.4]));
result.Passed(1e-6);   // bila false, dicetak entri mana yang berbeda dan sebesar apa
```

### Dua hal yang perlu diketahui

**Gradien broadcast itu dijumlahkan.** Bias berbentuk `[3]` yang ditambahkan ke batch `[64, 3]`
memengaruhi 64 keluaran, sehingga gradiennya adalah *jumlah* atas batch — bukan satu baris, bukan
rata-ratanya. Ini cara paling umum sebuah backward pass tulisan tangan menjadi salah, dan gagalnya
diam-diam: gradiennya meleset persis sebesar ukuran batch, dan modelnya tetap tampak terlatih.

**Ini tape, bukan framework.** Tidak ada pengoptimal graf, tidak ada kernel tergabung, tidak ada
penempatan perangkat. Tujuannya agar lapisan baru atau log-densitas baru cukup ditulis sekali,
maju. Mode mundur berbiaya satu backward pass berapa pun jumlah parameternya, sementara beda hingga
berbiaya `2d` evaluasi maju tambahan — tetapi tape mengalokasikan satu node per operasi, sehingga
pada model *kecil* beda hingga justru benar-benar lebih cepat. Diukur pada log posterior sebuah
model Bayesian, titik silangnya ada di sekitar 50 parameter. Di bawah itu, tape memberi Anda
kebenaran dan kemudahan, bukan kecepatan.

## Bilangan acak

`GraviRandom` adalah xoshiro256++ yang di-seed lewat SplitMix64: cepat, kokoh secara statistik, dan
reprodusibel dari seed di semua platform.

```csharp
var rng = new GraviRandom(seed: 42);

rng.NextDouble();                       rng.Next(10);
rng.Normal(mean: 0, stdDev: 1);         rng.Uniform(-1, 1);
rng.Gamma(shape: 2, scale: 3);          rng.Beta(2, 5);
rng.Binomial(trials: 100, probability: 0.3);
rng.Poisson(lambda: 4.5);               rng.Exponential(rate: 2);

rng.StandardNormal(1000, 20);           // versi array
rng.Normal(100, 15, 50_000);
rng.MultivariateNormal(mean, covariance, samples: 1000);

rng.Permutation(100);                   rng.Shuffle(list);
rng.Choice(n: 100, count: 20, replace: false);
```

Pengambilan sampel memakai algoritma eksak, bukan aproksimasi normal — Marsaglia-Tsang untuk gamma,
pemecahan beta rekursif untuk binomial, transformed rejection untuk Poisson berlambda besar —
sehingga ekor distribusinya tetap benar.

## Statistik

```csharp
Statistics.Mean(a);        Statistics.Median(a);      Statistics.Mode(a);
Statistics.Var(a, ddof: 1);  Statistics.Std(a);
Statistics.Percentile(a, 95);  Statistics.Quantile(a, 0.95);
Statistics.Skewness(a);    Statistics.Kurtosis(a);
Statistics.Describe(a);    // count, mean, std, min, kuartil, max

Statistics.Sum(a, axis: 0);      // reduksi per sumbu
Statistics.Mean(a, axis: 1);
Statistics.ArgMax(a, axis: 0);

Statistics.Correlation(x, y);          Statistics.SpearmanCorrelation(x, y);
Statistics.CovarianceMatrix(data);     Statistics.CorrelationMatrix(data);
Statistics.Histogram(a, bins: 20);
Statistics.Standardize(a);             Statistics.MinMaxScale(a);
```

`Sum` memakai penjumlahan berpasangan dan `Var` memakai algoritma Welford, sehingga keduanya tidak
kehilangan presisi pada masukan yang panjang atau berskala buruk.

## Matriks sparse

CSR hanya menyimpan elemen tak-nol ditambah satu penunjuk baris per baris.

```csharp
var sparse = SparseMatrix.FromDense(dense, threshold: 0);
var built  = SparseMatrix.FromTriplets(rows, cols, triplets);   // duplikat dijumlahkan
var viaBuilder = new SparseBuilder(rows, cols).Add(0, 1, 2.5).Build();

sparse.Multiply(vector);                       // sparse · vektor dense
sparse.Multiply(matrix, denseIsMatrix: true);  // sparse · matriks dense
sparse.Transpose();  sparse.RowSums();  sparse.ToDense();
sparse.NonZeroCount;  sparse.Density;
```

Di bawah sekitar 10% kepadatan ini jelas menguntungkan; di atas sekitar 25% indireksi indeks lebih
mahal daripada nol yang dilewati.

## IO

```csharp
NdIO.SaveCsv(a, "matrix.csv");        NdIO.LoadCsv("matrix.csv");
NdIO.SaveBinary(a, "matrix.gnb");     NdIO.LoadBinary("matrix.gnb");
NdIO.SaveJson(a, "matrix.json");      NdIO.ToJson(a);
```

### Array memory-mapped

Untuk data yang tidak muat di RAM. Sistem operasi hanya memuat bagian yang benar-benar disentuh.

```csharp
using var mapped = MemoryMappedArray.Create("big.gmm", 1_000_000, 100);
mapped.Set(row: 5, column: 3, value: 42.0);
mapped.Flush();

using var opened = MemoryMappedArray.Open("big.gmm");
opened.ReadRow(42);                       // satu baris, tanpa memuat seluruh berkas
foreach (var chunk in opened.Chunks(1 << 16)) { /* streaming */ }
```

## Backend komputasi

```csharp
using Gravicode.Science.GraviNum.Compute;

Compute.DescribeDevices();       // perangkat yang tersedia
Compute.IsGpuAvailable;

Compute.Dot(a, b);               // dirutekan — CPU kecuali Anda mengaktifkan GPU
Compute.Dot(a, b, Compute.Gpu);  // dipaksa
```

**Pemilihan GPU otomatis mati secara bawaan.** Semua array di sini bertipe `double`, dan GPU
terintegrasi menjalankan float64 pada sebagian kecil laju float32-nya — terukur pada Intel UHD 620,
GPU 5–8× *lebih lambat* daripada CPU untuk beban kerja ini. Ukur perangkat keras Anda sendiri, lalu
aktifkan:

```csharp
Compute.AutomaticGpuDispatch = true;
```

Lihat [benchmarks.md](benchmarks.md) untuk hasil pengukurannya.

## Kesalahan yang sering terjadi

| Gejala | Penyebab |
|---|---|
| `*` memberi hasil salah untuk matriks | `*` adalah elemen demi elemen; gunakan `Dot` |
| Mengubah satu array ikut mengubah array lain | Keduanya view atas satu buffer; gunakan `Copy()` |
| `AsSpan()` melempar exception | Array-nya strided; panggil `AsContiguous()` dulu |
| GPU lebih lambat daripada CPU | Wajar untuk float64 pada perangkat terintegrasi — lihat di atas |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
