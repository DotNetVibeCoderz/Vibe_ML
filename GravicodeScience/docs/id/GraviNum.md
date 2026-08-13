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
