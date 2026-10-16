# SciNet

*[English](../SciNet.md)* · SciPy untuk .NET — fungsi khusus, optimisasi, integrasi dan ODE,
interpolasi, aljabar linear, matriks sparse dan graf, algoritma spasial, clustering, pemrosesan
sinyal, FFT, pemrosesan citra, statistika, konstanta fisika, dan format berkas ilmiah.

SciNet hanya bergantung pada GraviNum. Array adalah `NdArray` milik GraviNum, vektor 1-D adalah
`double[]`, dan setiap callback adalah delegate .NET — dan dari situlah sebagian besar kecepatan
SciNet berasal: SciPy memanggil balik ke Python untuk setiap evaluasi fungsi, SciNet tidak.

| Modul SciPy | Titik masuk SciNet | Namespace |
|---|---|---|
| `scipy.special` | `Special` | `Gravicode.Science.SciNet` |
| `scipy.constants` | `Constants` | `Gravicode.Science.SciNet` |
| `scipy.optimize` | `Optimize` | `Gravicode.Science.SciNet` |
| `scipy.integrate` | `Integrate` | `Gravicode.Science.SciNet` |
| `scipy.interpolate` | `Interpolate`, `CubicSpline`, `BSpline`, … | `Gravicode.Science.SciNet` |
| `scipy.linalg` | `Linalg` | `Gravicode.Science.SciNet` |
| `scipy.differentiate` | `Differentiate` | `Gravicode.Science.SciNet` |
| `scipy.fft` | `Fft`, `FftPlan`, `RfftPlan` | `Gravicode.Science.SciNet` |
| `scipy.signal` | `Signal` | `Gravicode.Science.SciNet` |
| `scipy.stats` | `Stats` | `Gravicode.Science.SciNet` |
| `scipy.ndimage` | `NdImage` | `Gravicode.Science.SciNet` |
| `scipy.cluster` | `Cluster` | `Gravicode.Science.SciNet` |
| `scipy.sparse`, `.linalg`, `.csgraph` | `CsrMatrix`, `SparseLinalg`, `CsGraph`, … | `Gravicode.Science.SciNet.Sparse` |
| `scipy.spatial` | `KDTree`, `Distance`, `Delaunay`, `Rotation`, … | `Gravicode.Science.SciNet.Spatial` |
| `scipy.io` | `MatFile`, `MatrixMarket`, `WavFile`, … | `Gravicode.Science.SciNet.IO` |

```bash
dotnet add package Gravicode.Science.SciNet
```

> **Dua nama yang perlu diperhatikan.** `Linalg` milik SciNet (scipy.linalg) dan `LinAlg` milik
> GraviNum (numpy.linalg) hanya berbeda huruf besar-kecilnya, dan itu disengaja. Lalu `fft()` milik
> scipy menjadi `Fft.Fft1D()` — C# tidak mengizinkan method bernama sama dengan kelasnya.

## Seberapa dekat dengan SciPy, dan bagaimana itu diperiksa

Setiap modul dipatok pada nilai yang dihitung dengan **SciPy 1.17.1** dan ditanam langsung di
tes, ditambah bentuk tertutup, identitas, dan rujukan brute-force yang ditulis di dalam tes. Di
banyak tempat SciNet mereproduksi SciPy *persis* — bukan hanya jawabannya, tetapi juga jalannya:

- `Integrate.Quad` mengembalikan nilai, taksiran galat, **dan jumlah evaluasi** milik SciPy
  (QUADPACK di-port baris per baris, termasuk pembukuan flag galatnya).
- `Integrate.SolveIvp` menempuh urutan langkah SciPy yang persis sama untuk RK23, RK45, DOP853,
  Radau, dan BDF.
- `Optimize.Minimize` dengan Nelder–Mead, Powell, CG, BFGS, Newton-CG, metode trust-region, dan
  L-BFGS-B memakai jumlah iterasi dan evaluasi fungsi yang sama.
- `Cluster.Linkage` menghasilkan matriks linkage SciPy secara persis, termasuk saat ada nilai
  kembar, untuk ketujuh metode.
- `Interpolate.Splrep` dengan smoothing adalah port dari `curfit` milik FITPACK: knot yang sama,
  koefisien sampai ~1e-12.

Di tempat SciNet sengaja berbeda, dokumen di bawah menyebutkannya.

## Fungsi khusus

```csharp
using Gravicode.Science.SciNet;

Special.Gamma(0.5);          Special.GammaLn(1e5);      Special.Digamma(3);
Special.GammaInc(2, 3);      Special.BetaIncInv(2, 3, 0.4);
Special.Erfc(6);             // 2.15e-17, dihitung langsung - bukan sebagai 1 - Erf(6)
Special.LogNdtr(-40);        // -804.6, di titik Ndtr sudah underflow
Special.Jv(2.5, 10);  Special.Kv(1, 3);  Special.Airy(1.0);   // Bessel dan Airy
Special.EllipK(0.99); Special.EllipJ(0.5, 0.3);                // eliptik
Special.Hyp2f1(1, 2, 3, 0.5);  Special.Zeta(3);  Special.LambertW(1);
Special.Stdtr(df: 5, t: 2.0);  Special.Chdtri(3, 0.05);        // fungsi bantu distribusi
var rule = Special.RootsLegendre(50);                          // node dan bobot kuadratur Gauss
rule.Integrate(Math.Cos);
Special.Erf(array);          // overload NdArray, paralel di atas suatu ambang
```

Keluarga yang tersedia: gamma dan beta (dengan bentuk tak lengkap teregularisasi dan inversnya),
fungsi galat dan Faddeeva `Wofz`, T milik Owen, Bessel J/Y/I/K berorde riil beserta varian
berskala, sferis, dan akar-akarnya, Hankel, Airy, polinomial ortogonal (Legendre, Legendre
terasosiasi, harmonik sferis dengan urutan argumen `sph_harm_y`, Hermite, Laguerre, Chebyshev,
Gegenbauer, Jacobi) beserta aturan Gauss-nya, hipergeometrik 0F1/1F1/2F1, zeta, integral
eksponensial dan sinus/kosinus, Lambert W, integral eliptik lengkap/tak lengkap, fungsi eliptik
Jacobi, bentuk Carlson, dan fungsi distribusi cephes (`Stdtr`, `Chdtr`, `Fdtr`, `Bdtr`, `Pdtr`,
`Nbdtr`, `Gdtr`, `Kolmogorov`, `Smirnov` beserta komplemen dan inversnya).

**Akurasi.** Umumnya 1e-15 sampai 5e-15 relatif terhadap SciPy. Bila SciPy dan mpmath tidak
sepakat, tes memakai mpmath, dan di beberapa tempat SciNet-lah yang lebih akurat (Bessel J0/Y0 di
x = 1e5, `Ndtr` dan `OwensT` jauh di ekor, `LambertW` dekat titik cabang, bobot Legendre orde
tinggi, `Hyp2f1` saat c − a − b bilangan bulat, di mana SciPy mengembalikan NaN). Titik lemahnya,
sekitar 1e-13 sampai 1e-12: `Smirnov` untuk n besar, ekor beta tak lengkap di bawah 1e-280, nilai
Bessel tepat di akarnya. `Hyp2f1` mencakup z riil ≤ 1. Belum diimplementasikan: Struve, `Hyperu`.

## Konstanta fisika

```csharp
Constants.SpeedOfLight;  Constants.Planck;  Constants.Avogadro;  Constants.Kilo;  Constants.Inch;
Constants.Value("electron mass");   Constants.Unit("Planck constant");   Constants.Precision("proton mass");
Constants.Find("Boltzmann");        // kunci yang cocok
Constants.ConvertTemperature(98.6, "F", "C");
```

Seluruh 445 entri `scipy.constants.physical_constants` (CODATA 2022), dibangkitkan dari SciPy
1.17.1, ditambah setiap nama satuan dan prefiks di tingkat modul.

## Optimisasi

```csharp
var result = Optimize.Minimize(Optimize.Rosen, [1.3, 0.7, 0.8, 1.9, 1.2],
    MinimizeMethod.Bfgs, jac: Optimize.RosenDer);
result.X;  result.Fun;  result.Nit;  result.Nfev;  result.Success;

// batas dan kendala
Optimize.Minimize(f, x0, MinimizeMethod.LBfgsB, bounds: new Bounds([0, 0], [1, 2]));
Optimize.Minimize(f, x0, MinimizeMethod.Slsqp, constraints: [
    new OptimizeConstraint(OptimizeConstraintType.Ineq, x => [x[0] - 2 * x[1] + 2]),
    new OptimizeConstraint(OptimizeConstraintType.Eq, x => [x[0] + x[1] - 1])]);

// masalah skalar dan pencarian akar
Optimize.MinimizeScalar(x => (x - 2) * (x - 2));
Optimize.Brentq(x => Math.Cos(x) - x, 0, 1);
Optimize.Root(F, x0, RootMethod.Hybr);

// fitting
var fit = Optimize.CurveFit((t, p) => p[0] * Math.Exp(-p[1] * t) + p[2], t, y, p0: [1, 1, 1]);
fit.Popt;  fit.Pcov;  fit.Perr;
Optimize.LeastSquares(residuals, x0, loss: LeastSquaresLoss.SoftL1, bounds: bounds);
Optimize.Nnls(A, b);   Optimize.LsqLinear(A, b, bounds);

// pemrograman linear dan bilangan bulat, assignment
Optimize.Linprog(c, aUb: A, bUb: b);
Optimize.Milp(c, integrality, bounds, constraints);
var (rows, cols) = Optimize.LinearSumAssignment(cost);

// global
Optimize.DifferentialEvolution(f, bounds, seed: 7);
Optimize.DualAnnealing(f, bounds);  Optimize.BasinHopping(f, x0);  Optimize.Direct(f, bounds);  Optimize.Brute(f, ranges);
```

Metode lokal: Nelder–Mead, Powell, CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, COBYLA, dogleg,
trust-ncg, trust-exact, trust-krylov. Akar: bisect, brentq, brenth, ridder, TOMS 748, Newton /
secant / Halley, titik tetap; untuk sistem persamaan hybr, lm, Broyden 1/2, Anderson, linear
mixing, Broyden diagonal, excitingmixing, Newton–Krylov, DF-SANE. Kuadrat terkecil: TRF, dogbox,
dan Levenberg–Marquardt dengan lima fungsi loss robust.

**Bukan port, sehingga jumlah iterasinya berbeda (solusinya tetap dites):** TNC (truncated Newton
terproyeksi), COBYLA (gaya Powell, bukan PRIMA), SLSQP (QP Goldfarb–Idnani, bukan LSQ milik
Kraft), Newton–Krylov (GMRES, bukan LGMRES), DIRECT. Belum diimplementasikan: `trust-constr`,
SHGO, quadratic assignment. `Linprog` tidak punya presolve; metode interior-point-nya membangun
ulang A·D·Aᵀ secara dense dan lambat pada masalah besar — simplex adalah bawaannya.

## Integrasi dan ODE

```csharp
var q = Integrate.Quad(x => Math.Log(x) / Math.Sqrt(x), 0, 1);       // q.Value, q.AbsErr, q.Neval
Integrate.Quad(x => Math.Exp(-x * x), double.NegativeInfinity, double.PositiveInfinity);
Integrate.Quad(f, 0, 1, weight: QuadWeight.Cos, wvar: 50);           // QAWO
Integrate.Dblquad((y, x) => x * y, 0, 1, x => 0, x => 1 - x);
Integrate.Simpson(y, x);   Integrate.CumulativeTrapezoid(y, x);   Integrate.TanhSinh(f, 0, 1);
Integrate.Cubature(f, [0, 0], [1, 1]);
```

`Quad` mencakup QAGS, QAGI (rentang tak hingga), QAGP (titik patah), QAWO/QAWF (bobot osilasi),
QAWS (bobot aljabar–logaritmik di titik ujung), dan QAWC (nilai utama Cauchy).

Ruas kanan ODE adalah delegate yang bebas alokasi:

```csharp
void Lorenz(double t, ReadOnlySpan<double> y, Span<double> dy)
{
    dy[0] = 10 * (y[1] - y[0]);
    dy[1] = y[0] * (28 - y[2]) - y[1];
    dy[2] = y[0] * y[1] - 8.0 / 3.0 * y[2];
}

var sol = Integrate.SolveIvp(Lorenz, 0, 40, [1.0, 1.0, 1.0], OdeMethod.Dop853,
    rtol: 1e-9, atol: 1e-12, denseOutput: true);
sol.T;  sol.Y;            // Y berukuran n × len(T), seperti di SciPy
sol.Sol!.Evaluate(12.5);  // dense output di titik mana pun
```

Metode: `Rk23`, `Rk45`, `Dop853`, `Radau`, `Bdf`, `Lsoda`. Event (terminal, arah), `tEval`,
integrasi mundur, `atol` per komponen, Jacobian dari pengguna, dan sparsitas Jacobian didukung.
Kelas solver langkah demi langkah (`Rk45Solver`, `BdfSolver`, …) mencerminkan
`scipy.integrate.RK45` dan kawan-kawannya. Loop langkah RK45 tidak mengalokasikan apa pun —
terukur 0 byte untuk 100.000 langkah.

`Integrate.SolveBvp` adalah kolokasi orde-4 milik SciPy untuk masalah nilai batas, dengan
parameter yang tidak diketahui; sistem Newton-nya dibuat berpita, sehingga mesh 10.000 node tidak
masalah.

**LSODA adalah implementasi ulang**, bukan ODEPACK: RK45 dengan detektor kekakuan Hairer yang
berpindah ke BDF dan kembali lagi. Hasilnya sesuai dengan SciPy dalam batas toleransi, bukan
langkah demi langkah. `TanhSinh`, `QuadVec`, dan `Cubature` mengikuti algoritma SciPy tetapi
berbeda di digit-digit terakhir.

![Atraktor Lorenz](../screenshots/scinet_lorenz.png)

## Interpolasi

```csharp
var cs = new CubicSpline(x, y);                         // not-a-knot; juga Natural, Clamped, Periodic, kustom
cs.Evaluate(2.5);  cs.Evaluate(xs, nu: 1);  cs.Integrate(0, 3);  cs.Roots();
new PchipInterpolator(x, y);   new Akima1DInterpolator(x, y);
new Interp1d(x, y, Interp1dKind.Cubic);
Interpolate.Interp(xq, xp, fp);                         // numpy.interp

var spl = Interpolate.MakeInterpSpline(x, y, k: 3);     // BSpline
var (t, c, k) = Interpolate.Splrep(x, y, s: 0.5);       // spline smoothing FITPACK
Interpolate.Splev(xq, (t, c, k));
Interpolate.MakeSmoothingSpline(x, y);                  // lambda dipilih lewat GCV

new RegularGridInterpolator([xs, ys, zs], values, RegularGridMethod.Cubic);
Interpolate.Griddata(points, values, xi, GriddataMethod.Linear);
new RBFInterpolator(points, values, kernel: RbfKernel.ThinPlateSpline);
```

Juga: `PPoly`, `BPoly`, `BSpline` (turunan, antiturunan, integral, matriks desain sparse),
`UnivariateSpline` dan kawan-kawannya, `Splprep` untuk kurva parametrik,
`BarycentricInterpolator`, `KroghInterpolator`, `FloaterHormannInterpolator`, `AAA`, `Pade`,
`Lagrange`, `NdBSpline`, `RectBivariateSpline`, `NearestNDInterpolator`, `LinearNDInterpolator`,
`CloughTocher2DInterpolator`.

Metode cubic dan quintic `RegularGridInterpolator` diselesaikan secara eksak; solver iteratif SciPy
berhenti di 1e-6, jadi SciPy sendiri hanya seakurat itu di sana. Batasan: tidak ada fitting FITPACK
periodik, `Bisplrep` memerlukan knot yang diberikan, `RectBivariateSpline` hanya interpolasi
(s = 0), interpolasi linear tersebar dan Clough–Tocher hanya 2-D (triangulasinya 2-D).
`BarycentricInterpolator.AddXi` sengaja **tidak** menyamai SciPy 1.17.1, yang punya bug tanda di
sana.

## Aljabar linear

```csharp
Linalg.Expm(a);  Linalg.Logm(a);  Linalg.Sqrtm(a);  Linalg.Funm(a, Math.Cos);  Linalg.FractionalMatrixPower(a, 0.5);
Linalg.Solve(a, b, assumeA: LinalgMatrixStructure.PositiveDefinite);
Linalg.SolveBanded(1, 1, ab, b);  Linalg.SolveToeplitz(c, r, b);  Linalg.SolveCirculant(c, b);
var (t, z) = Linalg.Schur(a);  Linalg.Hessenberg(a, calcQ: true);  Linalg.Eig(a);  Linalg.Eigh(a, b);
Linalg.Ldl(a);  Linalg.QrPivoting(a);  Linalg.Polar(a);  Linalg.NullSpace(a);  Linalg.Lstsq(a, b);
Linalg.SolveSylvester(a, b, q);  Linalg.SolveContinuousLyapunov(a, q);
Linalg.SolveContinuousAre(a, b, q, r);  Linalg.SolveDiscreteAre(a, b, q, r);
Linalg.Toeplitz(c);  Linalg.Hilbert(5);  Linalg.InvHilbertExact(5);  Linalg.Companion(p);
```

SciNet menambahkan apa yang dimiliki `scipy.linalg` di luar `numpy.linalg`, dengan memakai ulang
LU, QR, Cholesky, SVD, dan eigensolver simetris milik GraviNum di bawahnya. `Expm` adalah
scaling-and-squaring Al-Mohy–Higham dengan pemilihan derajat Padé; `ExpmFrechet` memberikan
turunan Fréchet-nya.

`Differentiate.Derivative(f, x)` adalah `scipy.differentiate.derivative` — beda pusat adaptif
dengan ekstrapolasi Richardson dan taksiran galat yang jujur — ditambah `Jacobian`, `Gradient`,
dan `Hessian`.

**Pendekatan yang perlu diketahui:** `FractionalMatrixPower` adalah `exp(t·log A)`, bukan
Higham–Lin; solver Riccati memakai fungsi tanda matriks / structured doubling, bukan QZ seperti
SciPy (solusi penstabilnya unik, sehingga hasilnya sesuai sampai ~1e-11). Belum
diimplementasikan: pengurutan ulang Schur, `Eig(a, b)` tergeneralisasi lewat QZ, vektor eigen
kiri.

## Matriks sparse, solver, dan graf

```csharp
using Gravicode.Science.SciNet.Sparse;

var d = SparseMatrices.Diags([lower, main, upper], [-1, 0, 1]);
var laplacian = SparseMatrices.Kronsum(d, d);          // CsrMatrix
var y = laplacian * x;                                  // SpMV;  sparse * sparse adalah perkalian matriks
var cg = SparseLinalg.Cg(laplacian, b, rtol: 1e-8);    // juga Gmres, Bicgstab, Minres, Lsqr, Lsmr, …
var lu = SparseLinalg.Splu(a);  lu.Solve(b);
var eig = SparseLinalg.Eigsh(laplacian, k: 6, sigma: 0);   // shift-invert
SparseLinalg.ExpmMultiply(a, v);

var (n, labels) = CsGraph.ConnectedComponents(graph, directed: true, connection: ComponentConnection.Strong);
CsGraph.ShortestPath(graph);  CsGraph.MinimumSpanningTree(graph);  CsGraph.MaximumFlow(graph, 0, 5);
CsGraph.ReverseCuthillMcKee(graph);
```

Format: `CsrMatrix`, `CscMatrix`, `CooMatrix` (immutable), builder `DokMatrix` dan `LilMatrix`,
serta `DiaMatrix`; semuanya bisa dikonversi satu sama lain, ke `NdArray` dense, dan ke
`SparseMatrix` milik GraviNum. Setiap format dan setiap solver bekerja lewat `ILinearOperator`,
sehingga operator tanpa matriks bisa dipasang ke solver yang sama. Dua perbedaan dengan SciPy:
`*` antara dua matriks sparse adalah perkalian matriks (elemen demi elemen memakai `Multiply`),
dan transpose CSR↔CSC bernilai O(1).

**Lebih lambat dari SciPy, terus terang:** `Splu` adalah Gilbert–Peierls left-looking tanpa
supernode — sekitar 13× lebih lambat dari SuperLU pada matriks Poisson 10k — dan `Eigsh` memakai
reortogonalisasi penuh, sekitar 4,5× lebih lambat dari ARPACK. `Svds` bekerja lewat AᵀA, sehingga
nilai singular di bawah √ε·σmax tidak akurat. COLAMD didekati dengan minimum degree pada AᵀA.

## Algoritma spasial

```csharp
using Gravicode.Science.SciNet.Spatial;

var tree = new KDTree(points);
var (dist, idx) = tree.Query(queries, k: 5);           // kueri batch memakai semua core
tree.QueryBallPoint(x, r: 0.1);  tree.QueryPairs(0.05);

Distance.Cdist(a, b, DistanceMetric.Cosine);  Distance.Pdist(x);  Distance.Squareform(y);
var tri = new Delaunay(points2d);  tri.FindSimplex(q);
var hull = new ConvexHull(points);  hull.Volume;     // 2-D dan 3-D
var vor = new Voronoi(points2d);   vor.FinitePolygons(box);

var r = Rotation.FromEuler("xyz", [90, 0, 45], degrees: true);
r.AsQuat();  r.Apply(vectors);  r.Inv();  Rotation.AlignVectors(a, b);
new Slerp(times, rotations).Evaluate(0.5);
```

Delaunay memakai predikat eksak, sehingga kisi dan titik-titik sekonsiklik tertriangulasi dengan
benar; Delaunay hanya 2-D (begitu pula Voronoi; hull mendukung 2-D dan 3-D). Nilai kembar di
KD-tree diputuskan berdasarkan (jarak, indeks), dan penomoran segitiga berbeda dari Qhull, yang
memang sembarang.

## Clustering

```csharp
var z = Cluster.Linkage(data, LinkageMethod.Ward);    // matriks linkage SciPy, persis
var labels = Cluster.Fcluster(z, 3, FclusterCriterion.Maxclust);
var tree = Cluster.Dendrogram(z, p: 30, truncateMode: DendrogramTruncateMode.LastP);
Cluster.Cophenet(z, y);  Cluster.OptimalLeafOrdering(z, y);
var (codebook, distortion) = Cluster.KMeans(Cluster.Whiten(obs), 3);
Cluster.KMeans2(data, 3, minit: KMeansInit.PlusPlus);
```

`Dendrogram` mengembalikan kamus koordinat milik SciPy (`Icoord`, `Dcoord`, `ColorList`, …)
untuk digambar. `OptimalLeafOrdering` menemukan optimum yang sebenarnya — milik SciPy melesetkannya
pada sebagian input, sehingga keduanya bisa berbeda di sana, dan hasil SciNet tidak pernah lebih
buruk.

![Dendrogram Ward untuk Iris](../screenshots/scinet_dendrogram.png)

## Transformasi Fourier cepat

```csharp
var spectrum = Fft.Fft1D(signal);      Fft.Ifft(spectrum);
Fft.Rfft(real);  Fft.Irfft(half, n);   Fft.Fft2(image);  Fft.FftN(volume);
Fft.Dct(x, type: 2, norm: FftNorm.Ortho);  Fft.Dst(x, type: 4);
Fft.FftFreq(n, d);  Fft.FftShift(x);  Fft.NextFastLen(1000);

var plan = FftPlan.Get(4096);          // di-cache, thread-safe, eksekusi tanpa alokasi
plan.Execute(input, output);
```

Mesinnya adalah transformasi Stockham mixed-radix bergaya pocketfft (radix 2, 3, 4, 5, dan
butterfly prima ganjil generik) dengan butterfly SIMD; Bluestein hanya dipilih bila lebih murah,
dan transformasi riil dihitung dengan panjang setengah. Mesin ini **2,3–28× lebih cepat dari FFT
GraviNum** (28× di n = 1000, tempat GraviNum harus memutar lewat Bluestein). Terhadap `scipy.fft`
(pocketfft dalam C++) hasilnya campuran: transformasi riil 2²⁰ titik 1,5× lebih cepat,
transformasi kompleks tunggal 65.536 titik dan DCT-II masing-masing 1,2× dan 1,7× lebih lambat,
dan transformasi batch sepanjang suatu sumbu menang lewat paralelisme. Belum diimplementasikan:
`fht`, `hfft2`/`hfftn`.

## Pemrosesan sinyal

```csharp
var lowpass = Signal.Butter(4, 20.0, FilterType.Lowpass, fs: 1000);    // FilterBA (B, A)
var sos = Signal.ButterSos(8, [5.0, 50.0], FilterType.Bandpass, fs: 1000);
var y = Signal.Filtfilt(lowpass.B, lowpass.A, x);
Signal.Sosfiltfilt(sos, x);  Signal.Lfilter(b, a, x);  Signal.Sosfilt(sos, x);
var (w, h) = Signal.Freqz(lowpass.B, lowpass.A, fs: 1000);
var taps = Signal.Firwin(101, 0.2, window: ("kaiser", 8.6));
Signal.Remez(numtaps, bands, desired);
var (order, wn) = Signal.Ellipord(wp, ws, gpass, gstop);
Signal.GetWindow("hann", 256);  Signal.Dpss(512, 4, 6);
var system = new TransferFunction([1], [1, 2, 1]);
Signal.Step(system);  Signal.Bode(system);  Signal.Cont2Discrete(system, 0.01, DiscretizationMethod.Zoh);
Signal.Chirp(t, 1, 10, 100);
```

Desain filter mencakup FIR (`Firwin`, `Firwin2`, `Remez`, `Firls`, fungsi bantu Kaiser, fase
minimum) dan IIR (Butterworth, Chebyshev I/II, eliptik, Bessel; analog dan digital; keluaran ba,
zpk, dan sos dengan pemasangan seksi seperti SciPy; pemilihan orde), konversi antar semua
representasi, respons frekuensi, filtering sepanjang sumbu mana pun dari sebuah `NdArray` secara
paralel, sistem LTI (kontinu dan diskret), dan bentuk gelombang. Semua window yang dimiliki
SciPy, termasuk DPSS.

![Filtering fase-nol](../screenshots/scinet_filtfilt.png)

### Analisis spektral, konvolusi, dan puncak

```csharp
var (f, pxx) = Signal.Welch(x, fs: 1000, nperseg: 2048);          // juga Periodogram, Csd, Coherence
var (ft, tt, sxx, _) = Signal.Spectrogram(x, fs: 1000);
Signal.Stft(x, fs: 1000);  Signal.Istft(zxx, fs: 1000);  Signal.Lombscargle(t, y, freqs);
Signal.Convolve(a, b);                  // memilih langsung atau FFT dari model biaya yang diukur pada mesin ini
Signal.Fftconvolve(a, b);  Signal.Oaconvolve(a, b);  Signal.Convolve2d(image, kernel, ConvolveMode.Same);
Signal.ResamplePoly(x, up: 3, down: 2);  Signal.Resample(x, 500);  Signal.Decimate(x, 4);
Signal.SavgolFilter(x, 31, 3);  Signal.Medfilt(x, 101);  Signal.Hilbert(x);  Signal.Detrend(x);
var peaks = Signal.FindPeaks(psdDb, prominence: 10.0, width: 2.0);
peaks.Peaks;  peaks.Properties.Prominences;  peaks.Properties.Widths;
Signal.PeakWidths(x, peaks.Peaks, relHeight: 0.5);  Signal.ArgRelMax(x, order: 3);  Signal.FindPeaksCwt(x, widths);
```

`FindPeaks` mengembalikan semua kunci properti yang dikembalikan SciPy (`PeakHeights`,
`Prominences`, `LeftBases`, `Widths`, `LeftIps`, `PlateauSizes`, …), bernilai null bila
kondisinya tidak diminta. Welch dan STFT memproses segmen secara batch lewat plan FFT riil yang
di-cache dan menjalankan segmen secara paralel — Welch atas 10⁷ sampel memakan 66 ms, dibanding
671 ms di SciPy. Fungsi spektral dan konvolusi menerima input riil (`Resample`, `Czt`, dan
`ZoomFft` juga menerima kompleks). Belum diimplementasikan: `ShortTimeFFT` sebagai kelas,
keluarga `residue`, callback detrend kustom.

![PSD Welch dengan find_peaks](../screenshots/scinet_welch.png)

## Statistika

### Distribusi

```csharp
var normal = Stats.Norm(loc: 100, scale: 15);           // frozen, seperti scipy.stats.norm(100, 15)
normal.Pdf(115);  normal.Cdf(130);  normal.Sf(160);  normal.Ppf(0.975);  normal.Isf(1e-9);
normal.LogPdf(x);  normal.LogSf(x);  normal.Mean();  normal.Var();  normal.Interval(0.95);
normal.Moment(4);  normal.Entropy();  normal.Expect(v => v * v);
double[] draws = Stats.Gamma(a: 2.5, scale: 1.5).Rvs(10_000, seed: 3);
var fitted = Stats.Gamma(a: 1).FitTo(draws, floc: 0);  // maximum likelihood
Stats.Binom(20, 0.3).Pmf(6);  Stats.Poisson(4).Ppf(0.99);
var mvn = Stats.MultivariateNormal([0, 0], cov);  mvn.Cdf([0, 0]);  mvn.LogPdf(x);
```

61 keluarga kontinu (semua yang umum ditambah t/χ²/F non-sentral, normal dan eksponensial
terpotong, generalised extreme value dan Pareto, skew-normal, Johnson SU/SB, studentized range,
distribusi Kolmogorov–Smirnov, …), 17 keluarga diskret, serta `MultivariateNormal` (dengan CDF
quasi-Monte-Carlo Genz), `MultivariateT`, `Dirichlet`, `Multinomial`, `Wishart`, `InvWishart`,
`OrthoGroup`, dan `SpecialOrthoGroup`. Lokasi dan skala ditangani persis seperti di SciPy; setiap
distribusi mengimplementasikan `IUnivariateDistribution`, yang diterima oleh uji-uji hipotesis.
Ekor dihitung lewat formulasi `Sf`/`LogSf`, tidak pernah `1 - Cdf`.

Fitting memakai maximum likelihood bentuk tertutup bila SciPy punya, dan Nelder–Mead berpenalti
milik SciPy untuk selebihnya; tidak ada fit di suite tes yang berakhir pada likelihood lebih buruk
daripada fit SciPy.

**SciPy sendiri tidak selalu menjadi rujukan.** Di ekor jauh, kuantil SciPy 1.17.1 untuk pearson3,
f, truncnorm, halfnorm, weibull_max, foldnorm, dan genlogistic salah di digit ke-4–5 (atau
mengembalikan ±inf); kasus-kasus itu dipatok pada mpmath 40 digit. Titik lemah SciNet: ekor atas
jauh t non-sentral, `MultivariateNormal.Cdf` (~3e-5, seperti metode acak milik SciPy), dan
`StudentizedRange` (kuadratur bersarang, ~1 ms per nilai CDF).

### Uji hipotesis, korelasi, dan statistik deskriptif

```csharp
var t = Stats.TtestInd(a, b, equalVar: false);            // t.Statistic, t.Pvalue, t.Df, t.ConfidenceInterval()
Stats.Ttest1samp(a, 0);  Stats.TtestRel(before, after);  Stats.FOneway(g1, g2, g3);  Stats.Kruskal(g1, g2, g3);
Stats.MannWhitneyU(a, b);  Stats.Wilcoxon(differences);  Stats.BrunnerMunzel(a, b);
Stats.Shapiro(a);  Stats.Normaltest(a);  Stats.Anderson(a, "norm");  Stats.JarqueBera(a);
Stats.Kstest(sample, Stats.Gamma(2.5));  Stats.Ks2samp(a, b);  Stats.CramerVonMises(sample, Stats.Norm());
Stats.Chi2Contingency(table);  Stats.FisherExact(table2x2);  Stats.Binomtest(7, 20, 0.5);
Stats.Levene(g1, g2);  Stats.Bartlett(g1, g2);  Stats.TukeyHsd(g1, g2, g3);  Stats.FalseDiscoveryControl(pvalues);
Stats.Pearsonr(x, y);  Stats.Spearmanr(x, y);  Stats.Kendalltau(x, y);  Stats.Linregress(x, y);  Stats.Theilslopes(y, x);
Stats.Describe(a);  Stats.Skew(a);  Stats.Kurtosis(a);  Stats.Iqr(a);  Stats.MedianAbsDeviation(a);  Stats.Rankdata(a);
Stats.Quantile(a, 0.9);  Stats.BinnedStatistic(x, values, "mean", bins: 10);  Stats.WassersteinDistance(a, b);
Stats.Bootstrap([a], s => s[0].Average(), method: "BCa");  Stats.PermutationTest([a, b], statistic);
```

Nilai-p mengikuti metode SciPy, termasuk distribusi eksaknya (Mann–Whitney, Wilcoxon, Kendall, KS
dua sampel) dan opsi `Alternative`; hasilnya sesuai dengan SciPy sampai sekitar 1e-10–1e-12. Dua
tempat di mana SciNet sengaja *lebih* eksak: Ansari–Bradley (frekuensi eksak SciPy berupa
float32) serta probabilitas kotak Dunnett/bivariat (SciPy mengintegrasikannya secara acak).
`Bootstrap`, `PermutationTest`, dan `MonteCarloTest` menerima flag `parallel` yang memberi hasil
identik dengan versi serialnya.

### Estimasi densitas kernel dan quasi-Monte Carlo

```csharp
var kde = new GaussianKde(data);                      // aturan Scott; "silverman", sebuah faktor, atau fungsi
kde.Evaluate(points);  kde.LogPdf(points);  kde.IntegrateBox1d(0, 2);  kde.Resample(1000, seed: 1);

var sobol = new SobolEngine(d: 5, scramble: true, seed: 7);
NdArray points = sobol.RandomBase2(m: 10);            // 1024 titik
new HaltonEngine(3).Random(500);  new LatinHypercubeEngine(4, strength: 2).Random(49);
Qmc.Discrepancy(points);  Qmc.Scale(points, lower, upper);
```

KDE eksak berjalan paralel dan sekitar 30× lebih cepat dari milik SciPy (10 ribu titik data
dievaluasi di 100 ribu titik: 1,1–1,5 detik dibanding 39,8 detik). Sobol dan Halton tanpa
scrambling mereproduksi titik-titik SciPy bit demi bit; Sobol dihitung langsung dari indeks titik
(kode Gray), sehingga fast-forward gratis dan pengambilan paralel identik dengan serial. Sobol
mendukung 4096 dimensi (tabel SciPy sampai 21201). Belum diimplementasikan: `median_test`,
statistik sirkular, Box–Cox / Yeo–Johnson, `weibull_min` untuk `Anderson`.

## Pemrosesan citra

```csharp
NdImage.GaussianFilter(image, sigma: 3.0);          // rank berapa pun; sigma dan orde turunan per sumbu
NdImage.MedianFilter(image, 5);  NdImage.Sobel(image, axis: 0);  NdImage.UniformFilter(image, 9);
NdImage.Zoom(image, 2.0, order: 3);  NdImage.Rotate(image, 30);  NdImage.MapCoordinates(image, coords);
NdImage.BinaryOpening(mask);  NdImage.BinaryFillHoles(mask);  NdImage.GreyClosing(image, size: [3, 3]);
NdImage.DistanceTransformEdt(mask);
var (labels, count) = NdImage.Label(mask);
NdImage.CenterOfMass(image, labels, [1, 2]);  NdImage.FindObjects(labels);  NdImage.WatershedIft(image, markers);
```

Kedelapan mode batas sesuai dengan SciPy, termasuk perbedaan halus antara semantik filter dan
interpolasi untuk `Mirror`, `Wrap`, `Constant`, dan mode `Grid*`. Nilai boolean adalah double
0/1, dan keluaran selalu float64 (SciPy mempertahankan dtype input). `WatershedIft` secara bawaan
mereproduksi tetangga yang melompat ke baris berikutnya seperti di SciPy; berikan
`exactNeighbours: true` untuk geometri yang benar.

## Format berkas ilmiah

```csharp
using Gravicode.Science.SciNet.IO;

var vars = MatFile.Load("data.mat");                       // MATLAB v4 dan v5, terkompresi atau tidak
var a = ((MatArray)vars["A"]).Real;                        // NdArray row-major; kelas MATLAB-nya dipertahankan
MatFile.Save("out.mat", new Dictionary<string, object?> { ["A"] = a, ["note"] = "text" }, doCompression: true);
MatFile.Whosmat("data.mat");

var m = MatrixMarket.MmRead("matrix.mtx");                 // NdArray dense atau CooMatrix
MatrixMarket.MmWrite("out.mtx", sparse, symmetry: "symmetric");
var hb = HarwellBoeing.HbRead("matrix.rua", out var hbInfo);
var wav = WavFile.Read("audio.wav", normalize: true);      // int 8/16/24/32/64-bit, float, extensible, RIFX, RF64
WavFile.Write("out.wav", 44100, samples, WavSampleFormat.Int16, normalized: true);
using var nc = NetCdfFile.Open("climate.nc");              // NetCDF klasik dan 64-bit offset
var arff = Arff.LoadArff("data.arff");
using var ff = new FortranFile("records.dat");  ff.ReadReals();
IdlSave.ReadSav("data.sav");
```

Data MATLAB bersifat column-major sedangkan array SciNet row-major, sehingga setiap array yang
dimuat ditranspose ke urutan C — itulah satu-satunya tempat `loadmat` lebih lambat dari SciPy
(41 ms dibanding 33 ms untuk 50 MB tanpa kompresi). Pemuatan terkompresi dan pembacaan Matrix
Market sama cepat atau lebih cepat. Jenis nilai: `MatArray` (mempertahankan kelas MATLAB-nya),
`MatLogical`, `MatChar`, `MatSparse` (sebuah `CscMatrix`), `MatStruct` (sebuah kamus), `MatCell`,
dan `MatUnsupported` untuk function handle dan objek classdef. Berkas yang ditulis SciNet dalam
semua format dibaca kembali oleh SciPy 1.17.1 dan cocok persis.

Ditolak dengan pesan yang jelas: MATLAB v7.3 dan NetCDF-4 (keduanya HDF5), CDF-5, ARFF sparse.
Nilai int64/uint64 di atas 2⁵³ kehilangan presisi saat dimuat sebagai double. Harwell–Boeing
melampaui SciPy (yang hanya membaca RUA): penyimpanan simetris, skew, pattern, integer, dan
persegi panjang.

## Performa dibandingkan SciPy

Diukur dengan `benchmarks/comparison` — input yang sama di kedua sisi (rumus deterministik, bukan
RNG milik salah satu ekosistem), median dari pengulangan, pada laptop 4-core yang mengalami
throttling. .NET 10.0.12, Release, jalur managed tanpa BLAS/LAPACK native; Python 3.12.10 dengan
SciPy 1.17.1 dan NumPy 2.4.4. Setiap beban kerja .NET mendapat 40 panggilan pemanasan, karena
tiered JIT baru mengoptimalkan sebuah method setelah sekitar 30 panggilan — dengan 5, solusi
Lorenz terukur 19 ms, bukan 5,6 ms dalam keadaan stabil. Laporan lengkapnya ada di
[benchmarks/comparison/comparison-report.md](../../benchmarks/comparison/comparison-report.md).

| Operasi | SciNet | SciPy | |
|---|---:|---:|---|
| `solve_ivp` Lorenz 0..100, RK45, rtol 1e-9 | 5,57 ms | 1.004 ms | **SciNet 180×** |
| `solve_ivp` Van der Pol μ=1000, BDF | 1,10 ms | 130 ms | **SciNet 119×** |
| `minimize` BFGS, Rosenbrock-100 dengan gradien | 28,6 ms | 317 ms | **SciNet 11×** |
| `welch`, 10⁷ sampel, nperseg 4096 | 66 ms | 671 ms | **SciNet 10×** |
| `quad` dari log(x)/√x di [0, 1] | 9 µs | 83 µs | **SciNet 9,3×** |
| `erf` atas 10⁶ nilai | 10,1 ms | 47,6 ms | **SciNet 4,7×** |
| `RegularGridInterpolator` linear, 10⁶ titik 3-D | 120 ms | 506 ms | **SciNet 4,2×** |
| `gaussian_filter` σ=3, 2048² | 52 ms | 208 ms | **SciNet 4,0×** |
| `cdist` 2000×2000 dalam 64-D | 27,5 ms | 106 ms | **SciNet 3,8×** |
| `Delaunay`, 100 ribu titik | 192 ms | 585 ms | **SciNet 3,1×** |
| `median_filter` 5×5, 1024² | 131 ms | 388 ms | **SciNet 3,0×** |
| `CubicSpline`, evaluasi 10⁶ titik | 40,9 ms | 71,6 ms | **SciNet 1,8×** |
| conjugate gradient, Poisson 2-D 200² | 148 ms | 257 ms | **SciNet 1,7×** |
| FFT riil, 2²⁰ titik | 21,9 ms | 31,8 ms | **SciNet 1,5×** |
| `fftconvolve` 10⁶ × 10⁴ | 70 ms | 88 ms | **SciNet 1,3×** |
| `find_peaks` dengan prominence, 10⁶ sampel | 43 ms | 55 ms | **SciNet 1,3×** |
| `lfilter` Butterworth 4, 10⁶ sampel | 11,1 ms | 10,1 ms | setara |
| `sosfiltfilt` Butterworth 8, 10⁶ sampel | 21,8 ms | 24,6 ms | setara |
| FFT kompleks, 65.536 titik | 2,29 ms | 1,98 ms | SciPy 1,2× |
| `linear_sum_assignment` 500×500 | 17,2 ms | 13,2 ms | SciPy 1,3× |
| KD-tree build + kueri 5-NN, 100 ribu titik 3-D | 133 ms | 102 ms | SciPy 1,3× |
| `linkage` Ward, 2000 titik | 103 ms | 75 ms | SciPy 1,4× |
| DCT-II, 65.536 titik | 0,98 ms | 0,59 ms | SciPy 1,7× |
| `curve_fit`, 10.000 titik | 5,6 ms | 3,3 ms | SciPy 1,7× |
| `expm`, 200×200 | 29,8 ms | 8,2 ms | SciPy 3,6× |
| `splu` faktorisasi + solve, Poisson 2-D 100² | 273 ms | 37 ms | SciPy 7,4× |

**Rekap: SciNet lebih cepat di 16, SciPy lebih cepat di 8, setara di 2.**

Polanya konsisten. SciNet menang telak di mana pun SciPy memanggil balik ke Python untuk setiap
evaluasi fungsi — ODE, kuadratur, minimizer — karena delegate .NET berbiaya nanodetik sedangkan
panggilan Python mikrodetik. SciNet menang sedang pada kernel yang diparalelkannya (filter,
jarak, interpolasi, estimasi spektral). SciNet kalah di mana SciPy menyerahkan pekerjaannya ke
pustaka terkompilasi yang matang — LAPACK untuk fungsi matriks dense, SuperLU, solver assignment
dan KD-tree C++ miliknya — serta pada `curve_fit`, di mana model SciPy adalah satu panggilan NumPy
tervektorisasi dan inti kuadrat terkecilnya adalah MINPACK. Ukuran yang lebih besar menggeser
sebagian baris: di n = 400 `expm` mencapai kesetaraan, dan keunggulan pocketfft pada transformasi
kompleks tunggal menyempit begitu transformasinya terbatas oleh memori.

Kasus yang lebih besar yang diukur selama pengembangan (sekali jalan, mesin yang sama, jadi lebih
kasar): KDE 10 ribu titik yang dievaluasi di 100 ribu titik sekitar 30× lebih cepat dari
`gaussian_kde` SciPy, `spearmanr` atas 10⁶ sekitar 3×, `norm.ppf` atas 10⁶ sekitar 7×; di sisi
lambat, `Eigsh` sekitar 4,5× di belakang ARPACK dan Schur/Eig di n = 400 sekitar 4,5× di belakang
LAPACK.

---

Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
