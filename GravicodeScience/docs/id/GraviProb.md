# GraviProb

*[English](../GraviProb.md)* · Probabilistic programming untuk .NET — distribusi, MCMC, inferensi variasional, dan model probabilistik.

## Distribusi

Setiap distribusi menyediakan `LogDensity`, bukan densitas biasa, karena inferensi mengalikan
banyak densitas: dalam ruang linear beberapa ratus observasi akan underflow menjadi nol, sedangkan
dalam ruang log cukup dijumlahkan.

```csharp
using Gravicode.Science.GraviProb;

var normal = Distribution.Normal(mean: 0, stdDev: 1);
normal.LogDensity(1.5);   normal.Density(1.5);
normal.Sample(rng);       normal.Sample(rng, 10_000);
normal.Mean;  normal.Variance;  normal.Cdf(1.96);
normal.LogLikelihood(data);
```

Tersedia: `Normal`, `Uniform`, `Bernoulli`, `Binomial`, `Poisson`, `Gamma`, `Beta`, `Exponential`,
`StudentT`, `LogNormal`, `HalfNormal`, `Categorical`.

`HalfNormal` adalah prior lemah-informatif yang lazim untuk parameter skala. `Beta` mengekspos
konjugasinya secara langsung:

```csharp
var exact = Distribution.Beta(1, 1).PosteriorAfter(successes: 125, failures: 75);
// Beta(126, 76) — posterior bentuk tertutup, dipakai untuk menguji sampler
```

## Menyusun model

```csharp
var model = new BayesianModel()
    .AddDistribution("theta", Distribution.Beta(1, 1))
    .AddObservation("data", DistributionSpec.Binomial(200, "theta"), 125);
```

`DistributionSpec.Binomial(200, "theta")` mencatat **ketergantungan** pada variabel laten, bukan
probabilitas tetap, sehingga sampler mengevaluasi ulang likelihood saat `theta` bergerak. Spesifikasi
yang tersedia: `Binomial`, `Bernoulli`, `Normal`, `Poisson`, ditambah `DistributionSpec.From` untuk
selebihnya:

```csharp
DistributionSpec.From(v => new Gamma(v["shape"], v["rate"]), "shape", "rate");
```

Model adalah log densitas gabungan: prior ditambah likelihood. Nilainya hanya diketahui **sampai
suatu konstanta**, karena marginal likelihood pada teorema Bayes tidak pernah dihitung — dan justru
itulah sebabnya MCMC bekerja, karena Metropolis-Hastings hanya melihat rasio sehingga konstantanya
saling meniadakan.

```csharp
model.LogPosterior(values);
model.LogPrior(values);
model.SampleFromPrior(rng);
model.PriorPredictive(draws: 1000);   // apakah prior menyiratkan data yang masuk akal?
```

## Pengambilan sampel

```csharp
var posterior = model.SampleMCMC(iterations: 20_000, chains: 4, warmup: 10_000, thin: 1, seed: 42);
var gibbs     = model.SampleGibbs(iterations: 20_000, chains: 4);
```

**Metropolis-Hastings** mengusulkan langkah Gaussian dan menerimanya dengan peluang
`min(1, exp(Δ logPosterior))`. Selama warmup skala usulan menyesuaikan diri menuju tingkat
penerimaan **0,234** — nilai optimal asimtotik untuk random walk. Langkah terlalu besar selalu
ditolak dan rantai macet; langkah terlalu kecil selalu diterima tetapi tidak menjelajah apa pun.
Adaptasi **berhenti** saat warmup selesai, karena usulan yang terus berubah merusak sifat Markov.

**Metropolis within Gibbs** memperbarui satu koordinat setiap kali, menargetkan penerimaan 0,44.
Tingkat penerimaan per langkah lebih tinggi, tetapi pencampurannya lambat bila parameter
berkorelasi, karena langkah satu koordinat tidak dapat menyusuri punggungan diagonal.

Rantai berjalan paralel dan saling bebas, itulah sebabnya sampling multi-rantai nyaris gratis dan
R-hat menjadi bermakna.

## Membaca posterior

```csharp
posterior["theta"];                  // semua draw, digabung
posterior.Chain("theta", 0);         // satu rantai
posterior.Mean("theta");  posterior.StandardDeviation("theta");  posterior.Median("theta");

posterior.CredibleInterval("theta", 0.95);           // ekor sama
posterior.HighestDensityInterval("theta", 0.95);     // interval terpendek yang memuat 95%

posterior.RHat("theta");                   // mendekati 1 berarti konvergen; di atas ~1,01 tidak
posterior.EffectiveSampleSize("theta");    // setara berapa draw independen
posterior.AcceptanceRate;

Console.WriteLine(posterior.Summary());
```

Utamakan **HDI** pada posterior yang miring: interval ekor-sama bisa mengecualikan modus, nilai
paling mungkin, sedangkan HDI tidak.

### Pemeriksaan prediktif posterior

```csharp
var replicated = posterior.PosteriorPredictive(
    (values, rng) => rng.Binomial(200, values["theta"]), draws: 5000);
```

Bisakah model yang sudah dilatih menghasilkan kembali data yang dipakainya? Bila nilai teramati
berada jauh di ekor replikasi, modelnya salah — sebagus apa pun konvergensi rantainya.

## Inferensi variasional

```csharp
var fit = model.FitVariational(iterations: 2000, learningRate: 0.05, monteCarloSamples: 8);
fit.Means["theta"];  fit.StandardDeviations["theta"];  fit.EvidenceLowerBound;
```

VI mengubah integrasi menjadi optimasi, sehingga konvergen dalam milidetik ketika MCMC memerlukan
detik. Harganya adalah asumsi mean-field: parameter dianggap saling bebas, sehingga VI secara
sistematis **meremehkan varians posterior** dan tidak dapat merepresentasikan korelasi. Gunakan
untuk iterasi cepat, lalu konfirmasi dengan MCMC.

Parameter berbatas dipetakan ke seluruh garis bilangan riil sebelum dicocokkan — logit untuk
parameter Beta pada (0,1), log untuk skala pada (0,∞) — dengan log Jacobian disertakan dalam fungsi
objektif. Tanpa transformasi itu, pengoptimal akan melangkah keluar dari support dan hasilnya tak
bermakna.

Gradien memakai beda hingga terpusat alih-alih diferensiasi otomatis, yang menjaga implementasi
bebas dependensi dan cukup akurat untuk jumlah parameter yang menjadi sasaran metode ini.

## Bayesian network

```csharp
using Gravicode.Science.GraviProb.Models;

var network = new BayesianNetwork()
    .AddVariable("rain", 0.8, 0.2)
    .AddVariable("sprinkler", 2, ["rain"], [[0.6, 0.4], [0.99, 0.01]])
    .AddVariable("wet", 2, ["rain", "sprinkler"],
    [
        [1.00, 0.00],   // tanpa hujan, tanpa penyiram
        [0.10, 0.90],   // tanpa hujan, ada penyiram
        [0.20, 0.80],   // hujan, tanpa penyiram
        [0.01, 0.99],   // keduanya
    ]);

network.Infer("wet");                                                  // 0,4484
network.Infer("rain", new Dictionary<string, int> { ["wet"] = 1 });     // 0,3577
network.Infer("rain", new() { ["wet"] = 1, ["sprinkler"] = 1 });        // 0,0068
network.Sample(rng);
```

Pasangan terakhir itu adalah **explaining away**: rumput basah menaikkan peluang hujan dari 0,20
menjadi 0,36, tetapi begitu penyiram menjelaskannya, peluang hujan turun ke 0,007.

Faktorisasi adalah intinya — distribusi gabungan atas *n* variabel biner punya `2^n - 1` parameter
bebas, sedangkan hasil kali kondisional per variabel jauh lebih kecil secara eksponensial bila
grafnya jarang. Inferensinya berupa enumerasi eksak: benar untuk jaringan kecil, eksponensial di
luar itu. Baris yang jumlahnya bukan satu ditolak saat konstruksi.

## Hidden Markov model

```csharp
var hmm = new HiddenMarkovModel(initial, transitions, emissions);

hmm.LogLikelihood(observations);      // rekursi forward, penskalaan per langkah
hmm.Viterbi(observations);            // jalur paling mungkin, di ruang log
hmm.StatePosteriors(observations);    // forward-backward
hmm.Generate(length, rng);

var learned = HiddenMarkovModel.Random(states: 2, symbols: 3, seed: 42);
learned.Fit(sequences, iterations: 50);   // Baum-Welch (EM)
```

Ketiga algoritma bekerja di ruang log atau dengan normalisasi per langkah, karena probabilitas
mentahnya underflow dalam beberapa puluh langkah waktu.

Viterbi **bukan** sama dengan mengambil state paling mungkin di tiap langkah secara terpisah: ia
memaksimalkan probabilitas gabungan seluruh jalur, sehingga tidak mungkin mengembalikan urutan yang
memuat transisi yang dilarang model.

Baum-Welch menjamin likelihood tidak menurun, tetapi hanya menemukan optimum **lokal** — itulah
sebabnya parameter awal penting dan pencocokan sungguhan memakai beberapa restart acak.

## Regresi linear Bayesian

```csharp
var model = new BayesianLinearRegression(priorPrecision: 1e-3, noisePrecision: 1.0).Fit(x, y);

model.CoefficientMeans;  model.CoefficientCovariance;
model.Predict(newX);
var (mean, deviation) = model.PredictWithUncertainty(newX);
var (lower, upper) = model.PredictInterval(newX, mass: 0.95);
model.SampleCoefficients(draws: 1000);
```

Prior normal-inverse-gamma bersifat konjugat, sehingga posteriornya berbentuk tertutup — tanpa
sampling, dan jawabannya eksak. Keunggulannya dibanding least squares adalah setiap koefisien
membawa posterior penuh, dan prediksinya membawa interval yang **melebar di wilayah yang jarang
dilihat model** — sesuatu yang tak bisa diungkapkan oleh estimasi titik.

## Kesalahan yang sering terjadi

| Gejala | Penyebab |
|---|---|
| "Could not find a starting point with finite posterior density" | Prior tidak memberi massa di tempat likelihood terdefinisi |
| `RHat` jauh di atas 1 | Belum konvergen — perpanjang sampling, atau ubah parameterisasi |
| Penerimaan mendekati 0 atau 1 | Warmup terlalu pendek untuk skala menyesuaikan diri |
| Varians VI terlihat terlalu kecil | Wajar — mean-field meremehkan sebaran secara desain |
| Hasil VI di luar rentang parameter | Sudah diperbaiki: parameter ditransformasi; periksa `SupportBounds` pada distribusi khusus |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
