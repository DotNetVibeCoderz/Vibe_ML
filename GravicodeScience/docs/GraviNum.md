# GraviNum

*[Bahasa Indonesia](id/GraviNum.md)* · NumPy for .NET — the numerical foundation everything else builds on.

## NdArray

An `NdArray` is a shared `double[]` buffer plus a shape, strides and an offset. Reshaping,
transposing and slicing produce **views**, so they cost a small header allocation and never move
data. Only `Copy()` and `AsContiguous()` allocate.

```csharp
var a = NdArray.Arange(12).Reshape(3, 4);

a[1, 2];              // 6
a.T[2, 1];            // 6 — the same element, through a view
a.Shape;              // [3, 4]
a.IsContiguous;       // true
```

Because a view aliases the original buffer, writing through one is visible in the other:

```csharp
a.T[0, 1] = 99.0;
Console.WriteLine(a[1, 0]);   // 99
```

### Creation

```csharp
NdArray.Zeros(3, 4);                    NdArray.Ones(2, 2);
NdArray.Full(7.5, 10);                  NdArray.Eye(4);
NdArray.Arange(0, 10, 2);               NdArray.Linspace(0, 1, 11);
NdArray.FromValues([1.0, 2.0, 3.0]);    NdArray.FromArray(new double[,] { { 1, 2 }, { 3, 4 } });
```

### Slicing

`Slice.At` picks one index and **drops** the axis; `Slice.Range` keeps it.

```csharp
a.Slice(Slice.All, Slice.Range(1, 3));   // 3x2 view
a.Slice(Slice.At(1), Slice.All);         // rank 1 — axis 0 removed
a.Slice(Slice.Range(0, 10, 2));          // every second element
a.Slice(Slice.Reversed);                 // reversed view
a.Row(1);  a.Column(2);                  // zero-copy row and column
```

### Reshaping and joining

```csharp
a.Reshape(2, -1);        // one dimension inferred
a.Ravel();               // flat view (or copy when strided)
a.Transpose(1, 0);       // permute axes
a.ExpandDims(0);         // insert a length-1 axis
a.Squeeze();             // drop every length-1 axis
a.BroadcastTo(5, 3, 4);  // stretch without copying

NdArray.Concatenate([x, y], axis: 0);
NdArray.Stack([x, y]);           // new leading axis
NdArray.VStack(x, y);  NdArray.HStack(x, y);
```

## Broadcasting

Shapes are aligned from the right; a dimension of 1 stretches to match.

```csharp
NdArray.Ones(3, 4) + NdArray.Arange(4);    // (3,4) + (4,) -> (3,4)
NdArray.Ones(3, 1) * NdArray.Ones(1, 5);   // (3,1) * (1,5) -> (3,5)
```

Incompatible shapes throw `InvalidOperationException` naming both shapes, rather than producing a
silently wrong result.

## Element-wise maths

`UFunc` dispatches through three tiers automatically: a `Vector<double>` SIMD loop when both
operands are contiguous and identically shaped, a threaded version of that loop above ~24,000
elements, and a strided broadcast walk otherwise.

```csharp
a + b;   a - b;   a * b;   a / b;      // element-wise, with broadcasting
a * 2.0; 1.0 / a;                      // scalars
a.Sqrt(); a.Exp(); a.Log(); a.Abs(); a.Pow(3);
a.Sigmoid(); a.Relu(); a.Tanh();
a.Clip(0, 1);
a.Map(x => x * x + 1);                 // arbitrary function

UFunc.AllClose(x, y, tolerance: 1e-9);
UFunc.Minimum(x, y);  UFunc.Maximum(x, y);
```

> `*` is the **Hadamard** (element-wise) product, matching NumPy. Use `LinAlg.Dot` or `a.Dot(b)`
> for a matrix product.

## Linear algebra

```csharp
LinAlg.Dot(a, b);           // matrix·matrix, matrix·vector, vector·vector
LinAlg.Inner(x, y);         // scalar dot product
LinAlg.Outer(x, y);

LinAlg.Determinant(m);      LinAlg.Trace(m);      LinAlg.Diagonal(m);
LinAlg.Inverse(m);          LinAlg.Solve(a, b);
LinAlg.PseudoInverse(m);    LinAlg.LeastSquares(a, b);
LinAlg.MatrixRank(m);       LinAlg.ConditionNumber(m);
LinAlg.Norm(v);             LinAlg.Norm(v, p: 1);
LinAlg.MatrixPower(m, 5);   LinAlg.Kron(a, b);
```

`Solve` and `Inverse` use LU with partial pivoting. `LeastSquares` goes through the SVD-based
pseudo-inverse, which stays well behaved when features are collinear — the case where the normal
equations quietly return nonsense.

### Decompositions

```csharp
var lu    = Decomposition.Lu(m);              // P A = L U
var qr    = Decomposition.Qr(m);              // A = Q R, Householder
var chol  = Decomposition.Cholesky(spd);      // A = L L'
var svd   = Decomposition.Svd(m);             // A = U diag(S) V'
var eigen = Decomposition.SymmetricEigen(s);  // descending eigenvalues
var (re, im) = Decomposition.Eigenvalues(m);  // general, possibly complex

lu.Solve(b);  qr.Solve(b);  svd.Reconstruct();
```

`Cholesky` throws when the input is not positive definite rather than returning NaNs, so it
doubles as a positive-definiteness check.

## Random numbers

`GraviRandom` is xoshiro256++ seeded through SplitMix64: fast, statistically solid and
reproducible from a seed on every platform.

```csharp
var rng = new GraviRandom(seed: 42);

rng.NextDouble();                       rng.Next(10);
rng.Normal(mean: 0, stdDev: 1);         rng.Uniform(-1, 1);
rng.Gamma(shape: 2, scale: 3);          rng.Beta(2, 5);
rng.Binomial(trials: 100, probability: 0.3);
rng.Poisson(lambda: 4.5);               rng.Exponential(rate: 2);

rng.StandardNormal(1000, 20);           // array overloads
rng.Normal(100, 15, 50_000);
rng.MultivariateNormal(mean, covariance, samples: 1000);

rng.Permutation(100);                   rng.Shuffle(list);
rng.Choice(n: 100, count: 20, replace: false);
```

Sampling uses exact algorithms rather than normal approximations — Marsaglia-Tsang for gamma,
recursive beta-splitting for binomial, transformed rejection for large-λ Poisson — so the tails
stay correct.

## Statistics

```csharp
Statistics.Mean(a);        Statistics.Median(a);      Statistics.Mode(a);
Statistics.Var(a, ddof: 1);  Statistics.Std(a);
Statistics.Percentile(a, 95);  Statistics.Quantile(a, 0.95);
Statistics.Skewness(a);    Statistics.Kurtosis(a);
Statistics.Describe(a);    // count, mean, std, min, quartiles, max

Statistics.Sum(a, axis: 0);      // axis reductions
Statistics.Mean(a, axis: 1);
Statistics.ArgMax(a, axis: 0);

Statistics.Correlation(x, y);          Statistics.SpearmanCorrelation(x, y);
Statistics.CovarianceMatrix(data);     Statistics.CorrelationMatrix(data);
Statistics.Histogram(a, bins: 20);
Statistics.Standardize(a);             Statistics.MinMaxScale(a);
```

`Sum` uses pairwise summation and `Var` uses Welford's algorithm, so neither loses precision on
long or badly scaled inputs.

## Sparse matrices

CSR stores only the non-zeros plus one row pointer per row.

```csharp
var sparse = SparseMatrix.FromDense(dense, threshold: 0);
var built  = SparseMatrix.FromTriplets(rows, cols, triplets);   // duplicates are summed
var viaBuilder = new SparseBuilder(rows, cols).Add(0, 1, 2.5).Build();

sparse.Multiply(vector);                       // sparse · dense vector
sparse.Multiply(matrix, denseIsMatrix: true);  // sparse · dense matrix
sparse.Transpose();  sparse.RowSums();  sparse.ToDense();
sparse.NonZeroCount;  sparse.Density;
```

Below about 10% density this is a clear win; above roughly 25% the index indirection costs more
than the skipped zeros.

## IO

```csharp
NdIO.SaveCsv(a, "matrix.csv");        NdIO.LoadCsv("matrix.csv");
NdIO.SaveBinary(a, "matrix.gnb");     NdIO.LoadBinary("matrix.gnb");
NdIO.SaveJson(a, "matrix.json");      NdIO.ToJson(a);
```

### Memory-mapped arrays

For data that will not fit in RAM. The OS pages in only the regions actually touched.

```csharp
using var mapped = MemoryMappedArray.Create("big.gmm", 1_000_000, 100);
mapped.Set(row: 5, column: 3, value: 42.0);
mapped.Flush();

using var opened = MemoryMappedArray.Open("big.gmm");
opened.ReadRow(42);                       // one row, without loading the file
foreach (var chunk in opened.Chunks(1 << 16)) { /* stream */ }
```

## Compute backends

```csharp
using Gravicode.Science.GraviNum.Compute;

Compute.DescribeDevices();       // what this machine has
Compute.IsGpuAvailable;

Compute.Dot(a, b);               // routed — CPU unless you opt in
Compute.Dot(a, b, Compute.Gpu);  // forced
```

**Automatic GPU dispatch is off by default.** Every array here is `double`, and integrated GPUs
run float64 at a fraction of their float32 rate — measured on an Intel UHD 620, the GPU is 5–8×
*slower* than the CPU for this workload. Benchmark your own hardware, then opt in:

```csharp
Compute.AutomaticGpuDispatch = true;
```

See [benchmarks.md](benchmarks.md) for the measurements.

## Common mistakes

| Symptom | Cause |
|---|---|
| `*` gives the wrong answer for matrices | `*` is element-wise; use `Dot` |
| Modifying one array changes another | They are views over one buffer; use `Copy()` |
| `AsSpan()` throws | The array is strided; call `AsContiguous()` first |
| GPU is slower than CPU | Expected for float64 on integrated hardware — see above |

---

*Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil*
