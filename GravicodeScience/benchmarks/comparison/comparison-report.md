# Gravicode.Science vs the Python stack

- .NET   : .NET 10.0.12, 8 processors, SIMD width 4
- Python : Python 3.12.10 (numpy 2.4.4, pandas 3.0.3, scipy 1.17.1, networkx 3.6.1)

### Dense linear algebra

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| 256x256 matrix product | 12.14 ms | 0.35 ms | Python 34.7x |
| 512x512 matrix product | 11.23 ms | 2.47 ms | Python 4.5x |
| 1024x1024 matrix product | 72.59 ms | 22.07 ms | Python 3.3x |
| LU factorisation, 256x256 | 87.20 ms | 2.63 ms | Python 33.1x |
| QR factorisation, 256x256 | 119.36 ms | 8.04 ms | Python 14.8x |
| Cholesky factorisation, 256x256 | 13.13 ms | 1.04 ms | Python 12.6x |
| SVD, 256x256 | 339.75 ms | 24.46 ms | Python 13.9x |
| symmetric eigen, 256x256 | 118.17 ms | 15.10 ms | Python 7.8x |
| solve Ax=b, 256x256 | 61.29 ms | 3.81 ms | Python 16.1x |
| matrix inverse, 256x256 | 142.83 ms | 5.31 ms | Python 26.9x |

### Array and statistics

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| element-wise add, 1,000,000 elements | 3.84 ms | 3.27 ms | Python 1.2x |
| element-wise add, 10,000,000 elements | 29.57 ms | 33.39 ms | parity |
| sparse matrix-vector, 2000x2000 @ 1% | 0.17 ms | 0.05 ms | Python 3.2x |
| dense matrix-vector, 2000x2000 | 5.63 ms | 1.81 ms | Python 3.1x |
| 1,000,000 normal deviates | 14.40 ms | 13.01 ms | parity |
| mean + std over 1,000,000 | 12.55 ms | 6.28 ms | Python 2.0x |

### DataFrames

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| read 200,000-row CSV | 402.02 ms | 136.45 ms | Python 2.9x |
| group-by mean, 500 groups | 44.52 ms | 5.04 ms | Python 8.8x |
| group-by two keys | 49.90 ms | 17.96 ms | Python 2.8x |
| sort by a numeric column | 69.62 ms | 23.50 ms | Python 3.0x |
| rolling mean, window 30 | 1.00 ms | 6.97 ms | **.NET 7.0x** |
| filter a numeric column | 6.72 ms | 4.80 ms | Python 1.4x |
| describe all numeric columns | 111.75 ms | 27.19 ms | Python 4.1x |

### Machine learning

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| logistic regression, 20k x 20, 100 iterations | 355.78 ms | 23.23 ms | Python 15.3x |
| random forest fit, 50 trees, depth 8 | 1,849.93 ms | 2,217.49 ms | **.NET 1.2x** |
| random forest predict, 20k rows | 106.92 ms | 78.58 ms | Python 1.4x |
| k-means, k=5, 3 restarts | 2,605.78 ms | 312.56 ms | Python 8.3x |
| PCA to 5 components, 20k x 20 | 70.90 ms | 3.13 ms | Python 22.6x |
| kNN predict, 2k queries against 5k points | 523.96 ms | 36.14 ms | Python 14.5x |

### Text

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| TF-IDF fit+transform, 20k documents | 254.60 ms | 553.10 ms | **.NET 2.2x** |
| regex tokenize 20k documents | 150.22 ms | 334.28 ms | **.NET 2.2x** |

### Graphs

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| PageRank, 50k nodes | 258.40 ms | 278.90 ms | parity |
| breadth-first search, 50k nodes | 9.24 ms | 346.94 ms | **.NET 37.5x** |
| connected components, 50k nodes | 9.12 ms | 38.65 ms | **.NET 4.2x** |
| Dijkstra from one source, 50k nodes | 23.40 ms | 181.68 ms | **.NET 7.8x** |

### Probabilistic

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| Metropolis-Hastings, 4 chains x 5000 draws | 19.30 ms | 283.84 ms | **.NET 14.7x** |
| 1,000,000 normal log densities | 2.08 ms | 270.76 ms | **.NET 130.3x** |

### SciNet vs SciPy

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| erf over 1,000,000 values | 10.58 ms | 47.56 ms | **.NET 4.5x** |
| complex FFT, 65,536 points | 2.02 ms | 1.98 ms | parity |
| real FFT, 2^20 points | 23.06 ms | 31.83 ms | **.NET 1.4x** |
| DCT-II, 65,536 points | 0.93 ms | 0.59 ms | Python 1.6x |
| quad of log(x)/sqrt(x) on [0,1] | 0.01 ms | 0.08 ms | **.NET 9.8x** |
| solve_ivp Lorenz 0..100, RK45, rtol 1e-9 | 5.47 ms | 1,004.49 ms | **.NET 183.6x** |
| solve_ivp Van der Pol mu=1000, BDF | 0.97 ms | 130.43 ms | **.NET 134.8x** |
| minimize BFGS, Rosenbrock-100 with gradient | 22.80 ms | 317.24 ms | **.NET 13.9x** |
| curve_fit exponential decay, 10,000 points | 5.56 ms | 3.30 ms | Python 1.7x |
| linear_sum_assignment 500x500 | 14.55 ms | 13.17 ms | parity |
| cdist 2000x2000 in 64-D, euclidean | 25.74 ms | 105.56 ms | **.NET 4.1x** |
| KD-tree build + 5-NN query, 100k points in 3-D | 76.98 ms | 101.83 ms | **.NET 1.3x** |
| Delaunay triangulation, 100k points | 158.60 ms | 585.46 ms | **.NET 3.7x** |
| hierarchical clustering, Ward, 2000 points | 44.62 ms | 75.10 ms | **.NET 1.7x** |
| cubic spline, evaluate 1,000,000 points | 19.57 ms | 71.63 ms | **.NET 3.7x** |
| RegularGridInterpolator linear, 1,000,000 points in 3-D | 109.88 ms | 505.74 ms | **.NET 4.6x** |
| conjugate gradient, 2-D Poisson 200x200, rtol 1e-8 | 136.02 ms | 257.40 ms | **.NET 1.9x** |
| sparse LU factor + solve, 2-D Poisson 100x100 | 16.42 ms | 37.17 ms | **.NET 2.3x** |
| matrix exponential, 200x200 | 6.42 ms | 8.19 ms | **.NET 1.3x** |
| gaussian_filter sigma 3, 2048x2048 | 55.03 ms | 208.04 ms | **.NET 3.8x** |
| median_filter 5x5, 1024x1024 | 122.32 ms | 387.86 ms | **.NET 3.2x** |
| lfilter Butterworth order 4, 1,000,000 samples | 14.41 ms | 10.06 ms | Python 1.4x |
| sosfiltfilt Butterworth order 8, 1,000,000 samples | 19.06 ms | 24.63 ms | **.NET 1.3x** |
| welch PSD, 10,000,000 samples, nperseg 4096 | 64.96 ms | 671.49 ms | **.NET 10.3x** |
| fftconvolve 1,000,000 x 10,000 | 65.16 ms | 87.72 ms | **.NET 1.3x** |
| find_peaks with prominence, 1,000,000 samples | 37.40 ms | 55.13 ms | **.NET 1.5x** |

**Tally**: .NET faster on 30, Python faster on 28, parity on 5.
