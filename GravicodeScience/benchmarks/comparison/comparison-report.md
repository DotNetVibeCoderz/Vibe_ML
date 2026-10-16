# Gravicode.Science vs the Python stack

- .NET   : .NET 10.0.12, 8 processors, SIMD width 4
- Python : Python 3.12.10 (numpy 2.4.4, pandas 3.0.3, scipy 1.17.1, networkx 3.6.1)

### Dense linear algebra

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| 256x256 matrix product | 13.63 ms | 0.35 ms | Python 38.9x |
| 512x512 matrix product | 11.40 ms | 2.47 ms | Python 4.6x |
| 1024x1024 matrix product | 79.10 ms | 22.07 ms | Python 3.6x |
| LU factorisation, 256x256 | 71.07 ms | 2.63 ms | Python 27.0x |
| QR factorisation, 256x256 | 157.66 ms | 8.04 ms | Python 19.6x |
| Cholesky factorisation, 256x256 | 15.34 ms | 1.04 ms | Python 14.8x |
| SVD, 256x256 | 403.56 ms | 24.46 ms | Python 16.5x |
| symmetric eigen, 256x256 | 137.40 ms | 15.10 ms | Python 9.1x |
| solve Ax=b, 256x256 | 54.08 ms | 3.81 ms | Python 14.2x |
| matrix inverse, 256x256 | 150.85 ms | 5.31 ms | Python 28.4x |

### Array and statistics

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| element-wise add, 1,000,000 elements | 4.06 ms | 3.27 ms | Python 1.2x |
| element-wise add, 10,000,000 elements | 33.10 ms | 33.39 ms | parity |
| sparse matrix-vector, 2000x2000 @ 1% | 0.20 ms | 0.05 ms | Python 3.8x |
| dense matrix-vector, 2000x2000 | 6.03 ms | 1.81 ms | Python 3.3x |
| 1,000,000 normal deviates | 19.22 ms | 13.01 ms | Python 1.5x |
| mean + std over 1,000,000 | 13.55 ms | 6.28 ms | Python 2.2x |

### DataFrames

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| read 200,000-row CSV | 383.88 ms | 136.45 ms | Python 2.8x |
| group-by mean, 500 groups | 46.59 ms | 5.04 ms | Python 9.2x |
| group-by two keys | 67.68 ms | 17.96 ms | Python 3.8x |
| sort by a numeric column | 52.70 ms | 23.50 ms | Python 2.2x |
| rolling mean, window 30 | 1.37 ms | 6.97 ms | **.NET 5.1x** |
| filter a numeric column | 9.52 ms | 4.80 ms | Python 2.0x |
| describe all numeric columns | 135.35 ms | 27.19 ms | Python 5.0x |

### Machine learning

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| logistic regression, 20k x 20, 100 iterations | 398.76 ms | 23.23 ms | Python 17.2x |
| random forest fit, 50 trees, depth 8 | 1,999.77 ms | 2,217.49 ms | parity |
| random forest predict, 20k rows | 147.01 ms | 78.58 ms | Python 1.9x |
| k-means, k=5, 3 restarts | 2,990.79 ms | 312.56 ms | Python 9.6x |
| PCA to 5 components, 20k x 20 | 109.53 ms | 3.13 ms | Python 35.0x |
| kNN predict, 2k queries against 5k points | 622.99 ms | 36.14 ms | Python 17.2x |

### Text

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| TF-IDF fit+transform, 20k documents | 302.73 ms | 553.10 ms | **.NET 1.8x** |
| regex tokenize 20k documents | 182.40 ms | 334.28 ms | **.NET 1.8x** |

### Graphs

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| PageRank, 50k nodes | 289.79 ms | 278.90 ms | parity |
| breadth-first search, 50k nodes | 9.60 ms | 346.94 ms | **.NET 36.1x** |
| connected components, 50k nodes | 10.86 ms | 38.65 ms | **.NET 3.6x** |
| Dijkstra from one source, 50k nodes | 36.80 ms | 181.68 ms | **.NET 4.9x** |

### Probabilistic

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| Metropolis-Hastings, 4 chains x 5000 draws | 19.69 ms | 283.84 ms | **.NET 14.4x** |
| 1,000,000 normal log densities | 2.17 ms | 270.76 ms | **.NET 124.7x** |

### SciNet vs SciPy

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| erf over 1,000,000 values | 10.14 ms | 47.56 ms | **.NET 4.7x** |
| complex FFT, 65,536 points | 2.29 ms | 1.98 ms | Python 1.2x |
| real FFT, 2^20 points | 21.88 ms | 31.83 ms | **.NET 1.5x** |
| DCT-II, 65,536 points | 0.98 ms | 0.59 ms | Python 1.7x |
| quad of log(x)/sqrt(x) on [0,1] | 0.01 ms | 0.08 ms | **.NET 9.3x** |
| solve_ivp Lorenz 0..100, RK45, rtol 1e-9 | 5.57 ms | 1,004.49 ms | **.NET 180.3x** |
| solve_ivp Van der Pol mu=1000, BDF | 1.10 ms | 130.43 ms | **.NET 119.1x** |
| minimize BFGS, Rosenbrock-100 with gradient | 28.63 ms | 317.24 ms | **.NET 11.1x** |
| curve_fit exponential decay, 10,000 points | 5.64 ms | 3.30 ms | Python 1.7x |
| linear_sum_assignment 500x500 | 17.20 ms | 13.17 ms | Python 1.3x |
| cdist 2000x2000 in 64-D, euclidean | 27.54 ms | 105.56 ms | **.NET 3.8x** |
| KD-tree build + 5-NN query, 100k points in 3-D | 132.52 ms | 101.83 ms | Python 1.3x |
| Delaunay triangulation, 100k points | 191.81 ms | 585.46 ms | **.NET 3.1x** |
| hierarchical clustering, Ward, 2000 points | 102.52 ms | 75.10 ms | Python 1.4x |
| cubic spline, evaluate 1,000,000 points | 40.92 ms | 71.63 ms | **.NET 1.8x** |
| RegularGridInterpolator linear, 1,000,000 points in 3-D | 120.42 ms | 505.74 ms | **.NET 4.2x** |
| conjugate gradient, 2-D Poisson 200x200, rtol 1e-8 | 147.92 ms | 257.40 ms | **.NET 1.7x** |
| sparse LU factor + solve, 2-D Poisson 100x100 | 273.44 ms | 37.17 ms | Python 7.4x |
| matrix exponential, 200x200 | 29.82 ms | 8.19 ms | Python 3.6x |
| gaussian_filter sigma 3, 2048x2048 | 51.95 ms | 208.04 ms | **.NET 4.0x** |
| median_filter 5x5, 1024x1024 | 130.96 ms | 387.86 ms | **.NET 3.0x** |
| lfilter Butterworth order 4, 1,000,000 samples | 11.10 ms | 10.06 ms | parity |
| sosfiltfilt Butterworth order 8, 1,000,000 samples | 21.81 ms | 24.63 ms | parity |
| welch PSD, 10,000,000 samples, nperseg 4096 | 66.45 ms | 671.49 ms | **.NET 10.1x** |
| fftconvolve 1,000,000 x 10,000 | 70.02 ms | 87.72 ms | **.NET 1.3x** |
| find_peaks with prominence, 1,000,000 samples | 43.44 ms | 55.13 ms | **.NET 1.3x** |

**Tally**: .NET faster on 24, Python faster on 34, parity on 5.
