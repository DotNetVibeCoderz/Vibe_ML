# Gravicode.Science vs the Python stack

- .NET   : .NET 10.0.11, 8 processors, SIMD width 4
- Python : Python 3.12.10 (numpy 2.4.4, pandas 3.0.3, scipy 1.17.1, networkx 3.6.1)

### Dense linear algebra

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| 256x256 matrix product | 2.58 ms | 0.41 ms | Python 6.3x |
| 512x512 matrix product | 15.64 ms | 3.07 ms | Python 5.1x |
| 1024x1024 matrix product | 93.45 ms | 27.76 ms | Python 3.4x |
| LU factorisation, 256x256 | 41.13 ms | 3.28 ms | Python 12.5x |
| QR factorisation, 256x256 | 153.63 ms | 10.36 ms | Python 14.8x |
| Cholesky factorisation, 256x256 | 12.24 ms | 1.23 ms | Python 10.0x |
| SVD, 256x256 | 308.41 ms | 27.11 ms | Python 11.4x |
| symmetric eigen, 256x256 | 110.22 ms | 19.74 ms | Python 5.6x |
| solve Ax=b, 256x256 | 66.78 ms | 6.18 ms | Python 10.8x |
| matrix inverse, 256x256 | 147.48 ms | 9.90 ms | Python 14.9x |

### Array and statistics

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| element-wise add, 1,000,000 elements | 2.52 ms | 4.80 ms | **.NET 1.9x** |
| element-wise add, 10,000,000 elements | 26.70 ms | 47.21 ms | **.NET 1.8x** |
| sparse matrix-vector, 2000x2000 @ 1% | 0.16 ms | 0.06 ms | Python 2.8x |
| dense matrix-vector, 2000x2000 | 5.54 ms | 1.97 ms | Python 2.8x |
| 1,000,000 normal deviates | 13.75 ms | 16.22 ms | **.NET 1.2x** |
| mean + std over 1,000,000 | 11.50 ms | 7.62 ms | Python 1.5x |

### DataFrames

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| read 200,000-row CSV | 308.79 ms | 135.88 ms | Python 2.3x |
| group-by mean, 500 groups | 41.35 ms | 5.86 ms | Python 7.1x |
| group-by two keys | 51.93 ms | 31.04 ms | Python 1.7x |
| sort by a numeric column | 40.30 ms | 27.24 ms | Python 1.5x |
| rolling mean, window 30 | 0.94 ms | 8.83 ms | **.NET 9.4x** |
| filter a numeric column | 6.49 ms | 6.06 ms | parity |
| describe all numeric columns | 117.24 ms | 35.42 ms | Python 3.3x |

### Machine learning

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| logistic regression, 20k x 20, 100 iterations | 359.77 ms | 31.16 ms | Python 11.5x |
| random forest fit, 50 trees, depth 8 | 1,700.45 ms | 2,584.07 ms | **.NET 1.5x** |
| random forest predict, 20k rows | 109.78 ms | 98.20 ms | parity |
| k-means, k=5, 3 restarts | 2,566.71 ms | 497.53 ms | Python 5.2x |
| PCA to 5 components, 20k x 20 | 67.90 ms | 5.92 ms | Python 11.5x |
| kNN predict, 2k queries against 5k points | 613.68 ms | 55.17 ms | Python 11.1x |

### Text

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| TF-IDF fit+transform, 20k documents | 252.59 ms | 691.75 ms | **.NET 2.7x** |
| regex tokenize 20k documents | 154.67 ms | 359.60 ms | **.NET 2.3x** |

### Graphs

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| PageRank, 50k nodes | 281.13 ms | 313.02 ms | parity |
| breadth-first search, 50k nodes | 8.96 ms | 376.63 ms | **.NET 42.0x** |
| connected components, 50k nodes | 9.32 ms | 38.60 ms | **.NET 4.1x** |
| Dijkstra from one source, 50k nodes | 20.67 ms | 180.42 ms | **.NET 8.7x** |

### Probabilistic

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| Metropolis-Hastings, 4 chains x 5000 draws | 41.66 ms | 279.18 ms | **.NET 6.7x** |
| 1,000,000 normal log densities | 2.11 ms | 275.67 ms | **.NET 130.7x** |

**Tally**: .NET faster on 12, Python faster on 22, parity on 3.
