# Gravicode.Science vs the Python stack

- .NET   : .NET 10.0.11, 8 processors, SIMD width 4
- Python : Python 3.12.10 (numpy 2.4.4, pandas 3.0.3, scipy 1.17.1, networkx 3.6.1)

### Dense linear algebra

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| 256x256 matrix product | 5.22 ms | 0.41 ms | Python 12.8x |
| 512x512 matrix product | 27.73 ms | 3.07 ms | Python 9.0x |
| 1024x1024 matrix product | 169.69 ms | 27.76 ms | Python 6.1x |
| LU factorisation, 256x256 | 49.36 ms | 3.28 ms | Python 15.0x |
| QR factorisation, 256x256 | 115.42 ms | 10.36 ms | Python 11.1x |
| Cholesky factorisation, 256x256 | 12.66 ms | 1.23 ms | Python 10.3x |
| SVD, 256x256 | 1,794.65 ms | 27.11 ms | Python 66.2x |
| symmetric eigen, 256x256 | 2,739.93 ms | 19.74 ms | Python 138.8x |
| solve Ax=b, 256x256 | 44.03 ms | 6.18 ms | Python 7.1x |
| matrix inverse, 256x256 | 121.99 ms | 9.90 ms | Python 12.3x |

### Array and statistics

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| element-wise add, 1,000,000 elements | 15.25 ms | 4.80 ms | Python 3.2x |
| element-wise add, 10,000,000 elements | 97.03 ms | 47.21 ms | Python 2.1x |
| sparse matrix-vector, 2000x2000 @ 1% | 0.17 ms | 0.06 ms | Python 3.0x |
| dense matrix-vector, 2000x2000 | 5.38 ms | 1.97 ms | Python 2.7x |
| 1,000,000 normal deviates | 17.01 ms | 16.22 ms | parity |
| mean + std over 1,000,000 | 12.28 ms | 7.62 ms | Python 1.6x |

### DataFrames

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| read 200,000-row CSV | 371.37 ms | 135.88 ms | Python 2.7x |
| group-by mean, 500 groups | 39.88 ms | 5.86 ms | Python 6.8x |
| group-by two keys | 51.09 ms | 31.04 ms | Python 1.6x |
| sort by a numeric column | 47.85 ms | 27.24 ms | Python 1.8x |
| rolling mean, window 30 | 1.03 ms | 8.83 ms | **.NET 8.6x** |
| filter a numeric column | 6.39 ms | 6.06 ms | parity |
| describe all numeric columns | 112.31 ms | 35.42 ms | Python 3.2x |

### Machine learning

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| logistic regression, 20k x 20, 100 iterations | 346.95 ms | 31.16 ms | Python 11.1x |
| random forest fit, 50 trees, depth 8 | 1,906.19 ms | 2,584.07 ms | **.NET 1.4x** |
| random forest predict, 20k rows | 134.54 ms | 98.20 ms | Python 1.4x |
| k-means, k=5, 3 restarts | 2,541.98 ms | 497.53 ms | Python 5.1x |
| PCA to 5 components, 20k x 20 | 375.40 ms | 5.92 ms | Python 63.5x |
| kNN predict, 2k queries against 5k points | 609.92 ms | 55.17 ms | Python 11.1x |

### Text

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| TF-IDF fit+transform, 20k documents | 262.50 ms | 691.75 ms | **.NET 2.6x** |
| regex tokenize 20k documents | 153.92 ms | 359.60 ms | **.NET 2.3x** |

### Graphs

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| PageRank, 50k nodes | 276.32 ms | 313.02 ms | parity |
| breadth-first search, 50k nodes | 9.54 ms | 376.63 ms | **.NET 39.5x** |
| connected components, 50k nodes | 9.28 ms | 38.60 ms | **.NET 4.2x** |
| Dijkstra from one source, 50k nodes | 32.34 ms | 180.42 ms | **.NET 5.6x** |

### Probabilistic

| Operation | Gravicode.Science | Python stack | Ratio |
|---|---:|---:|---|
| Metropolis-Hastings, 4 chains x 5000 draws | 21.37 ms | 279.18 ms | **.NET 13.1x** |
| 1,000,000 normal log densities | 2.08 ms | 275.67 ms | **.NET 132.2x** |

**Tally**: .NET faster on 9, Python faster on 25, parity on 3.
