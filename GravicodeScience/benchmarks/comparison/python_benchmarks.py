"""The Python half of the Gravicode.Science cross-stack comparison.

This mirrors ``Gravicode.Science.Comparison`` operation for operation, at the same
shapes, using the same harness: fixed warmup, fixed repeat count, report the median.

Using the same protocol on both sides is the point. Comparing BenchmarkDotNet's
statistics against ``timeit`` would mean comparing two measurement methodologies as
well as two implementations, and the difference between the two would no longer be
attributable to the code.

Usage:
    python python_benchmarks.py [output.json]
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

RESULTS: list[dict] = []


def measure(identifier: str, library: str, operation: str, action, repeats: int = 7, warmup: int = 2) -> None:
    """Times ``action`` exactly the way the .NET harness does."""
    for _ in range(warmup):
        action()

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        timings.append((time.perf_counter() - start) * 1000.0)

    timings.sort()
    median = timings[repeats // 2]
    RESULTS.append(
        {
            "Id": identifier,
            "Library": library,
            "Operation": operation,
            "MedianMs": median,
            "MinMs": timings[0],
            "MaxMs": timings[-1],
            "Repeats": repeats,
        }
    )
    print(f"  {library:<12}{identifier:<28}{median:>10.3f} ms")


def build_corpus(count: int, words_per_document: int, seed: int = 11) -> list[str]:
    vocabulary = [
        "the", "model", "learns", "from", "data", "and", "produces", "an", "embedding", "vector",
        "graph", "network", "attention", "layer", "token", "sentence", "document", "corpus",
        "bagus", "sangat", "menarik", "hasil", "jaringan", "kata", "kalimat", "analisis",
    ]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(vocabulary), size=(count, words_per_document))
    return [" ".join(vocabulary[i] for i in row) for row in picks]


def write_comparison_csv(path: str, rows: int) -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "id": np.arange(rows),
            "group": np.arange(rows) % 500,
            "label": [f"cat{i % 20}" for i in range(rows)],
            "value": rng.normal(100, 25, rows).round(4),
            "quantity": rng.integers(1, 50, rows),
        }
    )
    frame.to_csv(path, index=False)


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "python-results.json"
    rng = np.random.default_rng(42)

    print("Gravicode.Science comparison harness (Python)")
    print(f"  runtime : Python {platform.python_version()}")
    print(f"  numpy   : {np.__version__}   pandas: {pd.__version__}")
    print(f"  cpu     : {os.cpu_count()} logical processors")
    print()

    # ------------------------------------------------------------ numpy core

    for size in (256, 512, 1024):
        a = rng.standard_normal((size, size))
        b = rng.standard_normal((size, size))
        measure(f"matmul_{size}", "NumPy", f"{size}x{size} matrix product", lambda a=a, b=b: a @ b)

    for length in (1_000_000, 10_000_000):
        a = rng.standard_normal(length)
        b = rng.standard_normal(length)
        measure(
            f"elementwise_add_{length}",
            "NumPy",
            f"element-wise add, {length:,} elements",
            lambda a=a, b=b: a + b,
        )

    m = rng.standard_normal((256, 256)) + np.eye(256) * 256
    spd = m @ m.T + np.eye(256) * 256
    rhs = rng.standard_normal(256)

    measure("lu_256", "SciPy", "LU factorisation, 256x256", lambda: scipy.linalg.lu_factor(m))
    measure("qr_256", "NumPy", "QR factorisation, 256x256", lambda: np.linalg.qr(m))
    measure("cholesky_256", "NumPy", "Cholesky factorisation, 256x256", lambda: np.linalg.cholesky(spd))
    measure("svd_256", "NumPy", "SVD, 256x256", lambda: np.linalg.svd(m), repeats=3)
    measure("eigh_256", "NumPy", "symmetric eigen, 256x256", lambda: np.linalg.eigh(spd), repeats=3)
    measure("solve_256", "NumPy", "solve Ax=b, 256x256", lambda: np.linalg.solve(m, rhs))
    measure("inverse_256", "NumPy", "matrix inverse, 256x256", lambda: np.linalg.inv(m))

    # Sparse, matched to the .NET fixture: 2000x2000 at 1% density.
    size = 2000
    non_zeros = int(size * size * 0.01)
    rows_idx = rng.integers(0, size, non_zeros)
    cols_idx = rng.integers(0, size, non_zeros)
    values = rng.standard_normal(non_zeros)
    dense = np.zeros((size, size))
    dense[rows_idx, cols_idx] = values
    sparse = scipy.sparse.csr_matrix(dense)
    vector = rng.standard_normal(size)

    measure("sparse_matvec_2000", "SciPy", "sparse matrix-vector, 2000x2000 @ 1%", lambda: sparse @ vector)
    measure("dense_matvec_2000", "NumPy", "dense matrix-vector, 2000x2000", lambda: dense @ vector)

    measure(
        "random_normal_1m",
        "NumPy",
        "1,000,000 normal deviates",
        lambda: np.random.default_rng(1).standard_normal(1_000_000),
    )

    data = rng.standard_normal(1_000_000)
    measure("statistics_1m", "NumPy", "mean + std over 1,000,000", lambda: data.mean() + data.std())

    # ------------------------------------------------------------ pandas

    csv_path = os.path.join(tempfile.gettempdir(), "gravicode-comparison-py.csv")
    write_comparison_csv(csv_path, 200_000)
    print(f"  fixture : {csv_path} ({os.path.getsize(csv_path) / 1024 / 1024:.1f} MB)")
    print()

    measure("csv_read_200k", "pandas", "read 200,000-row CSV", lambda: pd.read_csv(csv_path), repeats=3)

    frame = pd.read_csv(csv_path)
    measure(
        "groupby_mean_200k",
        "pandas",
        "group-by mean, 500 groups",
        lambda: frame.groupby("group", observed=True)["value"].mean(),
    )
    measure(
        "groupby_composite_200k",
        "pandas",
        "group-by two keys",
        lambda: frame.groupby(["group", "label"], observed=True)["value"].sum(),
    )
    measure("sort_200k", "pandas", "sort by a numeric column", lambda: frame.sort_values("value"), repeats=3)
    measure(
        "rolling_mean_200k",
        "pandas",
        "rolling mean, window 30",
        lambda: frame["value"].rolling(30).mean(),
    )
    measure("filter_200k", "pandas", "filter a numeric column", lambda: frame[frame["value"] > 100])
    measure("describe_200k", "pandas", "describe all numeric columns", lambda: frame.describe(), repeats=3)

    # ------------------------------------------------------------ scikit-learn

    features = rng.standard_normal((20_000, 20))
    truth = rng.standard_normal(20)
    labels = (features @ truth > 0).astype(float)

    measure(
        "logistic_fit_20k",
        "scikit-learn",
        "logistic regression, 20k x 20, 100 iterations",
        lambda: LogisticRegression(max_iter=100, tol=1e-7).fit(features, labels),
        repeats=3,
    )
    measure(
        "randomforest_fit_20k",
        "scikit-learn",
        "random forest fit, 50 trees, depth 8",
        lambda: RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42).fit(features, labels),
        repeats=3,
    )

    forest = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42).fit(features, labels)
    measure(
        "randomforest_predict_20k",
        "scikit-learn",
        "random forest predict, 20k rows",
        lambda: forest.predict(features),
        repeats=3,
    )
    measure(
        "kmeans_20k",
        "scikit-learn",
        "k-means, k=5, 3 restarts",
        lambda: KMeans(n_clusters=5, n_init=3, random_state=42).fit(features),
        repeats=3,
    )
    measure(
        "pca_20k",
        "scikit-learn",
        "PCA to 5 components, 20k x 20",
        lambda: PCA(n_components=5).fit_transform(features),
        repeats=3,
    )

    knn_train = rng.standard_normal((5_000, 20))
    knn_labels = np.arange(5_000) % 3
    knn = KNeighborsClassifier(n_neighbors=5).fit(knn_train, knn_labels)
    knn_query = rng.standard_normal((2_000, 20))
    measure(
        "knn_predict_2k",
        "scikit-learn",
        "kNN predict, 2k queries against 5k points",
        lambda: knn.predict(knn_query),
        repeats=3,
    )

    # ------------------------------------------------------------ text

    documents = build_corpus(20_000, 40)
    measure(
        "tfidf_20k",
        "scikit-learn",
        "TF-IDF fit+transform, 20k documents",
        lambda: TfidfVectorizer(min_df=2).fit(documents),
        repeats=3,
    )

    import re

    pattern = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)

    def tokenize_all():
        total = 0
        for document in documents:
            total += len(pattern.findall(document.lower()))
        return total

    measure("tokenize_20k", "Python re", "regex tokenize 20k documents", tokenize_all, repeats=3)

    # ------------------------------------------------------------ graphs

    graph = nx.barabasi_albert_graph(50_000, 3, seed=42)
    measure("pagerank_50k", "NetworkX", "PageRank, 50k nodes", lambda: nx.pagerank(graph), repeats=3)
    measure(
        "bfs_50k",
        "NetworkX",
        "breadth-first search, 50k nodes",
        lambda: list(nx.bfs_tree(graph, 0)),
        repeats=3,
    )
    measure(
        "components_50k",
        "NetworkX",
        "connected components, 50k nodes",
        lambda: list(nx.connected_components(graph)),
        repeats=3,
    )
    measure(
        "dijkstra_50k",
        "NetworkX",
        "Dijkstra from one source, 50k nodes",
        lambda: nx.single_source_dijkstra_path_length(graph, 0),
        repeats=3,
    )

    # ------------------------------------------------------------ Bayesian

    # PyMC is not assumed to be installed - it is a heavy dependency and would also
    # be measuring a compiled backend rather than Python. This is the same
    # random-walk Metropolis algorithm the .NET side runs, written in NumPy, so the
    # comparison stays like for like.
    observations = rng.normal(5.0, 2.0, 500)

    def log_posterior(mu: float, sigma: float) -> float:
        if sigma <= 0:
            return -np.inf
        prior = -0.5 * (mu / 10.0) ** 2 - np.log(10.0)
        prior += 0.5 * np.log(2 / np.pi) - np.log(5.0) - sigma * sigma / 50.0
        residual = observations - mu
        likelihood = -0.5 * np.sum(residual * residual) / (sigma * sigma)
        likelihood -= observations.size * (np.log(sigma) + 0.5 * np.log(2 * np.pi))
        return prior + likelihood

    def metropolis():
        draws = []
        for chain in range(4):
            generator = np.random.default_rng(42 + chain)
            mu, sigma = 0.0, 1.0
            current = log_posterior(mu, sigma)
            scale_mu, scale_sigma = 1.0, 1.0
            accepted = 0
            for iteration in range(5_000):
                candidate_mu = mu + generator.normal(0, scale_mu)
                candidate_sigma = sigma + generator.normal(0, scale_sigma)
                proposed = log_posterior(candidate_mu, candidate_sigma)
                if proposed - current >= 0 or np.log(generator.random()) < proposed - current:
                    mu, sigma, current = candidate_mu, candidate_sigma, proposed
                    accepted += 1
                if iteration < 2_500 and (iteration + 1) % 50 == 0:
                    rate = accepted / 50.0
                    factor = np.exp((rate - 0.234) * 1.5)
                    scale_mu = float(np.clip(scale_mu * factor, 1e-6, 1e4))
                    scale_sigma = float(np.clip(scale_sigma * factor, 1e-6, 1e4))
                    accepted = 0
                if iteration >= 2_500:
                    draws.append(mu)
        return draws

    measure(
        "mcmc_4x5000",
        "NumPy",
        "Metropolis-Hastings, 4 chains x 5000 draws",
        metropolis,
        repeats=3,
    )

    grid = np.arange(1_000_000) * 1e-6

    def logpdf_loop():
        # A scalar loop, matching what the .NET side measures.
        total = 0.0
        constant = -0.5 * np.log(2 * np.pi)
        for value in grid:
            total += constant - 0.5 * value * value
        return total

    measure("logpdf_1m", "Python loop", "1,000,000 normal log densities", logpdf_loop, repeats=3)

    measure(
        "logpdf_1m_vectorised",
        "NumPy",
        "1,000,000 normal log densities (vectorised)",
        lambda: float(np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * grid * grid)),
    )

    # ------------------------------------------------------------ output

    payload = {
        "Stack": "python",
        "Runtime": f"Python {platform.python_version()}",
        "Processors": os.cpu_count(),
        "SimdWidth": 0,
        "TimestampUtc": datetime.now(timezone.utc).isoformat(),
        "Versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "networkx": nx.__version__,
        },
        "Results": RESULTS,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print()
    print(f"Wrote {len(RESULTS)} measurements to {output_path}")
    os.remove(csv_path)


if __name__ == "__main__":
    main()
