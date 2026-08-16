"""Merges the two result files into a comparison table.

Usage:
    python compare.py dotnet-results.json python-results.json [--markdown]
"""

from __future__ import annotations

import json
import sys


GROUPS = [
    ("Dense linear algebra", [
        "matmul_256", "matmul_512", "matmul_1024",
        "lu_256", "qr_256", "cholesky_256", "svd_256", "eigh_256",
        "solve_256", "inverse_256",
    ]),
    ("Array and statistics", [
        "elementwise_add_1000000", "elementwise_add_10000000",
        "sparse_matvec_2000", "dense_matvec_2000",
        "random_normal_1m", "statistics_1m",
    ]),
    ("DataFrames", [
        "csv_read_200k", "groupby_mean_200k", "groupby_composite_200k",
        "sort_200k", "rolling_mean_200k", "filter_200k", "describe_200k",
    ]),
    ("Machine learning", [
        "logistic_fit_20k", "randomforest_fit_20k", "randomforest_predict_20k",
        "kmeans_20k", "pca_20k", "knn_predict_2k",
    ]),
    ("Text", ["tfidf_20k", "tokenize_20k"]),
    ("Graphs", ["pagerank_50k", "bfs_50k", "components_50k", "dijkstra_50k"]),
    ("Probabilistic", ["mcmc_4x5000", "logpdf_1m", "logpdf_1m_vectorised"]),
]


def load(path: str) -> tuple[dict, dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, {r["Id"]: r for r in payload["Results"]}


def main() -> None:
    dotnet_payload, dotnet = load(sys.argv[1] if len(sys.argv) > 1 else "dotnet-results.json")
    python_payload, python = load(sys.argv[2] if len(sys.argv) > 2 else "python-results.json")

    print(f"# Gravicode.Science vs the Python stack\n")
    print(f"- .NET   : {dotnet_payload['Runtime']}, {dotnet_payload['Processors']} processors, "
          f"SIMD width {dotnet_payload['SimdWidth']}")
    versions = python_payload.get("Versions", {})
    print(f"- Python : {python_payload['Runtime']} "
          f"(numpy {versions.get('numpy')}, pandas {versions.get('pandas')}, "
          f"scipy {versions.get('scipy')}, networkx {versions.get('networkx')})")
    print()

    dotnet_wins = 0
    python_wins = 0
    parity = 0

    for group, ids in GROUPS:
        print(f"### {group}\n")
        print("| Operation | Gravicode.Science | Python stack | Ratio |")
        print("|---|---:|---:|---|")
        for identifier in ids:
            left = dotnet.get(identifier)
            right = python.get(identifier)
            if left is None or right is None:
                continue

            a = left["MedianMs"]
            b = right["MedianMs"]
            if a < b:
                ratio = b / a
                verdict = f"**.NET {ratio:.1f}x**" if ratio >= 1.15 else "parity"
            else:
                ratio = a / b
                verdict = f"Python {ratio:.1f}x" if ratio >= 1.15 else "parity"

            if verdict == "parity":
                parity += 1
            elif verdict.startswith("**"):
                dotnet_wins += 1
            else:
                python_wins += 1

            print(f"| {left['Operation']} | {a:,.2f} ms | {b:,.2f} ms | {verdict} |")
        print()

    print(f"**Tally**: .NET faster on {dotnet_wins}, Python faster on {python_wins}, "
          f"parity on {parity}.")


if __name__ == "__main__":
    main()
