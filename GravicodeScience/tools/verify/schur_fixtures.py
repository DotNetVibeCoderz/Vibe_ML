"""Reference eigenvalues for SciNet's Schur/Eig tests (tests/SciNet.Tests/Linalg/LinalgMultishiftTests.cs).

The matrices are defined by formula so the C# tests rebuild them exactly; the eigenvalues come
from scipy.linalg.schur (LAPACK dgees via SciPy's bundled OpenBLAS), read off the diagonal blocks
of T. Not eigvals: dgeev balances first, and on Frank's ill-conditioned small eigenvalues the two
differ by ~5e-8 - the unbalanced Schur form is the like-for-like reference. Prints C# literals.
Run:  python tools/verify/schur_fixtures.py
Dibuat oleh Gravicode Studios, dipimpin oleh Kang Fadhil.
"""
import numpy as np
import scipy
import scipy.linalg as sl


def frank(n):
    return np.array([[n - max(i, j) if j >= i - 1 else 0.0 for j in range(n)] for i in range(n)], float)


def grcar(n, k=3):
    a = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == i - 1:
                a[i, j] = -1.0
            elif i <= j <= i + k:
                a[i, j] = 1.0
    return a


def graded(n):
    return np.array([[(((7 * i + 3 * j) % 11) - 5 + 0.5) * 2.0 ** (-(i + j)) for j in range(n)] for i in range(n)])


def hashed(n):
    return np.array([[(((i + 1) * (j + 3) * (i + j + 5)) % 101) / 50.0 - 1.0 for j in range(n)] for i in range(n)])


def emit(name, a):
    t, _ = sl.schur(a, output="real")
    w = sl.eigvals(t)  # T is quasi-triangular: its 1x1/2x2 blocks give the eigenvalues exactly
    w = sorted(w, key=lambda z: (round(z.real, 9), z.imag))
    re = ", ".join(repr(float(z.real)) for z in w)
    im = ", ".join(repr(float(z.imag)) for z in w)
    print(f"    // {name}: scipy {scipy.__version__} schur")
    print(f"    private static readonly double[] {name}Re = [{re}];")
    print(f"    private static readonly double[] {name}Im = [{im}];")


emit("Frank12", frank(12))
emit("Grcar20", grcar(20))
emit("Graded16", graded(16))
emit("Hashed40", hashed(40))
emit("Hashed120", hashed(120))  # n >= 75: the multishift path
