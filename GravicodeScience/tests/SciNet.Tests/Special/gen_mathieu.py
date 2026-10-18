"""Generates MathieuReference.Generated.cs: Mathieu characteristic values and functions at 40 digits.

Run from this folder:  python gen_mathieu.py [--compare]
mpmath has no Mathieu functions, so this builds them from their definitions (DLMF 28.4, 28.24):
the characteristic values are eigenvalues of the tridiagonal recurrence matrices, found by Sturm
bisection at 40 digits; the Fourier coefficients are the eigenvectors (inverse iteration), normalised
so the functions have norm pi with ce_m(pi/2) of sign (-1)^n etc.; the modified (radial) functions
are the Bessel-product series, evaluated twice with different pivot indices s, which must agree.
With --compare it prints where scipy 1.17.1 disagrees with these values.
"""
import sys
import mpmath as mp
import numpy as np
import scipy
from scipy import special as sp

mp.mp.dps = 40


def matrix(kind, q, n):
    """Diagonal and off-diagonal of the symmetric recurrence matrix, and the coefficient indices."""
    q = mp.mpf(q)
    if kind == "a_even":    # ce_{2n}: A_0, A_2, ...; A_0 scaled by sqrt(2)
        d = [mp.mpf((2 * k) ** 2) for k in range(n)]
        e = [mp.sqrt(2) * q] + [q] * (n - 2)
        idx = [2 * k for k in range(n)]
    elif kind == "a_odd":   # ce_{2n+1}: A_1, A_3, ...
        d = [mp.mpf((2 * k + 1) ** 2) for k in range(n)]
        d[0] += q
        e = [q] * (n - 1)
        idx = [2 * k + 1 for k in range(n)]
    elif kind == "b_odd":   # se_{2n+1}: B_1, B_3, ...
        d = [mp.mpf((2 * k + 1) ** 2) for k in range(n)]
        d[0] -= q
        e = [q] * (n - 1)
        idx = [2 * k + 1 for k in range(n)]
    else:                   # se_{2n+2}: B_2, B_4, ...
        d = [mp.mpf((2 * k + 2) ** 2) for k in range(n)]
        e = [q] * (n - 1)
        idx = [2 * k + 2 for k in range(n)]
    return d, e, idx


def count_below(d, e, x):
    """Sturm count: eigenvalues < x."""
    c, p = 0, d[0] - x
    if p < 0:
        c += 1
    for i in range(1, len(d)):
        if p == 0:
            p = mp.mpf(10) ** (-mp.mp.dps)
        p = d[i] - x - e[i - 1] ** 2 / p
        if p < 0:
            c += 1
    return c


def eigen(kind, q, j):
    """The j-th smallest eigenvalue (j = 0, 1, ...) and its unit eigenvector."""
    n = j + 40 + int(3 * mp.sqrt(abs(q)))
    d, e, idx = matrix(kind, q, n)
    lo = min(d) - 2 * abs(mp.mpf(q)) * 2 - 1
    hi = max(d) + 4 * abs(mp.mpf(q)) + 1
    for _ in range(400):
        mid = (lo + hi) / 2
        if count_below(d, e, mid) > j:
            hi = mid
        else:
            lo = mid
        if hi - lo < mp.mpf(10) ** (-37) * max(1, abs(mid)):
            break
    lam = (lo + hi) / 2
    # inverse iteration with a tridiagonal (Thomas) solve; the shift keeps it non-singular and
    # 60-digit arithmetic makes pivoting unnecessary
    shift = lam + mp.mpf(10) ** (-30) * max(1, abs(lam))
    v = [mp.mpf(1)] * n
    for _ in range(3):
        cp = [mp.mpf(0)] * n
        dp = [mp.mpf(0)] * n
        diag = d[0] - shift
        cp[0] = e[0] / diag if n > 1 else 0
        dp[0] = v[0] / diag
        for i in range(1, n):
            den = d[i] - shift - e[i - 1] * cp[i - 1]
            cp[i] = e[i] / den if i < n - 1 else 0
            dp[i] = (v[i] - e[i - 1] * dp[i - 1]) / den
        x = [mp.mpf(0)] * n
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        nrm = mp.sqrt(sum(t * t for t in x))
        v = [t / nrm for t in x]
    return lam, v, idx


_cache = {}


def coefficients(fn, m, q):
    key = (fn, m, float(q))
    if key not in _cache:
        _cache[key] = _coefficients(fn, m, q)
    return _cache[key]


def _coefficients(fn, m, q):
    """(characteristic value, {index: coefficient}) for ce_m (fn='ce') or se_m (fn='se'), q >= 0."""
    if fn == "ce":
        kind, j = ("a_even", m // 2) if m % 2 == 0 else ("a_odd", (m - 1) // 2)
    else:
        kind, j = ("b_odd", (m - 1) // 2) if m % 2 == 1 else ("b_even", (m - 2) // 2)
    lam, v, idx = eigen(kind, q, j)
    coef = dict(zip(idx, v))
    if kind == "a_even":
        coef[0] = coef[0] / mp.sqrt(2)
    nn = j  # n in DLMF's ce_{2n}, ce_{2n+1}, se_{2n+1}, se_{2n+2}
    # Sign: as q -> 0 the function tends to cos(mx) / sin(mx); fix the sign at pi/2, where it never vanishes.
    if kind == "a_even":
        s = sum(c * (-1) ** (k // 2) for k, c in coef.items()); want = (-1) ** nn
    elif kind == "a_odd":
        s = sum(k * c * (-1) ** ((k - 1) // 2) for k, c in coef.items()); want = (-1) ** nn
    elif kind == "b_odd":
        s = sum(c * (-1) ** ((k - 1) // 2) for k, c in coef.items()); want = (-1) ** nn
    else:
        s = sum(k * c * (-1) ** (k // 2) for k, c in coef.items()); want = (-1) ** (nn + 1)
    if mp.sign(s) != want:
        coef = {k: -c for k, c in coef.items()}
    return lam, coef


def char_value(fn, m, q):
    """a_m(q) (fn='ce') or b_m(q) (fn='se'), any real q, via a_{2n+1}(-q) = b_{2n+1}(q) etc."""
    if q < 0 and m % 2 == 1:
        fn = "se" if fn == "ce" else "ce"
    return coefficients(fn, m, abs(q))[0]


def angular(fn, m, q, xdeg):
    """(value, d/dx in radians) of ce_m or se_m at x degrees; negative q by DLMF 28.2.34-37."""
    x = mp.radians(xdeg)
    if q < 0:
        n = m // 2 if m % 2 == 0 else (m - 1) // 2
        if fn == "ce":
            other = "ce" if m % 2 == 0 else "se"
            sgn = (-1) ** n
        else:
            other = "ce" if m % 2 == 1 else "se"
            sgn = (-1) ** ((m - 1) // 2) if m % 2 == 1 else (-1) ** ((m - 2) // 2)
        v, dv = angular(other, m, -q, mp.degrees(mp.pi / 2 - x))
        return sgn * v, -sgn * dv
    _, coef = coefficients(fn, m, q)
    if fn == "ce":
        return sum(c * mp.cos(k * x) for k, c in coef.items()), sum(-k * c * mp.sin(k * x) for k, c in coef.items())
    return sum(c * mp.sin(k * x) for k, c in coef.items()), sum(k * c * mp.cos(k * x) for k, c in coef.items())


def radial(fn, kind, m, q, z, s=None):
    """(value, d/dz) of Mc^(kind)_m or Ms^(kind)_m (DLMF 28.24.6-9), pivot index s."""
    _, coef = coefficients("ce" if fn == "mc" else "se", m, q)
    h = mp.sqrt(q)
    u1, u2 = h * mp.exp(-z), h * mp.exp(z)
    cache = {}

    def bes(f, n, u):
        key = (f, n, u)
        if key not in cache:
            sgn = (-1) ** (-n) if n < 0 else 1   # J_{-n} = (-1)^n J_n, same for Y
            cache[key] = sgn * (mp.besselj(abs(n), u) if f == "j" else mp.bessely(abs(n), u))
        return cache[key]
    J = lambda n, u: bes("j", n, u)
    Z = (lambda n, u: bes("j", n, u)) if kind == 1 else (lambda n, u: bes("y", n, u))
    dJ = lambda n, u: (J(n - 1, u) - J(n + 1, u)) / 2
    dZ = lambda n, u: (Z(n - 1, u) - Z(n + 1, u)) / 2
    big = max(abs(c) for c in coef.values())
    keys = sorted(k for k, c in coef.items() if abs(c) > big * mp.mpf(10) ** -38)
    if fn == "mc" and m % 2 == 0:
        ks = [k // 2 for k in keys]; off = 0; sign_minus = False
    elif fn == "mc":
        ks = [(k - 1) // 2 for k in keys]; off = 1; sign_minus = False
    elif m % 2 == 1:
        ks = [(k - 1) // 2 for k in keys]; off = 1; sign_minus = True
    else:
        ks = [(k - 2) // 2 for k in keys]; off = 2; sign_minus = True
    n = (m - (0 if fn == "mc" and m % 2 == 0 else 1 if m % 2 == 1 else 2)) // 2
    c = [coef[k] for k in keys]
    if s is None:
        s = max(range(len(c)), key=lambda i: abs(c[i]))
    eps = 2 if (fn == "mc" and m % 2 == 0 and s == 0) else 1
    val = dval = mp.mpf(0)
    for i, l in enumerate(ks):
        sg = (-1) ** (l + n)
        a, b = l - s, l + s + off
        t1 = J(a, u1) * Z(b, u2)
        t2 = J(b, u1) * Z(a, u2)
        d1 = -u1 * dJ(a, u1) * Z(b, u2) + J(a, u1) * u2 * dZ(b, u2)
        d2 = -u1 * dJ(b, u1) * Z(a, u2) + J(b, u1) * u2 * dZ(a, u2)
        if sign_minus:
            val += sg * c[i] * (t1 - t2); dval += sg * c[i] * (d1 - d2)
        else:
            val += sg * c[i] * (t1 + t2); dval += sg * c[i] * (d1 + d2)
    return val / (eps * c[s]), dval / (eps * c[s])


CHAR_Q = [1e-3, 0.5, 1.0, 5.0, 25.0, 100.0, 1000.0, -3.0, -40.0]
CHAR_M = list(range(0, 11)) + [25]
ANG_M = [0, 1, 2, 3, 5, 10]
ANG_Q = [0.5, 5.0, 50.0, -2.0]
ANG_X = [0.0, 17.0, 90.0, 133.3, 270.5]
RAD_M = [0, 1, 2, 5]
RAD_Q = [0.5, 2.0, 10.0, 50.0]
RAD_Z = [0.1, 0.5, 1.0, 2.0]
# Where scipy 1.17.1 is wrong: its characteristic values are off by 3-175% for q >= 5000 (e.g.
# b_5(20000) = +28258, reference -37464.7; the large-q asymptote -2q + 2(2m+1)sqrt(q) - ((2m+1)^2+1)/8
# agrees with the reference), and ce/se lose six digits at m = 15, q = 1000.
EXTRA_CHAR = [(5, 5000.0), (15, 5000.0), (30, 5000.0), (5, 20000.0), (15, 20000.0), (50, 20000.0)]
EXTRA_ANG = [("ce", 15, 1000.0, 10.0), ("se", 15, 1000.0, 10.0), ("ce", 30, 1000.0, 45.0), ("se", 30, 200.0, 80.0)]
EXTRA_RAD = [("mc", 2, 3, 400.0, 0.05), ("ms", 2, 10, 100.0, 0.3), ("mc", 1, 10, 400.0, 1.5)]


def lit(v):
    v = float(v)
    s = repr(v)
    return s if ("." in s or "e" in s or "inf" in s or "nan" in s) else s + ".0"


def main(compare):
    rows = []
    worst = {}

    def note(name, ref, got):
        ref = float(ref)
        got = float(got)
        err = abs(got - ref) / max(abs(ref), 1e-300)
        if err > worst.get(name, (0,))[0]:
            worst[name] = (err, ref, got)

    for m in CHAR_M:
        for q in CHAR_Q:
            a = char_value("ce", m, q)
            rows.append(f"new(\"a\", {m}, {lit(q)}, 0.0, {lit(a)}, 0.0),")
            if compare: note(f"a(m={m},q={q})", a, sp.mathieu_a(m, q))
            if m >= 1:
                b = char_value("se", m, q)
                rows.append(f"new(\"b\", {m}, {lit(q)}, 0.0, {lit(b)}, 0.0),")
                if compare: note(f"b(m={m},q={q})", b, sp.mathieu_b(m, q))
    for m, q in EXTRA_CHAR:
        for fn, f, name in (("ce", sp.mathieu_a, "a"), ("se", sp.mathieu_b, "b")):
            v = char_value(fn, m, q)
            rows.append(f"new(\"{name}\", {m}, {lit(q)}, 0.0, {lit(v)}, 0.0),")
            if compare: note(f"{name}(m={m},q={q})", v, f(m, q))
    for fn, m, q, x in EXTRA_ANG:
        v, d = angular(fn, m, q, x)
        rows.append(f"new(\"{fn}\", {m}, {lit(q)}, {lit(x)}, {lit(v)}, {lit(d)}),")
    for fn, kind, m, q, z in EXTRA_RAD:
        v, d = radial(fn, kind, m, q, mp.mpf(z))
        rows.append(f"new(\"{fn}{kind}\", {m}, {lit(q)}, {lit(z)}, {lit(v)}, {lit(d)}),")
    for fn in ("ce", "se"):
        for m in ANG_M:
            if fn == "se" and m == 0:
                continue
            for q in ANG_Q:
                for x in ANG_X:
                    v, d = angular(fn, m, q, x)
                    rows.append(f"new(\"{fn}\", {m}, {lit(q)}, {lit(x)}, {lit(v)}, {lit(d)}),")
                    if compare:
                        sv, sd = (sp.mathieu_cem if fn == "ce" else sp.mathieu_sem)(m, q, x)
                        note(f"{fn}(m={m},q={q},x={x}) value", v, sv)
                        note(f"{fn}(m={m},q={q},x={x}) deriv", d, sd)
    funcs = {("mc", 1): sp.mathieu_modcem1, ("mc", 2): sp.mathieu_modcem2, ("ms", 1): sp.mathieu_modsem1, ("ms", 2): sp.mathieu_modsem2}
    for (fn, kind), sfun in funcs.items():
        for m in RAD_M:
            if fn == "ms" and m == 0:
                continue
            for q in RAD_Q:
                for z in RAD_Z:
                    v, d = radial(fn, kind, m, q, mp.mpf(z))
                    if z == 0.5:   # the series must not depend on the pivot index
                        v0, d0 = radial(fn, kind, m, q, mp.mpf(z), s=0)
                        assert abs(v - v0) <= mp.mpf(10) ** -25 * max(1, abs(v)), (fn, kind, m, q, z, v, v0)
                    rows.append(f"new(\"{fn}{kind}\", {m}, {lit(q)}, {lit(z)}, {lit(v)}, {lit(d)}),")
                    if compare:
                        sv, sd = sfun(m, q, z)
                        note(f"{fn}{kind}(m={m},q={q},z={z}) value", v, sv)
                        note(f"{fn}{kind}(m={m},q={q},z={z}) deriv", d, sd)
    # Wronskian W{Mc1, Mc2} = W{Ms1, Ms2} = 2/pi (DLMF 28.20.21): checks both normalisations at once.
    for fn in ("mc", "ms"):
        v1, d1 = radial(fn, 1, 2, mp.mpf(3), mp.mpf("0.7"))
        v2, d2 = radial(fn, 2, 2, mp.mpf(3), mp.mpf("0.7"))
        assert abs(v1 * d2 - d1 * v2 - 2 / mp.pi) < mp.mpf(10) ** -25
    if compare:
        bad = sorted(((e, k, r, g) for k, (e, r, g) in worst.items() if e > 1e-10), reverse=True)
        print(f"{len(bad)} values where scipy {scipy.__version__} differs from the reference by more than 1e-10 (relative):")
        for e, k, r, g in bad[:60]:
            print(f"  {k}: reference {r!r}, scipy {g!r}, rel {e:.2e}")
    with open("MathieuReference.Generated.cs", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("// <auto-generated> Mathieu reference values at 40 digits (mpmath, from the definitions) by gen_mathieu.py. Do not edit. </auto-generated>\n")
        fh.write("namespace Gravicode.Science.Tests.SciNet;\n")
        fh.write("/// <summary>One Mathieu value: function, order, q, argument (degrees for ce/se, z for the radial ones), value, derivative.</summary>\n")
        fh.write("internal sealed record MathieuCase(string Function, int M, double Q, double X, double Value, double Derivative);\n")
        fh.write("internal static class MathieuReference\n{\n    public static readonly MathieuCase[] Cases =\n    [\n")
        for r in rows:
            fh.write("        " + r + "\n")
        fh.write("    ];\n}\n")


if __name__ == "__main__":
    main("--compare" in sys.argv)
