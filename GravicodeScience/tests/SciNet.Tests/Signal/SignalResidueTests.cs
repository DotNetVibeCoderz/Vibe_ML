using System.Numerics;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>residue/residuez/invres/invresz/unique_roots against scipy 1.17.1.</summary>
public class SignalResidueTests
{
    private static Complex C(double re, double im = 0) => new(re, im);

    /// <summary>Each scipy (pole, residue) pair must appear in the result; repeated-pole blocks keep their order.</summary>
    private static void SamePairs(Complex[] eR, Complex[] eP, Complex[] r, Complex[] p, double tol)
    {
        Assert.Equal(eP.Length, p.Length);
        Assert.Equal(eR.Length, r.Length);
        var used = new bool[p.Length];
        for (var i = 0; i < eP.Length; i++)
        {
            var found = false;
            for (var j = 0; j < p.Length && !found; j++)
                if (!used[j] && (p[j] - eP[i]).Magnitude < tol && (r[j] - eR[i]).Magnitude < tol) { used[j] = true; found = true; }
            Assert.True(found, $"pair {i}: pole {eP[i]}, residue {eR[i]} not found in [{string.Join(", ", p)}] / [{string.Join(", ", r)}]");
        }
    }

    private static void Close(double[] e, Complex[] a, double tol, string what)
    {
        Assert.Equal(e.Length, a.Length);
        for (var i = 0; i < e.Length; i++) Assert.True((a[i] - e[i]).Magnitude < tol, $"{what}[{i}]: {e[i]} vs {a[i]}");
    }

    [Fact]
    public void Residue_SimplePolesAndDirectTerm()
    {
        var (r, p, k) = Signal.Residue([1, 2, 3], [1, 3, 2]);
        Assert.Equal([C(-1), C(-2)], p.Select(v => C(Math.Round(v.Real, 12), Math.Round(v.Imaginary, 12))).ToArray());
        SamePairs([C(2), C(-3)], [C(-1), C(-2)], r, p, 1e-12);
        Assert.Equal([1.0], k);
        var (b, a) = Signal.Invres(r, p, k);
        Close([1, 2, 3], b, 1e-12, "b");
        Close([1, 3, 2], a, 1e-12, "a");
    }

    [Fact]
    public void Residue_RepeatedPoles()
    {
        // (s² + 1)/(s − 1)³: scipy gives residues [1, 2, 2] on the triple pole 1 (powers 1, 2, 3).
        // A triple root is only determined to ~ε^(1/3) ≈ 6e-6 by the companion eigenvalues, so the
        // averaged pole (and the residues) carry that error; scipy's 4e-15 is luck of LAPACK's rounding.
        var (r, p, k) = Signal.Residue([1, 0, 1], [1, -3, 3, -1]);
        Assert.Empty(k);
        Close([1, 2, 2], r, 1e-4, "r");
        foreach (var v in p) Assert.True((v - 1).Magnitude < 1e-4);
        var (b, a) = Signal.Invres(r, p, k);
        Close([1, 0, 1], b, 1e-4, "b");
        Close([1, -3, 3, -1], a, 1e-4, "a");

        // (s + 2)/(s + 1)⁴: residues [0, 0, 1, 1].
        var (r4, p4, _) = Signal.Residue([1, 2], [1, 4, 6, 4, 1]);
        Close([0, 0, 1, 1], r4, 1e-3, "r4");
        foreach (var v in p4) Assert.True((v + 1).Magnitude < 1e-3);
    }

    [Fact]
    public void Residue_ComplexPolesMatchScipy()
    {
        var (r, p, k) = Signal.Residue([2, 1, 0, 5], [1, 0.5, 0.3]);
        // scipy 1.17.1 (conjugate pair, +imag first).
        Assert.True(p[0].Imaginary > 0);
        SamePairs([C(-0.3, -5.2837885132385445), C(-0.3, 5.2837885132385445)],
            [C(-0.24999999999999997, 0.48733971724044817), C(-0.24999999999999997, -0.48733971724044817)], r, p, 1e-12);
        Assert.Equal(2, k.Length);
        Assert.True(Math.Abs(k[0] - 2) < 1e-14 && Math.Abs(k[1]) < 1e-14);
        var (b, a) = Signal.Invres(r, p, k);
        Close([2, 1, 0, 5], b, 1e-12, "b");
        Close([1, 0.5, 0.3], a, 1e-12, "a");

        // 1/(s⁴ + s² + 1): four poles on the unit circle; numpy's eigenvalue order is not reproduced, the pairs are.
        var (r2, p2, k2) = Signal.Residue([1], [1, 0, 1, 0, 1]);
        Assert.Empty(k2);
        SamePairs([C(0.2499999999999999, -0.14433756729740668), C(0.2499999999999999, 0.14433756729740668), C(-0.2499999999999999, -0.14433756729740638), C(-0.2499999999999999, 0.14433756729740638)],
            [C(-0.5000000000000003, 0.8660254037844382), C(-0.5000000000000003, -0.8660254037844382), C(0.5000000000000003, 0.8660254037844386), C(0.5000000000000003, -0.8660254037844386)],
            r2, p2, 1e-12);
    }

    [Fact]
    public void Residuez_MatchesScipy()
    {
        var (r, p, k) = Signal.Residuez([1, 2, 3], [1, -0.5, 0.06]);
        SamePairs([C(-172), C(123)], [C(0.2), C(0.3)], r, p, 1e-9);
        Assert.Single(k);
        Assert.True(Math.Abs(k[0] - 50) < 1e-10);
        var (b, a) = Signal.Invresz(r, p, k);
        Close([1, 2, 3], b, 1e-9, "b");
        Close([1, -0.5, 0.06], a, 1e-12, "a");

        // Double pole at 0.5: residues [-8, 6] (powers 1 and 2), k = [4].
        var (r2, p2, k2) = Signal.Residuez([2, 0, 1], [1, -1, 0.25]);
        Close([-8, 6], r2, 1e-6, "r2");
        Close([0.5, 0.5], p2, 1e-6, "p2");
        Assert.True(Math.Abs(k2[0] - 4) < 1e-9);
        var (b2, _) = Signal.Invresz(r2, p2, k2);
        Close([2, 0, 1], b2, 1e-6, "b2");

        var (r3, p3, k3) = Signal.Residuez([1, 1, 1, 1], [1, 0.2]);
        Close([-104], r3, 1e-10, "r3");
        Close([-0.2], p3, 1e-14, "p3");
        Assert.Equal(3, k3.Length);
        Assert.True(Math.Abs(k3[0] - 105) < 1e-10 && Math.Abs(k3[1] + 20) < 1e-10 && Math.Abs(k3[2] - 5) < 1e-10);
    }

    [Fact]
    public void UniqueRoots_GroupsWithinTolerance()
    {
        var (u, m) = Signal.UniqueRoots([C(1), C(1.0005), C(2), C(2, 1e-4), C(3)], 1e-3, ResidueRootType.Avg);
        Assert.Equal([2, 2, 1], m);
        Assert.True((u[0] - 1.00025).Magnitude < 1e-15);
        Assert.True((u[1] - C(2, 5e-5)).Magnitude < 1e-15);
        var (umax, _) = Signal.UniqueRoots([C(1), C(1.0005)], 1e-3, ResidueRootType.Max);
        Assert.Equal(C(1.0005), umax[0]);
        Assert.Throws<ArgumentException>(() => Signal.Residue([1], [0, 0]));
    }
}
