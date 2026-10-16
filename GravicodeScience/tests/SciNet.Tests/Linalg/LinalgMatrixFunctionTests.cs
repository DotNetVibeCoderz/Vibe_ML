using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.LinalgAssert;

namespace Gravicode.Science.Tests.SciNet;

public class LinalgMatrixFunctionTests
{
    private static NdArray Scale(NdArray a, double f)
    {
        var d = a.ToArray();
        for (var i = 0; i < d.Length; i++) d[i] *= f;
        return new NdArray(d, a.Shape.ToArray());
    }

    [Fact]
    public void Expm_SymmetricMatchesEigenClosedForm_AtEveryPadeDegree()
    {
        var s = M(3, 3, 4, 1, 2, 1, 5, 3, 2, 3, 6);
        // Norms from tiny (degree 3) to large (degree 13 with squaring).
        foreach (var f in new[] { 1e-4, 1e-2, 0.08, 0.2, 0.4, 3.0 })
        {
            var a = Scale(s, f);
            var expected = Linalg.FunmSymmetric(a, Math.Exp);
            Close(expected, Linalg.Expm(a), 1e-13, $"scale {f}");
        }
    }

    [Fact]
    public void Expm_JordanBlockIsExact()
    {
        // exp(λI + N) = e^λ (I + N + N²/2) for a 3x3 nilpotent shift N.
        var j = M(3, 3, 0.5, 1, 0, 0, 0.5, 1, 0, 0, 0.5);
        var e = Math.Exp(0.5);
        Close(M(3, 3, e, e, e / 2, 0, e, e, 0, 0, e), Linalg.Expm(j), 1e-15, "jordan");
    }

    [Fact]
    public void Expm_NonNormalAndBadlyScaled_MatchScipy()
    {
        // scipy 1.17.1 expm values.
        Close([2.705000055080607, -2.1098658782121382, 3.0749249105457945, 0.7467962958844766, 1.7492867239306407, 1.6688545284891698, -3.067631053663166, 5.614472573217561, 2.1081887532596406, 2.8803642990137566, 4.626904045483847, -5.565247769114031, 0.6524777256470972, -2.7792836952428486, 4.725196549690374, 2.5184085530970544],
            Linalg.Expm(Scale(A4, 0.25)).ToArray(), 1e-13, "A4/4");
        Close([-36.47671114989274, 62.694563593682496, 162.18352486137383, -68.14167947085008, 11.558954396556102, 57.24744771651494, -62.69456359368252, -19.758977806296297, -18.079512312888365],
            Linalg.Expm(M(3, 3, 3, 6, 9, 0, -3, 12, -6, 3, 1.5)).ToArray(), 1e-12, "N");
        // Moler & Van Loan's classic example: large norm, eigenvalues -1 and -17.
        Close([-0.7357587581447559, 0.5518190996580998, -1.4715175990882672, 1.1036382407155778],
            Linalg.Expm(M(2, 2, -49, 24, -64, 31)).ToArray(), 1e-12, "Moler–Van Loan");

        // Al-Mohy–Higham's badly scaled triangular case: the exact off-diagonal is
        // b (e^a − e^d)/(a − d), written as b e^((a+d)/2) sinh(δ)/δ with δ = (a−d)/2 so the
        // reference itself does not cancel; the triangular fix is what keeps expm accurate here.
        double a = -1, d = -1.0001, b = 1e4;
        var delta = (a - d) / 2;
        var exact12 = b * Math.Exp((a + d) / 2) * Math.Sinh(delta) / delta;
        var t = Linalg.Expm(M(2, 2, a, b, 0, d));
        Assert.True(Math.Abs(t[0, 1] - exact12) <= 1e-13 * Math.Abs(exact12), $"off-diagonal {t[0, 1]} vs {exact12}");
        Close(Math.Exp(a), t[0, 0], 1e-15);
        Close(3678.6104781250074, t[0, 1], 1e-13, "scipy 1.17.1");
    }

    [Fact]
    public void Expm_LargerMatrixUsesBlockedProducts()
    {
        var a = Scale(Random(60, 60, 9), 0.3);
        var e = Linalg.Expm(a);
        var em = Linalg.Expm(Scale(a, -1));
        Identity(Mul(e, em), 1e-11, "exp(A) exp(-A) = I");
    }

    [Fact]
    public void Expm_AboveTwoHundred_UsesNormEstimates()
    {
        var a = Scale(Random(210, 210, 17), 0.15);
        Identity(Mul(Linalg.Expm(a), Linalg.Expm(Scale(a, -1))), 1e-10, "exp(A) exp(-A) = I at n = 210");
    }

    [Fact]
    public void ExpmFrechet_MatchesBlockMatrixIdentity()
    {
        // L(A, E) is the upper-right block of exp([[A, E], [0, A]]), at every Padé degree.
        var a0 = M(3, 3, 1, 2, 3, 0, -1, 4, -2, 1, 0.5);
        var e = M(3, 3, 0, 1, 0, 0, 0, 1, 1, 0, 0);
        foreach (var f in new[] { 1e-3, 0.02, 0.1, 0.25, 0.5, 1.0 })
        {
            var a = Scale(a0, f);
            var big = NdArray.Zeros(6, 6);
            for (var i = 0; i < 3; i++)
                for (var j = 0; j < 3; j++) { big[i, j] = a[i, j]; big[i + 3, j + 3] = a[i, j]; big[i, j + 3] = e[i, j]; }
            var block = Linalg.Expm(big).Slice(Slice.Range(0, 3), Slice.Range(3, 6)).Copy();
            Close(block, Linalg.ExpmFrechet(a, e).Frechet, 1e-12, $"scale {f}");
        }
    }

    [Fact]
    public void ExpmFrechet_MatchesScipyAndFiniteDifference()
    {
        var a = M(3, 3, 1, 2, 3, 0, -1, 4, -2, 1, 0.5);
        var e = M(3, 3, 0, 1, 0, 0, 0, 1, 1, 0, 0);
        var (ex, l) = Linalg.ExpmFrechet(a, e);
        Close(Linalg.Expm(a), ex, 1e-13, "expm part");
        // scipy 1.17.1 expm_frechet
        Close([-0.4314843262945769, 0.8538649154424827, 4.469043866155394, -0.4017516527508534, -0.41395837589522044, 0.6932171042879332, -0.7321493701786601, -0.2596051475098833, 0.1427538477786987],
            l.ToArray(), 1e-12, "Frechet");

        // Small norm (low Padé degree) against a central difference.
        var small = Scale(a, 0.05);
        var (_, ls) = Linalg.ExpmFrechet(small, e);
        const double h = 1e-6;
        var fd = Scale(Sub(Linalg.Expm(Add(small, Scale(e, h))), Linalg.Expm(Sub(small, Scale(e, h)))), 0.5 / h);
        Close(fd, ls, 1e-8, "central difference");
    }

    [Fact]
    public void Logm_Sqrtm_FractionalPower_MatchScipy()
    {
        // scipy 1.17.1 logm/sqrtm/fractional_matrix_power of A4 (has a complex pair).
        Close([1.393519022369908, -0.2508639785281317, 0.02785536281396965, 0.5888432338949194, 0.6972561876619984, 1.9098781981677113, -0.5614840412371412, -0.0811933859114681, 0.4086955533909986, 0.03315036293315349, 2.1471258706679723, -0.737591127288701, 0.2925929486238219, -0.4136114357548654, 0.13866296746196255, 1.9604287843780455],
            Linalg.Logm(A4).ToArray(), 1e-12, "logm");
        var sq = Linalg.Sqrtm(A4);
        Close([2.0066620428064326, -0.3594738357781598, 0.11034790442931319, 0.6746275712303635, 0.7259544964516731, 2.5362838260300773, -0.7607674516762328, 0.15008194995877644, 0.44579368008529185, 0.1147874666159074, 2.8815803613331124, -0.9567651293958532, 0.2743390618575237, -0.5626754603289044, 0.2773629604704869, 2.676703559240304],
            sq.ToArray(), 1e-12, "sqrtm");
        Close(A4, Mul(sq, sq), 1e-12, "sqrtm² = A");
        Close([6.0069958578245295, -4.731839656960547, 2.9308593181723723, 6.149204491382252, 5.983518104358318, 9.536104854205751, -9.129796460002185, 6.174901299952493, 4.282720232142763, 2.888115845629861, 14.427358885962574, -11.556874782601751, 1.833240044473094, -6.936777107587603, 5.4230829783845556, 12.08576750844802],
            Linalg.FractionalMatrixPower(A4, 1.3).ToArray(), 1e-11, "A^1.3");
        Close(sq, Linalg.FractionalMatrixPower(A4, 0.5), 1e-11, "A^0.5 = sqrtm");
        Close(Linalg.Inv(A4), Linalg.FractionalMatrixPower(A4, -1), 1e-13, "A^-1");
    }

    [Theory]
    [InlineData(0.1)]
    [InlineData(1.0)]
    [InlineData(2.5)]
    public void Logm_InvertsExpm(double scale)
    {
        // Spectrum of scale*B kept inside |Im| < pi so log(exp(X)) = X is the principal branch.
        var x = Scale(M(3, 3, 0.2, 0.5, -0.3, -0.4, 0.1, 0.6, 0.3, -0.2, -0.5), scale);
        Close(x, Linalg.Logm(Linalg.Expm(x)), 1e-11, "logm(expm(X))");
    }

    [Fact]
    public void Logm_And_Sqrtm_RejectWhatHasNoRealPrincipalValue()
    {
        var neg = M(2, 2, -1, 0, 0, 2);
        Assert.Throws<LinalgException>(() => Linalg.Logm(neg));
        Assert.Throws<LinalgException>(() => Linalg.Sqrtm(neg));
        var c = Linalg.SqrtmComplex(neg).ToArray();
        Assert.True(Complex.Abs(c[0] - Complex.ImaginaryOne) < 1e-14, $"sqrt(-1) = {c[0]}");
        var lc = Linalg.LogmComplex(neg).ToArray();
        Assert.True(Complex.Abs(lc[0] - new Complex(0, Math.PI)) < 1e-13, $"log(-1) = {lc[0]}");
        Assert.Throws<LinalgSingularMatrixException>(() => Linalg.Logm(M(2, 2, 1, 1, 1, 1)));
        // [[0,1],[0,0]] has no square root at all.
        Assert.Throws<LinalgException>(() => Linalg.Sqrtm(M(2, 2, 0, 1, 0, 0)));
    }

    [Fact]
    public void Funm_TrigAndHyperbolic_MatchScipy()
    {
        var a4 = Scale(A4, 0.25);
        Close(Linalg.Expm(a4), Linalg.Funm(a4, Complex.Exp), 1e-12, "funm(exp) = expm");
        // scipy 1.17.1
        Close([0.5595241066696657, 0.6261043197185203, -0.20874318658611946, -0.951632066254403, -0.7157662167678193, -0.09577726689361676, 1.3068846316601932, -0.636188514350205, -0.4410216437900968, -0.22360650987839434, -0.6459725365021125, 1.404153807411571, -0.1604387566722075, 0.9371218047886531, -0.47446275041173375, -0.5189767141661847],
            Linalg.Cosm(a4).ToArray(), 1e-12, "cosm");
        Close([0.569226034738315, -0.3302021356645858, 0.4719004875355267, 0.18728219097820414, 0.4365644398374638, 0.4918357166479304, -0.4879598624705023, 0.7617980310353628, 0.4275339900725731, 0.43981194992537354, 0.9677837875572635, -0.9466370363826851, 0.19535721621928467, -0.44831431427813345, 0.7361657989989686, 0.5732836888653877],
            Linalg.Tanm(Scale(A4, 0.125)).ToArray(), 1e-12, "tanm");
        Close([0.7711710467692809, -0.02100390971185543, -0.040190034000328126, 0.16844228967293617, 0.26819973722696017, 0.9615427542335874, -0.07418807947741653, -0.19658041631002904, 0.14668733685420446, -0.038575634816530195, 0.983048889132625, -0.1443507458607027, 0.13825421004547686, -0.056068557565901934, -0.03337130637539847, 0.9096574243817428],
            Linalg.Tanhm(a4).ToArray(), 1e-12, "tanhm");
        Close([1.172244076372547, -1.0887589960508428, 1.526950925401745, 0.47903258804417626, 1.0028155635258174, 0.7568747772048035, -1.6180549504906956, 2.7559676080924937, 1.1243015077017744, 1.4313663487618533, 2.264515078104632, -2.891931238825144, 0.38266235668166615, -1.4499232593099372, 2.364174653061744, 1.1818079430499653],
            Linalg.Sinhm(a4).ToArray(), 1e-12, "sinhm");
        var c = Linalg.Cosm(a4);
        var s = Linalg.Sinm(a4);
        Identity(Add(Mul(c, c), Mul(s, s)), 1e-12, "cos² + sin² = I");
        var ch = Linalg.Coshm(a4);
        var sh = Linalg.Sinhm(a4);
        Identity(Sub(Mul(ch, ch), Mul(sh, sh)), 1e-12, "cosh² − sinh² = I");
        // A Jordan block: the Parlett recurrence would silently drop f'(λ); it must refuse instead.
        Assert.Throws<LinalgException>(() => Linalg.Funm(M(2, 2, 1, 1, 0, 1), Complex.Exp));
        Close(Scale(NdArray.Eye(2), Math.E), Linalg.Funm(NdArray.Eye(2), Complex.Exp), 1e-15, "repeated but diagonal");
    }

    [Fact]
    public void Signm_MatchesScipyAndSquaresToIdentity()
    {
        var a = Sub(A4, Scale(NdArray.Eye(4), 6));
        var s = Linalg.Signm(a);
        // scipy 1.17.1 signm(A4 - 6I)
        Close([-1.0572441160233406, -0.3043328904992262, -0.6963912192021241, 1.189811920799709, 0.3060203703254404, 0.23376975517477538, -1.0805352985732002, -0.6877312047517323, -0.16031815071704894, -0.6025556271423811, 0.10110040182054678, -0.27159055724624365, -0.11453676591142893, -0.5633031310203636, -0.8360081960674859, 0.7223739590280169],
            s.ToArray(), 1e-11, "signm");
        Identity(Mul(s, s), 1e-12, "S² = I");
        Close(Mul(a, s), Mul(s, a), 1e-11, "commutes with A");
    }

    [Fact]
    public void FunmSymmetric_HandlesRepeatedEigenvalues()
    {
        var i3 = Scale(NdArray.Eye(3), 4);
        Close(Scale(NdArray.Eye(3), 2), Linalg.FunmSymmetric(i3, Math.Sqrt), 1e-15, "sqrt(4I)");
    }
}
