using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class FftTrigTests
{
    private static readonly double[] X7 = [0.5, -1.25, 2.0, 3.5, -0.75, 1.0, 0.25];
    private static readonly double[] X8 = [1.0, -2.0, 0.5, 3.0, -1.5, 2.5, 0.0, -0.25];

    // cos(π p/q) and sin(π p/q) with p reduced mod 2q first, so the reference does not lose
    // accuracy to argument rounding at large N (that was 1e-13 at N = 1000).
    private static double CosPi(long p, long q) => Math.Cos(Math.PI * (p % (2 * q)) / q);
    private static double SinPi(long p, long q) => Math.Sin(Math.PI * (p % (2 * q)) / q);

    /// <summary>scipy's unnormalised definitions, written out directly (O(N²)).</summary>
    private static double[] NaiveDct(double[] x, int type)
    {
        var n = x.Length;
        var y = new double[n];
        for (var k = 0; k < n; k++)
        {
            double s = 0;
            switch (type)
            {
                case 1:
                    s = x[0] + ((k & 1) == 0 ? 1 : -1) * x[n - 1];
                    for (var j = 1; j < n - 1; j++) s += 2 * x[j] * CosPi((long)k * j, n - 1);
                    break;
                case 2:
                    for (var j = 0; j < n; j++) s += 2 * x[j] * CosPi((long)k * (2 * j + 1), 2L * n);
                    break;
                case 3:
                    s = x[0];
                    for (var j = 1; j < n; j++) s += 2 * x[j] * CosPi((long)j * (2 * k + 1), 2L * n);
                    break;
                default:
                    for (var j = 0; j < n; j++) s += 2 * x[j] * CosPi((long)(2 * k + 1) * (2 * j + 1), 4L * n);
                    break;
            }
            y[k] = s;
        }
        return y;
    }

    private static double[] NaiveDst(double[] x, int type)
    {
        var n = x.Length;
        var y = new double[n];
        for (var k = 0; k < n; k++)
        {
            double s = 0;
            switch (type)
            {
                case 1:
                    for (var j = 0; j < n; j++) s += 2 * x[j] * SinPi((long)(k + 1) * (j + 1), n + 1);
                    break;
                case 2:
                    for (var j = 0; j < n; j++) s += 2 * x[j] * SinPi((long)(k + 1) * (2 * j + 1), 2L * n);
                    break;
                case 3:
                    s = ((k & 1) == 0 ? 1 : -1) * x[n - 1];
                    for (var j = 0; j < n - 1; j++) s += 2 * x[j] * SinPi((long)(2 * k + 1) * (j + 1), 2L * n);
                    break;
                default:
                    for (var j = 0; j < n; j++) s += 2 * x[j] * SinPi((long)(2 * k + 1) * (2 * j + 1), 4L * n);
                    break;
            }
            y[k] = s;
        }
        return y;
    }

    public static IEnumerable<object[]> TypesAndLengths()
    {
        foreach (var type in new[] { 1, 2, 3, 4 })
        foreach (var n in new[] { 1, 2, 3, 4, 5, 8, 15, 16, 17, 31, 64, 97, 100, 1000 })
            if (!(type == 1 && n < 2)) yield return [type, n];
    }

    [Theory]
    [MemberData(nameof(TypesAndLengths))]
    public void Dct_MatchesDefinition(int type, int n)
    {
        var x = FftAssert.RandomReal(n, 31 * n + type);
        FftAssert.Close(NaiveDct(x, type), Fft.Dct(x, type), 1e-13, $"DCT-{type} n={n}");
        FftAssert.Close(x, Fft.Idct(Fft.Dct(x, type), type), 1e-13, $"IDCT-{type} n={n}");
    }

    [Theory]
    [MemberData(nameof(TypesAndLengths))]
    public void Dst_MatchesDefinition(int type, int n)
    {
        var x = FftAssert.RandomReal(n, 17 * n + type);
        FftAssert.Close(NaiveDst(x, type), Fft.Dst(x, type), 1e-13, $"DST-{type} n={n}");
        FftAssert.Close(x, Fft.Idst(Fft.Dst(x, type), type), 1e-13, $"IDST-{type} n={n}");
    }

    // Reference values below came from scipy 1.17.1 (scipy.fft.dct / dst) on X7.
    [Theory]
    [InlineData(1, FftNorm.Backward, new[] { 9.75, -0.897114317029974, -7.75, -5.25, 6.75, 6.897114317029974, -3.25 })]
    [InlineData(1, FftNorm.Ortho, new[] { 2.0536236580334934, -0.22908130724829895, -2.1475524262080126, -1.4856511676555053, 2.0382370254167745, 2.0209186927517013, -0.5999902299816164 })]
    [InlineData(1, FftNorm.Forward, new[] { 0.8125, -0.0747595264191645, -0.6458333333333333, -0.4375, 0.5625, 0.5747595264191645, -0.2708333333333333 })]
    [InlineData(2, FftNorm.Backward, new[] { 10.5, -0.6444171498686524, -7.318531669771362, -3.0187109497370033, 7.829416801848523, 8.90419062795115, -4.102051528380114 })]
    [InlineData(2, FftNorm.Ortho, new[] { 1.984313483298443, -0.172227727783561, -1.955959863038503, -0.8067844374013456, 2.0924996579120383, 2.3797450454511946, -1.096319385863628 })]
    [InlineData(2, FftNorm.Forward, new[] { 0.75, -0.04602979641918943, -0.5227522621265258, -0.2156222106955002, 0.5592440572748945, 0.6360136162822251, -0.29300368059857956 })]
    [InlineData(3, FftNorm.Backward, new[] { 7.183169310856499, -4.511828567607109, -7.555235140679853, -5.5, 5.135848395581348, 9.371412841106148, -0.6233668392570326 })]
    [InlineData(3, FftNorm.Ortho, new[] { 1.975134366435123, -1.1504852907262297, -1.9638699110900866, -1.4145852149699325, 1.427964836025405, 2.5599670499364717, -0.11125018007845577 })]
    [InlineData(3, FftNorm.Forward, new[] { 0.5130835222040356, -0.32227346911479354, -0.5396596529057037, -0.39285714285714285, 0.3668463139700963, 0.669386631507582, -0.04452620280407377 })]
    [InlineData(4, FftNorm.Backward, new[] { 6.889140396530277, -6.151821848293326, -5.840623534941347, 1.7677669529663689, 11.350292364658255, 1.8963667050145423, -4.544953885765801 })]
    [InlineData(4, FftNorm.Ortho, new[] { 1.8412002180857336, -1.6441435471988606, -1.5609722994913586, 0.472455591261534, 3.0334932334476736, 0.5068253207035588, -1.214690019944467 })]
    [InlineData(4, FftNorm.Forward, new[] { 0.49208145689501975, -0.4394158463066662, -0.41718739535295335, 0.12626906806902632, 0.810735168904161, 0.13545476464389586, -0.32463956326898574 })]
    public void Dct_MatchesScipy(int type, FftNorm norm, double[] expected)
    {
        FftAssert.Abs(expected, Fft.Dct(X7, type, norm: norm), 1e-13, $"dct{type} {norm}");
        FftAssert.Abs(X7, Fft.Idct(expected, type, norm: norm), 1e-13, $"idct{type} {norm}");
    }

    // scipy 1.17.1, scipy.fft.dst on X7.
    [Theory]
    [InlineData(1, FftNorm.Backward, new[] { 9.530170589232577, -0.25735931288071434, -6.924442672739069, -5.0, 7.782664108447479, 8.742640687119286, -3.7627226295808747 })]
    [InlineData(1, FftNorm.Ortho, new[] { 2.3825426473081444, -0.06433982822017859, -1.7311106681847672, -1.25, 1.9456660271118698, 2.1856601717798214, -0.9406806573952187 })]
    [InlineData(1, FftNorm.Forward, new[] { 0.5956356618270361, -0.016084957055044646, -0.4327776670461918, -0.3125, 0.48641650677796744, 0.5464150429449554, -0.23517016434880467 })]
    [InlineData(2, FftNorm.Backward, new[] { 9.274458669761152, 0.12983941831473533, -7.071552066053894, -6.923664601795024, 6.903989264184951, 6.392066192343615, -2.5 })]
    [InlineData(2, FftNorm.Ortho, new[] { 2.478703342145817, 0.03470104418798303, -1.8899517874219347, -1.8504272000608295, 1.8451687448961152, 1.7083515489521766, -0.472455591261534 })]
    [InlineData(2, FftNorm.Forward, new[] { 0.662461333554368, 0.009274244165338286, -0.5051108618609925, -0.4945474715567875, 0.4931420902989251, 0.45657615659597245, -0.17857142857142855 })]
    [InlineData(3, FftNorm.Backward, new[] { 7.952993693383581, 3.3473501028351706, -6.715186778342462, -4.75, 9.107426448072985, 5.274943245970594, -4.722940014308339 })]
    [InlineData(3, FftNorm.Ortho, new[] { 2.153202779189576, 0.8669411378252085, -1.7670333502794977, -1.2971667068582167, 2.461737910912247, 1.3821120751614555, -1.2345830059277296 })]
    [InlineData(3, FftNorm.Forward, new[] { 0.5680709780988272, 0.2390964359167979, -0.4796561984530329, -0.3392857142857143, 0.6505304605766418, 0.3767816604264709, -0.33735285816488136 })]
    [InlineData(4, FftNorm.Backward, new[] { 7.478679062303501, 5.434112339423441, -3.965621525832857, -8.838834764831844, -0.47579981841918983, 9.79473026780816, 0.3879458636397799 })]
    [InlineData(4, FftNorm.Ortho, new[] { 1.9987610540556788, 1.4523276125259388, -1.0598569339487327, -2.3622779563076706, -0.1271628503724187, 2.617751775571622, 0.10368289331115554 })]
    [InlineData(4, FftNorm.Forward, new[] { 0.5341913615931071, 0.38815088138738857, -0.2832586804166326, -0.6313453403451317, -0.033985701315656425, 0.6996235905577255, 0.027710418831412897 })]
    public void Dst_MatchesScipy(int type, FftNorm norm, double[] expected)
    {
        FftAssert.Abs(expected, Fft.Dst(X7, type, norm: norm), 1e-13, $"dst{type} {norm}");
        FftAssert.Abs(X7, Fft.Idst(expected, type, norm: norm), 1e-13, $"idst{type} {norm}");
    }

    // scipy 1.17.1, norm="ortho" on the even-length X8 (exercises the half-length DCT-IV path).
    [Theory]
    [InlineData(false, 1, new[] { 1.1453571139573864, -0.6220182875904838, -1.4622747829452234, -0.30196498802151683, -0.42350213566707096, 3.544025638002513, 2.0680438781019945, -1.3667635761509922 })]
    [InlineData(false, 2, new[] { 1.14904851942814, -0.3350958205338397, -1.3031634056044568, 0.44551108581434473, 0.44194173824159216, 3.00373298170508, 2.166192544141308, -2.3607348089301317 })]
    [InlineData(false, 3, new[] { 0.9906888269042591, -1.3010860821288686, -1.0986063342501933, -0.27425707454140447, -0.5412360823075125, 2.675031571034017, 3.2601947512777825, -0.8823024512418893 })]
    [InlineData(false, 4, new[] { 1.021846839103652, -1.3847683118763199, -0.33393910405222704, 0.46145905215304134, 1.400034956036425, 3.6191169085539228, -0.5065765920385885, -2.0522088242431984 })]
    [InlineData(true, 1, new[] { 1.436005522806204, -0.6406817153504558, -1.1226827987756236, -0.28921800201902303, -0.09958359002890207, 3.163924251094939, 2.138645501339661, -2.0980260294517454 })]
    [InlineData(true, 2, new[] { 1.5003820164926867, -0.7475441969729406, -1.103413169223596, -1.1490485194281397, -0.9377894632428181, 3.4215205207001294, 1.8863017004710636, -1.14904851942814 })]
    [InlineData(true, 3, new[] { 1.3572733174415768, 0.12347839155900309, -1.3511859954380614, -0.33425827849620693, 0.06595340321876139, 3.751378689353516, 0.6298489986380216, -2.1316022973694655 })]
    [InlineData(true, 4, new[] { 1.2264086529416518, 0.5873585273901526, -1.3951130260477886, -0.7637583000056691, -1.6596958566647042, 1.2647821576009068, 3.7135991142958416, -0.5372914065027455 })]
    public void OrthoEvenLength_MatchesScipy(bool sine, int type, double[] expected)
    {
        var y = sine ? Fft.Dst(X8, type, norm: FftNorm.Ortho) : Fft.Dct(X8, type, norm: FftNorm.Ortho);
        FftAssert.Abs(expected, y, 1e-13, $"{(sine ? "dst" : "dct")}{type}");
    }

    [Fact]
    public void Orthogonalize_IsIndependentOfNorm_AsInScipy()
    {
        // scipy 1.17.1: dct/dst(X7, type, norm="backward", orthogonalize=True), and DCT-I ortho without it on X7[:6].
        FftAssert.Abs([7.4246212024587495, -0.6444171498686524, -7.318531669771362, -3.0187109497370033, 7.829416801848523, 8.90419062795115, -4.102051528380114],
            Fft.Dct(X7, 2, orthogonalize: true), 1e-13, "dct2");
        FftAssert.Abs([7.390276092043047, -4.304721786420563, -7.348128359493305, -5.292893218813452, 5.342955176767896, 9.578519622292696, -0.4162600580704856],
            Fft.Dct(X7, 3, orthogonalize: true), 1e-13, "dct3");
        FftAssert.Abs([9.274458669761152, 0.12983941831473533, -7.071552066053894, -6.923664601795024, 6.903989264184951, 6.392066192343615, -1.7677669529663689],
            Fft.Dst(X7, 2, orthogonalize: true), 1e-13, "dst2");
        FftAssert.Abs([8.056547083976856, 3.2437967122418967, -6.611633387749187, -4.853553390593275, 9.210979838666258, 5.1713898553773205, -4.619386623715066],
            Fft.Dst(X7, 3, orthogonalize: true), 1e-13, "dst3");
        FftAssert.Abs([2.6879360111431225, -0.7071067811865475, -2.7307073709537626, 0.7071067811865475, 2.5725934879453436, -0.7905694150420949],
            Fft.Dct(X7[..6], 1, norm: FftNorm.Ortho, orthogonalize: false), 1e-13, "dct1 ortho, not orthogonalised");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void OrthoMatrices_AreOrthogonal(int type)
    {
        // With norm=Ortho each transform preserves the Euclidean norm.
        var x = FftAssert.RandomReal(50, type);
        double e0 = x.Sum(v => v * v);
        Assert.True(Math.Abs(Fft.Dct(x, type, norm: FftNorm.Ortho).Sum(v => v * v) - e0) < 1e-12 * e0, "dct");
        Assert.True(Math.Abs(Fft.Dst(x, type, norm: FftNorm.Ortho).Sum(v => v * v) - e0) < 1e-12 * e0, "dst");
    }

    [Fact]
    public void DctN_IsSeparable_AndInverts()
    {
        var data = FftAssert.RandomReal(5 * 6 * 4, 3);
        var x = new NdArray(data, 5, 6, 4);
        var y = Fft.DctN(x, 2, norm: FftNorm.Ortho);
        var manual = Fft.Dct(Fft.Dct(Fft.Dct(x, 2, axis: 0, norm: FftNorm.Ortho), 2, axis: 1, norm: FftNorm.Ortho), 2, axis: 2, norm: FftNorm.Ortho);
        FftAssert.Close(manual.ToArray(), y.ToArray(), 1e-14, "separable");
        FftAssert.Close(data, Fft.IdctN(y, 2, norm: FftNorm.Ortho).ToArray(), 1e-13, "idctn");
        var z = Fft.DstN(x, 4, axes: [0, 2]);
        FftAssert.Close(data, Fft.IdstN(z, 4, axes: [0, 2]).ToArray(), 1e-13, "idstn");
    }

    [Fact]
    public void DctAlongAxis_MatchesRowByRow()
    {
        var data = FftAssert.RandomReal(3 * 10, 8);
        var x = new NdArray(data, 3, 10);
        var y = Fft.Dct(x, 3, axis: 0).ToArray();
        for (var c = 0; c < 10; c++)
        {
            var column = new[] { data[c], data[10 + c], data[20 + c] };
            var expected = NaiveDct(column, 3);
            for (var r = 0; r < 3; r++) Assert.True(Math.Abs(expected[r] - y[r * 10 + c]) < 1e-13);
        }
    }

    [Fact]
    public void Dct1_RejectsSinglePoint_AndBadType()
    {
        Assert.Throws<ArgumentException>(() => Fft.Dct([1.0], 1));
        Assert.Throws<ArgumentException>(() => Fft.Dct([1.0, 2.0], 5));
    }
}
