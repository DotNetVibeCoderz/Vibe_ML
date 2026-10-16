using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet;
using Xunit;
using static Gravicode.Science.Tests.SciNet.InterpolateAssert;

namespace Gravicode.Science.Tests.SciNet;

// Expected values: scipy 1.17.1 RBFInterpolator on 30 random points (default_rng(2)).
public class RbfTests
{
    private static readonly double[,] Y = {{0.523224, 0.298491}, {1.628451, 0.091916}, {1.200201, 0.728561}, {0.375802, 0.055147}, {0.549939, 0.657433}, {1.124531, 0.150062}, {0.865262, 0.669297}, {0.845569, 0.633184}, {1.934872, 0.683065}, {0.78325, 0.187253}, {0.691921, 0.511066}, {1.782419, 0.775564}, {0.636293, 0.924217}, {0.94182, 0.693759}, {0.214415, 0.104544}, {0.403815, 0.88445}, {1.359623, 0.849236}, {1.288873, 0.406542}, {1.033156, 0.593444}, {1.724236, 0.438186}, {1.78448, 0.613717}, {1.658712, 0.498056}, {1.385036, 0.339025}, {1.045657, 0.216223}, {0.201407, 0.038604}, {1.403899, 0.456431}, {1.795468, 0.835183}, {0.77019, 0.973679}, {1.184124, 0.765883}, {0.814389, 0.19617}};
    private static readonly double[] D = [1.4908000924791338, 0.8471678252891195, 0.09830073744293943, 1.669157335751303, 0.500350058897097, 1.6790282645560035, 0.5639615615302807, 0.6699020834549343, -1.1258491422883674, 1.8463102150377018, 1.0201643511513772, -1.0966622774499262, 0.023154305720732515, 0.4628655381045217, 1.3670263454605944, -0.16050751796998697, -0.4188711132209447, 0.8784383023228242, 0.6717178803222287, -0.048641754378833335, -0.6815496122307825, -0.09837336971497525, 0.8888888850481422, 1.6644504381359888, 1.385309945221925, 0.5277730558372097, -1.2388263448507368, 0.023761420585113213, 0.03401147430060558, 1.8300882679904222];
    private static readonly double[,] X = {{0.5, 0.5}, {1.2, 0.3}, {0.05, 0.9}, {1.9, 0.95}, {2.4, -0.2}};

    [Fact]
    public void EachKernel_MatchesScipy()
    {
        AllClose([0.9206743301543953, 1.2836856594442536, -0.391761143916122, -1.6332525879524389, 0.20930032450038197], new RBFInterpolator(Y, D, kernel: RbfKernel.ThinPlateSpline, epsilon: null).Evaluate(X), 1e-9, "thin_plate_spline");
        AllClose([0.9325849690206424, 1.2713224204130071, 0.05706913201686409, -1.342789084725411, 0.16451069667258222], new RBFInterpolator(Y, D, kernel: RbfKernel.Linear, epsilon: null).Evaluate(X), 1e-9, "linear");
        AllClose([0.9207204408761361, 1.2906230251768298, -0.5828977263878488, -1.623570711042356, -0.31791761244537203], new RBFInterpolator(Y, D, kernel: RbfKernel.Cubic, epsilon: null).Evaluate(X), 1e-9, "cubic");
        AllClose([0.9137849320224305, 1.2967331653163128, -0.7894041141880166, -1.5747920145077323, -0.7118993661723607], new RBFInterpolator(Y, D, kernel: RbfKernel.Quintic, epsilon: null).Evaluate(X), 1e-9, "quintic");
        AllClose([0.9190823181969137, 1.2964366042160052, -0.40942451372393407, -1.5697301161184356, -0.08845747479418264], new RBFInterpolator(Y, D, kernel: RbfKernel.Multiquadric, epsilon: 2.0).Evaluate(X), 1e-9, "multiquadric");
        AllClose([0.9135039338972379, 1.2998533014988238, -0.13891960726778269, -1.4741794212583104, 0.12898945144488128], new RBFInterpolator(Y, D, kernel: RbfKernel.InverseMultiquadric, epsilon: 2.0).Evaluate(X), 1e-9, "inverse_multiquadric");
        AllClose([0.9089731473369096, 1.3040542906820143, -0.03087045471467792, -1.406689597110378, 0.2310077694245972], new RBFInterpolator(Y, D, kernel: RbfKernel.InverseQuadratic, epsilon: 2.0).Evaluate(X), 1e-9, "inverse_quadratic");
        AllClose([0.8668285371740418, 1.307378247199243, 0.1648974470727871, -1.2979605398924443, 0.4114489509005037], new RBFInterpolator(Y, D, kernel: RbfKernel.Gaussian, epsilon: 3.0).Evaluate(X), 1e-9, "gaussian");
    }

    [Fact]
    public void DegreeSmoothingNeighbors_MatchScipy()
    {
        AllClose([0.9372549209486429, 1.2663343299442031, -1.723453104945312, -2.1020656088572185, -2.1300648445279724], new RBFInterpolator(Y, D, degree: -1).Evaluate(X), 1e-8, "degree -1");
        AllClose([0.9186411738993809, 1.2716939189784906, -0.8531866278737046, -1.8348508100590009, 2.0810726466683542], new RBFInterpolator(Y, D, degree: 0).Evaluate(X), 1e-8, "degree 0");
        AllClose([0.9217026671805852, 1.2818273081182228, -0.7012038549560939, -1.7043284364634892, -0.888438494155795], new RBFInterpolator(Y, D, degree: 2).Evaluate(X), 1e-8, "degree 2");
        AllClose([0.9131565553901242, 1.2986715399737427, -1.0360260627804072, -1.5768852304138368, -0.7801619398396245], new RBFInterpolator(Y, D, degree: 3).Evaluate(X), 1e-8, "degree 3");
        AllClose([0.9126058069376481, 1.2591281342157659, -0.32303261912852166, -1.6641463451667953, 0.3263168152347597], new RBFInterpolator(Y, D, smoothing: 0.1).Evaluate(X), 1e-9, "smoothing");
        var sm = new double[30];
        for (var i = 0; i < 30; i++) sm[i] = 0.5 * i / 29;
        AllClose([0.9417198950826671, 1.2896410694412181, 0.03920671916941937, -1.0898302839831147, 0.34250094672946524], new RBFInterpolator(NdArray.FromArray(Y), new NdArray((double[])D.Clone(), 30), smoothing: sm, kernel: RbfKernel.Gaussian, epsilon: 1.5).Evaluate(X), 1e-9, "per-point smoothing");
        AllClose([0.9434557889931274, 1.2753712891779392, -0.5826475488655007, -1.6951984856851032, -0.05845221354975583], new RBFInterpolator(Y, D, neighbors: 8).Evaluate(X), 1e-8, "neighbors");
        AllClose([0.8861767946314998, 1.277057324706253, -1.1252402456265944, -1.639323555771688, -1.3135285885425692], new RBFInterpolator(Y, D, neighbors: 9, kernel: RbfKernel.Quintic, smoothing: 0.01).Evaluate(X), 1e-8, "neighbors quintic");
        var d2 = new double[60];
        for (var i = 0; i < 30; i++) { d2[2 * i] = D[i]; d2[2 * i + 1] = D[i] * D[i]; }
        var v = new RBFInterpolator(NdArray.FromArray(Y), new NdArray(d2, 30, 2), kernel: RbfKernel.Cubic).Evaluate(NdArray.FromArray(X));
        Assert.Equal([5, 2], v.Shape.ToArray());
        AllClose([0.9207204408761388, 0.9300718675244712, 1.2906230251768258, 1.664313000900544, -0.5828977263878431, -0.20195276038549018, -1.6235707110423803, 2.4973097143148557, -0.31791761244541933, 0.3515615660466088], v.ToArray(), 1e-9, "vector values");
        // interpolates the data without smoothing
        AllClose(D, new RBFInterpolator(Y, D).Evaluate(Y), 1e-10, "reproduces data");
        Assert.Throws<ArgumentException>(() => new RBFInterpolator(Y, D, kernel: RbfKernel.Gaussian));
        // 400 points take the blocked, threaded LU; the interpolant must still pass through the data.
        var rng = new Random(5);
        var big = new double[400, 2];
        var bd = new double[400];
        for (var i = 0; i < 400; i++) { big[i, 0] = rng.NextDouble(); big[i, 1] = rng.NextDouble(); bd[i] = Math.Sin(5 * big[i, 0]) * big[i, 1]; }
        AllClose(bd, new RBFInterpolator(big, bd).Evaluate(big), 1e-8, "blocked LU path reproduces data");
    }
}
