using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.Sparse;
using Xunit;
using static Gravicode.Science.Tests.SciNet.SparseAssert;

namespace Gravicode.Science.Tests.SciNet;

public class SparseFormatTests
{
    private static readonly NdArray Dense = RandomDense(7, 5, 0.4, seed: 3);

    [Fact]
    public void EveryFormat_RoundTripsThroughDense()
    {
        var csr = CsrMatrix.FromDense(Dense);
        SparseMatrixBase[] all = [csr, csr.ToCsc(), csr.ToCoo(), csr.ToDok(), csr.ToLil(), csr.ToDia(), CscMatrix.FromDense(Dense), CooMatrix.FromDense(Dense)];
        foreach (var m in all)
        {
            Close(Dense, m.ToDense(), 0, m.Format);
            Close(Dense, m.ToCsr().ToDense(), 0, m.Format + "->csr");
            Close(Dense, m.ToCsc().ToDense(), 0, m.Format + "->csc");
            Assert.Equal(csr.Nnz, m.ToCsr().Nnz);
        }
    }

    [Fact]
    public void GraviNumSparseMatrix_RoundTrips()
    {
        var g = SparseMatrix.FromDense(Dense);
        var csr = CsrMatrix.FromGraviNum(g);
        Close(Dense, csr.ToDense(), 0, "from GraviNum");
        Close(Dense, csr.ToGraviNum().ToDense(), 0, "to GraviNum");
        Close(Dense, CscMatrix.FromGraviNum(g).ToDense(), 0, "csc from GraviNum");
    }

    [Fact]
    public void Constructor_ValidatesAndKeepsDuplicatesUntilCanonicalised()
    {
        // Row 0 holds column 2 twice and out of order: SciPy semantics say they add up.
        var m = new CsrMatrix([1.0, 2.0, 3.0, 4.0], [2, 0, 2, 1], [0, 3, 4], 2, 3);
        Assert.False(m.HasSortedIndices);
        Assert.False(m.HasCanonicalFormat);
        Assert.Equal(4.0, m[0, 2]);
        var c = m.SumDuplicates();
        Assert.True(c.HasCanonicalFormat);
        Assert.Equal(3, c.Nnz);
        Close(new NdArray([2.0, 0, 4, 0, 4, 0], 2, 3), m.ToDense(), 0);
        Assert.Throws<ArgumentException>(() => new CsrMatrix([1.0], [5], [0, 1], 1, 3));
        Assert.Throws<ArgumentException>(() => new CsrMatrix([1.0], [0], [0, 2], 1, 3));
    }

    [Fact]
    public void Transpose_IsZeroCopyBetweenCsrAndCsc()
    {
        var csr = CsrMatrix.FromDense(Dense);
        var t = csr.Transpose();
        Assert.Equal("csc", t.Format);
        Close(Dense.T.Copy(), t.ToDense(), 0);
        Assert.True(t.Data == csr.Data); // same backing span
        Close(Dense, t.Transpose().ToDense(), 0);
        Close(Dense.T.Copy(), csr.ToCoo().Transpose().ToDense(), 0);
    }

    [Fact]
    public void Arithmetic_MatchesDense()
    {
        var a = RandomDense(6, 6, 0.4, seed: 11);
        var b = RandomDense(6, 6, 0.4, seed: 12);
        var sa = CsrMatrix.FromDense(a);
        var sb = CscMatrix.FromDense(b);
        var da = a.ToArray(); var db = b.ToArray();
        Close(new NdArray(da.Zip(db, (x, y) => x + y).ToArray(), 6, 6), (sa + sb).ToDense(), 1e-15, "+");
        Close(new NdArray(da.Zip(db, (x, y) => x - y).ToArray(), 6, 6), (sa - sb).ToDense(), 1e-15, "-");
        Close(new NdArray(da.Zip(db, (x, y) => x * y).ToArray(), 6, 6), sa.Multiply(sb).ToDense(), 1e-15, "hadamard");
        Close(new NdArray(da.Zip(db, Math.Max).ToArray(), 6, 6), sa.Maximum(sb).ToDense(), 0, "max");
        Close(new NdArray(da.Zip(db, Math.Min).ToArray(), 6, 6), sa.Minimum(sb).ToDense(), 0, "min");
        Close(new NdArray(da.Select(x => -2.5 * x).ToArray(), 6, 6), (-2.5 * sa).ToDense(), 1e-15, "scale");
        Close(new NdArray(da.Select(x => x / 4).ToArray(), 6, 6), (sa / 4).ToDense(), 1e-15, "divide");
        Close(new NdArray(da.Select(x => -x).ToArray(), 6, 6), (-sa).ToDense(), 0, "negate");
        Close(new NdArray(da.Zip(db, (x, y) => x * y).ToArray(), 6, 6), sa.Multiply(b).ToDense(), 1e-15, "sparse∘dense");
        Close(new NdArray(da.Zip(db, (x, y) => x + y).ToArray(), 6, 6), sa + b, 1e-15, "sparse+dense");
        Close(new NdArray(da.Select(x => x * x * x).ToArray(), 6, 6), sa.Power(3).ToDense(), 1e-15, "power");
        Close(new NdArray(da.Select(Math.Abs).ToArray(), 6, 6), sa.Abs().ToDense(), 0, "abs");
        Close(new NdArray(da.Select(x => Math.Sqrt(Math.Abs(x))).ToArray(), 6, 6), sa.Abs().Sqrt().ToDense(), 0, "sqrt");
        // Exact cancellation leaves no stored entry.
        Assert.Equal(0, (sa - sa).Nnz);
        Assert.Throws<ArgumentException>(() => sa.Power(0));
    }

    [Fact]
    public void Products_MatchNaiveReference()
    {
        var a = RandomDense(9, 7, 0.35, seed: 21);
        var b = RandomDense(7, 8, 0.35, seed: 22);
        var sa = CsrMatrix.FromDense(a);
        var expected = NaiveMatMul(a, b);
        Close(expected, sa.MatMul(CsrMatrix.FromDense(b)).ToDense(), 1e-14, "SpGEMM");
        Close(expected, (sa * CscMatrix.FromDense(b)).ToDense(), 1e-14, "SpGEMM via operator");
        Close(expected, sa.Dot(b), 1e-14, "SpMM");
        var x = Enumerable.Range(0, 7).Select(i => Math.Sin(i + 1.0)).ToArray();
        var y = NaiveMatMul(a, new NdArray(x, 7, 1)).ToArray();
        SparseMatrixBase[] formats = [sa, sa.ToCsc(), sa.ToCoo(), sa.ToDok(), sa.ToLil(), sa.ToDia()];
        foreach (var f in formats) Close(y, f.Dot(x), 1e-14, "SpMV " + f.Format);
        var z = Enumerable.Range(0, 9).Select(i => Math.Cos(i + 1.0)).ToArray();
        var yt = NaiveMatMul(a.T.Copy(), new NdArray(z, 9, 1)).ToArray();
        foreach (var f in formats) Close(yt, f.RDot(z), 1e-14, "RMatVec " + f.Format);
    }

    [Fact]
    public void SpGemm_DropsExactCancellation_LikeScipy()
    {
        // [1 1] · [1; -1] = 0: the structural entry is not stored.
        var a = CsrMatrix.FromDense(new NdArray([1.0, 1.0], 1, 2));
        var b = CsrMatrix.FromDense(new NdArray([1.0, -1.0], 2, 1));
        Assert.Equal(0, a.MatMul(b).Nnz);
    }

    [Fact]
    public void LargeProducts_ParallelPathsAgreeWithScalarReference()
    {
        // Big enough to cross both parallel thresholds.
        var a = SparseMatrices.Random(3000, 3000, 0.01, seed: 5);
        var x = Enumerable.Range(0, 3000).Select(i => 1.0 / (i + 1)).ToArray();
        var fast = a.Dot(x);
        var reference = a.DotReference(x);
        Close(reference, fast, 1e-12, "SpMV parallel");

        var b = SparseMatrices.Random(3000, 400, 0.02, seed: 6);
        var c = a.MatMul(b);
        Assert.True(c.HasCanonicalFormat);
        // Check a sample of rows against a direct dot product.
        var bd = b.ToDense().AsContiguous().ToArray();
        for (var i = 0; i < 3000; i += 97)
        {
            var row = new double[400];
            for (var p = a.Indptr[i]; p < a.Indptr[i + 1]; p++)
                for (var j = 0; j < 400; j++) row[j] += a.Data[p] * bd[a.Indices[p] * 400 + j];
            for (var j = 0; j < 400; j++) Close(row[j], c[i, j], 1e-12, $"C[{i},{j}]");
        }
    }

    [Fact]
    public void Reductions_IncludeImplicitZeros()
    {
        var d = new NdArray([3.0, 0, -1, 0, 0, 0, 2, 5, 4], 3, 3);
        var s = CsrMatrix.FromDense(d);
        Assert.Equal(13.0, s.Sum());
        Close([5.0, 5, 3], s.Sum(0), 0);
        Close([2.0, 0, 11], s.Sum(1), 0);
        Close([3.0, 5, 4], s.Max(0), 0, "max axis 0");
        Assert.Equal(-1.0, s.Min());
        Assert.Equal(5.0, s.Max());
        Close([3.0, 0, 4], s.Diagonal(), 0);
        Close([0.0, 0], s.Diagonal(1), 0);
        Close([0.0, 5], s.Diagonal(-1), 0);
        Assert.Equal(7.0, s.Trace());
        Assert.Equal(5, s.CountNonzero());
        Close([13.0 / 9], [s.Mean()], 1e-15);
        var (r, c) = s.Nonzero();
        Assert.Equal([0, 0, 2, 2, 2], r);
        Assert.Equal([0, 2, 0, 1, 2], c);
        // A row with every column stored does not see an implicit zero.
        Close([-1.0, 0, 2], s.Min(1), 0);
    }

    [Fact]
    public void Norms_MatchDenseDefinitions()
    {
        var d = RandomDense(8, 6, 0.5, seed: 31);
        var s = CsrMatrix.FromDense(d);
        var arr = d.ToArray();
        Close(Math.Sqrt(arr.Sum(x => x * x)), s.Norm(), 1e-14, "fro");
        var colSums = Enumerable.Range(0, 6).Select(j => Enumerable.Range(0, 8).Sum(i => Math.Abs(arr[i * 6 + j]))).ToArray();
        var rowSums = Enumerable.Range(0, 8).Select(i => Enumerable.Range(0, 6).Sum(j => Math.Abs(arr[i * 6 + j]))).ToArray();
        Close(colSums.Max(), s.Norm(SparseNormKind.One), 1e-14, "1");
        Close(rowSums.Max(), s.Norm(SparseNormKind.Infinity), 1e-14, "inf");
        Close(colSums.Min(), s.Norm(SparseNormKind.MinusOne), 1e-14, "-1");
        Close(rowSums.Min(), s.Norm(SparseNormKind.MinusInfinity), 1e-14, "-inf");
        var sv = Decomposition.SingularValues(d).At(0);
        Close(sv, SparseLinalg.SpectralNorm(s), 1e-8, "2-norm");
    }

    [Fact]
    public void Slicing_MatchesDenseIndexing()
    {
        var s = CsrMatrix.FromDense(Dense);
        var full = Dense.To2DArray();
        var rows = s.GetRows([4, 0, 4]);
        for (var j = 0; j < 5; j++) { Assert.Equal(full[4, j], rows[0, j]); Assert.Equal(full[0, j], rows[1, j]); Assert.Equal(full[4, j], rows[2, j]); }
        var cols = s.GetColumns([3, 1, 3]);
        for (var i = 0; i < 7; i++) { Assert.Equal(full[i, 3], cols[i, 0]); Assert.Equal(full[i, 1], cols[i, 1]); Assert.Equal(full[i, 3], cols[i, 2]); }
        var sl = s.Slice(2, 6, 1, 4);
        for (var i = 0; i < 4; i++) for (var j = 0; j < 3; j++) Assert.Equal(full[i + 2, j + 1], sl[i, j]);
        var sub = s.Submatrix([6, 1], [0, 4]);
        Assert.Equal(full[6, 4], sub[0, 1]);
        Assert.Equal(full[1, 0], sub[1, 0]);
        var csc = s.ToCsc().GetColumns([2]);
        for (var i = 0; i < 7; i++) Assert.Equal(full[i, 2], csc[i, 0]);
        var tril = s.Tril(-1).ToDense().To2DArray();
        var triu = s.Triu(1).ToDense().To2DArray();
        for (var i = 0; i < 7; i++)
            for (var j = 0; j < 5; j++)
            {
                Assert.Equal(j - i <= -1 ? full[i, j] : 0.0, tril[i, j]);
                Assert.Equal(j - i >= 1 ? full[i, j] : 0.0, triu[i, j]);
            }
    }

    [Fact]
    public void SetDiag_AndCleanup()
    {
        var s = CsrMatrix.FromDense(new NdArray([1.0, 2, 0, 0, 3, 4, 5, 0, 6], 3, 3));
        var d = s.SetDiag([9.0, 8.0], k: 0);
        Close(new NdArray([9.0, 2, 0, 0, 8, 4, 5, 0, 6], 3, 3), d.ToDense(), 0);
        var z = s.SetDiag(0.0, 1);
        Assert.Equal(0.0, z[0, 1]);
        Assert.Equal(s.Nnz, z.Nnz); // explicit zeros kept, as SciPy's setdiag does
        Assert.Equal(s.Nnz - 2, z.EliminateZeros().Nnz);
        Assert.True(s.ValueEquals(s.ToCoo()));
        Assert.False(s.ValueEquals(d));
        Assert.True(s.AllClose(s.Scale(1 + 1e-9)));
    }

    [Fact]
    public void Builders_DokAndLil()
    {
        var dok = new DokMatrix(4, 4);
        dok[1, 2] = 3;
        dok[3, 0] = -1;
        dok[1, 2] += 2;
        dok[0, 0] = 0; // assigning zero stores nothing
        Assert.Equal(2, dok.Nnz);
        Assert.Equal(5.0, dok[1, 2]);
        var lil = new LilMatrix(4, 4);
        lil[1, 3] = 1; lil[1, 0] = 2; lil[1, 2] = 3;
        Assert.Equal([0, 2, 3], lil.RowIndices(1));
        lil[1, 2] = 0;
        Assert.Equal(2, lil.Nnz);
        Assert.True(lil.ToCsr().HasCanonicalFormat);
        Close(new NdArray([0.0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 4, 4), lil.ToDense(), 0);
    }

    [Fact]
    public void DiaMatrix_FollowsScipysColumnIndexedConvention()
    {
        // scipy: dia_matrix((data, offsets), shape=(4, 4)) with data = [[1,2,3,4],[1,2,3,4],[1,2,3,4]], offsets = [0,-1,2]
        var data = new NdArray([1.0, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], 3, 4);
        var dia = new DiaMatrix(data, [0, -1, 2], 4, 4);
        // Expected toarray() from scipy 1.17.1.
        var expected = new NdArray([1.0, 0, 3, 0, 1, 2, 0, 4, 0, 2, 3, 0, 0, 0, 3, 4], 4, 4);
        Close(expected, dia.ToDense(), 0);
        Close(expected, SparseMatrices.SpDiags(data, [0, -1, 2], 4, 4).ToCsr().ToDense(), 0);
        var x = new[] { 1.0, -1, 2, 0.5 };
        Close(NaiveMatMul(expected, new NdArray(x, 4, 1)).ToArray(), dia.Dot(x), 1e-15);
    }

    [Fact]
    public void Constructors_MatchDenseDefinitions()
    {
        Close(new NdArray([0.0, 1, 0, 0, 0, 1, 0, 0, 0], 3, 3), SparseMatrices.Eye(3, k: 1).ToDense(), 0, "eye k=1");
        Close(new NdArray([0.0, 0, 1, 0, 0, 0, 0, 1], 2, 4), SparseMatrices.Eye(2, 4, 2).ToDense(), 0, "eye 2x4");
        // scipy.sparse.diags([[1,2,3],[4,5],[6]], [0,1,-2]).toarray()
        var d = SparseMatrices.Diags([[1.0, 2, 3], [4.0, 5], [6.0]], [0, 1, -2]);
        Close(new NdArray([1.0, 4, 0, 0, 2, 5, 6, 0, 3], 3, 3), d.ToDense(), 0, "diags");
        var a = RandomDense(2, 3, 0.7, 41);
        var b = RandomDense(3, 2, 0.7, 42);
        Close(LinAlg.Kron(a, b), SparseMatrices.Kron(CsrMatrix.FromDense(a), CsrMatrix.FromDense(b)).ToDense(), 1e-15, "kron");
        var p = SparseAssert.Poisson2D(3, 4).ToDense();
        var lx = Laplacian1D(3).ToDense(); var ly = Laplacian1D(4).ToDense();
        var expect = LinAlg.Kron(NdArray.Eye(4), lx).ToArray().Zip(LinAlg.Kron(ly, NdArray.Eye(3)).ToArray(), (x, y) => x + y).ToArray();
        Close(new NdArray(expect, 12, 12), p, 0, "kronsum");
        var h = SparseMatrices.HStack(CsrMatrix.FromDense(a), CsrMatrix.Zeros(2, 1));
        Assert.Equal((2, 4), h.Shape);
        var v = SparseMatrices.VStack(CsrMatrix.FromDense(a), SparseMatrices.Eye(1, 3));
        Assert.Equal(1.0, v[2, 0]);
        var bd = SparseMatrices.BlockDiag(CsrMatrix.FromDense(a), CsrMatrix.FromDense(b));
        Assert.Equal((5, 5), bd.Shape);
        Assert.Equal(b.To2DArray()[2, 1], bd[4, 4]);
        var bm = SparseMatrices.Bmat(new SparseMatrixBase?[,] { { CsrMatrix.FromDense(a), null }, { null, CsrMatrix.FromDense(b) } });
        Assert.True(bm.ValueEquals(bd));
        Assert.Throws<ArgumentException>(() => SparseMatrices.Bmat(new SparseMatrixBase?[,] { { null } }));
    }

    [Fact]
    public void Random_HasExactNnzAndIsReproducible()
    {
        var r1 = SparseMatrices.Random(50, 40, 0.1, seed: 9);
        var r2 = SparseMatrices.Random(50, 40, 0.1, seed: 9);
        Assert.Equal(200, r1.Nnz);
        Assert.True(r1.ValueEquals(r2));
        Assert.All(r1.Data.ToArray(), v => Assert.InRange(v, 0.0, 1.0));
        var (rr, cc, vv) = SparseMatrices.Find(r1);
        Assert.Equal(200, rr.Length);
        Assert.Equal(r1[rr[7], cc[7]], vv[7]);
    }

    [Fact]
    public void LinearOperator_CompositionAlgebra()
    {
        var a = CsrMatrix.FromDense(RandomDense(5, 5, 0.6, 51));
        var b = CsrMatrix.FromDense(RandomDense(5, 5, 0.6, 52));
        var la = a.AsLinearOperator();
        var lb = b.AsLinearOperator();
        var x = new[] { 1.0, 2, -1, 0.5, 3 };
        Close((a + b).Dot(x), (la + lb).Dot(x), 1e-14, "sum");
        Close((a - b).Dot(x), (la - lb).Dot(x), 1e-14, "difference");
        Close(a.MatMul(b).Dot(x), (la * lb).Dot(x), 1e-14, "product");
        Close((3.0 * a).Dot(x), (3.0 * la).Dot(x), 1e-14, "scaled");
        Close(a.RDot(x), la.T.Dot(x), 1e-14, "transpose");
        Close(a.MatMul(a).MatMul(a).Dot(x), LinearOperator.Power(a, 3).Dot(x), 1e-13, "power");
        Close(a.ToDense(), la.ToDenseArray(), 0, "to dense");
        var dense = LinearOperator.AsLinearOperator(a.ToDense());
        Close(a.Dot(x), dense.Dot(x), 1e-14, "dense operator");
        Close(a.RDot(x), dense.RDot(x), 1e-14, "dense operator transpose");
        Close(a.Dot(a.ToDense()), la.MatMat(a.ToDense()), 1e-14, "matmat");
    }
}
