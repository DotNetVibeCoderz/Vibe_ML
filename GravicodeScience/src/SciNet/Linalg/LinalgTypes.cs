using System.Numerics;
using Gravicode.Science.GraviNum;

namespace Gravicode.Science.SciNet;

/// <summary>scipy's <c>LinAlgError</c>: a factorisation or solve that cannot be completed.</summary>
public class LinalgException : InvalidOperationException
{
    /// <summary>Creates the exception with a message.</summary>
    public LinalgException(string message) : base(message) { }
}

/// <summary>The matrix is singular (or a factor has an exactly zero pivot).</summary>
public sealed class LinalgSingularMatrixException : LinalgException
{
    /// <summary>Creates the exception with a message.</summary>
    public LinalgSingularMatrixException(string message) : base(message) { }
}

/// <summary>Structure assumed by <see cref="Linalg.Solve"/>, scipy's <c>assume_a</c>.</summary>
public enum LinalgMatrixStructure
{
    /// <summary>LU with partial pivoting (<c>'gen'</c>).</summary>
    General,
    /// <summary>Symmetric indefinite: Bunch–Kaufman LDLᵀ (<c>'sym'</c>).</summary>
    Symmetric,
    /// <summary>Hermitian; for real input identical to <see cref="Symmetric"/> (<c>'her'</c>).</summary>
    Hermitian,
    /// <summary>Symmetric positive definite: Cholesky (<c>'pos'</c>).</summary>
    PositiveDefinite,
    /// <summary>Only the diagonal is read (<c>'diagonal'</c>).</summary>
    Diagonal,
    /// <summary>Tridiagonal: banded LU with partial pivoting (<c>'tridiagonal'</c>).</summary>
    Tridiagonal,
    /// <summary>Only the upper triangle is read (<c>'upper triangular'</c>).</summary>
    UpperTriangular,
    /// <summary>Only the lower triangle is read (<c>'lower triangular'</c>).</summary>
    LowerTriangular,
}

/// <summary>Packed LU from <see cref="Linalg.LuFactor"/>: scipy's <c>(lu, piv)</c>.</summary>
/// <param name="Lu">L strictly below the diagonal (unit diagonal implied), U on and above.</param>
/// <param name="Piv">LAPACK-style row <em>swaps</em>, 0-based: at step i row i was exchanged with row Piv[i].</param>
public sealed record LuFactorResult(NdArray Lu, int[] Piv);

/// <summary>Cholesky factor from <see cref="Linalg.ChoFactor"/>: scipy's <c>(c, lower)</c>.</summary>
/// <param name="C">The triangular factor; the other triangle is zero (scipy leaves garbage there).</param>
/// <param name="Lower">True when <c>A = C Cᵀ</c>, false when <c>A = Cᵀ C</c>.</param>
public sealed record ChoFactorResult(NdArray C, bool Lower);

/// <summary>Bunch–Kaufman factorisation <c>A = Lu D Luᵀ</c> from <see cref="Linalg.Ldl"/>.</summary>
/// <param name="Lu">Permuted unit-triangular factor; <c>Lu[Perm]</c> is triangular.</param>
/// <param name="D">Block diagonal with 1×1 and symmetric 2×2 blocks.</param>
/// <param name="Perm">Row order that makes <paramref name="Lu"/> triangular.</param>
public sealed record LdlResult(NdArray Lu, NdArray D, int[] Perm);

/// <summary>Column-pivoted QR, <c>A[:, P] = Q R</c>.</summary>
/// <param name="Q">Orthogonal factor.</param>
/// <param name="R">Upper-triangular factor with non-increasing |diagonal|.</param>
/// <param name="P">Column permutation.</param>
public sealed record QrPivotingResult(NdArray Q, NdArray R, int[] P);

/// <summary>RQ factorisation <c>A = R Q</c>.</summary>
public sealed record RqResult(NdArray R, NdArray Q);

/// <summary>Hessenberg form <c>A = Q H Qᵀ</c>; <see cref="Q"/> is null when it was not requested.</summary>
public sealed record HessenbergResult(NdArray H, NdArray? Q);

/// <summary>Real Schur form <c>A = Z T Zᵀ</c>.</summary>
/// <param name="T">Quasi-upper-triangular; each complex pair sits in a standardised 2×2 block
/// <c>[[a, b], [c, a]]</c> with <c>b·c &lt; 0</c>.</param>
/// <param name="Z">Orthogonal Schur vectors.</param>
public sealed record SchurResult(NdArray T, NdArray Z);

/// <summary>Complex Schur form <c>A = Z T Zᴴ</c> with T upper triangular.</summary>
public sealed record ComplexSchurResult(ComplexNdArray T, ComplexNdArray Z);

/// <summary>Eigenvalues and right eigenvectors of a general matrix, from <see cref="Linalg.Eig"/>.</summary>
/// <param name="Values">Eigenvalues; for a real matrix complex pairs are adjacent, positive imaginary part first.</param>
/// <param name="Vectors">Unit-norm eigenvectors as columns (n×n complex).</param>
public sealed record LinalgEigResult(Complex[] Values, ComplexNdArray Vectors);

/// <summary>Symmetric / generalised symmetric-definite eigen decomposition from <see cref="Linalg.Eigh"/>.</summary>
/// <param name="Values">Eigenvalues in ascending order (scipy's order, the reverse of GraviNum's).</param>
/// <param name="Vectors">Eigenvectors as columns; for a generalised problem they are B-orthonormal.</param>
public sealed record EighResult(double[] Values, NdArray Vectors);

/// <summary>Singular value decomposition <c>A = U diag(S) Vh</c>, scipy's <c>(U, s, Vh)</c>.</summary>
public sealed record LinalgSvdResult(NdArray U, double[] S, NdArray Vh);

/// <summary>Polar decomposition from <see cref="Linalg.Polar"/>: <c>A = U P</c> (right) or <c>A = P U</c> (left).</summary>
public sealed record PolarResult(NdArray U, NdArray P);

/// <summary>Least-squares solution from <see cref="Linalg.Lstsq"/>.</summary>
/// <param name="X">Solution, shape of b with n rows.</param>
/// <param name="Residues">Squared 2-norm of each residual column; empty unless rank is n and m &gt; n (as scipy).</param>
/// <param name="Rank">Effective rank.</param>
/// <param name="SingularValues">Singular values of a, descending.</param>
public sealed record LstsqResult(NdArray X, double[] Residues, int Rank, double[] SingularValues);
