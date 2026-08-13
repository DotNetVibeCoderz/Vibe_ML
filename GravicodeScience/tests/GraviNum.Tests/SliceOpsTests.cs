using Gravicode.Science.GraviNum;
using Xunit;
using Sel = Gravicode.Science.GraviNum.Slice;

namespace GraviNum.Tests;

/// <summary>
/// Tests for slice assignment, axis selection and element-wise choice.
/// </summary>
/// <remarks>
/// Most of these check the boundary between view and copy, because that is where the whole design
/// can be silently wrong: an assignment that writes into a detached copy leaves the original
/// untouched and reports success, and no shape check catches it.
/// </remarks>
public class SliceOpsTests
{
    private static NdArray Grid() => NdArray.FromArray(new double[,]
    {
        { 1, 2, 3, 4 },
        { 5, 6, 7, 8 },
        { 9, 10, 11, 12 },
    });

    // ---------------------------------------------------------------- assignment

    [Fact]
    public void AssigningThroughAViewChangesTheOriginal()
    {
        // The whole point. If Slice returned a copy this would pass every shape check and change
        // nothing.
        var grid = Grid();
        grid.Slice(Sel.Range(1, 3)).Assign(0.0);

        Assert.Equal(1.0, grid[0, 0]);      // row 0 untouched
        Assert.Equal(0.0, grid[1, 0]);
        Assert.Equal(0.0, grid[2, 3]);
    }

    [Fact]
    public void AssigningAnArrayCopiesItElementForElement()
    {
        var grid = Grid();
        var replacement = NdArray.FromArray(new double[,] { { 100, 200, 300, 400 } });

        grid.Slice(Sel.Range(0, 1)).Assign(replacement);

        Assert.Equal(100.0, grid[0, 0]);
        Assert.Equal(400.0, grid[0, 3]);
        Assert.Equal(5.0, grid[1, 0]);      // the next row is untouched
    }

    [Fact]
    public void ARowIsBroadcastAcrossEveryRowOfTheTarget()
    {
        // Matching axes from the right and stretching a missing leading one — the ordinary
        // broadcasting rule, applied to the destination rather than to a binary operation.
        var grid = Grid();
        grid.Assign(NdArray.FromValues([1.0, 2.0, 3.0, 4.0]));

        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 4; j++)
                Assert.Equal(j + 1.0, grid[i, j]);
    }

    [Fact]
    public void ALengthOneAxisIsStretched()
    {
        var grid = Grid();
        var column = NdArray.FromArray(new double[,] { { 7 }, { 8 }, { 9 } });

        grid.Assign(column);

        for (var j = 0; j < 4; j++)
        {
            Assert.Equal(7.0, grid[0, j]);
            Assert.Equal(9.0, grid[2, j]);
        }
    }

    [Fact]
    public void AScalarArrayFillsTheWholeTarget()
    {
        var grid = Grid();
        grid.Slice(Sel.All, Sel.Range(2, 4)).Assign(NdArray.FromValues([-1.0]));

        Assert.Equal(1.0, grid[0, 0]);
        Assert.Equal(-1.0, grid[0, 2]);
        Assert.Equal(-1.0, grid[2, 3]);
    }

    [Fact]
    public void AShapeThatCannotBroadcastIsRejected()
    {
        var grid = Grid();

        // Three values cannot stretch across four columns.
        Assert.Throws<InvalidOperationException>(() => grid.Assign(NdArray.FromValues([1.0, 2.0, 3.0])));

        // And a higher-rank source has nowhere to go.
        Assert.Throws<InvalidOperationException>(() => grid.Row(0).Assign(NdArray.Zeros(2, 4)));
    }

    [Fact]
    public void AssignmentIntoAStridedViewSkipsTheRightElements()
    {
        // A stride of two means every other column, so a bug that walks the buffer contiguously
        // writes into the wrong half and this catches it.
        var grid = Grid();
        grid.Slice(Sel.All, Sel.Range(0, null, 2)).Assign(0.0);

        Assert.Equal(0.0, grid[0, 0]);
        Assert.Equal(2.0, grid[0, 1]);      // untouched
        Assert.Equal(0.0, grid[0, 2]);
        Assert.Equal(4.0, grid[0, 3]);      // untouched
    }

    [Fact]
    public void CopyDetachesFromTheOriginal()
    {
        // The escape hatch, and the counterpart to every test above.
        var grid = Grid();
        var detached = grid.Slice(Sel.Range(0, 2)).Copy();

        detached.Assign(0.0);
        Assert.Equal(1.0, grid[0, 0]);
    }

    // ----------------------------------------------------------------- ellipsis

    [Fact]
    public void AnEllipsisFillsWhateverAxesAreLeft()
    {
        var volume = NdArray.Zeros(2, 3, 4);
        for (var i = 0; i < volume.Size; i++) volume.SetAt(i, i);

        // The last position of the trailing axis, without counting the leading ones.
        var last = volume.SliceEllipsis([], [Sel.At(3)]);

        Assert.Equal([2, 3], last.Shape.ToArray());
        Assert.Equal(3.0, last[0, 0]);
        Assert.Equal(volume[1, 2, 3], last[1, 2]);
    }

    [Fact]
    public void AnEllipsisStaysCorrectWhenTheRankChanges()
    {
        // The reason it exists: the same expression works on rank 3 and rank 4, where positional
        // selectors would have to be rewritten.
        var rank3 = NdArray.Zeros(2, 3, 5);
        var rank4 = NdArray.Zeros(2, 3, 4, 5);

        Assert.Equal([2, 3], rank3.SliceEllipsis([], [Sel.At(0)]).Shape.ToArray());
        Assert.Equal([2, 3, 4], rank4.SliceEllipsis([], [Sel.At(0)]).Shape.ToArray());
    }

    [Fact]
    public void SelectorsOnBothSidesOfTheEllipsisAreHonoured()
    {
        var volume = NdArray.Zeros(2, 3, 4, 5);
        var view = volume.SliceEllipsis([Sel.At(1)], [Sel.At(0)]);

        Assert.Equal([3, 4], view.Shape.ToArray());
    }

    [Fact]
    public void TooManySelectorsForTheRankIsRejected()
    {
        var grid = Grid();
        Assert.Throws<ArgumentException>(() => grid.SliceEllipsis([Sel.At(0), Sel.At(0)], [Sel.At(0)]));
    }

    // --------------------------------------------------------------------- axes

    [Fact]
    public void TakeAlongSelectsColumnsAsEasilyAsRows()
    {
        // NdArray.Take only walks axis 0; this is the general form.
        var columns = Grid().TakeAlong([3, 0], axis: 1);

        Assert.Equal([3, 2], columns.Shape.ToArray());
        Assert.Equal(4.0, columns[0, 0]);
        Assert.Equal(1.0, columns[0, 1]);
        Assert.Equal(12.0, columns[2, 0]);
    }

    [Fact]
    public void TakeAlongMayRepeatAndReorder()
    {
        var repeated = Grid().TakeAlong([1, 1, 0], axis: 0);

        Assert.Equal([3, 4], repeated.Shape.ToArray());
        Assert.Equal(5.0, repeated[0, 0]);
        Assert.Equal(5.0, repeated[1, 0]);
        Assert.Equal(1.0, repeated[2, 0]);
    }

    [Fact]
    public void TakeAlongAcceptsNegativeAxesAndIndices()
    {
        // -1 is the last axis and -1 is the last position, both following the usual convention.
        var last = Grid().TakeAlong([-1], axis: -1);

        Assert.Equal([3, 1], last.Shape.ToArray());
        Assert.Equal(4.0, last[0, 0]);
        Assert.Equal(12.0, last[2, 0]);
    }

    [Fact]
    public void TakeAlongRejectsAnOutOfRangeIndex()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Grid().TakeAlong([9], axis: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Grid().TakeAlong([0], axis: 5));
    }

    [Fact]
    public void AxisAtIsAViewAndDropsItsAxis()
    {
        var grid = Grid();
        var column = grid.AxisAt(1, 2);

        Assert.Equal([3], column.Shape.ToArray());
        Assert.Equal([3.0, 7.0, 11.0], column.ToArray());

        // A view, so writing through it lands in the original.
        column.SetAt(0, 99);
        Assert.Equal(99.0, grid[0, 2]);
    }

    [Fact]
    public void AxisRangeKeepsItsAxis()
    {
        var block = Grid().AxisRange(1, 1, 3);

        Assert.Equal([3, 2], block.Shape.ToArray());
        Assert.Equal(2.0, block[0, 0]);
        Assert.Equal(3.0, block[0, 1]);
    }

    // ------------------------------------------------------------------ masking

    [Fact]
    public void SelectChoosesElementWiseAndKeepsTheShape()
    {
        // Distinct from NdArray.Where, which filters and returns only the survivors. Keeping the
        // shape is what makes this composable.
        var condition = NdArray.FromArray(new double[,] { { 1, 0 }, { 0, 1 } });
        var a = NdArray.FromArray(new double[,] { { 10, 20 }, { 30, 40 } });
        var b = NdArray.FromArray(new double[,] { { -1, -2 }, { -3, -4 } });

        var result = SliceOps.Select(condition, a, b);

        Assert.Equal([2, 2], result.Shape.ToArray());
        Assert.Equal(10.0, result[0, 0]);
        Assert.Equal(-2.0, result[0, 1]);
        Assert.Equal(-3.0, result[1, 0]);
        Assert.Equal(40.0, result[1, 1]);
    }

    [Fact]
    public void SelectAcceptsAConstantAlternative()
    {
        var condition = NdArray.FromValues([1, 0, 1]);
        var values = NdArray.FromValues([5, 6, 7]);

        Assert.Equal([5.0, 0.0, 7.0], SliceOps.Select(condition, values, 0.0).ToArray());
    }

    [Fact]
    public void SetWhereWritesInPlaceAndConfinesToAView()
    {
        // Restricting a masked write to a region is the reason this takes a view rather than always
        // working on the whole array.
        var grid = Grid();
        grid.Slice(Sel.Range(0, 1)).SetWhere(v => v > 2, -1);

        Assert.Equal(1.0, grid[0, 0]);
        Assert.Equal(2.0, grid[0, 1]);
        Assert.Equal(-1.0, grid[0, 2]);
        Assert.Equal(7.0, grid[1, 2]);      // outside the view, untouched
    }

    [Fact]
    public void SetWhereAcceptsAnExplicitMask()
    {
        var values = NdArray.FromValues([1, 2, 3, 4]);
        values.SetWhere([true, false, false, true], 0);

        Assert.Equal([0.0, 2.0, 3.0, 0.0], values.ToArray());
    }

    [Fact]
    public void IndicesWhereReportsPositionsRatherThanValues()
    {
        // Positions are what you need to look the same rows up in a second array.
        var values = NdArray.FromValues([5, 1, 9, 3]);
        Assert.Equal([0, 2], values.IndicesWhere(v => v > 4));
    }

    [Fact]
    public void FilterRowsKeepsWholeRows()
    {
        var kept = Grid().FilterRows(row => row.At(0) > 4);

        Assert.Equal([2, 4], kept.Shape.ToArray());
        Assert.Equal(5.0, kept[0, 0]);
        Assert.Equal(9.0, kept[1, 0]);
    }

    [Fact]
    public void FilterRowsCopiesSoTheOriginalIsSafe()
    {
        var grid = Grid();
        var kept = grid.FilterRows(row => row.At(0) > 4);

        kept.Assign(0.0);
        Assert.Equal(5.0, grid[1, 0]);
    }

    [Fact]
    public void ClipBoundsBothEnds()
    {
        var values = NdArray.FromValues([-5, 0, 5, 10]);
        Assert.Equal([0.0, 0.0, 5.0, 6.0], values.Clip(0, 6).ToArray());

        // And the original is untouched, because Clip copies.
        Assert.Equal(-5.0, values.At(0));
    }

    [Fact]
    public void ClipRejectsInvertedBounds()
    {
        Assert.Throws<ArgumentException>(() => NdArray.Zeros(3).Clip(5, 1));
    }

    // --------------------------------------------------------------- shorthands

    [Fact]
    public void ShorthandSelectorsAgreeWithTheirLongForms()
    {
        var values = NdArray.FromValues([0, 1, 2, 3, 4, 5]);

        Assert.Equal([3.0, 4.0, 5.0], values.Slice(SliceShorthand.Last(3)).ToArray());
        Assert.Equal([0.0, 1.0], values.Slice(SliceShorthand.First(2)).ToArray());
        Assert.Equal([0.0, 2.0, 4.0], values.Slice(SliceShorthand.Every(2)).ToArray());
        Assert.Equal([0.0, 1.0, 2.0, 3.0], values.Slice(SliceShorthand.DropLast(2)).ToArray());
    }

    [Fact]
    public void ShorthandCountsMustBePositive()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => SliceShorthand.Last(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => SliceShorthand.First(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => SliceShorthand.DropLast(0));
    }

    [Fact]
    public void RangeSyntaxStillWorksAlongsideTheShorthands()
    {
        // The implicit conversion from System.Range, which is the most natural spelling when the
        // bounds are literal.
        var values = NdArray.FromValues([0, 1, 2, 3, 4, 5]);
        Assert.Equal([1.0, 2.0], values.Slice(1..3).ToArray());
    }
}
