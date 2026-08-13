using Gravicode.Science.GraviNum;
using Xunit;

namespace Gravicode.Science.Tests.GraviNum;

public class NdArrayTests
{
    [Fact]
    public void Zeros_HasRequestedShapeAndSize()
    {
        var a = NdArray.Zeros(2, 3, 4);
        Assert.Equal(3, a.Rank);
        Assert.Equal(24, a.Size);
        Assert.Equal(0.0, a.Sum());
    }

    [Fact]
    public void Arange_And_Linspace_ProduceExpectedValues()
    {
        var r = NdArray.Arange(0, 5);
        Assert.Equal(5, r.Size);
        Assert.Equal(4.0, r.At(4));

        var l = NdArray.Linspace(0, 1, 5);
        Assert.Equal(0.25, l.At(1), 12);
        Assert.Equal(1.0, l.At(4), 12);
    }

    [Fact]
    public void Reshape_KeepsElementsInRowMajorOrder()
    {
        var a = NdArray.Arange(6).Reshape(2, 3);
        Assert.Equal(0.0, a[0, 0]);
        Assert.Equal(2.0, a[0, 2]);
        Assert.Equal(3.0, a[1, 0]);
        Assert.Equal(5.0, a[1, 2]);
    }

    [Fact]
    public void Reshape_InfersTheMinusOneDimension()
    {
        var a = NdArray.Arange(12).Reshape(3, -1);
        Assert.Equal(3, a.Shape[0]);
        Assert.Equal(4, a.Shape[1]);
    }

    [Fact]
    public void Transpose_IsAZeroCopyViewOverTheSameBuffer()
    {
        var a = NdArray.Arange(6).Reshape(2, 3);
        var t = a.T;

        Assert.Equal(3, t.Shape[0]);
        Assert.Equal(2, t.Shape[1]);
        Assert.Equal(a[1, 2], t[2, 1]);

        // Writing through the view is visible in the original: it is a view, not a copy.
        t[0, 1] = 99.0;
        Assert.Equal(99.0, a[1, 0]);
    }

    [Fact]
    public void Slice_SelectsARangeAndDropsIndexedAxes()
    {
        var a = NdArray.Arange(12).Reshape(3, 4);

        var rows = a.Slice(Slice.Range(1, 3), Slice.All);
        Assert.Equal(2, rows.Shape[0]);
        Assert.Equal(4.0, rows[0, 0]);

        var single = a.Slice(Slice.At(1), Slice.All);
        Assert.Equal(1, single.Rank);
        Assert.Equal(4, single.Size);
        Assert.Equal(5.0, single.At(1));
    }

    [Fact]
    public void Slice_WithStepAndNegativeDirection()
    {
        var a = NdArray.Arange(10);

        var evens = a.Slice(Slice.Range(0, 10, 2));
        Assert.Equal(5, evens.Size);
        Assert.Equal(8.0, evens.At(4));

        var reversed = a.Slice(Slice.Reversed);
        Assert.Equal(10, reversed.Size);
        Assert.Equal(9.0, reversed.At(0));
        Assert.Equal(0.0, reversed.At(9));
    }

    [Fact]
    public void StridedViews_CopyCorrectlyToContiguousStorage()
    {
        var a = NdArray.Arange(12).Reshape(3, 4);
        var view = a.Slice(Slice.All, Slice.Range(1, 3));
        Assert.False(view.IsContiguous);

        var flat = view.ToArray();
        Assert.Equal(new[] { 1.0, 2.0, 5.0, 6.0, 9.0, 10.0 }, flat);
    }

    [Fact]
    public void Broadcasting_StretchesLengthOneAxes()
    {
        var matrix = NdArray.Ones(3, 4);
        var rowVector = NdArray.Arange(4);

        var sum = matrix + rowVector;
        Assert.Equal(3, sum.Shape[0]);
        Assert.Equal(4, sum.Shape[1]);
        Assert.Equal(1.0, sum[0, 0]);
        Assert.Equal(4.0, sum[2, 3]);
    }

    [Fact]
    public void Broadcasting_RejectsIncompatibleShapes()
    {
        var a = NdArray.Ones(3, 4);
        var b = NdArray.Ones(5);
        Assert.Throws<InvalidOperationException>(() => a + b);
    }

    [Fact]
    public void Concatenate_And_Stack_JoinAlongTheExpectedAxis()
    {
        var a = NdArray.Zeros(2, 3);
        var b = NdArray.Ones(2, 3);

        var vertical = NdArray.Concatenate([a, b], axis: 0);
        Assert.Equal(4, vertical.Shape[0]);
        Assert.Equal(1.0, vertical[3, 0]);

        var horizontal = NdArray.Concatenate([a, b], axis: 1);
        Assert.Equal(6, horizontal.Shape[1]);
        Assert.Equal(1.0, horizontal[0, 5]);

        var stacked = NdArray.Stack([a, b]);
        Assert.Equal(3, stacked.Rank);
        Assert.Equal(2, stacked.Shape[0]);
    }

    [Fact]
    public void BooleanMask_KeepsOnlySelectedElements()
    {
        var a = NdArray.Arange(6);
        var mask = a.Mask(x => x % 2 == 0);
        var kept = a.BooleanMask(mask);
        Assert.Equal(3, kept.Size);
        Assert.Equal(new[] { 0.0, 2.0, 4.0 }, kept.ToArray());
    }

    [Fact]
    public void Take_SelectsRowsByIndex()
    {
        var a = NdArray.Arange(12).Reshape(4, 3);
        var picked = a.Take([2, 0]);
        Assert.Equal(2, picked.Shape[0]);
        Assert.Equal(6.0, picked[0, 0]);
        Assert.Equal(0.0, picked[1, 0]);
    }
}
