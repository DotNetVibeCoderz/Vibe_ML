using Gravicode.Science.GraviNum;
using static Gravicode.Science.SciNet.NdImageSupport;

namespace Gravicode.Science.SciNet;

public static partial class NdImage
{
    // ------------------------------------------------------------------ structuring elements

    /// <summary>
    /// The 3×…×3 structuring element whose entries lie within <paramref name="connectivity"/>
    /// city-block steps of the centre (scipy.ndimage.generate_binary_structure): connectivity 1 is
    /// the cross, connectivity = rank is the full cube.
    /// </summary>
    /// <param name="rank">Number of dimensions (≥ 1).</param>
    /// <param name="connectivity">Maximum squared distance in unit steps.</param>
    public static NdArray GenerateBinaryStructure(int rank, int connectivity)
    {
        if (rank < 1) throw new ArgumentException("Rank must be at least 1.");
        var shape = Enumerable.Repeat(3, rank).ToArray();
        var size = SizeOf(shape);
        var d = new double[size];
        for (var i = 0; i < size; i++)
        {
            var rem = i;
            var dist = 0;
            for (var a = 0; a < rank; a++) { dist += Math.Abs(rem % 3 - 1); rem /= 3; }
            d[i] = dist <= connectivity ? 1.0 : 0.0;
        }
        return new NdArray(d, shape);
    }

    /// <summary>
    /// Dilates a structure by itself <c>iterations − 1</c> times (scipy.ndimage.iterate_structure),
    /// giving the element equivalent to applying the original that many times.
    /// </summary>
    /// <param name="structure">Structuring element.</param>
    /// <param name="iterations">Number of applications to fold together.</param>
    public static NdArray IterateStructure(NdArray structure, int iterations)
    {
        ArgumentNullException.ThrowIfNull(structure);
        var sshape = ShapeOf(structure);
        if (iterations < 2) return FromMask(ToMask(structure), sshape);
        var ni = iterations - 1;
        var shape = sshape.Select(s => s + ni * (s - 1)).ToArray();
        var pos = sshape.Select(s => ni * (s / 2)).ToArray();
        var outv = new bool[SizeOf(shape)];
        var smask = ToMask(structure);
        var strides = StridesOf(shape);
        var coord = new int[sshape.Length];
        for (var i = 0; i < smask.Length; i++)
        {
            var idx = 0;
            for (var a = 0; a < sshape.Length; a++) idx += (coord[a] + pos[a]) * strides[a];
            outv[idx] = smask[i];
            for (var a = sshape.Length - 1; a >= 0; a--)
            {
                if (++coord[a] < sshape[a]) break;
                coord[a] = 0;
            }
        }
        return BinaryDilation(FromMask(outv, shape), structure, ni);
    }

    // ------------------------------------------------------------------ binary erosion / dilation

    /// <summary>
    /// Binary erosion (scipy.ndimage.binary_erosion): an element stays true only if every
    /// structure position over it is true, with <paramref name="borderValue"/> outside the array.
    /// </summary>
    /// <remarks>
    /// Repeated erosion with a structure containing its own centre is monotone, so instead of
    /// re-scanning the image once per iteration the erosion time of every element is found by a
    /// breadth-first pass from the false set — any iteration count, including "until stable"
    /// (<paramref name="iterations"/> &lt; 1), costs one sweep. <paramref name="bruteForce"/> forces
    /// the literal repeated passes; a structure without its centre always uses them.
    /// </remarks>
    /// <param name="input">Binary image (non-zero = true).</param>
    /// <param name="structure">Structuring element (default: connectivity-1 cross).</param>
    /// <param name="iterations">Repetitions; &lt; 1 repeats until nothing changes.</param>
    /// <param name="mask">Only elements true in the mask may change.</param>
    /// <param name="borderValue">Value assumed outside the array.</param>
    /// <param name="origin">Per-axis origin of the structure.</param>
    /// <param name="bruteForce">Use repeated full passes instead of the propagation shortcut.</param>
    public static NdArray BinaryErosion(NdArray input, NdArray? structure = null, int iterations = 1, NdArray? mask = null,
        bool borderValue = false, int[]? origin = null, bool bruteForce = false)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var (sm, ss) = StructureOf(structure, shape.Length);
        var o = Expand(origin, shape.Length, 0, nameof(origin));
        var r = BinaryErosionCore(ToMask(input), shape, sm, ss, iterations, MaskOf(mask, shape), borderValue, o, invert: false, bruteForce);
        return FromMask(r, shape);
    }

    /// <summary>
    /// Binary dilation (scipy.ndimage.binary_dilation): true wherever the reflected structure
    /// touches a true element. Implemented, as in scipy, as an inverted erosion with the structure
    /// reversed and the origin negated (shifted by one more for even extents).
    /// </summary>
    /// <param name="input">Binary image.</param>
    /// <param name="structure">Structuring element (default: connectivity-1 cross).</param>
    /// <param name="iterations">Repetitions; &lt; 1 repeats until nothing changes.</param>
    /// <param name="mask">Only elements true in the mask may change.</param>
    /// <param name="borderValue">Value assumed outside the array.</param>
    /// <param name="origin">Per-axis origin of the structure.</param>
    /// <param name="bruteForce">Use repeated full passes instead of the propagation shortcut.</param>
    public static NdArray BinaryDilation(NdArray input, NdArray? structure = null, int iterations = 1, NdArray? mask = null,
        bool borderValue = false, int[]? origin = null, bool bruteForce = false)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var r = DilateMask(ToMask(input), shape, structure, iterations, MaskOf(mask, shape), borderValue, origin, bruteForce);
        return FromMask(r, shape);
    }

    private static bool[] DilateMask(bool[] input, int[] shape, NdArray? structure, int iterations, bool[]? mask, bool borderValue,
        int[]? origin, bool bruteForce)
    {
        var (sm, ss) = StructureOf(structure, shape.Length);
        var rev = (bool[])sm.Clone();
        Array.Reverse(rev);
        var o = Expand(origin, shape.Length, 0, nameof(origin));
        for (var a = 0; a < o.Length; a++)
        {
            o[a] = -o[a];
            if ((ss[a] & 1) == 0) o[a] -= 1;
        }
        return BinaryErosionCore(input, shape, rev, ss, iterations, mask, borderValue, o, invert: true, bruteForce);
    }

    /// <summary>Erosion followed by dilation with the same structure (scipy.ndimage.binary_opening).</summary>
    /// <param name="input">Binary image.</param>
    /// <param name="structure">Structuring element (default: connectivity-1 cross).</param>
    /// <param name="iterations">Iterations of each step.</param>
    /// <param name="mask">Only elements true in the mask may change.</param>
    /// <param name="borderValue">Value assumed outside the array.</param>
    /// <param name="origin">Per-axis origin.</param>
    /// <param name="bruteForce">Use repeated full passes.</param>
    public static NdArray BinaryOpening(NdArray input, NdArray? structure = null, int iterations = 1, NdArray? mask = null,
        bool borderValue = false, int[]? origin = null, bool bruteForce = false)
    {
        ArgumentNullException.ThrowIfNull(input);
        structure ??= GenerateBinaryStructure(input.Rank, 1);
        var tmp = BinaryErosion(input, structure, iterations, mask, borderValue, origin, bruteForce);
        return BinaryDilation(tmp, structure, iterations, mask, borderValue, origin, bruteForce);
    }

    /// <summary>Dilation followed by erosion with the same structure (scipy.ndimage.binary_closing).</summary>
    /// <param name="input">Binary image.</param>
    /// <param name="structure">Structuring element (default: connectivity-1 cross).</param>
    /// <param name="iterations">Iterations of each step.</param>
    /// <param name="mask">Only elements true in the mask may change.</param>
    /// <param name="borderValue">Value assumed outside the array.</param>
    /// <param name="origin">Per-axis origin.</param>
    /// <param name="bruteForce">Use repeated full passes.</param>
    public static NdArray BinaryClosing(NdArray input, NdArray? structure = null, int iterations = 1, NdArray? mask = null,
        bool borderValue = false, int[]? origin = null, bool bruteForce = false)
    {
        ArgumentNullException.ThrowIfNull(input);
        structure ??= GenerateBinaryStructure(input.Rank, 1);
        var tmp = BinaryDilation(input, structure, iterations, mask, borderValue, origin, bruteForce);
        return BinaryErosion(tmp, structure, iterations, mask, borderValue, origin, bruteForce);
    }

    /// <summary>
    /// Dilation repeated until stable inside <paramref name="mask"/> (scipy.ndimage.binary_propagation):
    /// a flood fill from the true seeds through the mask.
    /// </summary>
    /// <param name="input">Seeds.</param>
    /// <param name="structure">Connectivity (default: connectivity-1 cross).</param>
    /// <param name="mask">Region the propagation may enter.</param>
    /// <param name="borderValue">Value assumed outside the array.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray BinaryPropagation(NdArray input, NdArray? structure = null, NdArray? mask = null, bool borderValue = false,
        int[]? origin = null)
        => BinaryDilation(input, structure, -1, mask, borderValue, origin);

    /// <summary>
    /// Fills holes — background regions not connected to the array border
    /// (scipy.ndimage.binary_fill_holes). Background connectivity follows <paramref name="structure"/>.
    /// </summary>
    /// <param name="input">Binary image.</param>
    /// <param name="structure">Connectivity of the background (default: connectivity-1 cross).</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray BinaryFillHoles(NdArray input, NdArray? structure = null, int[]? origin = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var inMask = ToMask(input);
        var mask = new bool[inMask.Length];
        for (var i = 0; i < mask.Length; i++) mask[i] = !inMask[i];
        var reached = DilateMask(new bool[inMask.Length], shape, structure, -1, mask, true, origin, false);
        for (var i = 0; i < reached.Length; i++) reached[i] = !reached[i];
        return FromMask(reached, shape);
    }

    /// <summary>
    /// Hit-or-miss transform (scipy.ndimage.binary_hit_or_miss): true where <paramref name="structure1"/>
    /// fits the foreground and <paramref name="structure2"/> (default: its complement) fits the background.
    /// </summary>
    /// <param name="input">Binary image.</param>
    /// <param name="structure1">Pattern that must be foreground (default: connectivity-1 cross).</param>
    /// <param name="structure2">Pattern that must be background (default: NOT structure1).</param>
    /// <param name="origin1">Origin of structure1.</param>
    /// <param name="origin2">Origin of structure2 (default: origin1).</param>
    public static NdArray BinaryHitOrMiss(NdArray input, NdArray? structure1 = null, NdArray? structure2 = null, int[]? origin1 = null,
        int[]? origin2 = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        var shape = ShapeOf(input);
        var rank = shape.Length;
        var (s1, ss1) = StructureOf(structure1, rank);
        bool[] s2;
        int[] ss2;
        if (structure2 is null)
        {
            s2 = s1.Select(b => !b).ToArray();
            ss2 = ss1;
        }
        else (s2, ss2) = StructureOf(structure2, rank);
        var o1 = Expand(origin1, rank, 0, nameof(origin1));
        var o2 = origin2 is null ? o1 : Expand(origin2, rank, 0, nameof(origin2));
        var mask = ToMask(input);
        var t1 = BinaryErosionCore(mask, shape, s1, ss1, 1, null, false, o1, invert: false, false);
        var t2 = BinaryErosionCore(mask, shape, s2, ss2, 1, null, false, o2, invert: true, false);
        var r = new bool[mask.Length];
        for (var i = 0; i < r.Length; i++) r[i] = t1[i] && !t2[i];
        return FromMask(r, shape);
    }

    private static (bool[] Mask, int[] Shape) StructureOf(NdArray? structure, int rank)
    {
        structure ??= GenerateBinaryStructure(rank, 1);
        if (structure.Rank != rank) throw new ArgumentException("Structure and input must have the same rank.");
        if (structure.Size < 1) throw new ArgumentException("Structure must not be empty.");
        return (ToMask(structure), ShapeOf(structure));
    }

    private static bool[]? MaskOf(NdArray? mask, int[] shape)
    {
        if (mask is null) return null;
        if (!ShapeOf(mask).SequenceEqual(shape)) throw new ArgumentException("Mask and input must have equal shapes.");
        return ToMask(mask);
    }

    /// <summary>
    /// scipy's <c>_binary_erosion</c>. Works in the "domain" <c>d = invert ? ¬input : input</c>,
    /// eroding the true set of d (border <c>invert ? ¬border : border</c>); dilation is the inverted case.
    /// </summary>
    private static bool[] BinaryErosionCore(bool[] input, int[] shape, bool[] structure, int[] sshape, int iterations, bool[]? mask,
        bool borderValue, int[] origin, bool invert, bool bruteForce)
    {
        var rank = shape.Length;
        CheckOrigins(sshape, origin);
        var size = input.Length;
        var d = new bool[size];
        for (var i = 0; i < size; i++) d[i] = input[i] ^ invert;
        var bd = borderValue ^ invert;
        // Is the structure's own centre (shifted by origin) part of it?
        var cidx = 0;
        var sstr = StridesOf(sshape);
        for (var a = 0; a < rank; a++) cidx += (sshape[a] / 2 + origin[a]) * sstr[a];
        var centreIsTrue = structure[cidx];

        bool[] result;
        if (iterations == 1 || !centreIsTrue || bruteForce)
        {
            result = ErodeOnce(d, shape, structure, sshape, origin, bd, mask);
            var it = 1;
            var cap = iterations < 1 ? size + 2 : iterations;
            while (it < cap)
            {
                var next = ErodeOnce(result, shape, structure, sshape, origin, bd, mask);
                it++;
                var changed = !next.AsSpan().SequenceEqual(result);
                result = next;
                if (!changed) break;
            }
        }
        else
        {
            result = ErodeByPropagation(d, shape, structure, sshape, origin, bd, mask, iterations < 1 ? int.MaxValue : iterations);
        }
        if (invert) for (var i = 0; i < size; i++) result[i] = !result[i];
        return result;
    }

    /// <summary>One erosion pass over a padded copy, parallel over rows.</summary>
    private static bool[] ErodeOnce(bool[] d, int[] shape, bool[] structure, int[] sshape, int[] origin, bool bd, bool[]? mask)
    {
        var rank = shape.Length;
        var before = new int[rank];
        var after = new int[rank];
        for (var a = 0; a < rank; a++)
        {
            before[a] = sshape[a] / 2 + origin[a];
            after[a] = sshape[a] - 1 - before[a];
        }
        var pshape = new int[rank];
        for (var a = 0; a < rank; a++) pshape[a] = shape[a] + before[a] + after[a];
        var pstr = StridesOf(pshape);
        var padded = new byte[SizeOf(pshape)];
        if (bd) padded.AsSpan().Fill(1);
        var size = d.Length;
        var last = shape[rank - 1];
        var rows = size / Math.Max(1, last);
        for (var r = 0; r < rows; r++)
        {
            var rem = r;
            var pb = before[rank - 1];
            for (var a = rank - 2; a >= 0; a--) { pb += (rem % shape[a] + before[a]) * pstr[a]; rem /= shape[a]; }
            for (var k = 0; k < last; k++) padded[pb + k] = d[r * last + k] ? (byte)1 : (byte)0;
        }
        var offsets = FootprintOffsets(sshape, structure, pstr);
        var result = new bool[size];
        Run(rows, (long)size * Math.Max(1, offsets.Length), (lo, hi) =>
        {
            for (var r = lo; r < hi; r++)
            {
                var rem = r;
                var pb = 0;
                for (var a = rank - 2; a >= 0; a--) { pb += rem % shape[a] * pstr[a]; rem /= shape[a]; }
                for (var k = 0; k < last; k++)
                {
                    var x = r * last + k;
                    if (mask is not null && !mask[x]) { result[x] = d[x]; continue; }
                    var v = true;
                    var b = pb + k;
                    foreach (var off in offsets)
                        if (padded[b + off] == 0) { v = false; break; }
                    result[x] = v;
                }
            }
        });
        return result;
    }

    /// <summary>
    /// Iterated erosion as a breadth-first search. With the centre in the structure an element is
    /// removed in iteration <c>1 + min over the structure of the removal time of what it covers</c>
    /// (0 for initially-false elements and a false border), so one BFS gives the result for any
    /// iteration count: an element survives <paramref name="iterations"/> passes iff its removal
    /// time exceeds it. Elements outside the mask never change.
    /// </summary>
    private static bool[] ErodeByPropagation(bool[] d, int[] shape, bool[] structure, int[] sshape, int[] origin, bool bd, bool[]? mask,
        int iterations)
    {
        var rank = shape.Length;
        var size = d.Length;
        var strides = StridesOf(shape);
        // structure offsets as per-axis deltas
        var deltas = new List<int[]>();
        var coord = new int[rank];
        for (var i = 0; i < structure.Length; i++)
        {
            if (structure[i])
            {
                var dl = new int[rank];
                for (var a = 0; a < rank; a++) dl[a] = coord[a] - sshape[a] / 2 - origin[a];
                deltas.Add(dl);
            }
            for (var a = rank - 1; a >= 0; a--)
            {
                if (++coord[a] < sshape[a]) break;
                coord[a] = 0;
            }
        }
        var time = new int[size];
        var queue = new int[size];
        int head = 0, tail = 0;
        const int Unknown = int.MaxValue;
        for (var x = 0; x < size; x++)
        {
            if (!d[x]) { time[x] = 0; queue[tail++] = x; }
            else time[x] = Unknown; // masked-out true elements stay Unknown forever: they are never enqueued
        }
        var c = new int[rank];
        if (!bd)
        {
            // elements whose structure reaches outside the array are removed in the first pass
            for (var x = 0; x < size; x++)
            {
                if (time[x] != Unknown || (mask is not null && !mask[x])) continue;
                var rem = x;
                for (var a = rank - 1; a >= 0; a--) { c[a] = rem % shape[a]; rem /= shape[a]; }
                foreach (var dl in deltas)
                {
                    var outside = false;
                    for (var a = 0; a < rank; a++)
                    {
                        var v = c[a] + dl[a];
                        if (v < 0 || v >= shape[a]) { outside = true; break; }
                    }
                    if (outside) { time[x] = -1; break; } // provisional: time 1, enqueued after the zeros
                }
            }
            for (var x = 0; x < size; x++)
                if (time[x] == -1) { time[x] = 1; queue[tail++] = x; }
        }
        while (head < tail)
        {
            var y = queue[head++];
            var ty = time[y];
            if (ty >= iterations) continue; // its dependants would only be removed after the last iteration
            var rem = y;
            for (var a = rank - 1; a >= 0; a--) { c[a] = rem % shape[a]; rem /= shape[a]; }
            foreach (var dl in deltas)
            {
                // x + dl = y  =>  x = y - dl
                var x = 0;
                var inside = true;
                for (var a = 0; a < rank; a++)
                {
                    var v = c[a] - dl[a];
                    if (v < 0 || v >= shape[a]) { inside = false; break; }
                    x += v * strides[a];
                }
                if (!inside || time[x] != Unknown || (mask is not null && !mask[x])) continue;
                time[x] = ty + 1;
                queue[tail++] = x;
            }
        }
        var result = new bool[size];
        for (var x = 0; x < size; x++) result[x] = time[x] == Unknown || time[x] > iterations;
        return result;
    }

    // ------------------------------------------------------------------ grey morphology

    /// <summary>
    /// Grey erosion (scipy.ndimage.grey_erosion): minimum over the footprint, of
    /// <c>input − structure</c> when a (non-flat) structure is given.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Flat window size per axis (one value broadcasts).</param>
    /// <param name="footprint">Flat structuring element.</param>
    /// <param name="structure">Non-flat structuring element (values subtracted).</param>
    /// <param name="mode">Boundary mode (default Reflect, as scipy).</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray GreyErosion(NdArray input, int[]? size = null, NdArray? footprint = null, NdArray? structure = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
    {
        if (size is null && footprint is null && structure is null) throw new ArgumentException("size, footprint, or structure must be specified.");
        return MinOrMaxFilter(input, size, footprint, structure, mode, cval, origin, minimum: true);
    }

    /// <summary>
    /// Grey dilation (scipy.ndimage.grey_dilation): maximum over the reflected footprint, of
    /// <c>input + structure</c> when a structure is given.
    /// </summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Flat window size per axis.</param>
    /// <param name="footprint">Flat structuring element.</param>
    /// <param name="structure">Non-flat structuring element (values added).</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray GreyDilation(NdArray input, int[]? size = null, NdArray? footprint = null, NdArray? structure = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (size is null && footprint is null && structure is null) throw new ArgumentException("size, footprint, or structure must be specified.");
        var rank = input.Rank;
        if (structure is not null) structure = ReverseAll(structure);
        if (footprint is not null) footprint = ReverseAll(footprint);
        var o = Expand(origin, rank, 0, nameof(origin));
        var sizes = size is null ? null : Expand(size, rank, 1, nameof(size));
        for (var a = 0; a < rank; a++)
        {
            o[a] = -o[a];
            var sz = footprint is not null ? footprint.Shape[a] : structure is not null ? structure.Shape[a] : sizes![a];
            if ((sz & 1) == 0) o[a] -= 1;
        }
        return MinOrMaxFilter(input, sizes, footprint, structure, mode, cval, o, minimum: false);
    }

    private static NdArray ReverseAll(NdArray a)
    {
        var d = a.ToArray();
        Array.Reverse(d);
        return new NdArray(d, ShapeOf(a));
    }

    /// <summary>Grey erosion then grey dilation (scipy.ndimage.grey_opening).</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Flat window size per axis.</param>
    /// <param name="footprint">Flat structuring element.</param>
    /// <param name="structure">Non-flat structuring element.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray GreyOpening(NdArray input, int[]? size = null, NdArray? footprint = null, NdArray? structure = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
        => GreyDilation(GreyErosion(input, size, footprint, structure, mode, cval, origin), size, footprint, structure, mode, cval, origin);

    /// <summary>Grey dilation then grey erosion (scipy.ndimage.grey_closing).</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Flat window size per axis.</param>
    /// <param name="footprint">Flat structuring element.</param>
    /// <param name="structure">Non-flat structuring element.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray GreyClosing(NdArray input, int[]? size = null, NdArray? footprint = null, NdArray? structure = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
        => GreyErosion(GreyDilation(input, size, footprint, structure, mode, cval, origin), size, footprint, structure, mode, cval, origin);

    /// <summary>Dilation minus erosion (scipy.ndimage.morphological_gradient).</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Flat window size per axis.</param>
    /// <param name="footprint">Flat structuring element.</param>
    /// <param name="structure">Non-flat structuring element.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray MorphologicalGradient(NdArray input, int[]? size = null, NdArray? footprint = null, NdArray? structure = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
    {
        var dil = GreyDilation(input, size, footprint, structure, mode, cval, origin).ToArray();
        var ero = GreyErosion(input, size, footprint, structure, mode, cval, origin).AsContiguous().AsSpan();
        for (var i = 0; i < dil.Length; i++) dil[i] -= ero[i];
        return new NdArray(dil, ShapeOf(input));
    }

    /// <summary>Dilation + erosion − 2·input (scipy.ndimage.morphological_laplace).</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Flat window size per axis.</param>
    /// <param name="footprint">Flat structuring element.</param>
    /// <param name="structure">Non-flat structuring element.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray MorphologicalLaplace(NdArray input, int[]? size = null, NdArray? footprint = null, NdArray? structure = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
    {
        var dil = GreyDilation(input, size, footprint, structure, mode, cval, origin).ToArray();
        var ero = GreyErosion(input, size, footprint, structure, mode, cval, origin).AsContiguous().AsSpan();
        var x = input.AsContiguous().AsSpan();
        for (var i = 0; i < dil.Length; i++) dil[i] = dil[i] + ero[i] - 2 * x[i];
        return new NdArray(dil, ShapeOf(input));
    }

    /// <summary>Input minus its grey opening — bright details smaller than the element (scipy.ndimage.white_tophat).</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Flat window size per axis.</param>
    /// <param name="footprint">Flat structuring element.</param>
    /// <param name="structure">Non-flat structuring element.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray WhiteTophat(NdArray input, int[]? size = null, NdArray? footprint = null, NdArray? structure = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
    {
        var open = GreyOpening(input, size, footprint, structure, mode, cval, origin).ToArray();
        var x = input.AsContiguous().AsSpan();
        for (var i = 0; i < open.Length; i++) open[i] = x[i] - open[i];
        return new NdArray(open, ShapeOf(input));
    }

    /// <summary>Grey closing minus input — dark details smaller than the element (scipy.ndimage.black_tophat).</summary>
    /// <param name="input">Input array.</param>
    /// <param name="size">Flat window size per axis.</param>
    /// <param name="footprint">Flat structuring element.</param>
    /// <param name="structure">Non-flat structuring element.</param>
    /// <param name="mode">Boundary mode.</param>
    /// <param name="cval">Fill value.</param>
    /// <param name="origin">Per-axis origin.</param>
    public static NdArray BlackTophat(NdArray input, int[]? size = null, NdArray? footprint = null, NdArray? structure = null,
        NdImageMode mode = NdImageMode.Reflect, double cval = 0.0, int[]? origin = null)
    {
        var close = GreyClosing(input, size, footprint, structure, mode, cval, origin).ToArray();
        var x = input.AsContiguous().AsSpan();
        for (var i = 0; i < close.Length; i++) close[i] -= x[i];
        return new NdArray(close, ShapeOf(input));
    }
}
