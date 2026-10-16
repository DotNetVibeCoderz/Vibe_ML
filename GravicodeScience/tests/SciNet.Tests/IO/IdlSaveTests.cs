using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// IDL SAVE tests on files from SciPy's test suite (BSD-3-Clause), written by IDL itself.
/// Expected values are scipy 1.17.1 readsav output.
/// </summary>
public class IdlSaveTests
{
    private static IReadOnlyDictionary<string, object?> Read(string name) => IdlSave.ReadSav(Fixture(Path.Combine("scipy", name)));

    [Fact]
    public void Scalars_MatchScipy()
    {
        Assert.Equal((byte)234, Read("scalar_byte.sav")["i8u"]);
        Assert.Equal((byte)234, Read("scalar_byte_descr.sav")["i8u"]);
        Assert.Equal((short)-23456, Read("scalar_int16.sav")["i16s"]);
        Assert.Equal(-1234567890, Read("scalar_int32.sav")["i32s"]);
        Assert.Equal(-9223372036854774567L, Read("scalar_int64.sav")["i64s"]);
        Assert.Equal((ushort)65511, Read("scalar_uint16.sav")["i16u"]);
        Assert.Equal(4294967233u, Read("scalar_uint32.sav")["i32u"]);
        Assert.Equal(18446744073709529285ul, Read("scalar_uint64.sav")["i64u"]);
        Assert.Equal(-3.1234566e+37f, Read("scalar_float32.sav")["f32"]);
        Assert.Equal(-1.1976931348623156e+307, Read("scalar_float64.sav")["f64"]);
        Assert.Equal(new Complex(3.124442e+13f, -2.312442e+31f), Read("scalar_complex32.sav")["c32"]);
        Assert.Equal(new Complex(1.1987253647623157e+112, -5.198725888772916e+307), Read("scalar_complex64.sav")["c64"]);
        Assert.Equal("The quick brown fox jumps over the lazy python", Read("scalar_string.sav")["s"]);
    }

    [Fact]
    public void Pointers_AreResolved()
    {
        var h = Read("scalar_heap_pointer.sav");
        var expected = new Complex(1.1987253647623157e+112, -5.198725888772916e+307);
        Assert.Equal(expected, h["c64_pointer1"]);
        Assert.Equal(expected, h["c64_pointer2"]);

        var n = Read("null_pointer.sav");
        Assert.Null(n["point"]);
        Assert.Equal((short)5, n["check"]);

        var pa = (IdlObjectArray)Read("array_float32_pointer_1d.sav")["array1d"]!;
        Assert.Equal([123], pa.Shape);
        Assert.All(pa.Items, x => Assert.Equal(4.0f, x));

        var sp = (IdlStructure)Read("struct_pointers.sav")["pointers"]!;
        Assert.Equal(4.0f, sp["g"]);
        Assert.Equal(4.0f, sp["h"]);
        var spa = (IdlStructure)Read("struct_pointer_arrays.sav")["arrays"]!;
        Assert.Equal(2, ((IdlObjectArray)spa["g"]!).Items.Length);
        Assert.All(((IdlObjectArray)spa["h"]!).Items, x => Assert.Equal(4.0f, x));
    }

    [Fact]
    public void Arrays_HaveScipyShapes()
    {
        var a1 = (NdArray)Read("array_float32_1d.sav")["array1d"]!;
        Assert.Equal([123], a1.Shape.ToArray());
        Assert.All(a1.ToArray(), v => Assert.Equal(0.0, v));
        Assert.Equal([22, 12], ((NdArray)Read("array_float32_2d.sav")["array2d"]!).Shape.ToArray());
        Assert.Equal([2, 1, 2, 3, 4, 3, 2], ((NdArray)Read("array_float32_7d.sav")["array7d"]!).Shape.ToArray());
    }

    [Fact]
    public void Structures_MatchScipy()
    {
        var s = (IdlStructure)Read("struct_scalars.sav")["scalars"]!;
        Assert.Equal(["A", "B", "C", "D", "E", "F"], s.FieldNames);
        Assert.Equal((short)1, s["a"]);
        Assert.Equal(2, s["B"]);
        Assert.Equal(3.0f, s["c"]);
        Assert.Equal(4.0, s["d"]);
        Assert.Equal("spam", s["e"]);
        Assert.Equal(new Complex(-1, 3), s["f"]);

        var rep = (IdlStructure)Read("struct_scalars_replicated.sav")["scalars_rep"]!;
        Assert.Equal(5, rep.Count);
        Assert.All(rep.Column("e"), x => Assert.Equal("spam", x));

        foreach (var (file, name, count) in new[] { ("struct_arrays.sav", "arrays", 1), ("struct_arrays_replicated.sav", "arrays_rep", 5) })
        {
            var a = (IdlStructure)Read(file)[name]!;
            Assert.Equal(count, a.Count);
            for (var i = 0; i < count; i++)
            {
                Close([1, 2, 3], ((NdArray)a[i, "a"]!).ToArray(), 0);
                Close([4, 5, 6, 7], ((NdArray)a[i, "b"]!).ToArray(), 0);
                var c = (ComplexNdArray)a[i, "c"]!;
                Assert.Equal(new Complex(1, 2), c[0]);
                Assert.Equal(new Complex(7, 8), c[1]);
                Assert.Equal(["cheese", "bacon", "spam"], ((IdlObjectArray)a[i, "d"]!).Items.Cast<string>());
            }
        }

        var inh = (IdlStructure)Read("struct_inherit.sav")["fc"]!;
        Assert.Equal(["C", "X", "Y", "R"], inh.FieldNames);
        Assert.Equal((short)4, inh["c"]);
        Assert.Equal((short)0, inh["r"]);

        var b80 = (IdlStructure)Read("struct_arrays_byte_idl80.sav")["y"]!;
        Close([55, 66], ((NdArray)b80["x"]!).ToArray(), 0);
    }

    [Fact]
    public void Compressed_MatchesScipy()
    {
        var v = Read("various_compressed.sav");
        Assert.Equal((byte)234, v["i8u"]);
        Assert.Equal(-3.1234566e+37f, v["f32"]);
        Assert.Equal(new Complex(1.1987253647623157e+112, -5.198725888772916e+307), v["c64"]);
        Assert.Equal([4, 3, 4, 6, 5], ((NdArray)v["array5d"]!).Shape.ToArray());
        var a = (IdlStructure)v["arrays"]!;
        Assert.Equal(["cheese", "bacon", "spam"], ((IdlObjectArray)a["d"]!).Items.Cast<string>());
        Assert.Throws<InvalidDataException>(() => IdlSave.ReadSav(new MemoryStream("XX\0\u0004"u8.ToArray())));
    }
}
