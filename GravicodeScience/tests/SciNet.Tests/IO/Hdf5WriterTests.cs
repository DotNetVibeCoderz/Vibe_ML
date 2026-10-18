using System.Buffers.Binary;
using System.Numerics;
using Gravicode.Science.GraviNum;
using Gravicode.Science.SciNet.IO;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// The HDF5 writer. Files are read back with SciNet's own reader here; that the reference HDF5
/// library accepts them (h5py 3.16 / HDF5 2.0.0 opens, reads and then modifies them in place) is
/// checked by tools/verify/hdf5_interop.py, which these tests cannot replace. The checksum is
/// pinned against checksums HDF5 itself wrote into the fixtures.
/// </summary>
public class Hdf5WriterTests
{
    private static byte[] Bytes(Hdf5Writer w)
    {
        var ms = new MemoryStream();
        w.Save(ms);
        return ms.ToArray();
    }

    private static Hdf5File Reopen(Hdf5Writer w) => Hdf5File.Open(new MemoryStream(Bytes(w)));

    [Fact]
    public void Lookup3_MatchesChecksumsHdf5Wrote()
    {
        // h5_v2.h5 was written by HDF5 2.0.0: its superblock and root object header carry lookup3 checksums.
        var raw = File.ReadAllBytes(Fixture("h5_v2.h5"));
        Assert.Equal(BinaryPrimitives.ReadUInt32LittleEndian(raw.AsSpan(44)), H5Checksum.Lookup3(raw.AsSpan(0, 44)));
        var root = (int)BinaryPrimitives.ReadInt64LittleEndian(raw.AsSpan(36));
        Assert.Equal((byte)'O', raw[root]);
        var flags = raw[root + 5];
        Assert.Equal(0, flags & 0x30); // no times or phase change fields in this header
        var sizeBytes = 1 << (flags & 3);
        var size = 0;
        for (var i = 0; i < sizeBytes; i++) size |= raw[root + 6 + i] << (8 * i);
        var end = root + 6 + sizeBytes + size;
        Assert.Equal(BinaryPrimitives.ReadUInt32LittleEndian(raw.AsSpan(end)), H5Checksum.Lookup3(raw.AsSpan(root, end - root)));
        // Bob Jenkins' own test vectors from lookup3.c (driver5).
        Assert.Equal(0xdeadbeefu, H5Checksum.Lookup3([]));
        Assert.Equal(0xbd5b7ddeu, H5Checksum.Lookup3([], 0xdeadbeef));
        Assert.Equal(0x17770551u, H5Checksum.Lookup3("Four score and seven years ago"u8));
        Assert.Equal(0xcd628161u, H5Checksum.Lookup3("Four score and seven years ago"u8, 1));
    }

    [Fact]
    public void Layout_SuperblockAndHeaderPins()
    {
        var w = new Hdf5Writer(userBlockSize: 1024) { UserBlock = "user text"u8.ToArray() };
        w.Root.CreateDataset("x", new[] { 1.0, 2.0 });
        var b = Bytes(w);
        Assert.Equal("user text"u8.ToArray(), b[..9]);
        var sb = b.AsSpan(1024);
        Assert.True(sb[..8].SequenceEqual(new byte[] { 0x89, (byte)'H', (byte)'D', (byte)'F', 0x0D, 0x0A, 0x1A, 0x0A }));
        Assert.Equal(2, sb[8]);                                                        // superblock version 2
        Assert.Equal(8, sb[9]);                                                        // offsets
        Assert.Equal(8, sb[10]);                                                       // lengths
        Assert.Equal(1024L, BinaryPrimitives.ReadInt64LittleEndian(sb[12..]));         // base address = user block
        Assert.Equal(-1L, BinaryPrimitives.ReadInt64LittleEndian(sb[20..]));           // no extension
        Assert.Equal(b.LongLength, BinaryPrimitives.ReadInt64LittleEndian(sb[28..])); // end of file, absolute
        Assert.Equal(48L, BinaryPrimitives.ReadInt64LittleEndian(sb[36..]));           // root header right after
        Assert.Equal(BinaryPrimitives.ReadUInt32LittleEndian(sb[44..]), H5Checksum.Lookup3(sb[..44]));
        var oh = sb[48..];
        Assert.True(oh[..4].SequenceEqual("OHDR"u8));
        Assert.Equal(2, oh[4]);
        Assert.Equal(0x06, oh[5]); // 4-byte chunk size, attribute creation order tracked
        var size = BinaryPrimitives.ReadInt32LittleEndian(oh[6..]);
        Assert.Equal(BinaryPrimitives.ReadUInt32LittleEndian(oh[(10 + size)..]), H5Checksum.Lookup3(oh[..(10 + size)]));

        using var f = Hdf5File.Open(new MemoryStream(b));
        Assert.Equal(2, f.SuperblockVersion);
        Assert.Equal(1024, f.UserBlockSize);
        Assert.True(Hdf5File.IsHdf5(new MemoryStream(b)));
        Assert.Equal(new[] { 1.0, 2.0 }, f.Root.Dataset("x").Read().ToArray());
        Assert.Throws<ArgumentException>(() => new Hdf5Writer(userBlockSize: 600));
    }

    [Fact]
    public void EveryElementType_RoundTrips()
    {
        var w = new Hdf5Writer();
        var g = w.Root.CreateGroup("t");
        g.CreateDataset("f8", new NdArray(Enumerable.Range(0, 24).Select(i => i * 0.5 - 3).ToArray(), 2, 3, 4));
        g.CreateDataset("f4", new[] { 1.5f, -2.25f, float.MaxValue });
        g.CreateDataset("f2", new[] { (Half)0.5, (Half)(-1.25), Half.MaxValue });
        g.CreateDataset("i1", new sbyte[] { -128, -1, 127 });
        g.CreateDataset("u1", new byte[] { 0, 200, 255 });
        g.CreateDataset("i2", new short[,] { { -32768, -2 }, { 3, 32767 } });
        g.CreateDataset("u2", new ushort[] { 0, 65535 });
        g.CreateDataset("i4", new[] { int.MinValue, int.MaxValue });
        g.CreateDataset("u4", new[] { 0u, uint.MaxValue });
        g.CreateDataset("i8", new[] { long.MinValue, long.MaxValue });
        g.CreateDataset("u8", new[] { 0UL, ulong.MaxValue });
        g.CreateDataset("bool", new[] { true, false, true });
        g.CreateDataset("c16", new ComplexNdArray([new(1, 2), new(-3, 0.5), new(0, -1), new(4.25, 0)], 2, 2));
        g.CreateDataset("vlen", new[] { "alpha", "日本語 €", "" });
        g.CreateDataset("fixed", new[] { "ab", "cde" }, fixedLengthStrings: true);
        g.CreateDataset("scalar", 2.5);
        g.CreateDataset("empty", new double[0, 3]);
        g.CreateDataset("reshaped", new[] { 1, 2, 3, 4, 5, 6 }, shape: [2, 3]);
        var refs = g.CreateDataset("refs", new Hdf5WriterNode[] { g, w.Root });

        using var f = Reopen(w);
        var t = f.Root.Group("t");
        Assert.Equal(["f8", "f4", "f2", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "bool", "c16", "vlen", "fixed", "scalar", "empty", "reshaped", "refs"], t.Keys); // creation order
        var f8 = t.Dataset("f8");
        Assert.Equal(new[] { 2, 3, 4 }, f8.Shape);
        Assert.Equal(Enumerable.Range(0, 24).Select(i => i * 0.5 - 3), f8.Read().ToArray());
        Assert.Equal("<f8", f8.DataType.ToString());
        Assert.Equal(new[] { 1.5f, -2.25f, float.MaxValue }, (float[])t.Dataset("f4").ReadArray());
        Assert.Equal(new[] { (Half)0.5, (Half)(-1.25), Half.MaxValue }, (Half[])t.Dataset("f2").ReadArray());
        Assert.Equal(new sbyte[] { -128, -1, 127 }, (sbyte[])t.Dataset("i1").ReadArray());
        Assert.Equal(new byte[] { 0, 200, 255 }, (byte[])t.Dataset("u1").ReadArray());
        Assert.Equal(new short[] { -32768, -2, 3, 32767 }, (short[])t.Dataset("i2").ReadArray());
        Assert.Equal(new[] { 2, 2 }, t.Dataset("i2").Shape);
        Assert.Equal(new ushort[] { 0, 65535 }, (ushort[])t.Dataset("u2").ReadArray());
        Assert.Equal(new[] { int.MinValue, int.MaxValue }, (int[])t.Dataset("i4").ReadArray());
        Assert.Equal(new[] { 0u, uint.MaxValue }, (uint[])t.Dataset("u4").ReadArray());
        Assert.Equal(new[] { long.MinValue, long.MaxValue }, (long[])t.Dataset("i8").ReadArray());
        Assert.Equal(new[] { 0UL, ulong.MaxValue }, (ulong[])t.Dataset("u8").ReadArray());
        Assert.True(t.Dataset("bool").DataType.IsBoolean);
        Assert.Equal(new[] { true, false, true }, (bool[])t.Dataset("bool").ReadArray());
        var c = t.Dataset("c16").ReadCompound();
        Assert.Equal(["r", "i"], c.Keys);
        Assert.Equal(new[] { 1.0, -3, 0, 4.25 }, (double[])c["r"]);
        Assert.Equal(new[] { 2.0, 0.5, -1, 0 }, (double[])c["i"]);
        Assert.True(t.Dataset("vlen").DataType.IsVariableLengthString);
        // An empty variable-length string is written as HDF5 writes it (a null heap ID), which SciNet's reader reports as null.
        Strings(["alpha", "日本語 €", null], t.Dataset("vlen").ReadStrings(), "vlen");
        Strings(["ab", "cde"], t.Dataset("fixed").ReadStrings(), "fixed");
        Assert.Equal(Hdf5TypeClass.String, t.Dataset("fixed").DataType.Class);
        Assert.True(t.Dataset("scalar").IsScalar);
        Assert.Equal(2.5, t.Dataset("scalar").Read().ToArray().Single());
        Assert.Equal(new[] { 0, 3 }, t.Dataset("empty").Shape);
        Assert.Equal(new[] { 2, 3 }, t.Dataset("reshaped").Shape);
        var r = t.Dataset("refs").ReadReferences();
        Assert.Equal(f.Root.Address, f.Dereference(r[1]).Address);
        Assert.Equal(t.Address, f.Dereference(r[0]).Address);
    }

    [Fact]
    public void Attributes_RoundTripInCreationOrder()
    {
        var w = new Hdf5Writer();
        var d = w.Root.CreateDataset("d", new[] { 1.0 });
        w.Root.SetAttribute("zeta", "last letter, first attribute");
        w.Root.SetAttribute("alpha", 3);
        w.Root.SetAttribute("pi", Math.PI);
        w.Root.SetAttribute("arr", new short[] { 1, -2, 3 });
        w.Root.SetAttribute("mat", new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        w.Root.SetAttribute("flag", true);
        w.Root.SetAttribute("z", new Complex(1, -2));
        w.Root.SetAttribute("names", new[] { "x", "yy" });
        w.Root.SetAttribute("fixed", "text", fixedLengthStrings: true);
        w.Root.SetAttribute("ref", d);
        w.Root.SetAttribute("alpha", 4); // replaces, keeps its place
        for (var i = 0; i < 12; i++) d.SetAttribute($"a{i:00}", i); // more than HDF5's default compact limit of 8

        using var f = Reopen(w);
        var a = f.Root.Attributes;
        Assert.Equal(["zeta", "alpha", "pi", "arr", "mat", "flag", "z", "names", "fixed", "ref"], a.Keys);
        Assert.Equal("last letter, first attribute", a["zeta"].Value);
        Assert.Equal(4, a["alpha"].Value);
        Assert.True(a["alpha"].IsScalar);
        Assert.Equal(Math.PI, a["pi"].Value);
        Assert.Equal(new short[] { 1, -2, 3 }, (short[])a["arr"].Value!);
        Assert.Equal(new[] { 2, 3 }, a["mat"].Shape);
        Assert.Equal(true, a["flag"].Value);
        var z = (Dictionary<string, object?>)a["z"].Value!;
        Assert.Equal(1.0, z["r"]);
        Assert.Equal(-2.0, z["i"]);
        Strings(["x", "yy"], (string?[])a["names"].Value!, "names");
        Assert.Equal("text", a["fixed"].AsString());
        Assert.Equal(f.Root.Dataset("d").Address, ((Hdf5Reference)a["ref"].Value!).Address);
        var da = f.Root.Dataset("d").Attributes;
        Assert.Equal(Enumerable.Range(0, 12).Select(i => $"a{i:00}"), da.Keys);
        Assert.Equal(Enumerable.Range(0, 12), da.Values.Select(x => (int)x.Value!));
    }

    [Fact]
    public void ChunkedDeflatedAndExtendible()
    {
        var big = new double[37 * 23];
        for (var k = 0; k < big.Length; k++) big[k] = k % 17 - 8;
        var w = new Hdf5Writer();
        w.Root.CreateDataset("z", big, shape: [37, 23], compressionLevel: 6);
        w.Root.CreateDataset("ext", Enumerable.Range(0, 15).Select(i => (double)i).ToArray(), shape: [5, 3], maxShape: [-1, 3]);
        w.Root.CreateDataset("ext0", new double[0, 2], maxShape: [-1, 2]);
        w.Root.CreateDataset("fixedmax", new[] { 1, 2 }, maxShape: [2]); // same as the shape: stays contiguous
        var bytes = Bytes(w);
        Assert.True(bytes.Length < 37 * 23 * 8, $"deflate should shrink the file, got {bytes.Length} bytes");

        using var f = Hdf5File.Open(new MemoryStream(bytes));
        var z = f.Root.Dataset("z");
        Assert.Equal(Hdf5Layout.Chunked, z.Layout);
        Assert.Equal(new[] { 37, 23 }, z.ChunkShape);
        Assert.Equal("deflate", Assert.Single(z.Filters).Name);
        Assert.Equal(6u, z.Filters[0].ClientData[0]);
        Assert.Equal(big, z.Read().ToArray());
        Assert.Equal(new[] { -7.0, -6, -5 }, z.ReadSlice([0, 1], [1, 3]).ToArray()); // row 0, columns 1..3: k % 17 - 8 for k = 1..3
        var ext = f.Root.Dataset("ext");
        Assert.Equal(new long[] { -1, 3 }, ext.MaxShape);
        Assert.Equal(Hdf5Layout.Chunked, ext.Layout);
        Assert.Equal(Enumerable.Range(0, 15).Select(i => (double)i), ext.Read().ToArray());
        var e0 = f.Root.Dataset("ext0");
        Assert.Equal(new[] { 0, 2 }, e0.Shape);
        Assert.Equal(0, e0.Read().Size);
        Assert.Equal(Hdf5Layout.Contiguous, f.Root.Dataset("fixedmax").Layout);
    }

    [Fact]
    public void ManyMembersAndNesting()
    {
        var w = new Hdf5Writer();
        var g = w.Root;
        for (var depth = 0; depth < 10; depth++) g = g.CreateGroup($"level{depth}");
        g.CreateDataset("deep", new[] { 42 });
        var wide = w.Root.CreateGroup("wide");
        for (var i = 0; i < 500; i++) wide.CreateDataset($"m{i:000}", (double)i);
        w.Root.CreateGroup("ünï").SetAttribute("ñame", "ü");

        using var f = Reopen(w);
        Assert.Equal(new[] { 42 }, (int[])f.Root.Dataset(string.Join("/", Enumerable.Range(0, 10).Select(i => $"level{i}")) + "/deep").ReadArray());
        var wg = f.Root.Group("wide");
        Assert.Equal(500, wg.Keys.Count);
        Assert.Equal(437.0, wg.Dataset("m437").Read().ToArray().Single());
        Assert.Equal("ü", f.Root.Group("ünï").Attributes["ñame"].Value);
    }

    [Fact]
    public void RefusesWhatItCannotWrite()
    {
        var w = new Hdf5Writer();
        Assert.Throws<ArgumentException>(() => w.Root.CreateGroup("a/b"));
        Assert.Throws<ArgumentException>(() => w.Root.CreateGroup(""));
        w.Root.CreateGroup("x");
        Assert.Throws<ArgumentException>(() => w.Root.CreateDataset("x", 1.0));
        Assert.Throws<ArgumentException>(() => w.Root.CreateDataset("y", new[] { 1.0, 2 }, shape: [3]));
        Assert.Throws<ArgumentException>(() => w.Root.CreateDataset("y", new[] { 1.0, 2 }, maxShape: [1]));
        Assert.Throws<ArgumentException>(() => w.Root.CreateDataset("y", new Hdf5WriterNode[] { w.Root }, compressionLevel: 4));
        Assert.Throws<ArgumentException>(() => w.Root.CreateDataset("y", new[] { DateTime.Now }));
        Assert.Throws<ArgumentException>(() => w.Root.SetAttribute("big", new double[10000]));

        var other = new Hdf5Writer();
        w.Root.CreateDataset("foreign", new Hdf5WriterNode[] { other.Root });
        Assert.Throws<InvalidOperationException>(() => Bytes(w));
    }

    [Fact]
    public void WritesToANonSeekableStream()
    {
        var w = new Hdf5Writer();
        w.Root.CreateDataset("d", new[] { 1, 2, 3 }, compressionLevel: 1);
        var sink = new MemoryStream();
        w.Save(new ForwardOnly(sink));
        using var f = Hdf5File.Open(new MemoryStream(sink.ToArray()));
        Assert.Equal(new[] { 1, 2, 3 }, (int[])f.Root.Dataset("d").ReadArray());
    }

    private sealed class ForwardOnly(Stream inner) : Stream
    {
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() => inner.Flush();
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => inner.Write(buffer, offset, count);
    }
}
