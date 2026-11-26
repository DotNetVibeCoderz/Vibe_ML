using Gravicode.Science.DaskNet;
using Xunit;

namespace Gravicode.Science.Tests.DaskNet;

public sealed class BagTests
{
    private static readonly int[] Data = Enumerable.Range(0, 1000).Select(i => (i * 7919) % 1013).ToArray();

    [Fact]
    public void Map_filter_flatmap_match_linq_in_order()
    {
        var bag = Bag.FromSequence(Data, 7);
        Assert.Equal(7, bag.PartitionCount);
        var expected = Data.Select(x => x * 2).Where(x => x % 3 != 0).SelectMany(x => new[] { x, -x }).ToList();
        Assert.Equal(expected, bag.Map(x => x * 2).Filter(x => x % 3 != 0).FlatMap(x => new[] { x, -x }).Compute());
    }

    [Fact]
    public void Linq_query_syntax_works()
    {
        var bag = Bag.FromSequence(Data, 4);
        var query = from x in bag where x > 500 select x / 2;
        Assert.Equal(Data.Where(x => x > 500).Select(x => x / 2).ToList(), query.Compute());

        var pairs = from x in Bag.FromSequence(new[] { 1, 2 }, 2) from y in new[] { 10, 20 } select x * y;
        Assert.Equal(new[] { 10, 20, 20, 40 }, pairs.Compute());
    }

    [Fact]
    public void Reductions_match_linq()
    {
        var bag = Bag.FromSequence(Data, 13);
        Assert.Equal(Data.Length, bag.Count().Compute());
        Assert.Equal(Data.Sum(x => (double)x), bag.Sum(x => x).Compute());
        Assert.Equal(Data.Min(), bag.Min(x => x).Compute());
        Assert.Equal(Data.Max(), bag.Max(x => x).Compute());
        Assert.True(Math.Abs(Data.Average() - bag.Mean(x => x).Compute()) < 1e-12);
        Assert.Equal(Data.Aggregate((a, b) => Math.Max(a, b)), bag.Reduce(Math.Max).Compute());
        Assert.True(bag.Any(x => x == 1012).Compute());
        Assert.False(bag.All(x => x < 1000).Compute());
    }

    [Fact]
    public void Variance_is_stable_far_from_zero()
    {
        // Values near 1e9 with unit spread: a sum-of-squares formula has no correct digits left.
        var values = Enumerable.Range(0, 10_000).Select(i => 1e9 + (i % 10)).ToArray();
        var mean = values.Average();
        var expected = values.Sum(v => (v - mean) * (v - mean)) / values.Length;
        var actual = Bag.FromSequence(values, 17).Var(x => x).Compute();
        Assert.True(Math.Abs(actual - expected) < 1e-9 * expected, $"{actual} vs {expected}");
        Assert.True(Math.Abs(Bag.FromSequence(values, 17).Std(x => x, ddof: 1).Compute()
            - Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / (values.Length - 1))) < 1e-9);
    }

    [Fact]
    public void Fold_uses_a_fresh_seed_per_partition()
    {
        var lists = Bag.FromSequence(Data, 8).Fold(() => new List<int>(), (acc, x) => { acc.Add(x); return acc; },
            (a, b) => { a.AddRange(b); return a; }).Compute();
        Assert.Equal(Data, lists);
    }

    [Fact]
    public void Reduce_of_empty_partitions_skips_them()
    {
        var bag = Bag.FromSequence(Data, 8).Filter(x => x == 1012);
        Assert.Equal(1012, bag.Reduce((a, b) => a + b).Compute());
        Assert.Throws<DaskTaskException>(() => bag.Filter(_ => false).Reduce((a, b) => a + b).Compute());
    }

    [Fact]
    public void Frequencies_and_foldby_agree_with_groupby()
    {
        var words = "the quick brown fox jumps over the lazy dog the end".Split(' ');
        var bag = Bag.FromSequence(Enumerable.Repeat(words, 50).SelectMany(w => w), 6);

        var freq = bag.Frequencies().Compute();
        Assert.Equal(("the", 150L), freq[0]);
        Assert.Equal(words.Distinct().Count(), freq.Length);

        var byLength = bag.FoldBy(w => w.Length, () => 0, (acc, _) => acc + 1, (a, b) => a + b).Compute()
            .OrderBy(p => p.Key).ToArray();
        var expected = Enumerable.Repeat(words, 50).SelectMany(w => w).GroupBy(w => w.Length)
            .OrderBy(g => g.Key).Select(g => (g.Key, g.Count())).ToArray();
        Assert.Equal(expected, byLength);
    }

    [Fact]
    public void GroupBy_puts_each_key_in_one_partition()
    {
        var bag = Bag.FromSequence(Data, 9);
        var groups = bag.GroupBy(x => x % 10, partitions: 4);
        Assert.Equal(4, groups.PartitionCount);
        var result = groups.Compute();
        Assert.Equal(10, result.Count);
        Assert.Equal(10, result.Select(g => g.Key).Distinct().Count());
        foreach (var (key, items) in result)
            Assert.Equal(Data.Where(x => x % 10 == key).Order(), items.Order());
    }

    [Fact]
    public void Distinct_removes_duplicates_across_partitions()
    {
        var bag = Bag.FromSequence(Data.Concat(Data), 11);
        Assert.Equal(Data.Distinct().Order(), bag.Distinct().Compute().Order());
    }

    [Fact]
    public void TopK_and_take()
    {
        var bag = Bag.FromSequence(Data, 10);
        Assert.Equal(Data.OrderByDescending(x => x).Take(5), bag.TopK(5, x => x).Compute());
        Assert.Equal(Data.Take(150), bag.Take(150));
    }

    [Fact]
    public void Broadcast_value_reaches_every_partition()
    {
        var calls = 0;
        var offset = Dask.Delayed(() => { Interlocked.Increment(ref calls); return 1000; });
        var result = Bag.FromSequence(Data, 6).Map(offset, (x, o) => x + o).Compute();
        Assert.Equal(Data.Select(x => x + 1000), result);
        Assert.Equal(1, calls);
    }

    [Fact]
    public void Repartition_zip_and_concat_keep_order()
    {
        var bag = Bag.FromSequence(Data, 3);
        var re = bag.Repartition(10);
        Assert.Equal(10, re.PartitionCount);
        Assert.Equal(Data, re.Compute());
        Assert.Equal(Data.Zip(Data.Select(x => -x)), bag.Zip(bag.Map(x => -x)).Compute());
        Assert.Equal(Data.Concat(Data), Bag.Concat(bag, re).Compute());
    }

    [Fact]
    public void ReadText_reads_globs_and_blocks()
    {
        var dir = Directory.CreateTempSubdirectory("dasknet-text-").FullName;
        try
        {
            var all = new List<string>();
            for (var f = 0; f < 3; f++)
            {
                var lines = Enumerable.Range(0, 300).Select(i => $"file{f}-line{i}").ToArray();
                all.AddRange(lines);
                File.WriteAllLines(Path.Combine(dir, $"part{f}.txt"), lines);
            }
            File.WriteAllText(Path.Combine(dir, "ignore.csv"), "nope");

            var whole = Bag.ReadText(Path.Combine(dir, "*.txt"));
            Assert.Equal(3, whole.PartitionCount);
            Assert.Equal(all, whole.Compute());

            var blocks = Bag.ReadText(Path.Combine(dir, "*.txt"), blockSize: 500);
            Assert.True(blocks.PartitionCount > 10);
            Assert.Equal(all, blocks.Compute());
            Assert.Equal(900, blocks.Count().Compute());
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void Persist_breaks_lineage()
    {
        var calls = 0;
        var bag = Bag.FromSequence(Data, 4).Map(x => { Interlocked.Increment(ref calls); return x; }).Persist();
        Assert.Equal(Data.Length, calls);
        Assert.Equal(Data.Length, bag.Count().Compute());
        Assert.Equal(Data.Length, calls);
    }
}
