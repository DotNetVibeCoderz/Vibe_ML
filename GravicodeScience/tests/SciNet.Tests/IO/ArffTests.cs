using Gravicode.Science.SciNet.IO;
using Xunit;
using static Gravicode.Science.Tests.SciNet.IOAssert;

namespace Gravicode.Science.Tests.SciNet;

/// <summary>
/// ARFF tests. scipy/*.arff come from SciPy's test suite (BSD-3-Clause); expected values are
/// scipy 1.17.1 loadarff output. scinet.arff exercises what SciPy refuses (string attributes)
/// alongside quoting, comments, missing values and dates.
/// </summary>
public class ArffTests
{
    private static string Scipy(string name) => Fixture(Path.Combine("scipy", name));

    [Fact]
    public void Iris_MatchesScipy()
    {
        var d = Arff.LoadArff(Scipy("iris.arff"));
        Assert.Equal(150, d.Count);
        Assert.Equal("iris", d.Meta.Name);
        Assert.Equal(["sepallength", "sepalwidth", "petallength", "petalwidth", "class"], d.Meta.Names());
        Assert.Equal(["numeric", "numeric", "numeric", "numeric", "nominal"], d.Meta.Types());
        Close(876.5, d.Numeric("sepallength").Sum(), 1e-9); // scipy: d['sepallength'].sum()
        Close([5.1, 4.9, 4.7], d.Numeric("sepallength")[..3], 0);
        var cls = d.Strings("class");
        IOAssert.Strings(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], new[] { cls[0], cls[50], cls[149] });
        Assert.Equal(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], d.Meta["class"].NominalValues);
        Assert.Equal(2, d.Codes("class")[149]);
    }

    [Fact]
    public void ScipyEdgeCases()
    {
        var m = Arff.LoadArff(Scipy("missing.arff"));
        Close([1, 2, double.NaN], m.Numeric("yop"), 0);

        var spaces = Arff.LoadArff(Scipy("quoted_nominal_spaces.arff"));
        IOAssert.Strings(["no  ", "  yes", "no  "], spaces.Strings("smoker")[..3]); // spaces inside quotes are kept
        var quoted = Arff.LoadArff(Scipy("quoted_nominal.arff"));
        IOAssert.Strings(["no", "yes", "no"], quoted.Strings("smoker")[..3]);

        foreach (var f in new[] { "test1.arff", "test4.arff", "test5.arff", "test11.arff" })
        {
            var t = Arff.LoadArff(Scipy(f));
            Assert.Equal(["class0", "class1", "class2", "class3"], t.Meta["class"].NominalValues);
        }
        var t4 = Arff.LoadArff(Scipy("test4.arff"));
        Close([0.1, -0.1, 1], t4.Numeric("attr0"), 0);
        IOAssert.Strings(["class1", "class2", "class3"], t4.Strings("class"));
        Assert.Equal(3, Arff.LoadArff(Scipy("test5.arff")).Count); // comments interleaved with data
        IOAssert.Strings(["C", "C", "C"], Arff.LoadArff(Scipy("test6.arff")).Strings("class"));

        var empty = Arff.LoadArff(Scipy("nodata.arff"));
        Assert.Equal(0, empty.Count);
        Assert.Equal(5, empty.Meta.Attributes.Count);

        Assert.Throws<InvalidDataException>(() => Arff.LoadArff(Scipy("test3.arff"))); // unknown type 'crap'
        // test2 declares every type spelling; its single data row is deliberately short.
        Assert.Throws<InvalidDataException>(() => Arff.LoadArff(Scipy("test2.arff")));
    }

    [Fact]
    public void Dates_MatchScipy()
    {
        var d = Arff.LoadArff(Scipy("test7.arff"));
        Assert.Equal(["date", "date", "date", "date", "date"], d.Meta.Types());
        Assert.Equal("yyyy-MM-dd HH:mm", d.Meta["attr_datetime_local"].DateFormat);
        Assert.Equal(new DateTime(1999, 1, 1), d.Dates("attr_year")[0]);
        Assert.Equal(new DateTime(2004, 12, 1), d.Dates("attr_month")[1]);
        Assert.Equal(new DateTime(1817, 4, 28), d.Dates("attr_date")[2]);
        Assert.Equal(new DateTime(1999, 1, 31, 0, 1, 0), d.Dates("attr_datetime_local")[0]);
        Assert.Null(d.Dates("attr_datetime_missing")[0]);
        Assert.Equal(new DateTime(2004, 12, 1, 23, 59, 0), d.Dates("attr_datetime_missing")[1]);

        // Time-zone patterns are refused, as by SciPy.
        Assert.Throws<NotSupportedException>(() => Arff.LoadArff(Scipy("test8.arff")));
    }

    [Fact]
    public void Relational_MatchesScipy()
    {
        var d = Arff.LoadArff(Scipy("test9.arff"));
        Assert.Equal(["relational"], d.Meta.Types());
        var bags = d.Relational("attr_date_number");
        Assert.Equal(6, bags.Length);
        var b3 = bags[3]!;
        Assert.Equal(3, b3.Count);
        Assert.Equal([new DateTime(2100, 9, 10), new DateTime(1957, 4, 17), new DateTime(1721, 1, 14)], b3.Dates("attr_date").Select(x => x!.Value));
        Close([4, 40, 400], b3.Numeric("attr_number"), 0);
    }

    [Fact]
    public void StringsQuotesAndMissing()
    {
        var d = Arff.LoadArff(Fixture("scinet.arff"));
        Assert.Equal("weather data", d.Meta.Name);
        Assert.Equal(4, d.Count);
        Assert.Equal(["outlook", "air temperature", "humidity", "note", "when", "windy"], d.Meta.Names());
        IOAssert.Strings(["sunny", "overcast", "light rain", null], d.Strings("outlook"));
        Close([85.5, double.NaN, -32.5, 1e-3], d.Numeric("air temperature"), 0);
        Close([85, 86, double.NaN, 70], d.Numeric("humidity"), 0);
        IOAssert.Strings(["hot, dry", null, "it's \"quoted\"", "?"], d.Strings("note")); // a quoted '?' is text, not missing
        Assert.Equal([new DateTime(2024, 7, 1, 14, 30, 0), null, new DateTime(2024, 1, 31, 0, 0, 5), new DateTime(1999, 12, 31, 23, 59, 59)], d.Dates("when"));
        Assert.Equal([1, 0, -1, 1], d.Codes("windy"));
        Assert.Throws<InvalidOperationException>(() => d.Numeric("note"));
    }

    [Fact]
    public void SparseAndBadValues_AreRefused()
    {
        Assert.Throws<NotSupportedException>(() => Arff.LoadArff(Fixture("sparse.arff")));
        var bad = "@relation r\n@attribute c {a, b}\n@data\nz\n";
        Assert.Throws<InvalidDataException>(() => Arff.LoadArff(new StringReader(bad)));
        var badNum = "@relation r\n@attribute x numeric\n@data\nabc\n";
        Assert.Throws<InvalidDataException>(() => Arff.LoadArff(new StringReader(badNum)));
    }
}
