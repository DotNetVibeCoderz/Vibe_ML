using Gravicode.Science.DaskNet;
using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;
using static Gravicode.Science.DaskNet.Expr;

namespace ScienceGallery.Showcases.Finance;

/// <summary>A month of card transactions as a partitioned, lazily planned DaskFrame: filter, join, group.</summary>
public sealed class TransactionLedgerDask : Showcase
{
    public override string Title => "Summarise a month of card payments by customer segment";
    public override Domain Domain => Domain.Finance;
    public override GraviLibrary[] Libraries => [GraviLibrary.DaskNet, GraviLibrary.GraviFrame, GraviLibrary.GraviNum];
    public override string Summary => "300,000 transactions in 16 partitions: filter, broadcast-join the customer table, group by segment and channel.";
    public override string Scenario =>
        "A bank's payments analyst needs settled card volume by customer segment and channel every month, plus " +
        "how much of it is high-value. The ledger is too big to eyeball and arrives in partitions, so it becomes " +
        "a DaskFrame: a filter written as an expression, a join with the small customer table, and a grouped " +
        "aggregate. Nothing runs until Compute, and Explain shows what the optimiser did with the plan - the " +
        "filter moved below the join and the customer table is broadcast to every partition instead of shuffled.";

    public override void Run(ShowcaseOutput output)
    {
        // The customer table: 6,000 customers in four segments.
        string[] segments = ["Mass", "Affluent", "Private", "SME"];
        double[] segmentShare = [0.62, 0.22, 0.04, 0.12];
        double[] segmentTicket = [0.35, 0.9, 3.2, 1.8];          // typical payment, Rp million
        string[] channels = ["Card present", "E-commerce", "Mobile wallet", "Recurring"];
        const int customers = 6_000, transactions = 300_000;
        var rng = new GraviRandom(91);
        var segmentOf = Enumerable.Range(0, customers).Select(_ => rng.Categorical(segmentShare)).ToArray();
        var customerTable = new DataFrame(
        [
            new NumericSeries("customer_id", Enumerable.Range(0, customers).Select(i => (double)i).ToArray()),
            new TextSeries("segment", segmentOf.Select(s => segments[s]).ToArray()),
        ]);

        // The ledger: one row per payment attempt. About 3% are declined or reversed.
        var (id, channel, status, amount) = (new double[transactions], new string[transactions], new string[transactions], new double[transactions]);
        for (var t = 0; t < transactions; t++)
        {
            var c = rng.Next(customers);
            var s = segmentOf[c];
            id[t] = c;
            channel[t] = channels[rng.Categorical(s == 3 ? [0.15, 0.30, 0.10, 0.45] : s == 2 ? [0.35, 0.45, 0.15, 0.05] : [0.38, 0.27, 0.28, 0.07])];
            status[t] = rng.NextBernoulli(0.03) ? (rng.NextBernoulli(0.7) ? "declined" : "reversed") : "settled";
            amount[t] = Math.Round(rng.LogNormal(Math.Log(segmentTicket[s]), 0.9), 4);
        }
        var ledger = new DataFrame(
        [
            new NumericSeries("customer_id", id), new TextSeries("channel", channel),
            new TextSeries("status", status), new NumericSeries("amount", amount),
        ]);

        // The plan. Each line is a step in a recipe - building it does no work.
        var payments = DaskFrame.FromFrame(ledger, partitions: 16);
        var report = payments
            .Join(DaskFrame.FromFrame(customerTable, partitions: 1), "customer_id")
            .Filter(Col("status").Eq("settled") & (Col("amount") > 0))
            .WithColumn("high_value", If(Col("amount") >= 10, Col("amount"), 0))
            .GroupBy("segment", "channel")
            .Agg(("amount", "sum"), ("amount", "mean"), ("amount", "count"), ("high_value", "sum"))
            .SortBy("amount_sum", ascending: false);

        var result = report.Compute();             // thread-pool scheduler: 16 partitions in parallel
        var settledTotal = payments.Filter(Col("status").Eq("settled")).Sum("amount").Compute();

        // A plain GraviFrame check of one number, so the partitioned answer is not taken on trust.
        var direct = ledger.Filter(r => r.String("status") == "settled").Numeric("amount").Values.Sum();

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var x = Enumerable.Range(0, segments.Length).Select(i => (double)i).ToArray();
        var bottom = new double[segments.Length];
        for (var c = 0; c < channels.Length; c++)
        {
            var values = segments.Select(s => Cell(result, s, channels[c], "amount_sum") / 1000).ToArray();     // Rp billion
            left.Bar(x, values, 0.7, bottoms: (double[])bottom.Clone(), label: channels[c], color: GalleryCharts.Cycle[c]);
            for (var i = 0; i < values.Length; i++) bottom[i] += values[i];
        }
        left.SetXTicks(x, segments);
        left.SetTitle("Settled volume by segment and channel").SetYLabel("Rp billion");
        left.Legend(LegendLocation.UpperRight);

        var highShare = segments.Select(s => channels.Sum(c => Cell(result, s, c, "high_value_sum")) /
                                             channels.Sum(c => Cell(result, s, c, "amount_sum")) * 100).ToArray();
        right.Bar(x, highShare, 0.6, color: GalleryCharts.Of(GraviLibrary.DaskNet));
        right.SetXTicks(x, segments);
        right.SetTitle("Share of volume in payments of Rp 10 million or more").SetYLabel("% of settled volume");
        output.Chart(figure);

        var top = result.Text("segment")[0] + " / " + result.Text("channel")[0];
        output.Metric("Transactions", transactions.ToString("N0"), "16 partitions");
        output.Metric("Settled volume", $"Rp {settledTotal / 1000:F1} billion", $"GraviFrame on one frame agrees to {Math.Abs(settledTotal - direct):E1}");
        output.Metric("Largest cell", top, $"Rp {result.Numeric("amount_sum")[0] / 1000:F1} billion");
        var privateAverage = channels.Sum(c => Cell(result, "Private", c, "amount_sum")) / channels.Sum(c => Cell(result, "Private", c, "amount_count"));
        output.Metric("Average Private payment", $"Rp {privateAverage:F2} million", "from the per-cell sums and counts");
        output.Line($"Private customers are {segmentShare[2] * 100:F0}% of the base, but {highShare[2]:F0}% of their volume is in high-value payments - " +
                    "the segment a large-payment review rule should be tuned for.");
        output.Line("Explain() - the optimised plan (the status filter now sits below the join, and the customer table is broadcast):\n" + report.Explain());
    }

    /// <summary>One aggregated value for a (segment, channel) pair, or 0 when the pair has no rows.</summary>
    private static double Cell(DataFrame result, string segment, string channel, string column)
    {
        for (var i = 0; i < result.RowCount; i++)
            if (result.Text("segment")[i] == segment && result.Text("channel")[i] == channel)
                return result.Numeric(column)[i];
        return 0;
    }
}
