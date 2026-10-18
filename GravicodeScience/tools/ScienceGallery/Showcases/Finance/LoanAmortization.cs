using Gravicode.Science.GraviFrame;
using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Finance;

/// <summary>A mortgage schedule, its true annual cost including fees, and what an extra payment saves.</summary>
public sealed class LoanAmortization : Showcase
{
    public override string Title => "What does this mortgage really cost per year?";
    public override Domain Domain => Domain.Finance;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet, GraviLibrary.GraviFrame];
    public override string Summary => "Build the amortisation schedule, then solve for the effective rate with fees using Brent's method.";
    public override string Scenario =>
        "A family compares a 15-year home loan of Rp 600 million advertised at 8.5% a year. The bank also " +
        "charges a 1.5% arrangement fee and monthly insurance, so the advertised rate is not what the loan " +
        "costs. The effective annual rate is the discount rate at which everything the family receives equals " +
        "everything it pays back; there is no formula for it, so a root finder solves for it. The schedule also " +
        "shows how slowly the principal falls at first, and what paying a little extra each month saves.";

    public override void Run(ShowcaseOutput output)
    {
        const double principal = 600_000_000, annualRate = 0.085, feeShare = 0.015, insurance = 95_000;   // Rp
        const int months = 15 * 12;
        var monthlyRate = annualRate / 12;

        // The level payment that clears the loan in exactly n months: P r / (1 − (1 + r)^−n).
        var payment = principal * monthlyRate / (1 - Math.Pow(1 + monthlyRate, -months));

        // The schedule, month by month, with an optional extra payment towards the principal.
        DataFrame Schedule(double extra)
        {
            var (month, interest, repaid, balance) = (new List<double>(), new List<double>(), new List<double>(), new List<double>());
            var owed = principal;
            for (var m = 1; owed > 0.5; m++)
            {
                var i = owed * monthlyRate;
                var p = Math.Min(payment + extra - i, owed);
                owed -= p;
                month.Add(m); interest.Add(i); repaid.Add(p); balance.Add(owed);
            }
            return DataFrame.FromColumns(new Dictionary<string, double[]>
            {
                ["month"] = month.ToArray(), ["year"] = month.Select(m => Math.Ceiling(m / 12)).ToArray(),
                ["interest"] = interest.ToArray(), ["principal"] = repaid.ToArray(), ["balance"] = balance.ToArray(),
            });
        }
        var plan = Schedule(0);
        var faster = Schedule(1_000_000);
        var yearly = plan.GroupBy("year").Sum("interest", "principal");

        // Effective rate: the monthly rate r at which the net amount received (principal minus the fee)
        // equals the present value of every payment plus insurance. Brent's method finds the root.
        var cashOut = payment + insurance;
        double Gap(double r) => principal * (1 - feeShare) - cashOut * (1 - Math.Pow(1 + r, -months)) / r;
        var effectiveMonthly = Optimize.Brentq(Gap, 1e-6, 0.05).Root;
        var effectiveAnnual = Math.Pow(1 + effectiveMonthly, 12) - 1;

        var totalInterest = plan.Numeric("interest").Values.Sum();
        var savedInterest = totalInterest - faster.Numeric("interest").Values.Sum();
        var monthsSaved = plan.RowCount - faster.RowCount;
        var halfway = plan.Numeric("balance").Values.ToList().FindIndex(b => b <= principal / 2) + 1;

        var figure = GalleryCharts.Figure(11, 4.8);
        var (left, right) = (figure.AddSubplot(1, 2, 1), figure.AddSubplot(1, 2, 2));
        var years = yearly.Numeric("year").Values;
        var interestPerYear = yearly.Numeric("interest").Values.Select(v => v / 1e6).ToArray();
        var principalPerYear = yearly.Numeric("principal").Values.Select(v => v / 1e6).ToArray();
        left.Bar(years, principalPerYear, 0.8, label: "principal repaid", color: GalleryCharts.Of(GraviLibrary.GraviLearn));
        left.Bar(years, interestPerYear, 0.8, bottoms: principalPerYear, label: "interest", color: Color.Parse("#E8A317"));
        left.SetTitle("Where each year's payments go").SetXLabel("year").SetYLabel("Rp million");
        left.Legend(LegendLocation.UpperLeft, columns: 2);
        left.SetYLim(0, (interestPerYear[0] + principalPerYear[0]) * 1.25);

        right.Plot(plan.Numeric("month").Values.Select(m => m / 12).ToArray(), plan.Numeric("balance").Values.Select(v => v / 1e6).ToArray(), "standard schedule");
        right.Plot(faster.Numeric("month").Values.Select(m => m / 12).ToArray(), faster.Numeric("balance").Values.Select(v => v / 1e6).ToArray(), "+ Rp 1 million a month");
        right.AxVLine(halfway / 12.0, Color.Parse("#8A94A3"), label: $"half repaid after {halfway / 12.0:F1} years");
        right.SetTitle("Outstanding balance").SetXLabel("years").SetYLabel("Rp million");
        right.Legend(LegendLocation.UpperRight);
        output.Chart(figure);

        output.Metric("Monthly instalment", $"Rp {payment:N0}", $"plus Rp {insurance:N0} insurance");
        output.Metric("Effective annual rate", $"{effectiveAnnual * 100:F2}%", $"advertised {annualRate * 100:F1}% nominal; fee and insurance included");
        output.Metric("Total interest", $"Rp {totalInterest / 1e6:F0} million", $"{totalInterest / principal * 100:F0}% of the amount borrowed");
        output.Metric("Extra Rp 1 million / month", $"saves Rp {savedInterest / 1e6:F0} million", $"and finishes {monthsSaved / 12} years {monthsSaved % 12} months early");
        output.Line($"In the first year {interestPerYear[0] / (interestPerYear[0] + principalPerYear[0]) * 100:F0}% of the payments are interest, so half the " +
                    $"loan is only repaid after {halfway / 12.0:F1} of 15 years. Comparing offers by effective rate, not advertised rate, puts fees on the same footing.");
    }
}
