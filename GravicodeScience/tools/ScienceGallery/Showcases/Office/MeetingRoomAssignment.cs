using Gravicode.Science.SciNet;
using Matplotlib.Net.Artists;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Office;

/// <summary>Giving every meeting a room at once, optimally, instead of first come first served.</summary>
public sealed class MeetingRoomAssignment : Showcase
{
    public override string Title => "Fit ten meetings into ten rooms without anyone squeezing in";
    public override Domain Domain => Domain.Office;
    public override GraviLibrary[] Libraries => [GraviLibrary.SciNet];
    public override string Summary => "The Hungarian algorithm assigns rooms to minimise empty seats, stair climbs and missing video kit.";
    public override string Scenario =>
        "On Monday morning ten teams want a room at 10:00. Booked first come first served, the early bookers " +
        "grab the nicest rooms and the last team ends up in a room too small, or without the video-conference " +
        "kit it needs. Scoring every meeting–room pair and solving the assignment problem in one go gives every " +
        "team a workable room and the lowest total inconvenience.";

    public override void Run(ShowcaseOutput output)
    {
        (string Team, int People, int Floor, bool NeedsVideo)[] meetings =
        [
            ("Sales", 6, 1, true), ("Finance", 4, 3, false), ("Design", 11, 1, true), ("Legal", 3, 3, false),
            ("Support", 12, 1, false), ("IT", 5, 2, true), ("HR", 7, 3, false), ("Marketing", 8, 2, true),
            ("Board", 14, 4, true), ("Interns", 10, 1, false),
        ];
        (string Room, int Seats, int Floor, bool HasVideo)[] rooms =
        [
            ("Merapi", 6, 1, true), ("Rinjani", 16, 4, true), ("Bromo", 4, 3, false), ("Kerinci", 12, 1, false),
            ("Semeru", 10, 2, true), ("Lawu", 8, 2, false), ("Slamet", 20, 1, true), ("Sindoro", 8, 3, true),
            ("Tambora", 6, 2, true), ("Agung", 6, 3, false),
        ];

        // The cost of putting meeting i in room j: one point per empty seat, three per floor the team has
        // to climb or descend, forty for a video meeting in a room without the kit. A room too small is
        // not allowed at all, so it gets a cost no sensible assignment will pay.
        const double tooSmall = 1_000;
        var cost = new double[meetings.Length, rooms.Length];
        for (var i = 0; i < meetings.Length; i++)
            for (var j = 0; j < rooms.Length; j++)
                cost[i, j] = rooms[j].Seats < meetings[i].People ? tooSmall
                    : rooms[j].Seats - meetings[i].People
                      + 3 * Math.Abs(rooms[j].Floor - meetings[i].Floor)
                      + (meetings[i].NeedsVideo && !rooms[j].HasVideo ? 40 : 0);

        var (_, optimal) = Optimize.LinearSumAssignment(cost);

        // For comparison: teams book in the order listed, each taking the cheapest room still free.
        var greedy = new int[meetings.Length];
        var taken = new bool[rooms.Length];
        for (var i = 0; i < meetings.Length; i++)
        {
            greedy[i] = Enumerable.Range(0, rooms.Length).Where(j => !taken[j]).MinBy(j => cost[i, j]);
            taken[greedy[i]] = true;
        }
        double Total(int[] plan) => plan.Select((j, i) => cost[i, j]).Sum();
        int Cramped(int[] plan) => plan.Where((j, i) => cost[i, j] >= tooSmall).Count();

        var shown = new double[meetings.Length, rooms.Length];       // cap the forbidden cells for the colour scale
        for (var i = 0; i < meetings.Length; i++)
            for (var j = 0; j < rooms.Length; j++) shown[i, j] = Math.Min(cost[i, j], 60);

        var figure = GalleryCharts.Figure(12, 6.2);
        var ax = figure.AddSubplot();
        ax.Heatmap(shown, meetings.Select(m => $"{m.Team} ({m.People})").ToArray(), rooms.Select(r => $"{r.Room} {r.Seats}").ToArray(),
                   colormap: "Blues");
        ax.AddColorbar("inconvenience (60 = too small)");
        ax.Scatter(greedy.Select(j => (double)j).ToArray(), Enumerable.Range(0, meetings.Length).Select(i => (double)i).ToArray(),
                   "first come, first served", Color.Parse("#E8A317"), sizes: [260], marker: MarkerStyle.Square, alpha: 0.9);
        ax.Scatter(optimal.Select(j => (double)j).ToArray(), Enumerable.Range(0, meetings.Length).Select(i => (double)i).ToArray(),
                   "optimal assignment", Color.Parse("#D9363E"), sizes: [90], marker: MarkerStyle.Circle);
        ax.SetTitle("Meeting × room cost (room name and seats), with both plans").SetYLabel("meeting (people)");
        ax.Legend(LegendLocation.OutsideBottom, columns: 2);
        output.Chart(figure);

        output.Metric("Optimal total cost", Total(optimal).ToString("F0"), $"{Cramped(optimal)} meetings in a room too small");
        output.Metric("First-come total cost", Cramped(greedy) > 0 ? $"{Total(greedy) - Cramped(greedy) * tooSmall:F0} + {Cramped(greedy)} too small" : Total(greedy).ToString("F0"),
                      "each team takes its best free room in booking order");
        var victim = Enumerable.Range(0, meetings.Length).MaxBy(i => cost[i, greedy[i]] - cost[i, optimal[i]]);
        output.Line($"First come, first served leaves {meetings[victim].Team} ({meetings[victim].People} people) in {rooms[greedy[victim]].Room} " +
                    $"({rooms[greedy[victim]].Seats} seats); the optimal plan moves them to {rooms[optimal[victim]].Room} and still serves everyone else.");
    }
}
