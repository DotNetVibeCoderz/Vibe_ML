using Gravicode.Science.GraviNum;
using Gravicode.Science.GraviText.Tasks;
using Matplotlib.Net.Artists;
using ScienceGallery.Models;
using ScienceGallery.Services;

namespace ScienceGallery.Showcases.Office;

/// <summary>Training a sequence tagger to pull vendor, invoice number, date and amount out of payment emails.</summary>
public sealed class InvoiceFieldExtraction : Showcase
{
    public override string Title => "Pull the invoice number, vendor, date and amount out of emails";
    public override Domain Domain => Domain.Office;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviText, GraviLibrary.GraviNum];
    public override string Summary => "A CRF tagger learns the four fields accounts payable retypes by hand, and is tested on unseen vendors and phrasing.";
    public override string Scenario =>
        "An accounts-payable clerk copies four fields from every invoice email into the finance system. A " +
        "sequence tagger (a conditional random field) learns to mark them from a few hundred annotated lines. " +
        "The honest test is not the training data: it is vendors the model has never seen, and a way of " +
        "phrasing the email that no training example used. Only fields extracted exactly are counted.";

    public override void Run(ShowcaseOutput output)
    {
        var rng = new GraviRandom(12);
        string[] knownVendors = ["PT Sinar Jaya", "CV Maju Bersama", "PT Kencana Abadi", "Toko Sumber Rejeki", "PT Mitra Data", "CV Cahaya Baru"];
        string[] newVendors = ["PT Nusantara Logistik", "CV Berkah Mandiri", "PT Global Tekno Indah", "Toko Sejahtera"];

        // Each template is a sequence of literal words and {FIELD} slots. The last one is held out.
        string[][] templates =
        [
            ["Invoice", "{NO}", "from", "{VENDOR}", "dated", "{DATE}", ",", "amount", "due", "Rp", "{AMOUNT}", "."],
            ["{VENDOR}", "invoice", "number", "{NO}", "issued", "{DATE}", "for", "Rp", "{AMOUNT}", "."],
            ["Please", "pay", "Rp", "{AMOUNT}", "to", "{VENDOR}", "for", "invoice", "{NO}", "dated", "{DATE}", "."],
            ["Attached", "is", "invoice", "{NO}", "(", "{DATE}", ")", "from", "{VENDOR}", "totalling", "Rp", "{AMOUNT}", "."],
        ];
        string[] unseenTemplate = ["Reminder", ":", "{VENDOR}", "sent", "{NO}", "on", "{DATE}", ";", "balance", "Rp", "{AMOUNT}", "outstanding", "."];

        (TaggedSentence Sentence, Dictionary<string, string> Fields) Generate(string[] template, string[] vendors)
        {
            var fields = new Dictionary<string, string>
            {
                ["VENDOR"] = vendors[rng.Next(vendors.Length)],
                ["NO"] = $"INV-{rng.Next(2024, 2026)}-{rng.Next(1, 9999):D4}",
                ["DATE"] = $"{rng.Next(1, 29):D2}/{rng.Next(1, 13):D2}/2025",
                ["AMOUNT"] = (rng.Next(150, 95_000) * 1000).ToString("N0").Replace(',', '.'),   // Rp 4.250.000
            };
            var tokens = new List<TaggedToken>();
            foreach (var word in template)
            {
                if (!word.StartsWith('{')) { tokens.Add(new TaggedToken(word, "O")); continue; }
                var field = word[1..^1];
                var parts = fields[field].Split(' ');
                for (var i = 0; i < parts.Length; i++) tokens.Add(new TaggedToken(parts[i], (i == 0 ? "B-" : "I-") + field));
            }
            return (new TaggedSentence(tokens), fields);
        }

        var training = Enumerable.Range(0, 240).Select(i => Generate(templates[i % templates.Length], knownVendors).Sentence).ToList();
        var newVendorTest = Enumerable.Range(0, 60).Select(i => Generate(templates[i % templates.Length], newVendors)).ToList();
        var newPhrasingTest = Enumerable.Range(0, 60).Select(_ => Generate(unseenTemplate, newVendors)).ToList();

        var tagger = new TrainedNer().Fit(training, epochs: 15, crfEpochs: 30);

        // Exact match per field: what the finance system would receive, compared with the truth.
        string[] fieldNames = ["NO", "VENDOR", "DATE", "AMOUNT"];
        double[] ExactRate(List<(TaggedSentence Sentence, Dictionary<string, string> Fields)> tests) => fieldNames.Select(field =>
            tests.Average(t =>
            {
                var found = tagger.Recognize(t.Sentence.Words).FirstOrDefault(e => e.Type == field);
                return found is not null && found.Text == t.Fields[field] ? 100.0 : 0.0;
            })).ToArray();
        var vendorRates = ExactRate(newVendorTest);
        var phrasingRates = ExactRate(newPhrasingTest);
        var strict = tagger.Evaluate(newPhrasingTest.Select(t => t.Sentence).ToList());

        var figure = GalleryCharts.Figure(9, 4.6);
        var ax = figure.AddSubplot();
        string[] labels = ["invoice no.", "vendor", "date", "amount"];
        var x = Enumerable.Range(0, labels.Length).Select(i => (double)i).ToArray();
        ax.Bar(x.Select(v => v - 0.2).ToArray(), vendorRates, 0.4, label: "unseen vendors, familiar phrasing");
        ax.Bar(x.Select(v => v + 0.2).ToArray(), phrasingRates, 0.4, label: "unseen vendors and phrasing");
        ax.SetXTicks(x, labels).SetYLim(0, 118);
        ax.SetTitle("Fields extracted exactly (60 test emails each)").SetYLabel("% exact");
        ax.Legend(LegendLocation.UpperLeft, columns: 2);
        output.Chart(figure);

        var sample = newPhrasingTest[0].Sentence;
        output.Metric("Training lines", training.Count.ToString(), $"{knownVendors.Length} vendors, {templates.Length} phrasings");
        output.Metric("Learned features", tagger.FeatureCount.ToString("N0"), "word shape, affixes and neighbours");
        output.Metric("Entity F1, unseen phrasing", $"{strict.F1 * 100:F1}%", "strict CoNLL scoring: span and type must match");
        output.Line($"Example: \"{sample}\" → " + string.Join(", ", tagger.Recognize(sample.Words).Select(e => $"{e.Type} = {e.Text}")) + ".");
        var weakest = Enumerable.Range(0, labels.Length).MinBy(i => phrasingRates[i]);
        output.Line($"The weakest field on new phrasing is the {labels[weakest]} ({phrasingRates[weakest]:F0}% exact). The corpus is generated, " +
                    "so these figures say the tagger learned the structure, not that it would match them on real mail; " +
                    "annotate a sample of real emails before trusting it.");
    }
}
