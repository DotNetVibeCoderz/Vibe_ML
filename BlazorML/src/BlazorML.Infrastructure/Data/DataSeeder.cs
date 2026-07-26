using BlazorML.Core.Abstractions;
using BlazorML.Core.Data;
using BlazorML.Core.Designer;
using BlazorML.Core.Domain;
using BlazorML.Core.Modules;
using BlazorML.Infrastructure.Identity;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace BlazorML.Infrastructure.Data;

/// <summary>
/// Fills an empty workspace with roles, users, datasets and ready-to-run experiments.
/// <para>
/// The sample datasets are generated here rather than shipped as files: they stay in step with
/// the experiments that consume them, they are deterministic (fixed seed) so screenshots and
/// documentation stay accurate, and the repository does not carry binary data.
/// </para>
/// </summary>
public static class DataSeeder
{
    public static async Task SeedAsync(IServiceProvider services, ILogger logger)
    {
        var db = services.GetRequiredService<AppDbContext>();

        await SeedRolesAsync(services);
        await SeedUsersAsync(services, logger);

        if (await db.Datasets.AnyAsync())
        {
            return;
        }

        logger.LogInformation("Seeding sample datasets and experiments…");

        var datasets = services.GetRequiredService<IDatasetService>();
        var admin = await db.Users.FirstOrDefaultAsync();

        var iris = await datasets.SaveAsync("Klasifikasi Bunga Iris", BuildIris(), DatasetFormat.Csv, admin?.Id);
        var houses = await datasets.SaveAsync("Harga Rumah Bandung", BuildHouses(), DatasetFormat.Csv, admin?.Id);
        var churn = await datasets.SaveAsync("Churn Pelanggan Telko", BuildChurn(), DatasetFormat.Csv, admin?.Id);
        var reviews = await datasets.SaveAsync("Ulasan Produk", BuildReviews(), DatasetFormat.Csv, admin?.Id);
        var ratings = await datasets.SaveAsync("Rating Film", BuildRatings(), DatasetFormat.Csv, admin?.Id);

        foreach (var dataset in (Dataset[])[iris, houses, churn, reviews, ratings])
        {
            var tracked = await db.Datasets.FirstAsync(d => d.Id == dataset.Id);
            tracked.IsSample = true;
        }

        await db.SaveChangesAsync();

        await SeedExperimentAsync(db, admin?.Id,
            "Prediksi churn pelanggan",
            "Menebak pelanggan mana yang akan berhenti berlangganan bulan depan, memakai boosted decision tree.",
            MlTask.BinaryClassification, "Prediksi", "📉",
            BuildSupervisedGraph(churn.Id, "Churn", "algo.bin.fastTree"), churn.Id);

        await SeedExperimentAsync(db, admin?.Id,
            "Prediksi harga rumah",
            "Memperkirakan harga rumah dari luas, kamar, dan jarak ke pusat kota.",
            MlTask.Regression, "Prediksi", "🏠",
            BuildSupervisedGraph(houses.Id, "HargaJuta", "algo.reg.fastTree"), houses.Id);

        await SeedExperimentAsync(db, admin?.Id,
            "Klasifikasi spesies iris",
            "Contoh klasik multiclass: menentukan spesies dari empat ukuran kelopak.",
            MlTask.MulticlassClassification, "Klasifikasi", "🌸",
            BuildSupervisedGraph(iris.Id, "Spesies", "algo.multi.sdca"), iris.Id);

        await SeedExperimentAsync(db, admin?.Id,
            "Analisis sentimen ulasan",
            "Melabeli ulasan produk sebagai positif atau negatif memakai Profesor Wicak, lalu mengevaluasi hasilnya.",
            MlTask.TextClassification, "Teks", "💬",
            BuildSentimentGraph(reviews.Id), reviews.Id);

        await SeedExperimentAsync(db, admin?.Id,
            "Rekomendasi film",
            "Matrix factorization atas rating pengguna untuk menyarankan film berikutnya.",
            MlTask.Recommendation, "Rekomendasi", "🎬",
            BuildRecommendationGraph(ratings.Id), ratings.Id);

        await SeedExperimentAsync(db, admin?.Id,
            "AutoML churn",
            "Membiarkan AutoML mencari algoritma dan setelan terbaik dalam anggaran waktu satu menit.",
            MlTask.BinaryClassification, "AutoML", "🤖",
            BuildAutoMlGraph(churn.Id, "Churn"), churn.Id);

        await db.SaveChangesAsync();
        logger.LogInformation("Seeded {Datasets} datasets and {Experiments} sample experiments.", 5, 6);
    }

    // ------------------------------------------------------------------ identity

    private static async Task SeedRolesAsync(IServiceProvider services)
    {
        var roles = services.GetRequiredService<RoleManager<IdentityRole>>();

        foreach (var role in AppRoles.All)
        {
            if (!await roles.RoleExistsAsync(role))
            {
                await roles.CreateAsync(new IdentityRole(role));
            }
        }
    }

    private static async Task SeedUsersAsync(IServiceProvider services, ILogger logger)
    {
        var users = services.GetRequiredService<UserManager<ApplicationUser>>();

        if (users.Users.Any())
        {
            return;
        }

        var seeds = new (string Email, string Name, string Role, string Org)[]
        {
            ("admin@gravicode.com", "Kang Fadhil", AppRoles.Administrator, "Gravicode Studios"),
            ("wina@gravicode.com", "Wina Prastiwi", AppRoles.DataScientist, "Gravicode Studios"),
            ("bagus@gravicode.com", "Bagus Nugroho", AppRoles.DataScientist, "Gravicode Studios"),
            ("sari@contoh.id", "Sari Rahmawati", AppRoles.DataScientist, "Universitas Contoh"),
            ("tamu@contoh.id", "Tamu Demo", AppRoles.Viewer, "Demo")
        };

        foreach (var (email, name, role, org) in seeds)
        {
            var user = new ApplicationUser
            {
                UserName = email,
                Email = email,
                DisplayName = name,
                Organisation = org,
                EmailConfirmed = true
            };

            // Development seed credentials. The README says plainly that these must be changed
            // before the app is exposed to anyone.
            var result = await users.CreateAsync(user, "StudioML#2026");

            if (result.Succeeded)
            {
                await users.AddToRoleAsync(user, role);
            }
            else
            {
                logger.LogWarning("Could not seed user {Email}: {Errors}", email,
                    string.Join("; ", result.Errors.Select(e => e.Description)));
            }
        }

        logger.LogInformation("Seeded {Count} sample users. Default password: StudioML#2026", seeds.Length);
    }

    // ----------------------------------------------------------------- graphs

    /// <summary>Import → clean → split → train → score → evaluate. The shape most experiments take.</summary>
    private static ExperimentGraph BuildSupervisedGraph(string datasetId, string label, string algorithmId)
    {
        var graph = new ExperimentGraph();

        // Stratify only when the label is a class. Stratifying on a continuous target puts
        // almost every row in a group of one and leaves the test half empty.
        var task = ModuleCatalog.Find(algorithmId)?.Task ?? MlTask.None;
        var stratify = task is MlTask.BinaryClassification or MlTask.MulticlassClassification
            ? label
            : string.Empty;

        var import = Node(graph, "data.dataset", 60, 40, ("datasetId", datasetId));
        var clean = Node(graph, "tf.cleanMissing", 60, 190, ("strategy", "mean"));
        var split = Node(graph, "tf.splitData", 60, 340, ("fraction", "0.8"), ("stratifyColumn", stratify));
        var algorithm = Node(graph, "algo.placeholder", 330, 340);
        var train = Node(graph, "train.model", 60, 500, ("labelColumn", label));
        var score = Node(graph, "score.model", 60, 650);
        var evaluate = Node(graph, "score.evaluate", 60, 800, ("task", "auto"));

        // The algorithm node is created last so its module id is set from the caller's choice.
        algorithm.ModuleId = algorithmId;
        algorithm.Label = ModuleCatalog.Find(algorithmId)?.Name ?? algorithmId;
        algorithm.Parameters = ModuleCatalog.Find(algorithmId)?.BuildDefaultParameters() ?? new();

        Edge(graph, import, 0, clean, 0);
        Edge(graph, clean, 0, split, 0);
        Edge(graph, algorithm, 0, train, 0);
        Edge(graph, split, 0, train, 1);
        Edge(graph, train, 0, score, 0);
        Edge(graph, split, 1, score, 1);
        Edge(graph, score, 0, evaluate, 0);

        return graph;
    }

    private static ExperimentGraph BuildAutoMlGraph(string datasetId, string label)
    {
        var graph = new ExperimentGraph();

        var import = Node(graph, "data.dataset", 80, 60, ("datasetId", datasetId));
        var clean = Node(graph, "tf.cleanMissing", 80, 210, ("strategy", "median"));
        var auto = Node(graph, "train.autoML", 80, 360,
            ("task", "BinaryClassification"), ("labelColumn", label), ("maxSeconds", "60"));

        Edge(graph, import, 0, clean, 0);
        Edge(graph, clean, 0, auto, 0);

        return graph;
    }

    private static ExperimentGraph BuildSentimentGraph(string datasetId)
    {
        var graph = new ExperimentGraph();

        var import = Node(graph, "data.dataset", 80, 60, ("datasetId", datasetId));
        var sentiment = Node(graph, "llm.sentiment", 80, 210,
            ("column", "Ulasan"), ("outputColumn", "PredictedLabel"), ("labels", "positif,negatif"));
        var evaluate = Node(graph, "score.evaluate", 80, 380, ("task", "BinaryClassification"));

        Edge(graph, import, 0, sentiment, 0);
        Edge(graph, sentiment, 0, evaluate, 0);

        return graph;
    }

    private static ExperimentGraph BuildRecommendationGraph(string datasetId)
    {
        var graph = new ExperimentGraph();

        var import = Node(graph, "data.dataset", 80, 60, ("datasetId", datasetId));
        var split = Node(graph, "tf.splitData", 80, 210, ("fraction", "0.8"));
        var algorithm = Node(graph, "algo.rec.matrixFactorization", 350, 210,
            ("userColumn", "PenggunaId"), ("itemColumn", "FilmId"), ("approximationRank", "24"));
        var train = Node(graph, "train.model", 80, 370, ("labelColumn", "Rating"));
        var score = Node(graph, "score.model", 80, 520);
        var evaluate = Node(graph, "score.evaluate", 80, 670, ("task", "Recommendation"));

        Edge(graph, import, 0, split, 0);
        Edge(graph, algorithm, 0, train, 0);
        Edge(graph, split, 0, train, 1);
        Edge(graph, train, 0, score, 0);
        Edge(graph, split, 1, score, 1);
        Edge(graph, score, 0, evaluate, 0);

        return graph;
    }

    private static GraphNode Node(ExperimentGraph graph, string moduleId, double x, double y,
        params (string Name, string Value)[] parameters)
    {
        var module = ModuleCatalog.Find(moduleId);

        var node = new GraphNode
        {
            ModuleId = moduleId,
            Label = module?.Name ?? moduleId,
            X = x,
            Y = y,
            Parameters = module?.BuildDefaultParameters() ?? new Dictionary<string, string?>()
        };

        foreach (var (name, value) in parameters)
        {
            node.Parameters[name] = value;
        }

        graph.Nodes.Add(node);
        return node;
    }

    private static void Edge(ExperimentGraph graph, GraphNode from, int fromPort, GraphNode to, int toPort) =>
        graph.Edges.Add(new GraphEdge
        {
            SourceNodeId = from.Id,
            SourcePort = fromPort,
            TargetNodeId = to.Id,
            TargetPort = toPort
        });

    private static async Task SeedExperimentAsync(AppDbContext db, string? ownerId, string name,
        string description, MlTask task, string category, string glyph, ExperimentGraph graph, string datasetId)
    {
        var experiment = new Experiment
        {
            Name = name,
            Description = description,
            Task = task,
            GraphJson = graph.ToJson(),
            OwnerId = ownerId,
            IsTemplate = true,
            TemplateCategory = category
        };

        db.Experiments.Add(experiment);

        db.ExperimentVersions.Add(new ExperimentVersion
        {
            ExperimentId = experiment.Id,
            Version = 1,
            GraphJson = experiment.GraphJson,
            Note = "Contoh bawaan",
            OwnerId = ownerId
        });

        db.MarketplaceItems.Add(new MarketplaceItem
        {
            Name = name,
            Summary = description,
            Category = category,
            Task = task,
            Glyph = glyph,
            ExperimentId = experiment.Id,
            DatasetId = datasetId,
            OwnerId = ownerId
        });

        await Task.CompletedTask;
    }

    // --------------------------------------------------------------- datasets

    /// <summary>Fixed seed everywhere: the sample data is the same on every install.</summary>
    private static Random Seeded() => new(20260726);

    private static TabularData BuildIris()
    {
        var random = Seeded();
        var table = TabularData.WithColumns(
            "PanjangSepal", "LebarSepal", "PanjangPetal", "LebarPetal", "Spesies");

        var species = new (string Name, double SepalL, double SepalW, double PetalL, double PetalW)[]
        {
            ("setosa", 5.0, 3.4, 1.5, 0.2),
            ("versicolor", 5.9, 2.8, 4.3, 1.3),
            ("virginica", 6.6, 3.0, 5.6, 2.0)
        };

        foreach (var (name, sl, sw, pl, pw) in species)
        {
            for (var i = 0; i < 50; i++)
            {
                table.AddRow(
                    Round(sl + Noise(random, 0.35)),
                    Round(sw + Noise(random, 0.3)),
                    Round(pl + Noise(random, 0.4)),
                    Round(pw + Noise(random, 0.2)),
                    name);
            }
        }

        return table;
    }

    private static TabularData BuildHouses()
    {
        var random = Seeded();
        var table = TabularData.WithColumns(
            "LuasM2", "KamarTidur", "KamarMandi", "UmurTahun", "JarakPusatKm", "Kecamatan", "HargaJuta");

        string[] districts = ["Coblong", "Sukajadi", "Antapani", "Cibiru", "Buahbatu", "Arcamanik"];

        for (var i = 0; i < 400; i++)
        {
            var area = 45 + random.Next(0, 210);
            var bedrooms = Math.Clamp(1 + area / 45, 1, 6);
            var bathrooms = Math.Clamp(1 + area / 80, 1, 4);
            var age = random.Next(0, 35);
            var distance = Math.Round(1 + random.NextDouble() * 17, 1);
            var district = districts[random.Next(districts.Length)];

            // A plausible relationship so a trained model actually finds something.
            var price = area * 11.5
                        + bedrooms * 42
                        + bathrooms * 28
                        - age * 7.5
                        - distance * 24
                        + (district is "Coblong" or "Sukajadi" ? 260 : 0)
                        + Noise(random, 90);

            table.AddRow(area, bedrooms, bathrooms, age, distance, district, Round(Math.Max(180, price)));
        }

        return table;
    }

    private static TabularData BuildChurn()
    {
        var random = Seeded();
        var table = TabularData.WithColumns(
            "LamaBerlanggananBulan", "TagihanBulanan", "TotalTagihan", "JumlahKomplain",
            "Paket", "PakaiInternet", "Churn");

        string[] plans = ["Prabayar", "Pascabayar", "Korporat"];

        for (var i = 0; i < 900; i++)
        {
            var tenure = random.Next(1, 61);
            var monthly = Math.Round(45 + random.NextDouble() * 260, 2);
            var complaints = random.Next(0, 6);
            var plan = plans[random.Next(plans.Length)];
            var internet = random.NextDouble() > 0.25;

            // Short tenure, high bills and complaints push churn up; the signal is real but noisy.
            var risk = 0.62
                       - tenure * 0.011
                       + complaints * 0.09
                       + (monthly > 200 ? 0.16 : 0)
                       + (plan == "Prabayar" ? 0.12 : -0.05)
                       + (internet ? 0 : 0.07)
                       + random.NextDouble() * 0.28 - 0.14;

            table.AddRow(tenure, monthly, Round(monthly * tenure), complaints, plan,
                internet ? "ya" : "tidak", risk > 0.5 ? "1" : "0");
        }

        return table;
    }

    private static TabularData BuildReviews()
    {
        var table = TabularData.WithColumns("Ulasan", "Label");

        var positive = new[]
        {
            "Barang sampai cepat dan kualitasnya jauh di atas ekspektasi saya.",
            "Sudah dipakai dua minggu, baterainya awet sekali. Puas.",
            "Penjual responsif, pengemasan rapi, tidak ada lecet sama sekali.",
            "Harga segini dapat bahan sebagus ini jelas worth it.",
            "Anak saya suka banget, warnanya persis seperti di foto.",
            "Pengiriman cepat, produk original, akan beli lagi di sini.",
            "Suaranya jernih, bass-nya mantap untuk kelas harga ini.",
            "Pelayanan ramah dan barang dikirim di hari yang sama."
        };

        var negative = new[]
        {
            "Barang datang dalam keadaan penyok, kemasannya asal-asalan.",
            "Baru tiga hari sudah mati total. Sangat mengecewakan.",
            "Warnanya berbeda jauh dari yang ditampilkan di foto produk.",
            "Chat tidak pernah dibalas, pengiriman molor sampai dua minggu.",
            "Bahannya tipis dan jahitannya sudah lepas sejak awal.",
            "Tidak sesuai deskripsi, ukurannya jauh lebih kecil.",
            "Kualitas suara pecah di volume sedang, tidak layak dibeli.",
            "Garansi katanya setahun tapi klaim ditolak tanpa alasan jelas."
        };

        foreach (var text in positive)
        {
            table.AddRow(text, "positif");
        }

        foreach (var text in negative)
        {
            table.AddRow(text, "negatif");
        }

        return table;
    }

    private static TabularData BuildRatings()
    {
        var random = Seeded();
        var table = TabularData.WithColumns("PenggunaId", "FilmId", "Rating");

        string[] films =
        [
            "Laskar Pelangi", "Pengabdi Setan", "Ada Apa Dengan Cinta", "The Raid",
            "Habibie & Ainun", "Dilan 1990", "Gundala", "Marlina", "Tilik", "Sultan Agung"
        ];

        // Two taste groups, so matrix factorization has real structure to recover.
        for (var user = 1; user <= 120; user++)
        {
            var group = user % 2;

            for (var film = 0; film < films.Length; film++)
            {
                if (random.NextDouble() > 0.55)
                {
                    continue;
                }

                var affinity = (film % 2 == group) ? 4.2 : 2.4;
                var rating = Math.Clamp(affinity + Noise(random, 0.8), 1, 5);

                table.AddRow($"U{user:D3}", films[film], Math.Round(rating, 1));
            }
        }

        return table;
    }

    private static double Noise(Random random, double scale) => (random.NextDouble() - 0.5) * 2 * scale;

    private static double Round(double value) => Math.Round(value, 2);
}
