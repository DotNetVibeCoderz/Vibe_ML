using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Entities;
using SplatStudio.Domain.Enums;
using SplatStudio.Infrastructure.Data;

namespace SplatStudio.Web;

/// <summary>
/// Populates a brand-new database with a demo community — several accounts, a full gallery,
/// and the comments/ratings that make the UI look like something people actually use. Runs
/// once: it no-ops the moment any user row exists, so it never touches a database that real
/// people have started using.
///
/// Sample imagery is generated in code by <see cref="SampleImageFactory"/> (drawn gradients
/// and shapes, not photographs) and pushed through the exact upload -> storage -> queue ->
/// background-worker path a real upload takes. That makes the seed double as a smoke test of
/// the whole conversion pipeline on every fresh deployment.
/// </summary>
public static class SeedData
{
    private const string DemoPassword = "Demo123!";

    private record SeedUser(string Email, string DisplayName, string Bio);

    /// <summary>
    /// The first entry is the documented demo login. The rest exist so the gallery has more
    /// than one voice in it — ratings from a single account would average to something
    /// meaningless.
    /// </summary>
    private static readonly SeedUser[] Users =
    {
        new("demo@splatstudio.local", "Demo User",
            "The account the README hands out. Everything here is generated, nothing is a real photo."),
        new("rani@splatstudio.local", "Rani Prameswari",
            "Product photographer. Mostly interested in whether this survives a plain studio backdrop."),
        new("tomas@splatstudio.local", "Tomas Lindqvist",
            "Graphics programmer. Here for the point-cloud renderer, not the pictures."),
        new("aiko@splatstudio.local", "Aiko Watanabe",
            "Generative artist. Feeding it things it was never designed to handle."),
        new("marcus@splatstudio.local", "Marcus Oyelaran",
            "Architect. Testing whether 2.5D depth is good enough for quick massing studies."),
        new("priya@splatstudio.local", "Priya Raghunathan",
            "Research engineer, photogrammetry background. Sceptical, politely.")
    };

    /// <summary>Comments keyed by recipe, so each scene's discussion is about that scene.</summary>
    private static readonly Dictionary<string, (int UserIndex, string Text)[]> Discussion = new()
    {
        ["orb"] =
        [
            (0, "Welcome to SplatStudio. Once a scene finishes baking, drag to orbit and scroll to zoom."),
            (2, "The falloff at the edge of the sphere reads really well in orbit — you can see the depth field bending."),
            (5, "Worth being clear that this is a depth heuristic, not reconstruction. It looks good because the subject suits it.")
        ],
        ["studio"] =
        [
            (1, "This is the case I care about and it holds up. The contact shadow gives it a floor to sit on."),
            (4, "Agreed. The moment there's a shadow anchoring the subject the illusion gets much stronger.")
        ],
        ["colour-study"] =
        [
            (3, "No single subject, so the radial prior has nothing to grab. Comes out as three floating shells."),
            (2, "Which is arguably the correct answer given the input.")
        ],
        ["nebula"] =
        [
            (3, "Pure luminance depth here and it's the best-looking one in the gallery to my eye."),
            (0, "This is the one I show people first.")
        ],
        ["crystal"] =
        [
            (2, "The facet edges terrace the depth field into steps. Looks intentional, isn't."),
            (5, "Nice example of the heuristic producing a plausible-but-wrong result.")
        ],
        ["torus"] =
        [
            (5, "The hole in the middle gets filled in by the centre bias. Good that this is kept in the gallery."),
            (1, "Failure cases being visible is the reason I trust the rest of it.")
        ],
        ["ripples"] =
        [
            (3, "Comes out as a rippled relief rather than an object. Still fun to orbit."),
        ],
        ["bokeh"] =
        [
            (3, "Each disc lands at its own depth and the parallax is genuinely convincing."),
            (2, "Because bright-and-small happens to correlate with near here. Lucky input.")
        ],
        ["dunes"] =
        [
            (4, "Almost flat, as expected. Not much to orbit around."),
        ],
        ["portrait-bust"] =
        [
            (1, "Closest thing to a real portrait subject and it's convincing from maybe 30 degrees off-axis."),
            (5, "Past that the lack of real geometry shows. Which is the honest limit of single-image depth.")
        ],
        ["prism"] =
        [
            (3, "The colour bands separate into layers. Accidental, but I'll take it."),
        ],
        ["moss"] =
        [
            (4, "Rolling surface, no subject. Reads like terrain."),
            (2, "Turn the point budget up on this one, it rewards density.")
        ]
    };

    public static async Task EnsureSeededAsync(IServiceProvider services, IWebHostEnvironment env)
    {
        var db = services.GetRequiredService<ApplicationDbContext>();

        // Already seeded, or a real user already signed up — never touch it again.
        if (await db.Users.AnyAsync()) return;

        var userManager = services.GetRequiredService<UserManager<ApplicationUser>>();
        var storage = services.GetRequiredService<IStorageService>();
        var imageProcessor = services.GetRequiredService<IImageProcessingService>();
        var queue = services.GetRequiredService<IConversionQueue>();
        // SeedData is a static class, so it can't be used as ILogger<T>'s type argument.
        var logger = services.GetRequiredService<ILoggerFactory>().CreateLogger("SplatStudio.Web.SeedData");

        var created = new List<ApplicationUser>();
        for (int i = 0; i < Users.Length; i++)
        {
            var spec = Users[i];
            var user = new ApplicationUser
            {
                UserName = spec.Email,
                Email = spec.Email,
                EmailConfirmed = true,
                DisplayName = spec.DisplayName,
                Bio = spec.Bio,
                // Spread join dates back over a few months so profiles don't all look identical.
                CreatedAtUtc = DateTime.UtcNow.AddDays(-120 + i * 17)
            };

            var result = await userManager.CreateAsync(user, DemoPassword);
            if (!result.Succeeded)
            {
                logger.LogWarning("Seed: could not create {Email} ({Errors}).",
                    spec.Email, string.Join(", ", result.Errors.Select(e => e.Description)));
                continue;
            }

            // Avatars go through the same storage port as everything else.
            try
            {
                var avatarBytes = SampleImageFactory.RenderAvatar(spec.Email);
                var avatarKey = $"avatars/{user.Id}/{Guid.NewGuid()}.jpg";
                using var avatarStream = new MemoryStream(avatarBytes);
                await storage.SaveAsync(avatarKey, avatarStream, "image/jpeg");
                user.AvatarStorageKey = avatarKey;
                await db.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "Seed: avatar generation failed for {Email}.", spec.Email);
            }

            created.Add(user);
        }

        if (created.Count == 0)
        {
            logger.LogWarning("Seed: no users could be created — skipping sample scenes.");
            return;
        }

        var catalogue = SampleImageFactory.Catalogue;
        var scenesByKey = new Dictionary<string, SplatScene>();

        for (int i = 0; i < catalogue.Length; i++)
        {
            var recipe = catalogue[i];
            var owner = created[i % created.Count];

            byte[] imageBytes;
            try
            {
                imageBytes = SampleImageFactory.Render(recipe.Key);
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "Seed: could not render sample image {Key}.", recipe.Key);
                continue;
            }

            using var inspectStream = new MemoryStream(imageBytes);
            var processed = await imageProcessor.ProcessUploadAsync(inspectStream, thumbnailMaxDimension: 480);

            var imageAsset = new ImageAsset
            {
                UserId = owner.Id,
                OriginalFileName = $"{recipe.Key}.jpg",
                ContentType = "image/jpeg",
                SizeBytes = imageBytes.Length,
                Width = processed.Width,
                Height = processed.Height,
                UploadedAtUtc = DateTime.UtcNow.AddDays(-catalogue.Length + i).AddHours(-i * 3)
            };
            imageAsset.StorageKey = $"images/{owner.Id}/{imageAsset.Id}.jpg";
            imageAsset.ThumbnailStorageKey = $"thumbnails/{owner.Id}/{imageAsset.Id}.jpg";

            using (var saveStream = new MemoryStream(imageBytes))
                await storage.SaveAsync(imageAsset.StorageKey, saveStream, imageAsset.ContentType);
            using (var thumbStream = new MemoryStream(processed.ThumbnailBytes))
                await storage.SaveAsync(imageAsset.ThumbnailStorageKey, thumbStream, processed.ThumbnailContentType);

            var scene = new SplatScene
            {
                ImageAssetId = imageAsset.Id,
                UserId = owner.Id,
                Title = recipe.Title,
                Description = recipe.Description,
                // Two of the twelve are private, so "My Scenes" has something to toggle.
                IsPublic = recipe.Key is not ("dunes" or "prism"),
                Status = SplatStatus.Queued,
                CreatedAtUtc = imageAsset.UploadedAtUtc,
                // Deterministic but uneven, so the gallery doesn't look uniform.
                ViewCount = 7 + (i * 53) % 400
            };

            db.ImageAssets.Add(imageAsset);
            db.SplatScenes.Add(scene);
            scenesByKey[recipe.Key] = scene;
        }

        await db.SaveChangesAsync();

        SeedDiscussion(db, created, scenesByKey);
        await db.SaveChangesAsync();

        foreach (var scene in scenesByKey.Values)
            queue.QueueSceneConversion(scene.Id);

        logger.LogInformation(
            "Seed: created {Users} users and queued {Scenes} sample scenes for conversion.",
            created.Count, scenesByKey.Count);
    }

    /// <summary>
    /// Adds the per-scene comments plus a spread of ratings. Ratings are derived from the
    /// scene and user index rather than randomised, so every deployment shows the same
    /// averages and screenshots stay reproducible.
    /// </summary>
    private static void SeedDiscussion(
        ApplicationDbContext db,
        List<ApplicationUser> users,
        Dictionary<string, SplatScene> scenesByKey)
    {
        int sceneIndex = 0;

        foreach (var (key, scene) in scenesByKey)
        {
            if (Discussion.TryGetValue(key, out var thread))
            {
                int commentIndex = 0;
                foreach (var (userIndex, text) in thread)
                {
                    var author = users[userIndex % users.Count];
                    db.Comments.Add(new Comment
                    {
                        SplatSceneId = scene.Id,
                        UserId = author.Id,
                        AuthorDisplayName = author.DisplayName,
                        Content = text,
                        CreatedAtUtc = scene.CreatedAtUtc.AddHours(4 + commentIndex * 9)
                    });
                    commentIndex++;
                }
            }

            // Three to five distinct raters per scene, walking the user list from a
            // per-scene offset so the same people don't always rate together.
            int raterCount = 3 + sceneIndex % 3;
            for (int r = 0; r < raterCount && r < users.Count; r++)
            {
                var rater = users[(sceneIndex + r) % users.Count];
                // Cycles 3..5 — enough variation that the gallery's star row isn't uniform.
                int stars = 3 + (sceneIndex + r * 2) % 3;

                db.Ratings.Add(new Rating
                {
                    SplatSceneId = scene.Id,
                    UserId = rater.Id,
                    Stars = stars,
                    CreatedAtUtc = scene.CreatedAtUtc.AddHours(6 + r * 5)
                });
            }

            sceneIndex++;
        }
    }
}
