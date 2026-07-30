using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using SplatStudio.Application.Abstractions;
using SplatStudio.Domain.Entities;
using SplatStudio.Domain.Enums;
using SplatStudio.Infrastructure.Data;

namespace SplatStudio.Web;

/// <summary>
/// Populates a brand-new database with a demo account and a handful of
/// sample scenes so the gallery isn't empty on first run. Runs once — it
/// no-ops the moment any user row already exists, so it never touches a
/// database that real people have started using.
///
/// The sample images are entirely synthetic (drawn gradients/shapes, see
/// scripts/generate_sample_images.py), not real photographs, and are pushed
/// through the exact same upload -> storage -> queue -> background-worker
/// pipeline a real user's upload would take, just so the demo data doubles
/// as a smoke test of that pipeline on every fresh deployment.
/// </summary>
public static class SeedData
{
    private static readonly (string FileName, string Title, string Description)[] Samples =
    {
        ("sample-orb-portrait.jpg", "Drifting orb",
            "A single soft sphere against a dark background — the kind of clear, centered subject the local heuristic engine handles best."),
        ("sample-object-plain.jpg", "Studio sphere",
            "A product-photo style object on a plain backdrop, with a soft drop shadow for depth cues."),
        ("sample-abstract-stilllife.jpg", "Colour study",
            "Three overlapping colour blobs — a more abstract test of the depth heuristic.")
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
        var logger = services.GetRequiredService<ILogger<SeedData>>();

        var demoUser = new ApplicationUser
        {
            UserName = "demo@splatstudio.local",
            Email = "demo@splatstudio.local",
            EmailConfirmed = true,
            DisplayName = "Demo User",
            Bio = "Exploring SplatStudio's sample gallery.",
            CreatedAtUtc = DateTime.UtcNow
        };

        var createResult = await userManager.CreateAsync(demoUser, "Demo123!");
        if (!createResult.Succeeded)
        {
            logger.LogWarning("Seed: could not create demo user ({Errors}) — skipping sample data.",
                string.Join(", ", createResult.Errors.Select(e => e.Description)));
            return;
        }

        var sampleDir = Path.Combine(env.ContentRootPath, "wwwroot", "sample-data", "images");
        var seededSceneIds = new List<Guid>();

        foreach (var (fileName, title, description) in Samples)
        {
            var sourcePath = Path.Combine(sampleDir, fileName);
            if (!File.Exists(sourcePath))
            {
                logger.LogWarning("Seed: sample image {File} not found, skipping.", fileName);
                continue;
            }

            var imageBytes = await File.ReadAllBytesAsync(sourcePath);

            using var inspectStream = new MemoryStream(imageBytes);
            var processed = await imageProcessor.ProcessUploadAsync(inspectStream, thumbnailMaxDimension: 480);

            var imageAsset = new ImageAsset
            {
                UserId = demoUser.Id,
                OriginalFileName = fileName,
                ContentType = "image/jpeg",
                SizeBytes = imageBytes.Length,
                Width = processed.Width,
                Height = processed.Height
            };
            imageAsset.StorageKey = $"images/{demoUser.Id}/{imageAsset.Id}.jpg";
            imageAsset.ThumbnailStorageKey = $"thumbnails/{demoUser.Id}/{imageAsset.Id}.jpg";

            using (var saveStream = new MemoryStream(imageBytes))
                await storage.SaveAsync(imageAsset.StorageKey, saveStream, imageAsset.ContentType);
            using (var thumbStream = new MemoryStream(processed.ThumbnailBytes))
                await storage.SaveAsync(imageAsset.ThumbnailStorageKey, thumbStream, processed.ThumbnailContentType);

            var scene = new SplatScene
            {
                ImageAssetId = imageAsset.Id,
                UserId = demoUser.Id,
                Title = title,
                Description = description,
                IsPublic = true,
                Status = SplatStatus.Queued
            };

            db.ImageAssets.Add(imageAsset);
            db.SplatScenes.Add(scene);
            seededSceneIds.Add(scene.Id);
        }

        await db.SaveChangesAsync();

        if (seededSceneIds.Count > 0)
        {
            db.Comments.Add(new Comment
            {
                SplatSceneId = seededSceneIds[0],
                UserId = demoUser.Id,
                AuthorDisplayName = demoUser.DisplayName,
                Content = "Welcome to SplatStudio! Once a scene finishes baking, drag to orbit and scroll to zoom."
            });
            db.Ratings.Add(new Rating { SplatSceneId = seededSceneIds[0], UserId = demoUser.Id, Stars = 5 });
            await db.SaveChangesAsync();
        }

        foreach (var sceneId in seededSceneIds)
            queue.QueueSceneConversion(sceneId);

        logger.LogInformation("Seed: created demo user and queued {Count} sample scenes for conversion.", seededSceneIds.Count);
    }
}
