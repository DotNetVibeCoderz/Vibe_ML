using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using SplatStudio.Domain.Entities;

namespace SplatStudio.Infrastructure.Data;

public class ApplicationDbContext : IdentityDbContext<ApplicationUser>
{
    public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options) : base(options) { }

    public DbSet<ImageAsset> ImageAssets => Set<ImageAsset>();
    public DbSet<SplatScene> SplatScenes => Set<SplatScene>();
    public DbSet<Comment> Comments => Set<Comment>();
    public DbSet<Rating> Ratings => Set<Rating>();

    protected override void OnModelCreating(ModelBuilder builder)
    {
        base.OnModelCreating(builder);

        builder.Entity<ImageAsset>(e =>
        {
            e.HasIndex(x => x.UserId);
            e.Property(x => x.OriginalFileName).HasMaxLength(260);
            e.Property(x => x.StorageKey).HasMaxLength(512);
            e.Property(x => x.ThumbnailStorageKey).HasMaxLength(512);
        });

        builder.Entity<SplatScene>(e =>
        {
            e.HasIndex(x => x.UserId);
            e.HasIndex(x => x.Status);
            e.HasIndex(x => x.CreatedAtUtc);
            e.Property(x => x.Title).HasMaxLength(160);
            e.Property(x => x.Description).HasMaxLength(2000);
            e.Property(x => x.SplatStorageKey).HasMaxLength(512);
            e.Property(x => x.MeshStorageKey).HasMaxLength(512);
            // Computed from the two keys above; nothing to persist.
            e.Ignore(x => x.OutputStorageKey);

            e.HasOne(x => x.ImageAsset)
                .WithOne(i => i.SplatScene)
                .HasForeignKey<SplatScene>(x => x.ImageAssetId)
                .OnDelete(DeleteBehavior.Cascade);

            e.HasMany(x => x.Comments)
                .WithOne(c => c.SplatScene!)
                .HasForeignKey(c => c.SplatSceneId)
                .OnDelete(DeleteBehavior.Cascade);

            e.HasMany(x => x.Ratings)
                .WithOne(r => r.SplatScene!)
                .HasForeignKey(r => r.SplatSceneId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        builder.Entity<Comment>(e =>
        {
            e.HasIndex(x => x.SplatSceneId);
            e.Property(x => x.Content).HasMaxLength(2000);
            e.Property(x => x.AuthorDisplayName).HasMaxLength(160);
        });

        builder.Entity<Rating>(e =>
        {
            // One rating per user per scene.
            e.HasIndex(x => new { x.SplatSceneId, x.UserId }).IsUnique();
        });
    }
}
