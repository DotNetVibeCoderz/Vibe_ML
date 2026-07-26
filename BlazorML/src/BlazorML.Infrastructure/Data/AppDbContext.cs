using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage.ValueConversion;

namespace BlazorML.Infrastructure.Data;

/// <summary>
/// One context for the whole app across all four database providers. Nothing here is
/// provider-specific: ids are strings, large payloads are <c>string</c> columns left unbounded,
/// and no provider-only column types are used.
/// </summary>
public class AppDbContext(DbContextOptions<AppDbContext> options) : IdentityDbContext<ApplicationUser>(options)
{
    public DbSet<Dataset> Datasets => Set<Dataset>();
    public DbSet<Experiment> Experiments => Set<Experiment>();
    public DbSet<ExperimentVersion> ExperimentVersions => Set<ExperimentVersion>();
    public DbSet<ExperimentRun> ExperimentRuns => Set<ExperimentRun>();
    public DbSet<RunLogEntry> RunLogs => Set<RunLogEntry>();
    public DbSet<TrainedModel> TrainedModels => Set<TrainedModel>();
    public DbSet<ServiceEndpoint> Endpoints => Set<ServiceEndpoint>();
    public DbSet<ApiKey> ApiKeys => Set<ApiKey>();
    public DbSet<ChatSession> ChatSessions => Set<ChatSession>();
    public DbSet<ChatMessage> ChatMessages => Set<ChatMessage>();
    public DbSet<MarketplaceItem> MarketplaceItems => Set<MarketplaceItem>();
    public DbSet<AppSetting> AppSettings => Set<AppSetting>();
    public DbSet<AuditEntry> AuditEntries => Set<AuditEntry>();

    protected override void OnModelCreating(ModelBuilder builder)
    {
        base.OnModelCreating(builder);

        builder.Entity<Dataset>(e =>
        {
            e.HasIndex(d => d.Name);
            e.HasIndex(d => d.OwnerId);
        });

        builder.Entity<Experiment>(e =>
        {
            e.HasIndex(x => x.Name);
            e.HasIndex(x => x.OwnerId);
            e.HasIndex(x => x.IsTemplate);

            e.HasMany(x => x.Runs)
                .WithOne(r => r.Experiment!)
                .HasForeignKey(r => r.ExperimentId)
                .OnDelete(DeleteBehavior.Cascade);

            e.HasMany(x => x.Versions)
                .WithOne(v => v.Experiment!)
                .HasForeignKey(v => v.ExperimentId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        builder.Entity<ExperimentVersion>(e =>
        {
            e.HasIndex(v => new { v.ExperimentId, v.Version }).IsUnique();
        });

        builder.Entity<ExperimentRun>(e =>
        {
            e.HasIndex(r => r.ExperimentId);
            e.HasIndex(r => r.Status);

            e.HasMany(r => r.Logs)
                .WithOne(l => l.Run!)
                .HasForeignKey(l => l.RunId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        builder.Entity<RunLogEntry>(e => e.HasIndex(l => new { l.RunId, l.Sequence }));

        builder.Entity<TrainedModel>(e =>
        {
            e.HasIndex(m => m.Name);
            e.HasIndex(m => new { m.Name, m.Version });
        });

        builder.Entity<ServiceEndpoint>(e =>
        {
            e.HasIndex(x => x.Slug).IsUnique();

            e.HasOne(x => x.Model)
                .WithMany()
                .HasForeignKey(x => x.ModelId)
                .OnDelete(DeleteBehavior.Restrict);

            e.HasMany(x => x.ApiKeys)
                .WithOne(k => k.Endpoint!)
                .HasForeignKey(k => k.EndpointId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        builder.Entity<ApiKey>(e =>
        {
            e.HasIndex(k => k.KeyHash).IsUnique();
            e.HasIndex(k => k.Prefix);
        });

        builder.Entity<ChatSession>(e =>
        {
            e.HasIndex(s => s.OwnerId);

            e.HasMany(s => s.Messages)
                .WithOne(m => m.Session!)
                .HasForeignKey(m => m.SessionId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        builder.Entity<ChatMessage>(e =>
        {
            e.HasIndex(m => new { m.SessionId, m.Sequence });
        });

        builder.Entity<AppSetting>(e => e.HasIndex(s => s.Key).IsUnique());

        builder.Entity<AuditEntry>(e => e.HasIndex(a => a.CreatedAt));

        builder.Entity<MarketplaceItem>(e => e.HasIndex(m => m.Category));

        ApplySqliteTimestampConversion(builder);
    }

    /// <summary>
    /// SQLite has no native date type, and EF refuses to translate <c>ORDER BY</c> over a
    /// <see cref="DateTimeOffset"/> against it. Since almost every list in the app is ordered by
    /// a timestamp, that would break the default provider outright.
    /// <para>
    /// Storing the value as UTC ticks fixes it: a <c>long</c> sorts correctly in SQL and converts
    /// back losslessly for anything written in UTC, which is every timestamp this app produces.
    /// The other three providers keep their native date types and are untouched.
    /// </para>
    /// </summary>
    private void ApplySqliteTimestampConversion(ModelBuilder builder)
    {
        if (!Database.IsSqlite())
        {
            return;
        }

        var toTicks = new ValueConverter<DateTimeOffset, long>(
            value => value.UtcTicks,
            ticks => new DateTimeOffset(ticks, TimeSpan.Zero));

        var toNullableTicks = new ValueConverter<DateTimeOffset?, long?>(
            value => value == null ? null : value.Value.UtcTicks,
            ticks => ticks == null ? null : new DateTimeOffset(ticks.Value, TimeSpan.Zero));

        foreach (var entity in builder.Model.GetEntityTypes())
        {
            foreach (var property in entity.GetProperties())
            {
                if (property.ClrType == typeof(DateTimeOffset))
                {
                    property.SetValueConverter(toTicks);
                }
                else if (property.ClrType == typeof(DateTimeOffset?))
                {
                    property.SetValueConverter(toNullableTicks);
                }
            }
        }
    }

    // The JSON payload properties (GraphJson, ProfileJson, MetricsJson, …) carry no MaxLength and
    // no explicit column type on purpose: each provider then picks its own unbounded text type —
    // TEXT on SQLite and PostgreSQL, nvarchar(max) on SQL Server, longtext on MySQL. Naming one of
    // those here would break the other three.

    public override int SaveChanges()
    {
        StampTimestamps();
        return base.SaveChanges();
    }

    public override Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
    {
        StampTimestamps();
        return base.SaveChangesAsync(cancellationToken);
    }

    private void StampTimestamps()
    {
        foreach (var entry in ChangeTracker.Entries<EntityBase>())
        {
            if (entry.State == EntityState.Modified)
            {
                entry.Entity.UpdatedAt = DateTimeOffset.UtcNow;
            }
        }
    }
}
