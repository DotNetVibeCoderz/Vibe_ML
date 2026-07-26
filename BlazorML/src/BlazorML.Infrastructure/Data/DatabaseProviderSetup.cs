using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;
using Microsoft.EntityFrameworkCore;

namespace BlazorML.Infrastructure.Data;

/// <summary>
/// The one place that knows how each of the four supported databases is opened. Everything else
/// in the app talks to <see cref="AppDbContext"/> without caring which engine is behind it.
/// </summary>
public static class DatabaseProviderSetup
{
    /// <param name="contentRoot">
    /// Where a relative SQLite <c>Data Source</c> is anchored. The web layer passes its content
    /// root so the database file sits beside the storage folder rather than inside <c>bin</c>,
    /// where a <c>dotnet clean</c> would delete the database and leave the datasets orphaned.
    /// </param>
    public static void Configure(DbContextOptionsBuilder builder, DatabaseOptions options,
        string? contentRoot = null)
    {
        var connectionString = options.Current;
        var timeout = options.CommandTimeoutSeconds;

        switch (options.Provider)
        {
            case DatabaseProviderKind.SqlServer:
                builder.UseSqlServer(connectionString, sql =>
                {
                    sql.CommandTimeout(timeout);
                    sql.EnableRetryOnFailure(3);
                });
                break;

            case DatabaseProviderKind.MySql:
                // Oracle's provider, not Pomelo: Pomelo caps at EF Core 9 and would hold the
                // whole solution back a major version.
                builder.UseMySQL(connectionString, my => my.CommandTimeout(timeout));
                break;

            case DatabaseProviderKind.PostgreSql:
                builder.UseNpgsql(connectionString, npg =>
                {
                    npg.CommandTimeout(timeout);
                    npg.EnableRetryOnFailure(3);
                });
                break;

            case DatabaseProviderKind.Sqlite:
            default:
                builder.UseSqlite(ResolveSqlitePath(connectionString, contentRoot),
                    lite => lite.CommandTimeout(timeout));
                break;
        }
    }

    /// <summary>
    /// Makes a relative SQLite <c>Data Source</c> resolve next to the app rather than against
    /// whatever the process working directory happens to be.
    /// </summary>
    private static string ResolveSqlitePath(string connectionString, string? contentRoot)
    {
        const string marker = "Data Source=";
        var index = connectionString.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (index < 0)
        {
            return connectionString;
        }

        var start = index + marker.Length;
        var end = connectionString.IndexOf(';', start);
        var path = end < 0 ? connectionString[start..] : connectionString[start..end];

        if (string.IsNullOrWhiteSpace(path) || Path.IsPathRooted(path) || path.Equals(":memory:", StringComparison.OrdinalIgnoreCase))
        {
            return connectionString;
        }

        var absolute = Path.Combine(contentRoot ?? AppContext.BaseDirectory, path);
        var directory = Path.GetDirectoryName(absolute);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        return connectionString[..start] + absolute + (end < 0 ? string.Empty : connectionString[end..]);
    }

    /// <summary>Friendly name for the Settings page and the dashboard footer.</summary>
    public static string DisplayName(DatabaseProviderKind kind) => kind switch
    {
        DatabaseProviderKind.SqlServer => "SQL Server",
        DatabaseProviderKind.MySql => "MySQL",
        DatabaseProviderKind.PostgreSql => "PostgreSQL",
        _ => "SQLite"
    };
}
