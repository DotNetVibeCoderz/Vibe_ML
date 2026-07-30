using Microsoft.EntityFrameworkCore;
using SplatStudio.Domain.Enums;

namespace SplatStudio.Infrastructure.Data;

/// <summary>
/// Single place that knows how to wire EF Core to each of the three
/// supported relational engines. Program.cs reads "Database:Provider"
/// from configuration and calls this once — nothing else in the app
/// branches on database vendor.
/// </summary>
public static class DatabaseConfiguration
{
    public static void ConfigureProvider(
        DbContextOptionsBuilder options,
        DatabaseProviderType provider,
        string connectionString)
    {
        switch (provider)
        {
            case DatabaseProviderType.Sqlite:
                options.UseSqlite(connectionString);
                break;

            case DatabaseProviderType.SqlServer:
                options.UseSqlServer(connectionString, sql => sql.EnableRetryOnFailure(3));
                break;

            case DatabaseProviderType.MySql:
                // A pinned ServerVersion avoids an extra round trip at startup
                // that AutoDetect would otherwise require.
                options.UseMySql(
                    connectionString,
                    Microsoft.EntityFrameworkCore.ServerVersion.AutoDetect(connectionString),
                    mysql => mysql.EnableRetryOnFailure(3));
                break;

            default:
                throw new NotSupportedException($"Unsupported database provider: {provider}");
        }
    }

    public static DatabaseProviderType ParseProvider(string? value) =>
        Enum.TryParse<DatabaseProviderType>(value, ignoreCase: true, out var parsed)
            ? parsed
            : DatabaseProviderType.Sqlite;
}
