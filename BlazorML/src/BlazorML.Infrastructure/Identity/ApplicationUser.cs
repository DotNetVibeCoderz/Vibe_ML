using Microsoft.AspNetCore.Identity;

namespace BlazorML.Infrastructure.Identity;

public class ApplicationUser : IdentityUser
{
    [PersonalData]
    public string? DisplayName { get; set; }

    [PersonalData]
    public string? Organisation { get; set; }

    /// <summary>Storage key of the uploaded avatar, resolved through <c>IStorageProvider</c>.</summary>
    public string? AvatarKey { get; set; }

    /// <summary><c>system</c>, <c>light</c> or <c>dark</c>; overrides the workspace default.</summary>
    public string? ThemePreference { get; set; }

    /// <summary>Two-letter UI language override.</summary>
    public string? LanguagePreference { get; set; }

    public DateTimeOffset CreatedAt { get; set; } = DateTimeOffset.UtcNow;
    public DateTimeOffset? LastSeenAt { get; set; }

    public string Initials
    {
        get
        {
            var source = !string.IsNullOrWhiteSpace(DisplayName) ? DisplayName : UserName ?? "?";
            var parts = source.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            return parts.Length >= 2
                ? string.Concat(parts[0][0], parts[1][0]).ToUpperInvariant()
                : source[..Math.Min(2, source.Length)].ToUpperInvariant();
        }
    }
}

public static class AppRoles
{
    public const string Administrator = "Administrator";
    public const string DataScientist = "Data Scientist";
    public const string Viewer = "Viewer";

    public static readonly string[] All = [Administrator, DataScientist, Viewer];
}
