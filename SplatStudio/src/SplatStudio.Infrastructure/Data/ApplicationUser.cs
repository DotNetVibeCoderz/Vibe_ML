using Microsoft.AspNetCore.Identity;

namespace SplatStudio.Infrastructure.Data;

/// <summary>
/// The single user/account entity for the app. Profile fields (display
/// name, bio, avatar) live directly on the Identity user instead of a
/// separate "UserProfile" table — one row, one round trip, simpler joins.
/// </summary>
public class ApplicationUser : IdentityUser
{
    public string DisplayName { get; set; } = string.Empty;
    public string? Bio { get; set; }
    public string? AvatarStorageKey { get; set; }
    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}
