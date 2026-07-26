using BlazorML.Core.Abstractions;
using BlazorML.Infrastructure.Identity;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;

namespace BlazorML.Web.Endpoints;

public static class SupportEndpoints
{
    /// <summary>
    /// Serves objects out of the configured storage provider for providers that cannot hand out
    /// a signed URL. Authorised, unlike a raw file path would be.
    /// </summary>
    public static void MapStorageDownloads(this WebApplication app)
    {
        app.MapGet("/storage/{**key}", async (
                string key,
                IStorageProviderFactory storage,
                HttpContext http,
                CancellationToken ct) =>
            {
                if (http.User.Identity?.IsAuthenticated != true)
                {
                    return Results.Unauthorized();
                }

                if (!await storage.Current.ExistsAsync(key, ct))
                {
                    return Results.NotFound();
                }

                var stream = await storage.Current.ReadAsync(key, ct);
                var name = Path.GetFileName(key);

                return Results.File(stream, ContentType(name), name);
            })
            .ExcludeFromDescription();
    }

    /// <summary>
    /// Sign-out has to be a real form post rather than a Blazor event: the auth cookie is written
    /// on the HTTP response, which an interactive circuit no longer has access to.
    /// </summary>
    public static void MapAccountEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/akun").ExcludeFromDescription();

        group.MapPost("/keluar", async (
            SignInManager<ApplicationUser> signIn,
            [FromForm] string? kembali) =>
        {
            await signIn.SignOutAsync();
            return Results.LocalRedirect(string.IsNullOrWhiteSpace(kembali) ? "/masuk" : kembali);
        }).DisableAntiforgery();

        group.MapPost("/masuk", async (
            HttpContext http,
            SignInManager<ApplicationUser> signIn,
            UserManager<ApplicationUser> users,
            [FromForm] string email,
            [FromForm] string password,
            [FromForm] bool? ingat,
            [FromForm] string? kembali) =>
        {
            var user = await users.FindByEmailAsync(email);
            if (user is null)
            {
                return Results.LocalRedirect($"/masuk?galat=1{Return(kembali)}");
            }

            var result = await signIn.PasswordSignInAsync(user, password, ingat ?? false, lockoutOnFailure: true);

            if (result.IsLockedOut)
            {
                return Results.LocalRedirect($"/masuk?galat=2{Return(kembali)}");
            }

            if (!result.Succeeded)
            {
                return Results.LocalRedirect($"/masuk?galat=1{Return(kembali)}");
            }

            user.LastSeenAt = DateTimeOffset.UtcNow;
            await users.UpdateAsync(user);

            return Results.LocalRedirect(string.IsNullOrWhiteSpace(kembali) ? "/" : "/" + kembali.TrimStart('/'));
        }).DisableAntiforgery();

        group.MapPost("/lupa-sandi", async (
            HttpContext http,
            UserManager<ApplicationUser> users,
            IEmailSender email,
            ILoggerFactory loggerFactory,
            [FromForm] string emailAddress) =>
        {
            var user = await users.FindByEmailAsync(emailAddress);

            // The reply is identical whether or not the account exists, and the work happens
            // afterwards. Anything else turns this endpoint into a way to discover which email
            // addresses have accounts here.
            if (user is not null)
            {
                var token = await users.GeneratePasswordResetTokenAsync(user);

                var link = $"{http.Request.Scheme}://{http.Request.Host}/atur-ulang-sandi" +
                           $"?email={Uri.EscapeDataString(emailAddress)}" +
                           $"&token={Uri.EscapeDataString(token)}";

                var body =
                    $"""
                     <p>Halo{(string.IsNullOrWhiteSpace(user.DisplayName) ? "" : " " + user.DisplayName)},</p>
                     <p>Ada permintaan untuk mengatur ulang kata sandi akun Blazor ML Studio kamu.
                        Buka tautan berikut untuk memilih kata sandi baru:</p>
                     <p><a href="{link}">{link}</a></p>
                     <p>Kalau bukan kamu yang meminta, abaikan saja email ini —
                        kata sandimu tidak berubah.</p>
                     """;

                try
                {
                    await email.SendAsync(emailAddress, "Atur ulang kata sandi Blazor ML Studio", body);
                }
                catch (Exception e)
                {
                    // A mail server that is down must not leak through as a different response
                    // than a successful send would give.
                    loggerFactory.CreateLogger("Account")
                        .LogError(e, "Could not send the password reset message to {Email}.", emailAddress);
                }
            }

            return Results.LocalRedirect("/lupa-sandi?terkirim=1");
        }).DisableAntiforgery();

        group.MapPost("/atur-ulang", async (
            SignInManager<ApplicationUser> signIn,
            UserManager<ApplicationUser> users,
            [FromForm] string email,
            [FromForm] string token,
            [FromForm] string password,
            [FromForm] string ulangi) =>
        {
            var back = $"/atur-ulang-sandi?email={Uri.EscapeDataString(email)}" +
                       $"&token={Uri.EscapeDataString(token)}";

            if (password != ulangi)
            {
                return Results.LocalRedirect($"{back}&galat=beda");
            }

            var user = await users.FindByEmailAsync(email);

            if (user is null)
            {
                // Same outcome as a stale token: the page must not distinguish the two.
                return Results.LocalRedirect($"{back}&galat=kedaluwarsa");
            }

            var result = await users.ResetPasswordAsync(user, token, password);

            if (!result.Succeeded)
            {
                var expired = result.Errors.Any(e => e.Code.Contains("Token", StringComparison.OrdinalIgnoreCase));
                return Results.LocalRedirect($"{back}&galat={(expired ? "kedaluwarsa" : "lemah")}");
            }

            await signIn.SignInAsync(user, isPersistent: false);

            return Results.LocalRedirect("/");
        }).DisableAntiforgery();

        group.MapPost("/daftar", async (
            SignInManager<ApplicationUser> signIn,
            UserManager<ApplicationUser> users,
            ISettingsService settings,
            [FromForm] string email,
            [FromForm] string password,
            [FromForm] string? nama) =>
        {
            var workspace = await settings.GetAsync<Core.Configuration.WorkspaceOptions>(
                Core.Configuration.SettingsSections.Workspace);

            if (!workspace.AllowRegistration)
            {
                return Results.LocalRedirect("/daftar?galat=3");
            }

            var user = new ApplicationUser
            {
                UserName = email,
                Email = email,
                DisplayName = string.IsNullOrWhiteSpace(nama) ? email.Split('@')[0] : nama,
                EmailConfirmed = true
            };

            var created = await users.CreateAsync(user, password);

            if (!created.Succeeded)
            {
                var reason = Uri.EscapeDataString(
                    string.Join(" ", created.Errors.Select(e => e.Description)));

                return Results.LocalRedirect($"/daftar?pesan={reason}");
            }

            await users.AddToRoleAsync(user, AppRoles.DataScientist);
            await signIn.SignInAsync(user, isPersistent: true);

            return Results.LocalRedirect("/");
        }).DisableAntiforgery();
    }

    private static string Return(string? kembali) =>
        string.IsNullOrWhiteSpace(kembali) ? string.Empty : $"&kembali={Uri.EscapeDataString(kembali)}";

    private static string ContentType(string name) => Path.GetExtension(name).ToLowerInvariant() switch
    {
        ".csv" => "text/csv",
        ".json" => "application/json",
        ".zip" => "application/zip",
        ".png" => "image/png",
        ".jpg" or ".jpeg" => "image/jpeg",
        ".gif" => "image/gif",
        ".webp" => "image/webp",
        ".pdf" => "application/pdf",
        ".txt" => "text/plain",
        _ => "application/octet-stream"
    };
}
