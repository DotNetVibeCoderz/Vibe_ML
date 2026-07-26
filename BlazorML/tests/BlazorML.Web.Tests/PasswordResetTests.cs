using System.Net;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Identity;
using BlazorML.Infrastructure.Identity;

namespace BlazorML.Web.Tests;

/// <summary>
/// The reset flow end to end, against the real application. What matters most here is what the
/// endpoint refuses to reveal: whether a given email address has an account.
/// </summary>
public class PasswordResetTests(ScoringApiFixture fixture) : IClassFixture<ScoringApiFixture>
{
    private HttpClient Client() => fixture.CreateClient(new Microsoft.AspNetCore.Mvc.Testing
        .WebApplicationFactoryClientOptions { AllowAutoRedirect = false });

    private static FormUrlEncodedContent Form(params (string Key, string Value)[] fields) =>
        new(fields.Select(f => new KeyValuePair<string, string>(f.Key, f.Value)));

    [Fact]
    public async Task The_forgot_password_page_is_reachable_without_signing_in()
    {
        var response = await Client().GetAsync("/lupa-sandi");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Contains("Lupa kata sandi", await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task The_sign_in_page_offers_the_way_in()
    {
        var html = await Client().GetStringAsync("/masuk");

        Assert.Contains("/lupa-sandi", html);
    }

    /// <summary>
    /// The security property of this endpoint. If a known address behaved differently from an
    /// unknown one, the page would become a way to enumerate who has an account here.
    /// </summary>
    [Fact]
    public async Task A_known_and_an_unknown_address_are_answered_identically()
    {
        var known = await Client().PostAsync("/akun/lupa-sandi",
            Form(("emailAddress", ScoringApiFixture.Email)));

        var unknown = await Client().PostAsync("/akun/lupa-sandi",
            Form(("emailAddress", "tidak-ada@contoh.id")));

        Assert.Equal(known.StatusCode, unknown.StatusCode);
        Assert.Equal(known.Headers.Location, unknown.Headers.Location);
    }

    [Fact]
    public async Task Requesting_a_reset_lands_on_the_same_confirmation_either_way()
    {
        var response = await Client().PostAsync("/akun/lupa-sandi",
            Form(("emailAddress", ScoringApiFixture.Email)));

        Assert.Equal(HttpStatusCode.Found, response.StatusCode);
        Assert.Contains("terkirim", response.Headers.Location!.ToString());
    }

    [Fact]
    public async Task An_incomplete_link_says_so_instead_of_offering_a_form()
    {
        var html = await Client().GetStringAsync("/atur-ulang-sandi");

        Assert.Contains("tidak lengkap", html);
        Assert.DoesNotContain("name=\"password\"", html);
    }

    [Fact]
    public async Task A_valid_link_offers_the_form_and_names_the_account()
    {
        using var scope = fixture.Services.CreateScope();
        var users = scope.ServiceProvider
            .GetRequiredService<UserManager<ApplicationUser>>();

        var user = await users.FindByEmailAsync(ScoringApiFixture.Email);
        var token = await users.GeneratePasswordResetTokenAsync(user!);

        var html = await Client().GetStringAsync(
            $"/atur-ulang-sandi?email={Uri.EscapeDataString(ScoringApiFixture.Email)}" +
            $"&token={Uri.EscapeDataString(token)}");

        Assert.Contains("name=\"password\"", html);
        Assert.Contains(ScoringApiFixture.Email, html);
    }

    [Fact]
    public async Task Mismatched_passwords_are_rejected_before_anything_is_changed()
    {
        var response = await Client().PostAsync("/akun/atur-ulang", Form(
            ("email", ScoringApiFixture.Email),
            ("token", "apa-saja"),
            ("password", "SandiBaru#2026"),
            ("ulangi", "SandiLain#2026")));

        Assert.Contains("galat=beda", response.Headers.Location!.ToString());
    }

    [Fact]
    public async Task A_made_up_token_is_refused()
    {
        var response = await Client().PostAsync("/akun/atur-ulang", Form(
            ("email", ScoringApiFixture.Email),
            ("token", "token-karangan"),
            ("password", "SandiBaru#2026"),
            ("ulangi", "SandiBaru#2026")));

        Assert.Contains("galat=kedaluwarsa", response.Headers.Location!.ToString());
    }

    [Fact]
    public async Task An_unknown_address_is_refused_the_same_way_a_stale_token_is()
    {
        // Again: the reset page must not become an account oracle.
        var response = await Client().PostAsync("/akun/atur-ulang", Form(
            ("email", "tidak-ada@contoh.id"),
            ("token", "token-karangan"),
            ("password", "SandiBaru#2026"),
            ("ulangi", "SandiBaru#2026")));

        Assert.Contains("galat=kedaluwarsa", response.Headers.Location!.ToString());
    }

    [Fact]
    public async Task A_real_token_changes_the_password_and_signs_the_user_in()
    {
        using var scope = fixture.Services.CreateScope();
        var users = scope.ServiceProvider
            .GetRequiredService<UserManager<ApplicationUser>>();

        // A separate account, so the fixture's own sign-in credentials keep working.
        var email = $"reset-{Guid.NewGuid():n}@contoh.id";
        var user = new ApplicationUser
        {
            UserName = email, Email = email, EmailConfirmed = true
        };

        var created = await users.CreateAsync(user, "SandiLama#2026");
        Assert.True(created.Succeeded,
            string.Join("; ", created.Errors.Select(e => $"{e.Code}: {e.Description}")));

        var token = await users.GeneratePasswordResetTokenAsync(user);

        var response = await Client().PostAsync("/akun/atur-ulang", Form(
            ("email", email), ("token", token),
            ("password", "SandiBaru#2026"), ("ulangi", "SandiBaru#2026")));

        Assert.Equal("/", response.Headers.Location!.ToString());
        Assert.NotNull(response.Headers.GetValues("Set-Cookie").FirstOrDefault());

        // A fresh scope: the one above still tracks the entity as it was when it was created, so
        // it would hand back the old password hash and the check would fail for the wrong reason.
        using var verify = fixture.Services.CreateScope();
        var freshUsers = verify.ServiceProvider.GetRequiredService<UserManager<ApplicationUser>>();
        var reloaded = await freshUsers.FindByEmailAsync(email);

        Assert.True(await freshUsers.CheckPasswordAsync(reloaded!, "SandiBaru#2026"));
        Assert.False(await freshUsers.CheckPasswordAsync(reloaded!, "SandiLama#2026"));
    }

    [Fact]
    public async Task A_token_cannot_be_used_twice()
    {
        using var scope = fixture.Services.CreateScope();
        var users = scope.ServiceProvider
            .GetRequiredService<UserManager<ApplicationUser>>();

        var email = $"sekali-{Guid.NewGuid():n}@contoh.id";
        var user = new ApplicationUser
        {
            UserName = email, Email = email, EmailConfirmed = true
        };

        await users.CreateAsync(user, "SandiLama#2026");
        var token = await users.GeneratePasswordResetTokenAsync(user);

        var first = await Client().PostAsync("/akun/atur-ulang", Form(
            ("email", email), ("token", token),
            ("password", "Pertama#2026"), ("ulangi", "Pertama#2026")));

        var second = await Client().PostAsync("/akun/atur-ulang", Form(
            ("email", email), ("token", token),
            ("password", "Kedua#2026"), ("ulangi", "Kedua#2026")));

        Assert.Equal("/", first.Headers.Location!.ToString());
        Assert.Contains("galat=kedaluwarsa", second.Headers.Location!.ToString());
    }

    [Fact]
    public void Without_a_mail_server_the_sender_reports_itself_unconfigured()
    {
        // The flow still has to work; the link goes to the log instead. This is what the UI
        // promises a self-hosted administrator.
        var email = fixture.Services.GetRequiredService<IEmailSender>();

        Assert.False(email.IsConfigured);
    }

    [Fact]
    public void Email_ships_switched_off_with_no_credentials()
    {
        var settings = fixture.Services.GetRequiredService<ISettingsService>();
        var options = settings.GetAsync<EmailOptions>(SettingsSections.Email).GetAwaiter().GetResult();

        Assert.False(options.Enabled);
        Assert.Empty(options.Password);
        Assert.False(options.IsConfigured);
    }
}
