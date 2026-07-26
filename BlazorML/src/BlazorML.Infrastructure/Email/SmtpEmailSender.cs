using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using MailKit.Net.Smtp;
using MailKit.Security;
using Microsoft.Extensions.Logging;
using MimeKit;

namespace BlazorML.Infrastructure.Email;

/// <summary>
/// Sends over SMTP when a server is configured, and writes the message to the log when one is
/// not.
/// <para>
/// The fallback is deliberate rather than a stub. This app is deployed by the people who run it,
/// often with no mail server at hand, and a password reset that simply fails would lock an
/// administrator out of their own installation. Logging the link keeps the flow usable; the log
/// line says plainly that it is there because no mail server is set up.
/// </para>
/// </summary>
public sealed class SmtpEmailSender(ISettingsService settings, ILogger<SmtpEmailSender> logger) : IEmailSender
{
    public bool IsConfigured =>
        settings.GetAsync<EmailOptions>(SettingsSections.Email).GetAwaiter().GetResult().IsConfigured;

    public async Task SendAsync(string toAddress, string subject, string htmlBody, CancellationToken ct = default)
    {
        var options = await settings.GetAsync<EmailOptions>(SettingsSections.Email, ct);

        if (!options.IsConfigured)
        {
            // Warning, not Information: an administrator needs to notice this and act on it.
            logger.LogWarning(
                "No mail server is configured, so this message was not sent. Pass the link on by hand, " +
                "or set one up under Settings → Email.\nTo: {To}\nSubject: {Subject}\n{Body}",
                toAddress, subject, StripHtml(htmlBody));

            return;
        }

        var message = new MimeMessage();
        message.From.Add(new MailboxAddress(options.FromName, options.FromAddress));
        message.To.Add(MailboxAddress.Parse(toAddress));
        message.Subject = subject;
        message.Body = new BodyBuilder { HtmlBody = htmlBody, TextBody = StripHtml(htmlBody) }.ToMessageBody();

        using var client = new SmtpClient();

        try
        {
            await client.ConnectAsync(options.Host, options.Port,
                options.UseStartTls ? SecureSocketOptions.StartTls : SecureSocketOptions.Auto, ct);

            if (!string.IsNullOrWhiteSpace(options.Username))
            {
                await client.AuthenticateAsync(options.Username, options.Password, ct);
            }

            await client.SendAsync(message, ct);
        }
        finally
        {
            if (client.IsConnected)
            {
                await client.DisconnectAsync(true, ct);
            }
        }
    }

    /// <summary>
    /// A readable plain-text alternative, and what goes to the log. Deliberately simple: the only
    /// messages this app sends are ones it wrote itself.
    /// </summary>
    private static string StripHtml(string html)
    {
        var text = System.Text.RegularExpressions.Regex.Replace(html, "<br\\s*/?>|</p>", "\n");
        text = System.Text.RegularExpressions.Regex.Replace(text, "<[^>]+>", string.Empty);

        return System.Net.WebUtility.HtmlDecode(text).Trim();
    }
}
