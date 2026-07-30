using System.Net;
using System.Net.Mail;
using Microsoft.Extensions.Logging;
using SplatStudio.Application.Abstractions;

namespace SplatStudio.Infrastructure.Email;

public class EmailOptions
{
    public const string SectionName = "Email";

    /// <summary>File | Smtp. "File" writes a .html file under App_Data/emails — zero setup for local dev.</summary>
    public string Provider { get; set; } = "File";

    public SmtpOptions Smtp { get; set; } = new();
}

public class SmtpOptions
{
    public string Host { get; set; } = string.Empty;
    public int Port { get; set; } = 587;
    public bool EnableSsl { get; set; } = true;
    public string Username { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
    public string FromAddress { get; set; } = "no-reply@splatstudio.local";
    public string FromName { get; set; } = "SplatStudio";
}

/// <summary>
/// Default, zero-configuration email "sender": writes each message as an
/// .html file so password-reset links are still usable while developing
/// without SMTP credentials on hand.
/// </summary>
public class FileEmailSender : IAppEmailSender
{
    private readonly string _folder;
    private readonly ILogger<FileEmailSender> _logger;

    public FileEmailSender(string contentRootPath, ILogger<FileEmailSender> logger)
    {
        _folder = Path.Combine(contentRootPath, "App_Data", "emails");
        Directory.CreateDirectory(_folder);
        _logger = logger;
    }

    public async Task SendAsync(string toEmail, string subject, string htmlBody, CancellationToken ct = default)
    {
        var fileName = $"{DateTime.UtcNow:yyyyMMdd_HHmmss_fff}_{Sanitize(toEmail)}.html";
        var path = Path.Combine(_folder, fileName);
        await File.WriteAllTextAsync(path, $"<h3>To: {toEmail}</h3><h3>Subject: {subject}</h3><hr/>{htmlBody}", ct);
        _logger.LogInformation("Email to {Email} written to {Path} (configure Email:Provider=Smtp for real delivery)", toEmail, path);
    }

    private static string Sanitize(string value) => string.Join("_", value.Split(Path.GetInvalidFileNameChars()));
}

public class SmtpEmailSender : IAppEmailSender
{
    private readonly SmtpOptions _options;

    public SmtpEmailSender(SmtpOptions options) => _options = options;

    public async Task SendAsync(string toEmail, string subject, string htmlBody, CancellationToken ct = default)
    {
        using var client = new SmtpClient(_options.Host, _options.Port)
        {
            EnableSsl = _options.EnableSsl,
            Credentials = new NetworkCredential(_options.Username, _options.Password)
        };

        using var message = new MailMessage
        {
            From = new MailAddress(_options.FromAddress, _options.FromName),
            Subject = subject,
            Body = htmlBody,
            IsBodyHtml = true
        };
        message.To.Add(toEmail);

        await client.SendMailAsync(message, ct);
    }
}
