using System.ComponentModel.DataAnnotations;
using System.Text;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Extensions.Configuration;
using SplatStudio.Application.Abstractions;
using SplatStudio.Infrastructure.Data;

namespace SplatStudio.Web.Pages.Account;

public class ForgotPasswordModel : PageModel
{
    private readonly UserManager<ApplicationUser> _userManager;
    private readonly IAppEmailSender _emailSender;
    private readonly IConfiguration _configuration;

    public ForgotPasswordModel(UserManager<ApplicationUser> userManager, IAppEmailSender emailSender, IConfiguration configuration)
    {
        _userManager = userManager;
        _emailSender = emailSender;
        _configuration = configuration;
    }

    [BindProperty]
    public InputModel Input { get; set; } = new();

    public bool EmailSent { get; set; }

    /// <summary>
    /// Only populated when Email:Provider is "File" (the zero-config dev
    /// default), so local testing doesn't require digging through
    /// App_Data/emails by hand. Never shown when a real SMTP sender is configured.
    /// </summary>
    public string? DebugResetLink { get; set; }

    public void OnGet()
    {
    }

    public async Task<IActionResult> OnPostAsync()
    {
        if (!ModelState.IsValid) return Page();

        var user = await _userManager.FindByEmailAsync(Input.Email);
        if (user is not null)
        {
            var token = await _userManager.GeneratePasswordResetTokenAsync(user);
            var encodedToken = WebEncoders.Base64UrlEncode(Encoding.UTF8.GetBytes(token));
            var resetLink = Url.Page("/Account/ResetPassword", pageHandler: null,
                values: new { code = encodedToken, email = user.Email }, protocol: Request.Scheme);

            await _emailSender.SendAsync(
                user.Email!,
                "Reset your SplatStudio password",
                $"<p>Click the link below to choose a new password:</p><p><a href=\"{resetLink}\">{resetLink}</a></p><p>If you didn't request this, you can ignore this email.</p>");

            if (string.Equals(_configuration["Email:Provider"], "File", StringComparison.OrdinalIgnoreCase))
                DebugResetLink = resetLink;
        }

        // Always show the same generic message — confirming or denying that
        // an email address has an account would leak account existence.
        EmailSent = true;
        return Page();
    }

    public class InputModel
    {
        [Required, EmailAddress]
        public string Email { get; set; } = string.Empty;
    }
}
