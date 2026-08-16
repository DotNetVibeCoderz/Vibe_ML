using System.ComponentModel;
using System.Globalization;
using Microsoft.SemanticKernel;

namespace LocalGen.Kernel.Plugins;

/// <summary>
/// Clock and calendar access.
/// </summary>
/// <remarks>
/// A model's only notion of "now" is its training cutoff, so anything date-relative — "next
/// Tuesday", "how many days until…" — is wrong unless it can ask. The clock is injected so tests
/// can pin it.
/// </remarks>
public sealed class TimePlugin(TimeProvider? timeProvider = null)
{
    private readonly TimeProvider _time = timeProvider ?? TimeProvider.System;

    [KernelFunction("now")]
    [Description("The current local date and time, including the time zone.")]
    public string Now() =>
        _time.GetLocalNow().ToString("dddd, d MMMM yyyy HH:mm:ss zzz", CultureInfo.InvariantCulture);

    [KernelFunction("utc_now")]
    [Description("The current UTC date and time in ISO 8601 format.")]
    public string UtcNow() => _time.GetUtcNow().ToString("O", CultureInfo.InvariantCulture);

    [KernelFunction("today")]
    [Description("Today's date in yyyy-MM-dd format.")]
    public string Today() =>
        _time.GetLocalNow().ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);

    [KernelFunction("add_time")]
    [Description("Adds an interval to a date and returns the result. Use negative values to go back.")]
    public string AddTime(
        [Description("Starting date in yyyy-MM-dd form, or \"today\"")] string date,
        [Description("Amount to add, may be negative")] int amount,
        [Description("Unit: days, weeks, months, years, hours or minutes")] string unit)
    {
        if (!TryParseDate(date, out var start))
        {
            return $"Error: '{date}' is not a date I can read. Use yyyy-MM-dd.";
        }

        var result = unit.ToLowerInvariant().TrimEnd('s') switch
        {
            "day" => start.AddDays(amount),
            "week" => start.AddDays(amount * 7),
            "month" => start.AddMonths(amount),
            "year" => start.AddYears(amount),
            "hour" => start.AddHours(amount),
            "minute" => start.AddMinutes(amount),
            _ => (DateTimeOffset?)null
        };

        return result is null
            ? $"Error: unknown unit '{unit}'. Use days, weeks, months, years, hours or minutes."
            : result.Value.ToString("dddd, yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture);
    }

    [KernelFunction("days_between")]
    [Description("The number of days between two dates. Negative when the second date is earlier.")]
    public string DaysBetween(
        [Description("First date, yyyy-MM-dd or \"today\"")] string from,
        [Description("Second date, yyyy-MM-dd or \"today\"")] string to)
    {
        if (!TryParseDate(from, out var start) || !TryParseDate(to, out var end))
        {
            return "Error: both dates must be in yyyy-MM-dd format (or \"today\").";
        }

        return ((int)(end.Date - start.Date).TotalDays).ToString(CultureInfo.InvariantCulture);
    }

    [KernelFunction("day_of_week")]
    [Description("The day of the week for a given date.")]
    public string DayOfWeek(
        [Description("Date in yyyy-MM-dd form, or \"today\"")] string date) =>
        TryParseDate(date, out var parsed)
            ? parsed.DayOfWeek.ToString()
            : $"Error: '{date}' is not a date I can read.";

    [KernelFunction("convert_timezone")]
    [Description("Converts a date and time from one IANA or Windows time zone to another.")]
    public string ConvertTimeZone(
        [Description("Date and time, e.g. \"2026-03-14 09:30\"")] string dateTime,
        [Description("Source zone, e.g. \"Asia/Jakarta\" or \"UTC\"")] string fromZone,
        [Description("Target zone, e.g. \"America/New_York\"")] string toZone)
    {
        if (!DateTime.TryParse(dateTime, CultureInfo.InvariantCulture,
                DateTimeStyles.None, out var parsed))
        {
            return $"Error: '{dateTime}' is not a date and time I can read.";
        }

        try
        {
            var source = TimeZoneInfo.FindSystemTimeZoneById(fromZone);
            var target = TimeZoneInfo.FindSystemTimeZoneById(toZone);

            var utc = TimeZoneInfo.ConvertTimeToUtc(
                DateTime.SpecifyKind(parsed, DateTimeKind.Unspecified), source);

            var converted = TimeZoneInfo.ConvertTimeFromUtc(utc, target);

            return $"{converted:yyyy-MM-dd HH:mm} {target.Id}";
        }
        catch (TimeZoneNotFoundException ex)
        {
            return $"Error: {ex.Message}";
        }
        catch (InvalidTimeZoneException ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    [KernelFunction("timestamp")]
    [Description("The current Unix timestamp in seconds.")]
    public string Timestamp() =>
        _time.GetUtcNow().ToUnixTimeSeconds().ToString(CultureInfo.InvariantCulture);

    /// <summary>Accepts an ISO date or the word "today", which models reach for constantly.</summary>
    private bool TryParseDate(string value, out DateTimeOffset result)
    {
        if (string.Equals(value?.Trim(), "today", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(value?.Trim(), "now", StringComparison.OrdinalIgnoreCase))
        {
            result = _time.GetLocalNow();
            return true;
        }

        return DateTimeOffset.TryParse(
            value, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out result);
    }
}
