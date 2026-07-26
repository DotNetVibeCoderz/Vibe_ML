namespace BlazorML.Web.Components.Shared;

/// <summary>
/// Inline SVG paths. Kept as constants rather than an icon font or sprite sheet: there are few
/// enough of them that the extra request and the FOUT are not worth paying for.
/// </summary>
public static class Icons
{
    private const string Open = """<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">""";
    private const string End = "</svg>";

    public const string Grid = $"""{Open}<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>{End}""";

    public const string Table = $"""{Open}<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 10v10"/>{End}""";

    public const string Flow = $"""{Open}<rect x="2" y="4" width="7" height="5" rx="1.5"/><rect x="15" y="4" width="7" height="5" rx="1.5"/><rect x="8" y="15" width="8" height="5" rx="1.5"/><path d="M5.5 9v3.5h13V9M12 12.5V15"/>{End}""";

    public const string Box = $"""{Open}<path d="M21 8 12 3 3 8v8l9 5 9-5Z"/><path d="m3 8 9 5 9-5M12 13v8"/>{End}""";

    public const string Plug = $"""{Open}<path d="M9 2v6M15 2v6M6 8h12v3a6 6 0 0 1-12 0Z"/><path d="M12 17v5"/>{End}""";

    public const string Sparkle = $"""{Open}<path d="M12 3 13.8 9.2 20 11l-6.2 1.8L12 19l-1.8-6.2L4 11l6.2-1.8Z"/><path d="M18 4v3M19.5 5.5h-3"/>{End}""";

    public const string Settings = $"""{Open}<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 9 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.6 9a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1Z"/>{End}""";

    public const string Book = $"""{Open}<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z"/>{End}""";

    public const string Sun = $"""{Open}<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>{End}""";

    public const string Moon = $"""{Open}<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/>{End}""";

    public const string Play = $"""{Open}<path d="m6 3 14 9-14 9V3Z"/>{End}""";

    public const string Plus = $"""{Open}<path d="M12 5v14M5 12h14"/>{End}""";

    public const string Trash = $"""{Open}<path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>{End}""";

    public const string Search = $"""{Open}<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>{End}""";

    public const string Chat = $"""{Open}<path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 9 9 0 0 1-3.9-.9L3 20.5l1.5-4.5A8.4 8.4 0 0 1 3.6 11.5 8.4 8.4 0 0 1 12 3.1a8.4 8.4 0 0 1 9 8.4Z"/>{End}""";

    public const string Close = $"""{Open}<path d="M18 6 6 18M6 6l12 12"/>{End}""";

    public const string Save = $"""{Open}<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z"/><path d="M17 21v-8H7v8M7 3v5h8"/>{End}""";

    public const string Upload = $"""{Open}<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>{End}""";

    public const string Copy = $"""{Open}<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>{End}""";

    public const string Check = $"""{Open}<path d="m20 6-11 11-5-5"/>{End}""";

    public const string Warning = $"""{Open}<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4M12 17h.01"/>{End}""";

    public const string History = $"""{Open}<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/>{End}""";

    public const string Send = $"""{Open}<path d="M22 2 11 13M22 2l-7 20-4-9-9-4Z"/>{End}""";

    public const string Paperclip = $"""{Open}<path d="M21 11.5 12.5 20a5 5 0 0 1-7-7l8.5-8.5a3.3 3.3 0 0 1 4.7 4.7l-8.5 8.5a1.7 1.7 0 0 1-2.4-2.4l7.8-7.8"/>{End}""";
}
