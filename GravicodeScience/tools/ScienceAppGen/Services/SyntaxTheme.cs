using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Styling;
using AvaloniaEdit.Highlighting;

namespace ScienceAppGen.Services;

/// <summary>
/// Recolours AvaloniaEdit's built-in highlighting definitions from the app palette.
/// </summary>
/// <remarks>
/// <para>
/// The bundled definitions were written for a white page: <c>MethodCall</c> is MidnightBlue,
/// <c>NumberLiteral</c> is DarkBlue and <c>ContextKeywords</c> is Navy. On the dark editor ground
/// those are all but invisible, and several of the light-mode choices (Pink operators, Silver
/// doc-comment strings) are barely better on white.
/// </para>
/// <para>
/// Rather than fork a parallel set of <c>.xshd</c> files that would then have to be kept in step
/// with the package, the definitions are kept as they are and only their <em>named colours</em>
/// are remapped onto the tokens in <c>Themes/Tokens.axaml</c>. That also means a language we have
/// not thought about still highlights, just with upstream's colours.
/// </para>
/// <para>
/// Definitions come from the process-wide <see cref="HighlightingManager.Instance"/>, so this
/// mutates shared state. That is intentional and safe here — the application renders one theme at
/// a time — but it does mean <see cref="Apply"/> has to run again whenever the theme changes.
/// </para>
/// </remarks>
internal static class SyntaxTheme
{
    /// <summary>Every definition <c>EditorViewModel.SyntaxFor</c> can hand to the editor.</summary>
    /// <remarks><c>XmlDoc</c> is not selected directly; C# imports it for <c>///</c> comments.</remarks>
    private static readonly string[] Definitions =
        ["C#", "XML", "JavaScript", "MarkDown", "Python", "HTML", "CSS", "XmlDoc"];

    /// <summary>
    /// Highlighting colour name to palette token.
    /// </summary>
    /// <remarks>
    /// Keyed by name across all languages at once, because upstream reuses the same names for the
    /// same ideas — <c>Comment</c>, <c>String</c> and <c>MethodCall</c> mean what they say in C#,
    /// Python and JavaScript alike. Names absent from a given definition are simply skipped.
    /// </remarks>
    private static readonly (string Name, string Token)[] Roles =
    [
        // ---- structure and prose ---------------------------------------------------------
        ("Comment", "CodeComment"),
        ("DocComment", "CodeComment"),
        ("BlockQuote", "CodeComment"),
        ("Punctuation", "CodePunctuation"),
        ("XmlPunctuation", "CodePunctuation"),
        ("CurlyBraces", "CodePunctuation"),
        ("Colon", "CodePunctuation"),
        ("Slash", "CodePunctuation"),
        ("Assignment", "CodePunctuation"),
        ("LineBreak", "CodePunctuation"),

        // ---- keywords --------------------------------------------------------------------
        ("Keywords", "CodeKeyword"),
        ("GotoKeywords", "CodeKeyword"),
        ("ContextKeywords", "CodeKeyword"),
        ("CheckedKeyword", "CodeKeyword"),
        ("OperatorKeywords", "CodeKeyword"),
        ("ParameterModifiers", "CodeKeyword"),
        ("Modifiers", "CodeKeyword"),
        ("Visibility", "CodeKeyword"),
        ("NamespaceKeywords", "CodeKeyword"),
        ("SemanticKeywords", "CodeKeyword"),
        ("ThisOrBaseReference", "CodeKeyword"),
        ("JavaScriptKeyWords", "CodeKeyword"),
        ("XmlTag", "CodeKeyword"),
        ("HtmlTag", "CodeKeyword"),
        ("Tags", "CodeKeyword"),
        ("ScriptTag", "CodeKeyword"),
        ("JavaScriptTag", "CodeKeyword"),
        ("JScriptTag", "CodeKeyword"),
        ("VBScriptTag", "CodeKeyword"),
        ("UnknownScriptTag", "CodeKeyword"),
        ("Selector", "CodeKeyword"),
        ("Heading", "CodeKeyword"),

        // ---- types -----------------------------------------------------------------------
        ("TypeKeywords", "CodeType"),
        ("ValueTypeKeywords", "CodeType"),
        ("ReferenceTypeKeywords", "CodeType"),
        ("JavaScriptIntrinsics", "CodeType"),
        ("Class", "CodeType"),
        ("Link", "CodeType"),

        // ---- callables and members -------------------------------------------------------
        ("MethodCall", "CodeMethod"),
        ("GetSetAddRemove", "CodeMethod"),
        ("JavaScriptGlobalFunctions", "CodeMethod"),
        ("AttributeName", "CodeMethod"),
        ("Attributes", "CodeMethod"),
        ("UnknownAttribute", "CodeMethod"),
        ("Property", "CodeMethod"),
        ("Image", "CodeMethod"),

        // ---- literals --------------------------------------------------------------------
        ("String", "CodeString"),
        ("Char", "CodeString"),
        ("Character", "CodeString"),
        ("CData", "CodeString"),
        ("AttributeValue", "CodeString"),
        ("XmlString", "CodeString"),
        ("Code", "CodeString"),
        ("NumberLiteral", "CodeNumber"),
        ("Digits", "CodeNumber"),
        ("TrueFalse", "CodeNumber"),
        ("NullOrValueKeywords", "CodeNumber"),
        ("JavaScriptLiterals", "CodeNumber"),
        ("Entity", "CodeNumber"),
        ("Entities", "CodeNumber"),
        ("EntityReference", "CodeNumber"),
        ("Value", "CodeNumber"),

        // ---- things that want to be noticed ----------------------------------------------
        ("Preprocessor", "CodeMeta"),
        ("ExceptionKeywords", "CodeMeta"),
        ("UnsafeKeywords", "CodeMeta"),
        ("StringInterpolation", "CodeMeta"),
        ("Regex", "CodeMeta"),
        ("DocType", "CodeMeta"),
        ("XmlDeclaration", "CodeMeta"),
        ("BrokenEntity", "CodeMeta"),
        ("KnownDocTags", "CodeMeta"),
    ];

    /// <summary>Comments read better set apart by shape as well as by colour.</summary>
    private static readonly string[] Italic = ["Comment", "DocComment", "BlockQuote", "Emphasis"];

    /// <summary>Applies the palette for <paramref name="variant"/> to every definition.</summary>
    /// <remarks>Never throws: a failure here should cost colour, not the editor.</remarks>
    public static void Apply(ThemeVariant variant)
    {
        var palette = new Dictionary<string, SimpleHighlightingBrush>(StringComparer.Ordinal);

        foreach (var token in Roles.Select(r => r.Token).Distinct(StringComparer.Ordinal))
        {
            if (Resolve(token, variant) is { } colour)
                palette[token] = new SimpleHighlightingBrush(colour);
        }

        foreach (var name in Definitions)
        {
            IHighlightingDefinition? definition;
            try { definition = HighlightingManager.Instance.GetDefinition(name); }
            catch { continue; }

            if (definition is null) continue;

            foreach (var (colourName, token) in Roles)
            {
                if (!palette.TryGetValue(token, out var brush)) continue;

                var target = definition.GetNamedColor(colourName);
                if (target is null) continue;

                try
                {
                    target.Foreground = brush;
                    if (Italic.Contains(colourName)) target.FontStyle = FontStyle.Italic;
                }
                catch (InvalidOperationException)
                {
                    // The definition was frozen; leave upstream's colour rather than fail.
                }
            }
        }
    }

    private static Color? Resolve(string token, ThemeVariant variant)
    {
        if (Application.Current is null) return null;

        return Application.Current.TryGetResource(token, variant, out var value) && value is Color colour
            ? colour
            : null;
    }
}
