using System.Text.RegularExpressions;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Media;

namespace MediaPipeNet.Gallery.Controls;

/// <summary>Read-only C# snippet with lightweight syntax coloring.</summary>
public sealed partial class CodeView : SelectableTextBlock
{
    private static readonly HashSet<string> Keywords =
    [
        "using", "var", "new", "foreach", "in", "await", "async", "return", "if", "else", "true", "false", "null",
        "float", "int", "string", "bool", "public", "private", "static", "class", "record", "sealed", "void", "is", "not", "with", "const",
    ];

    private static readonly IBrush Plain = new SolidColorBrush(Color.Parse("#D5DBE5"));
    private static readonly IBrush Keyword = new SolidColorBrush(Color.Parse("#8FA8FF"));
    private static readonly IBrush Type = new SolidColorBrush(Color.Parse("#35E0B5"));
    private static readonly IBrush Str = new SolidColorBrush(Color.Parse("#FFB28F"));
    private static readonly IBrush Comment = new SolidColorBrush(Color.Parse("#6B7585"));
    private static readonly IBrush Number = new SolidColorBrush(Color.Parse("#FFC53D"));

    public CodeView()
    {
        FontFamily = new FontFamily("avares://MediaPipeNet.Gallery/Assets/Fonts#JetBrains Mono");
        FontSize = 12.5;
        LineHeight = 20;
        Foreground = Plain;
        TextWrapping = TextWrapping.NoWrap;
    }

    [GeneratedRegex("""(//[^\n]*)|(\$?"(?:[^"\\\n]|\\.)*")|(\b\d+(?:\.\d+)?f?\b)|(\b[A-Za-z_][A-Za-z0-9_]*\b)|(\s+|.)""")]
    private static partial Regex TokenRegex();

    /// <summary>Replaces the displayed code.</summary>
    public void SetCode(string code)
    {
        var inlines = new InlineCollection();
        foreach (Match m in TokenRegex().Matches(code))
        {
            IBrush brush = Plain;
            if (m.Groups[1].Success) brush = Comment;
            else if (m.Groups[2].Success) brush = Str;
            else if (m.Groups[3].Success) brush = Number;
            else if (m.Groups[4].Success)
                brush = Keywords.Contains(m.Value) ? Keyword : char.IsUpper(m.Value[0]) ? Type : Plain;
            inlines.Add(new Run(m.Value) { Foreground = brush });
        }
        Inlines = inlines;
        CodeText = code;
    }

    /// <summary>The raw code (for copying).</summary>
    public string CodeText { get; private set; } = "";
}
