using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using ScienceAppGen.Plugins;

namespace ScienceAppGen.Services.Conversion;

/// <summary>
/// Assembles the API reference that goes into a conversion prompt.
/// </summary>
/// <remarks>
/// <para>
/// The failure this exists to prevent is the model writing calls that look right and do not exist
/// - <c>Signal.Butterworth</c>, <c>ax.SetXlabel</c> - which costs a build-repair round each. So the
/// Gravicode libraries contribute their curated references, and SciNet and Visual.Net contribute
/// <b>signatures read by reflection from the assemblies</b>, which cannot be out of date.
/// </para>
/// <para>
/// Reflection output is filtered to what the file uses: scipy names are followed through their
/// call chain (<c>stats.norm.pdf</c> → <c>Stats.Norm</c> → <c>NormalDistribution.Pdf</c>), and
/// plotting types contribute only the methods the file calls by name. SciNet's <c>Stats</c> facade
/// alone has hundreds of members; sending them all would bury the dozen that matter.
/// </para>
/// </remarks>
public static partial class ApiReferenceBuilder
{
    private const int MaxOverloadsPerName = 4;

    private static readonly Dictionary<string, string> ScipyFacades = new(StringComparer.Ordinal)
    {
        ["scipy.signal"] = "Signal", ["scipy.optimize"] = "Optimize", ["scipy.integrate"] = "Integrate",
        ["scipy.interpolate"] = "Interpolate", ["scipy.linalg"] = "Linalg", ["scipy.stats"] = "Stats",
        ["scipy.special"] = "Special", ["scipy.fft"] = "Fft", ["scipy.fftpack"] = "Fft",
        ["scipy.ndimage"] = "NdImage", ["scipy.cluster"] = "Cluster", ["scipy.cluster.hierarchy"] = "Cluster",
        ["scipy.cluster.vq"] = "Cluster", ["scipy.constants"] = "Constants", ["scipy.differentiate"] = "Differentiate",
        ["scipy.stats.qmc"] = "Qmc",
    };

    /// <summary>scipy submodules whose SciNet counterparts are namespaces of types rather than facades.</summary>
    private static readonly Dictionary<string, string> ScipyNamespaces = new(StringComparer.Ordinal)
    {
        ["scipy.sparse"] = "Gravicode.Science.SciNet.Sparse",
        ["scipy.sparse.linalg"] = "Gravicode.Science.SciNet.Sparse",
        ["scipy.sparse.csgraph"] = "Gravicode.Science.SciNet.Sparse",
        ["scipy.spatial"] = "Gravicode.Science.SciNet.Spatial",
        ["scipy.spatial.distance"] = "Gravicode.Science.SciNet.Spatial",
        ["scipy.spatial.transform"] = "Gravicode.Science.SciNet.Spatial",
        ["scipy.io"] = "Gravicode.Science.SciNet.IO",
        ["scipy.io.wavfile"] = "Gravicode.Science.SciNet.IO",
    };

    /// <summary>pyplot/Axes names whose Matplotlib.Net spelling is not just PascalCase.</summary>
    private static readonly (string Python, string CSharp)[] PyplotRenames =
    [
        ("xlabel", "SetXLabel"), ("ylabel", "SetYLabel"), ("title", "SetTitle"), ("xlim", "SetXLim"),
        ("ylim", "SetYLim"), ("xscale", "SetXScale"), ("yscale", "SetYScale"), ("xticks", "SetXTicks"),
        ("yticks", "SetYTicks"), ("set_xlabel", "SetXLabel"), ("set_ylabel", "SetYLabel"),
        ("set_title", "SetTitle"), ("set_xlim", "SetXLim"), ("set_ylim", "SetYLim"), ("suptitle", "SupTitle"),
        ("savefig", "Save"), ("colorbar", "AddColorbar"), ("axhline", "AxHLine"), ("axvline", "AxVLine"),
        ("axhspan", "AxHSpan"), ("axvspan", "AxVSpan"), ("fill_between", "FillBetween"), ("imshow", "ImShow"),
        ("errorbar", "ErrorBar"), ("barh", "BarH"), ("boxplot", "BoxPlot"), ("violinplot", "ViolinPlot"),
        ("contourf", "ContourF"), ("hist2d", "Hist2D"), ("stackplot", "StackPlot"), ("tight_layout", "TightLayout"),
        ("subplots", "Subplots"), ("add_subplot", "AddSubplot"), ("axis", "AxisOff"), ("invert_yaxis", "InvertYAxis"),
        ("set_aspect", "SetAspect"), ("legend", "Legend"), ("grid", "Grid"),
    ];

    [GeneratedRegex(@"\.([A-Za-z_]\w*)\s*\(")]
    private static partial Regex MethodCall();

    /// <summary>The reference text for one module's conversion prompt.</summary>
    /// <param name="module">The module being converted.</param>
    /// <param name="mappings">How the project's packages are handled.</param>
    public static string Build(ModuleAnalysis module, IReadOnlyList<PackageMapping> mappings)
    {
        var packages = module.ExternalPackages.Select(LibraryMap.Map).ToList();
        // The project's shared data libraries first (see LibraryMap.SharedTargets), then this
        // module's own - SciNet and the plotting libraries are described only where imported.
        var targets = LibraryMap.SharedTargets(mappings)
            .Concat(packages.Where(p => p.Target is not null && p.Target != LibraryMap.Bcl).Select(p => p.Target!))
            .Distinct()
            .ToList();

        var builder = new StringBuilder();
        var calledNames = CalledNames(module.File.Source);

        foreach (var target in targets)
        {
            switch (target.Id)
            {
                case "SciNet":
                    builder.AppendLine(SciNetOverview);
                    AppendScipy(builder, module.UsedSymbols);
                    break;
                case "Matplotlib.Net":
                    builder.AppendLine(MatplotlibOverview);
                    AppendPyplotTable(builder);
                    AppendCalledMembers(builder, calledNames, typeof(Matplotlib.Net.Figure), typeof(Matplotlib.Net.Artists.Axes));
                    break;
                case "Seaborn.Net":
                    builder.AppendLine(SeabornOverview);
                    AppendCalledMembers(builder, calledNames, typeof(Seaborn.Net.Sns));
                    break;
                case "Plotly.Net":
                    builder.AppendLine(PlotlyOverview);
                    AppendCalledMembers(builder, calledNames, typeof(Plotly.Net.Chart), typeof(Plotly.Net.PlotlyFigure));
                    break;
                default:
                    var curated = GravicodeReferencePlugin.Lookup(target.Id);
                    if (curated is not null) builder.AppendLine(curated.Trim()).AppendLine();
                    break;
            }
        }

        foreach (var note in packages.Where(p => p.Note is not null))
            builder.AppendLine($"NOTE on {note.Package}: {note.Note}");

        return builder.ToString().TrimEnd();
    }

    /// <summary>Method names called anywhere in the source (<c>.plot(</c>, <c>.set_xlabel(</c>).</summary>
    public static IReadOnlySet<string> CalledNames(string source)
        => MethodCall().Matches(PythonImportAnalyzer.StripStringsAndComments(source))
            .Select(m => m.Groups[1].Value)
            .ToHashSet(StringComparer.Ordinal);

    // ---------------------------------------------------------------- scipy

    private static void AppendScipy(StringBuilder builder, IReadOnlySet<string> used)
    {
        var assembly = typeof(Gravicode.Science.SciNet.Optimize).Assembly;
        var emitted = new HashSet<MemberInfo>();
        var lines = new List<string>();

        foreach (var symbol in used.Where(s => s.StartsWith("scipy.", StringComparison.Ordinal)).OrderBy(s => s, StringComparer.Ordinal))
        {
            // Longest submodule prefix that SciNet knows: scipy.stats.qmc before scipy.stats.
            var prefix = ScipyFacades.Keys.Concat(ScipyNamespaces.Keys)
                .Where(p => symbol == p || symbol.StartsWith(p + ".", StringComparison.Ordinal))
                .OrderByDescending(p => p.Length)
                .FirstOrDefault();
            if (prefix is null) continue;

            var chain = symbol.Length > prefix.Length ? symbol[(prefix.Length + 1)..].Split('.') : [];
            if (chain.Length == 0) continue;

            if (ScipyFacades.TryGetValue(prefix, out var facadeName))
            {
                var facade = assembly.GetType("Gravicode.Science.SciNet." + facadeName);
                if (facade is not null) FollowChain(facade, chain, prefix, emitted, lines, staticOnly: true);
            }
            else
            {
                // A type in a sub-namespace: scipy.spatial.KDTree → Spatial.KdTree / KDTree.
                var ns = ScipyNamespaces[prefix];
                var wanted = CSharpNaming.ToPascalCase(chain[0]);
                var types = assembly.GetExportedTypes()
                    .Where(t => t.Namespace == ns && (t.Name.Equals(wanted, StringComparison.OrdinalIgnoreCase)
                                                      || t.Name.Equals(chain[0], StringComparison.OrdinalIgnoreCase)))
                    .ToList();
                if (types.Count == 0)
                {
                    // A function rather than a type: look on every static class of the namespace.
                    foreach (var holder in assembly.GetExportedTypes().Where(t => t.Namespace == ns && t.IsAbstract && t.IsSealed))
                        FollowChain(holder, chain, prefix, emitted, lines, staticOnly: true);
                }
                foreach (var type in types)
                {
                    if (!emitted.Add(type)) continue;
                    lines.Add($"// {prefix}.{chain[0]} -> {type.FullName}");
                    lines.AddRange(DescribeType(type, chain.Length > 1 ? chain[1..] : null));
                }
            }
        }

        if (lines.Count == 0) return;
        builder.AppendLine("SciNet signatures for the scipy names this file uses (exact, from the assembly):");
        foreach (var line in lines) builder.Append("  ").AppendLine(line);
        builder.AppendLine();
        AppendParameterTypes(builder, emitted.OfType<MethodInfo>(), assembly);
    }

    /// <summary>
    /// How to build the library's own parameter types: an enum's members, and a class's implicit
    /// conversions and constructors. <c>CriticalFrequencies</c> converts from a double and from a
    /// collection expression, <c>PeakCondition</c> from a double - without saying so, the model
    /// constructs them by hand, wrongly.
    /// </summary>
    private static void AppendParameterTypes(StringBuilder builder, IEnumerable<MethodInfo> methods, Assembly assembly)
    {
        var types = methods.SelectMany(m => m.GetParameters())
            .Select(p => p.ParameterType.IsByRef ? p.ParameterType.GetElementType()! : p.ParameterType)
            .Select(t => Nullable.GetUnderlyingType(t) ?? t)
            .Where(t => t.Assembly == assembly && t.Name != "NdArray")
            .Distinct()
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();
        if (types.Count == 0) return;

        builder.AppendLine("How to pass the parameter types above:");
        foreach (var type in types)
        {
            if (type.IsEnum)
            {
                builder.AppendLine($"  enum {type.Name}: {string.Join(", ", Enum.GetNames(type))}");
                continue;
            }

            var ways = new List<string>();
            ways.AddRange(type.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "op_Implicit" && m.ReturnType == type)
                .Select(m => "implicit from " + TypeName(m.GetParameters()[0].ParameterType)));
            if (type.GetCustomAttributes().Any(a => a.GetType().Name == "CollectionBuilderAttribute"))
                ways.Add("a collection expression, e.g. [0.5, 15.0]");
            ways.AddRange(type.GetConstructors().Take(2).Select(c => "new(" + Parameters(c.GetParameters()) + ")"));
            ways.AddRange(type.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => !m.IsSpecialName && m.ReturnType == type && !IsNoise(m))
                .Take(5)
                .Select(m => $"{type.Name}.{m.Name}({Parameters(m.GetParameters())})"));

            if (ways.Count > 0) builder.AppendLine($"  {type.Name}: {string.Join("; ", ways)}");
        }
        builder.AppendLine();
    }

    private static void FollowChain(Type start, string[] chain, string prefix, HashSet<MemberInfo> emitted, List<string> lines, bool staticOnly)
    {
        IEnumerable<Type> current = [start];
        var path = prefix;

        for (var i = 0; i < chain.Length; i++)
        {
            var segment = chain[i];
            path += "." + segment;
            var next = new List<Type>();
            var found = false;

            foreach (var type in current)
            {
                var members = MatchMembers(type, segment, staticOnly && i == 0, prefixMatch: i == 0 && chain.Length == 1);
                foreach (var member in members)
                {
                    found = true;
                    if (emitted.Add(member)) lines.Add($"// {path} -> {Describe(member)}");
                    var result = ResultType(member);
                    if (result is not null && !IsSimple(result)) next.Add(result);
                }
            }

            if (!found) return;

            // The chain ended on something with a result type: list that type's members once, so
            // the model sees .Popt, .Success and the like without guessing.
            if (i == chain.Length - 1)
            {
                foreach (var result in next.Distinct())
                {
                    if (!emitted.Add(result)) continue;
                    lines.AddRange(DescribeType(result, null).Select(l => "    " + l));
                }
            }

            current = next.Distinct().ToList();
        }
    }

    /// <summary>Compiler-generated record members and object overrides carry nothing for the model.</summary>
    private static bool IsNoise(MemberInfo member)
        => member.Name.StartsWith('<') || member.Name is "Equals" or "GetHashCode" or "ToString" or "GetType" or "PrintMembers";

    private static IEnumerable<MemberInfo> MatchMembers(Type type, string pythonName, bool staticOnly, bool prefixMatch)
    {
        var pascal = CSharpNaming.ToPascalCase(pythonName);
        var flags = BindingFlags.Public | (staticOnly ? BindingFlags.Static : BindingFlags.Static | BindingFlags.Instance);
        var members = type.GetMembers(flags)
            .Where(m => m is MethodInfo { IsSpecialName: false } or PropertyInfo or FieldInfo)
            .Where(m => !IsNoise(m))
            .ToList();

        var exact = members.Where(m => m.Name.Equals(pascal, StringComparison.OrdinalIgnoreCase)).ToList();
        // A bare function name (butter) also brings its variants (ButterSos) - scipy picks the
        // output form with a keyword argument, SciNet with a separate method.
        var related = prefixMatch
            ? members.Where(m => m.Name.StartsWith(pascal, StringComparison.OrdinalIgnoreCase) && !exact.Contains(m)).Take(6)
            : [];

        return exact.Take(MaxOverloadsPerName).Concat(related);
    }

    // ---------------------------------------------------------------- plotting

    private static void AppendPyplotTable(StringBuilder builder)
    {
        var known = new[] { typeof(Matplotlib.Net.Figure), typeof(Matplotlib.Net.Artists.Axes) }
            .SelectMany(t => t.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static))
            .Select(m => m.Name)
            .ToHashSet(StringComparer.Ordinal);

        builder.AppendLine("pyplot/Axes names that are spelled differently in Matplotlib.Net:");
        foreach (var (python, csharp) in PyplotRenames.Where(r => known.Contains(r.CSharp)))
            builder.Append("  ").Append(python).Append(" -> ").AppendLine(csharp);
        builder.AppendLine();
    }

    private static void AppendCalledMembers(StringBuilder builder, IReadOnlySet<string> called, params Type[] types)
    {
        var renames = PyplotRenames.ToLookup(r => r.Python, r => r.CSharp, StringComparer.Ordinal);
        var wanted = called
            .SelectMany(n => renames[n].DefaultIfEmpty(CSharpNaming.ToPascalCase(n)))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        var lines = new List<string>();
        foreach (var type in types)
        {
            var members = type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static)
                .Where(m => m is MethodInfo { IsSpecialName: false } or PropertyInfo)
                .Where(m => wanted.Contains(m.Name))
                .GroupBy(m => m.Name)
                .SelectMany(g => g.Take(MaxOverloadsPerName));
            foreach (var member in members) lines.Add($"{type.Name}.{Describe(member)}");
        }

        if (lines.Count == 0) return;
        builder.AppendLine("Signatures of the plotting members this file needs (exact, from the assembly):");
        foreach (var line in lines.Distinct()) builder.Append("  ").AppendLine(line);
        builder.AppendLine();
    }

    // ---------------------------------------------------------------- formatting

    private static IEnumerable<string> DescribeType(Type type, string[]? onlyMembers)
    {
        var kind = type.IsAbstract && type.IsSealed ? "static class" : type.IsValueType ? "struct" : "class";
        yield return $"{kind} {TypeName(type)}";

        var wanted = onlyMembers?.Select(CSharpNaming.ToPascalCase).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var count = 0;

        foreach (var constructor in type.GetConstructors().Take(3))
            yield return "  new(" + Parameters(constructor.GetParameters()) + ")";

        foreach (var member in type.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly)
                     .Where(m => m is MethodInfo { IsSpecialName: false } or PropertyInfo)
                     .Where(m => !IsNoise(m))
                     .Where(m => wanted is null || wanted.Contains(m.Name))
                     .OrderBy(m => m is PropertyInfo ? 0 : 1).ThenBy(m => m.Name, StringComparer.Ordinal))
        {
            if (++count > 40) { yield return "  ... (more members)"; yield break; }
            yield return "  " + Describe(member);
        }
    }

    /// <summary>A one-line C# signature.</summary>
    public static string Describe(MemberInfo member) => member switch
    {
        MethodInfo method => $"{(method.IsStatic ? "static " : "")}{TypeName(method.ReturnType)} {method.Name}"
                             + (method.IsGenericMethodDefinition ? "<" + string.Join(", ", method.GetGenericArguments().Select(a => a.Name)) + ">" : "")
                             + $"({Parameters(method.GetParameters())})",
        PropertyInfo property => $"{TypeName(property.PropertyType)} {property.Name} {{ {(property.CanWrite && property.SetMethod?.IsPublic == true ? "get; set;" : "get;")} }}",
        FieldInfo field => $"{(field.IsStatic ? "static " : "")}{TypeName(field.FieldType)} {field.Name}",
        Type type => $"class {TypeName(type)}",
        _ => member.Name,
    };

    private static string Parameters(ParameterInfo[] parameters)
        => string.Join(", ", parameters.Select(p =>
        {
            var text = (IsParams(p) ? "params " : "") + (p.IsOut ? "out " : "") + TypeName(p.ParameterType.IsByRef ? p.ParameterType.GetElementType()! : p.ParameterType) + " " + p.Name;
            if (p.HasDefaultValue) text += " = " + Literal(p.DefaultValue, p.ParameterType);
            return text;
        }));

    private static bool IsParams(ParameterInfo parameter)
        => parameter.GetCustomAttributes().Any(a => a.GetType().Name is "ParamArrayAttribute" or "ParamCollectionAttribute");

    private static string Literal(object? value, Type? declared = null)
    {
        // An enum default arrives boxed as its underlying integer, and for a nullable enum it
        // would print as "1"; name the member instead.
        var enumType = declared is null ? null : Nullable.GetUnderlyingType(declared) ?? declared;
        if (value is not null and not Enum && enumType is { IsEnum: true })
            value = Enum.ToObject(enumType, value);
        return LiteralValue(value);
    }

    private static string LiteralValue(object? value) => value switch
    {
        null => "null",
        string s => "\"" + s + "\"",
        bool b => b ? "true" : "false",
        double d => d.ToString("R", System.Globalization.CultureInfo.InvariantCulture),
        float f => f.ToString("R", System.Globalization.CultureInfo.InvariantCulture) + "f",
        Enum e => e.GetType().Name + "." + e,
        _ => Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? "null",
    };

    /// <summary>C#-style type names: <c>double[]</c>, <c>IReadOnlyList&lt;string&gt;</c>, <c>int?</c>.</summary>
    public static string TypeName(Type type)
    {
        if (type == typeof(void)) return "void";
        if (type.IsArray) return TypeName(type.GetElementType()!) + "[" + new string(',', type.GetArrayRank() - 1) + "]";
        var nullable = Nullable.GetUnderlyingType(type);
        if (nullable is not null) return TypeName(nullable) + "?";

        var alias = Type.GetTypeCode(type) switch
        {
            TypeCode.Double => "double", TypeCode.Single => "float", TypeCode.Int32 => "int", TypeCode.Int64 => "long",
            TypeCode.Boolean => "bool", TypeCode.String => "string", TypeCode.Byte => "byte", TypeCode.Char => "char",
            _ => null,
        };
        if (alias is not null && !type.IsEnum) return alias;
        if (type == typeof(object)) return "object";

        if (type.IsGenericType)
        {
            var name = type.Name[..type.Name.IndexOf('`')];
            if (name.StartsWith("ValueTuple", StringComparison.Ordinal))
                return "(" + string.Join(", ", type.GetGenericArguments().Select(TypeName)) + ")";
            return name + "<" + string.Join(", ", type.GetGenericArguments().Select(TypeName)) + ">";
        }

        return type.IsNested ? TypeName(type.DeclaringType!) + "." + type.Name : type.Name;
    }

    private static Type? ResultType(MemberInfo member) => member switch
    {
        MethodInfo method => method.ReturnType,
        PropertyInfo property => property.PropertyType,
        FieldInfo field => field.FieldType,
        _ => null,
    };

    private static bool IsSimple(Type type)
        => type.IsPrimitive || type == typeof(string) || type == typeof(void) || type.IsArray || type.IsEnum
           || type.Namespace?.StartsWith("System", StringComparison.Ordinal) == true
           || type.Name == "NdArray";

    // ---------------------------------------------------------------- curated overviews

    private const string SciNetOverview = """
        SciNet - scipy for .NET. using Gravicode.Science.SciNet; (sub-namespaces .Sparse, .Spatial, .IO)
          Facades mirror the submodules: Signal, Optimize, Integrate, Interpolate, Linalg (scipy.linalg; note
          GraviNum's numpy-style one is LinAlg), Stats, Special, Fft (scipy.fft.fft is Fft.Fft1D), NdImage,
          Cluster, Constants, Differentiate. Functions are PascalCase of the scipy name.
          Arrays are GraviNum NdArray or double[]; keyword arguments become C# named arguments.
          A scipy function returning a tuple returns a record or tuple here - use its named members.
          Distributions: Stats.Norm(loc, scale).Pdf(x) / .Cdf / .Ppf / .Rvs(size, seed); fitting is static
          on the distribution type: NormalDistribution.Fit(data) (stats.norm.fit).
        """;

    private const string MatplotlibOverview = """
        Matplotlib.Net (Visual.Net) - static figures to PNG/SVG/PDF. No pyplot global state:
          var figure = new Figure(widthInches: 8, heightInches: 4.5);   // dpi default 100
          var ax = figure.AddSubplot();                  // or figure.Subplots(rows, cols) -> Axes[] row-major
          ax.Plot(x, y, label: "sin", color: ..., lineWidth: 2);   // x, y are double[]
          ax.SetTitle("t").SetXLabel("x").SetYLabel("y");          // fluent, returns the Axes
          ax.Legend(); ax.Grid(true);
          figure.TightLayout = true;
          figure.Save("plot.png");                       // format from extension
          Plot's third positional argument is the LABEL; a matplotlib format string goes through
          ax.PlotFormat(x, y, "r--o"). Colors: Color.Parse("#4c72b0") or Color.Parse("tab:blue")
          (using Visual.Net.Core.Colors). Scatter sizes are areas in points^2, as in matplotlib.
          plt.show() has no window here: save the figure to a PNG next to the program instead, and
          print its path. Convert NdArray to double[] with .ToArray() before plotting.
          Idioms without a same-name method: plt.subplots(r, c, figsize=(w, h)) -> var fig = new
          Figure(w, h); var axes = fig.Subplots(r, c) (Axes[], row-major; index 2-D grids as
          axes[row * c + col]); semilogy/semilogx/loglog -> Plot then SetYScale("log")/SetXScale("log");
          set_yticks + set_yticklabels -> SetYTicks(positions, labels); linestyle="--" -> style:
          LineStyle.Dashed; fig.tight_layout() -> fig.TightLayout = true (set before Save).
        """;

    private const string SeabornOverview = """
        Seaborn.Net (Visual.Net) - using Seaborn.Net; built on Matplotlib.Net, data is a GraviFrame DataFrame.
          Sns.SetTheme(SeabornStyle.WhiteGrid);
          var (figure, ax) = Sns.Subplots(8, 4.5);
          Sns.ScatterPlot(df, "x", "y", hue: "group", ax: ax);   // axes-level functions take ax and return Axes
          figure.Save("plot.png");
          Figure-level functions (PairPlot, JointPlot, RelPlot, CatPlot, DisPlot, LmPlot, ClusterMap) return a Figure.
        """;

    private const string PlotlyOverview = """
        Plotly.Net (Visual.Net) - interactive HTML. using Plotly.Net; using Plotly.Net.Traces;
          Quick: Chart.Line(x, y, ...) / Chart.Scatter / Chart.Bar / Chart.Histogram ... return a PlotlyFigure.
          Full: var fig = new PlotlyFigure(); fig.Add(new Scatter { X = x, Y = y, Mode = ScatterMode.Lines });
          fig.Layout.Title = "t"; fig.Save("chart.html");   // fig.show() becomes Save to an .html file
          Any attribute without a property: trace.With("marker.line.width", 2), fig.Layout.Set("dragmode", "pan").
        """;
}
