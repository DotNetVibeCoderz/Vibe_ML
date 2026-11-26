using System.Globalization;
using System.Text;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet;

/// <summary>
/// A column expression: <c>Expr.Col("price") * Expr.Col("qty") &gt; 100</c>. The language of
/// <see cref="DaskFrame"/> filters and projections, and what SQL compiles to.
/// </summary>
/// <remarks>
/// <para>
/// Expressions rather than lambdas, for two reasons. The optimiser can read them: a filter that
/// names only <c>region</c> can move below a join or into a file scan, and a projection tells the
/// scan which columns to keep - a <c>Func&lt;RowView, bool&gt;</c> is opaque to all of that. And
/// they serialise, so a plan built from them runs on cluster workers.
/// </para>
/// <para>
/// Missing values follow SQL: arithmetic with a missing operand is missing, a comparison with one
/// is unknown, and <c>AND</c>/<c>OR</c> use three-valued logic (<c>unknown AND false</c> is
/// false). A filter keeps only rows where the predicate is true.
/// </para>
/// </remarks>
public abstract class Expr
{
    private protected Expr() { }

    /// <summary>A column reference.</summary>
    public static Expr Col(string name) => new ColumnExpr(name);

    /// <summary>A literal number.</summary>
    public static Expr Lit(double value) => new LiteralExpr(Value.Of(value));

    /// <summary>A literal string.</summary>
    public static Expr Lit(string? value) => new LiteralExpr(Value.Of(value));

    /// <summary>A literal boolean.</summary>
    public static Expr Lit(bool value) => new LiteralExpr(Value.Of(value));

    /// <summary>A literal timestamp.</summary>
    public static Expr Lit(DateTime value) => new LiteralExpr(Value.Of(value));

    /// <summary>A missing value.</summary>
    public static Expr Null() => new LiteralExpr(Value.Missing);

    /// <summary><paramref name="then"/> where <paramref name="condition"/> is true, else <paramref name="otherwise"/>
    /// (SQL's <c>CASE WHEN</c>).</summary>
    public static Expr If(Expr condition, Expr then, Expr otherwise) => new FunctionExpr("if", [condition, then, otherwise]);

    /// <summary>The first non-missing argument.</summary>
    public static Expr Coalesce(params Expr[] values) => new FunctionExpr("coalesce", values);

    /// <summary>A number literal.</summary>
    public static implicit operator Expr(double value) => Lit(value);

    /// <summary>A string literal (not a column - use <see cref="Col"/> for that).</summary>
    public static implicit operator Expr(string value) => Lit(value);

    /// <summary>A boolean literal.</summary>
    public static implicit operator Expr(bool value) => Lit(value);

    /// <summary>A timestamp literal.</summary>
    public static implicit operator Expr(DateTime value) => Lit(value);

    /// <summary>a + b (numbers add, strings concatenate).</summary>
    public static Expr operator +(Expr a, Expr b) => new BinaryExpr(BinaryKind.Add, a, b);
    /// <summary>a - b</summary>
    public static Expr operator -(Expr a, Expr b) => new BinaryExpr(BinaryKind.Subtract, a, b);
    /// <summary>a * b</summary>
    public static Expr operator *(Expr a, Expr b) => new BinaryExpr(BinaryKind.Multiply, a, b);
    /// <summary>a / b</summary>
    public static Expr operator /(Expr a, Expr b) => new BinaryExpr(BinaryKind.Divide, a, b);
    /// <summary>a mod b</summary>
    public static Expr operator %(Expr a, Expr b) => new BinaryExpr(BinaryKind.Modulo, a, b);
    /// <summary>-a</summary>
    public static Expr operator -(Expr a) => new FunctionExpr("negate", [a]);
    /// <summary>a &gt; b</summary>
    public static Expr operator >(Expr a, Expr b) => new BinaryExpr(BinaryKind.Greater, a, b);
    /// <summary>a &lt; b</summary>
    public static Expr operator <(Expr a, Expr b) => new BinaryExpr(BinaryKind.Less, a, b);
    /// <summary>a &gt;= b</summary>
    public static Expr operator >=(Expr a, Expr b) => new BinaryExpr(BinaryKind.GreaterOrEqual, a, b);
    /// <summary>a &lt;= b</summary>
    public static Expr operator <=(Expr a, Expr b) => new BinaryExpr(BinaryKind.LessOrEqual, a, b);
    /// <summary>a AND b</summary>
    public static Expr operator &(Expr a, Expr b) => new BinaryExpr(BinaryKind.And, a, b);
    /// <summary>a OR b</summary>
    public static Expr operator |(Expr a, Expr b) => new BinaryExpr(BinaryKind.Or, a, b);
    /// <summary>NOT a</summary>
    public static Expr operator !(Expr a) => new FunctionExpr("not", [a]);

    /// <summary>a = b. A method, not <c>==</c>, so reference equality on expressions keeps working.</summary>
    public Expr Eq(Expr other) => new BinaryExpr(BinaryKind.Equal, this, other);
    /// <summary>a &lt;&gt; b</summary>
    public Expr Ne(Expr other) => new BinaryExpr(BinaryKind.NotEqual, this, other);
    /// <summary>Whether the value is missing.</summary>
    public Expr IsNull() => new FunctionExpr("isnull", [this]);
    /// <summary>Whether the value is present.</summary>
    public Expr IsNotNull() => new FunctionExpr("notnull", [this]);
    /// <summary>Replaces missing values.</summary>
    public Expr FillNull(Expr value) => new FunctionExpr("coalesce", [this, value]);
    /// <summary>Whether the value is one of <paramref name="values"/>.</summary>
    public Expr In(params Expr[] values) => new FunctionExpr("in", [this, .. values]);
    /// <summary><paramref name="low"/> &lt;= x &lt;= <paramref name="high"/>.</summary>
    public Expr Between(Expr low, Expr high) => (this >= low) & (this <= high);
    /// <summary>|x|</summary>
    public Expr Abs() => new FunctionExpr("abs", [this]);
    /// <summary>√x</summary>
    public Expr Sqrt() => new FunctionExpr("sqrt", [this]);
    /// <summary>ln x</summary>
    public Expr Log() => new FunctionExpr("log", [this]);
    /// <summary>e^x</summary>
    public Expr Exp() => new FunctionExpr("exp", [this]);
    /// <summary>x^p</summary>
    public Expr Pow(Expr p) => new FunctionExpr("pow", [this, p]);
    /// <summary>Rounded to <paramref name="digits"/> decimals (half away from zero, like SQL).</summary>
    public Expr Round(int digits = 0) => new FunctionExpr("round", [this, digits]);
    /// <summary>⌊x⌋</summary>
    public Expr Floor() => new FunctionExpr("floor", [this]);
    /// <summary>⌈x⌉</summary>
    public Expr Ceiling() => new FunctionExpr("ceil", [this]);
    /// <summary>Lower-cased text.</summary>
    public Expr Lower() => new FunctionExpr("lower", [this]);
    /// <summary>Upper-cased text.</summary>
    public Expr Upper() => new FunctionExpr("upper", [this]);
    /// <summary>Text with surrounding white space removed.</summary>
    public Expr Trim() => new FunctionExpr("trim", [this]);
    /// <summary>Text length.</summary>
    public Expr Length() => new FunctionExpr("length", [this]);
    /// <summary>Whether text contains <paramref name="s"/> (ordinal).</summary>
    public Expr Contains(string s) => new FunctionExpr("contains", [this, s]);
    /// <summary>Whether text starts with <paramref name="s"/> (ordinal).</summary>
    public Expr StartsWith(string s) => new FunctionExpr("startswith", [this, s]);
    /// <summary>Whether text ends with <paramref name="s"/> (ordinal).</summary>
    public Expr EndsWith(string s) => new FunctionExpr("endswith", [this, s]);
    /// <summary>SQL <c>LIKE</c>: <c>%</c> is any run, <c>_</c> one character.</summary>
    public Expr Like(string pattern) => new FunctionExpr("like", [this, pattern]);
    /// <summary>The year of a timestamp.</summary>
    public Expr Year() => new FunctionExpr("year", [this]);
    /// <summary>The month (1-12) of a timestamp.</summary>
    public Expr Month() => new FunctionExpr("month", [this]);
    /// <summary>The day of month of a timestamp.</summary>
    public Expr Day() => new FunctionExpr("day", [this]);
    /// <summary>The value as a number (text is parsed, booleans are 1/0).</summary>
    public Expr ToNumber() => new FunctionExpr("tonumber", [this]);
    /// <summary>The value as text.</summary>
    public Expr ToText() => new FunctionExpr("totext", [this]);

    /// <summary>The same expression producing a column called <paramref name="name"/>.</summary>
    public Expr As(string name) => new AliasExpr(this is AliasExpr a ? a.Inner : this, name);

    /// <summary>The column name this expression produces by default.</summary>
    public abstract string OutputName { get; }

    /// <summary>The columns this expression reads.</summary>
    public IReadOnlySet<string> Columns()
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        CollectColumns(set);
        return set;
    }

    internal abstract void CollectColumns(HashSet<string> columns);

    internal abstract Value Evaluate(DataFrame frame);

    /// <summary>Evaluates over a partition into a named column.</summary>
    public Series EvaluateSeries(DataFrame frame, string? name = null) => Evaluate(frame).ToSeries(name ?? OutputName, frame.RowCount);

    /// <summary>Evaluates a predicate over a partition: true rows only.</summary>
    public bool[] EvaluateMask(DataFrame frame)
    {
        var v = Evaluate(frame).Broadcast(frame.RowCount);
        if (v.Kind != DataType.Boolean) throw new InvalidOperationException($"Filter expression {this} is not a boolean.");
        var mask = new bool[frame.RowCount];
        for (var i = 0; i < mask.Length; i++) mask[i] = v.Bools![i] == true;
        return mask;
    }

    /// <summary>A copy with column references renamed through <paramref name="map"/>.</summary>
    internal abstract Expr RenameColumns(Func<string, string> map);

    internal abstract void Write(BinaryWriter writer);

    internal static Expr Read(BinaryReader reader) => reader.ReadByte() switch
    {
        1 => new ColumnExpr(reader.ReadString()),
        2 => new LiteralExpr(Value.ReadScalar(reader)),
        3 => new BinaryExpr((BinaryKind)reader.ReadByte(), Read(reader), Read(reader)),
        4 => ReadFunction(reader),
        5 => new AliasExpr(Read(reader), reader.ReadString()),
        var tag => throw new InvalidDataException($"Unknown expression tag {tag}."),
    };

    private static FunctionExpr ReadFunction(BinaryReader reader)
    {
        var name = reader.ReadString();
        var args = new Expr[reader.ReadInt32()];
        for (var i = 0; i < args.Length; i++) args[i] = Read(reader);
        return new FunctionExpr(name, args);
    }

    /// <summary>The expression as bytes, for shipping to workers.</summary>
    public byte[] ToBytes()
    {
        using var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true)) Write(w);
        return ms.ToArray();
    }

    /// <summary>An expression from <see cref="ToBytes"/>.</summary>
    public static Expr FromBytes(byte[] bytes)
    {
        using var r = new BinaryReader(new MemoryStream(bytes), Encoding.UTF8);
        return Read(r);
    }

    /// <summary>The conjuncts of an <c>AND</c> chain.</summary>
    internal static IEnumerable<Expr> SplitAnd(Expr e)
        => e is BinaryExpr { Kind: BinaryKind.And } b ? SplitAnd(b.Left).Concat(SplitAnd(b.Right)) : [e];

    /// <summary>Joins conjuncts back with <c>AND</c>.</summary>
    internal static Expr? JoinAnd(IEnumerable<Expr> parts)
    {
        Expr? result = null;
        foreach (var p in parts) result = result is null ? p : result & p;
        return result;
    }
}

internal enum BinaryKind : byte
{
    Add, Subtract, Multiply, Divide, Modulo,
    Equal, NotEqual, Greater, Less, GreaterOrEqual, LessOrEqual,
    And, Or,
}

internal sealed class ColumnExpr(string name) : Expr
{
    public string Name { get; } = name;
    public override string OutputName => Name;
    internal override void CollectColumns(HashSet<string> columns) => columns.Add(Name);
    internal override Value Evaluate(DataFrame frame)
        => frame.HasColumn(Name) ? Value.FromSeries(frame[Name])
            : throw new KeyNotFoundException($"Column '{Name}' not found; available: {string.Join(", ", frame.ColumnNames)}.");
    internal override Expr RenameColumns(Func<string, string> map) => new ColumnExpr(map(Name));
    internal override void Write(BinaryWriter writer) { writer.Write((byte)1); writer.Write(Name); }
    public override string ToString() => Name.All(c => char.IsLetterOrDigit(c) || c == '_') ? Name : $"\"{Name}\"";
}

internal sealed class LiteralExpr(Value value) : Expr
{
    public Value Value { get; } = value;
    public override string OutputName => "literal";
    internal override void CollectColumns(HashSet<string> columns) { }
    internal override Value Evaluate(DataFrame frame) => Value;
    internal override Expr RenameColumns(Func<string, string> map) => this;
    internal override void Write(BinaryWriter writer) { writer.Write((byte)2); Value.WriteScalar(writer); }
    public override string ToString() => Value.ScalarToString();
}

internal sealed class AliasExpr(Expr inner, string name) : Expr
{
    public Expr Inner { get; } = inner;
    public string Name { get; } = name;
    public override string OutputName => Name;
    internal override void CollectColumns(HashSet<string> columns) => Inner.CollectColumns(columns);
    internal override Value Evaluate(DataFrame frame) => Inner.Evaluate(frame);
    internal override Expr RenameColumns(Func<string, string> map) => new AliasExpr(Inner.RenameColumns(map), Name);
    internal override void Write(BinaryWriter writer) { writer.Write((byte)5); Inner.Write(writer); writer.Write(Name); }
    public override string ToString() => $"{Inner} AS {Name}";
}

internal sealed class BinaryExpr(BinaryKind kind, Expr left, Expr right) : Expr
{
    public BinaryKind Kind { get; } = kind;
    public Expr Left { get; } = left;
    public Expr Right { get; } = right;

    public override string OutputName => ToString();

    internal override void CollectColumns(HashSet<string> columns)
    {
        Left.CollectColumns(columns);
        Right.CollectColumns(columns);
    }

    internal override Expr RenameColumns(Func<string, string> map) => new BinaryExpr(Kind, Left.RenameColumns(map), Right.RenameColumns(map));

    internal override void Write(BinaryWriter writer)
    {
        writer.Write((byte)3);
        writer.Write((byte)Kind);
        Left.Write(writer);
        Right.Write(writer);
    }

    public override string ToString() => $"({Left} {Symbol} {Right})";

    private string Symbol => Kind switch
    {
        BinaryKind.Add => "+", BinaryKind.Subtract => "-", BinaryKind.Multiply => "*", BinaryKind.Divide => "/",
        BinaryKind.Modulo => "%", BinaryKind.Equal => "=", BinaryKind.NotEqual => "<>", BinaryKind.Greater => ">",
        BinaryKind.Less => "<", BinaryKind.GreaterOrEqual => ">=", BinaryKind.LessOrEqual => "<=",
        BinaryKind.And => "AND", _ => "OR",
    };

    internal override Value Evaluate(DataFrame frame)
    {
        var a = Left.Evaluate(frame);
        var b = Right.Evaluate(frame);
        var n = Math.Max(a.Length, b.Length);
        if (a.IsScalar && b.IsScalar) n = 1;
        else { a = a.Broadcast(frame.RowCount); b = b.Broadcast(frame.RowCount); n = frame.RowCount; }

        switch (Kind)
        {
            case BinaryKind.And or BinaryKind.Or:
            {
                var x = a.AsBool(); var y = b.AsBool();
                var r = new bool?[n];
                for (var i = 0; i < n; i++)
                    r[i] = Kind == BinaryKind.And
                        ? (x[i] == false || y[i] == false ? false : x[i] is null || y[i] is null ? null : true)
                        : (x[i] == true || y[i] == true ? true : x[i] is null || y[i] is null ? null : false);
                return Value.Bool(r, a.IsScalar && b.IsScalar);
            }
            case BinaryKind.Add when a.Kind == DataType.Text || b.Kind == DataType.Text:
            {
                var x = a.AsText(); var y = b.AsText();
                var r = new string?[n];
                for (var i = 0; i < n; i++) r[i] = x[i] is null || y[i] is null ? null : x[i] + y[i];
                return Value.Text(r, a.IsScalar && b.IsScalar);
            }
            case BinaryKind.Add or BinaryKind.Subtract or BinaryKind.Multiply or BinaryKind.Divide or BinaryKind.Modulo:
            {
                var x = a.AsNumbers(); var y = b.AsNumbers();
                var r = new double[n];
                switch (Kind)
                {
                    case BinaryKind.Add: for (var i = 0; i < n; i++) r[i] = x[i] + y[i]; break;
                    case BinaryKind.Subtract: for (var i = 0; i < n; i++) r[i] = x[i] - y[i]; break;
                    case BinaryKind.Multiply: for (var i = 0; i < n; i++) r[i] = x[i] * y[i]; break;
                    case BinaryKind.Divide: for (var i = 0; i < n; i++) r[i] = x[i] / y[i]; break;
                    default: for (var i = 0; i < n; i++) r[i] = x[i] % y[i]; break;
                }
                return Value.Numbers(r, a.IsScalar && b.IsScalar);
            }
            default:
                return Compare(a, b, n);
        }
    }

    private Value Compare(Value a, Value b, int n)
    {
        var r = new bool?[n];
        var kind = a.Kind == DataType.Numeric || b.Kind == DataType.Numeric ? DataType.Numeric
            : a.Kind == DataType.DateTime || b.Kind == DataType.DateTime ? DataType.DateTime
            : a.Kind == DataType.Boolean && b.Kind == DataType.Boolean ? DataType.Boolean
            : DataType.Text;

        // A missing literal has no type of its own; compare it as whatever the other side is.
        if (kind == DataType.Numeric && (a.IsTypedText || b.IsTypedText))
            throw new InvalidOperationException($"Cannot compare a number with text in {this}.");

        for (var i = 0; i < n; i++)
        {
            int? c = kind switch
            {
                DataType.Numeric => CompareNumbers(a.NumberAt(i), b.NumberAt(i)),
                DataType.DateTime => a.DateAt(i) is { } da && b.DateAt(i) is { } db ? da.CompareTo(db) : null,
                DataType.Boolean => a.BoolAt(i) is { } ba && b.BoolAt(i) is { } bb ? ba.CompareTo(bb) : null,
                _ => a.TextAt(i) is { } ta && b.TextAt(i) is { } tb ? string.CompareOrdinal(ta, tb) : null,
            };
            r[i] = c is null ? null : Kind switch
            {
                BinaryKind.Equal => c == 0,
                BinaryKind.NotEqual => c != 0,
                BinaryKind.Greater => c > 0,
                BinaryKind.Less => c < 0,
                BinaryKind.GreaterOrEqual => c >= 0,
                _ => c <= 0,
            };
        }
        return Value.Bool(r, a.IsScalar && b.IsScalar);
    }

    private static int? CompareNumbers(double x, double y) => double.IsNaN(x) || double.IsNaN(y) ? null : x.CompareTo(y);
}

internal sealed class FunctionExpr(string name, Expr[] args) : Expr
{
    public string Name { get; } = name;
    public Expr[] Args { get; } = args;

    public override string OutputName => Args.Length > 0 && Name is not ("if" or "coalesce" or "in") ? $"{Name}({Args[0].OutputName})" : ToString();

    internal override void CollectColumns(HashSet<string> columns)
    {
        foreach (var a in Args) a.CollectColumns(columns);
    }

    internal override Expr RenameColumns(Func<string, string> map) => new FunctionExpr(Name, Args.Select(a => a.RenameColumns(map)).ToArray());

    internal override void Write(BinaryWriter writer)
    {
        writer.Write((byte)4);
        writer.Write(Name);
        writer.Write(Args.Length);
        foreach (var a in Args) a.Write(writer);
    }

    public override string ToString() => Name switch
    {
        "not" => $"NOT {Args[0]}",
        "negate" => $"-{Args[0]}",
        "isnull" => $"{Args[0]} IS NULL",
        "notnull" => $"{Args[0]} IS NOT NULL",
        "in" => $"{Args[0]} IN ({string.Join(", ", Args.Skip(1))})",
        "if" => $"CASE WHEN {Args[0]} THEN {Args[1]} ELSE {Args[2]} END",
        "like" => $"{Args[0]} LIKE {Args[1]}",
        _ => $"{Name}({string.Join(", ", Args.Select(a => a.ToString()))})",
    };

    internal override Value Evaluate(DataFrame frame)
    {
        var values = Args.Select(a => a.Evaluate(frame)).ToArray();
        var scalar = values.All(v => v.IsScalar);
        var n = scalar ? 1 : frame.RowCount;
        if (!scalar) values = values.Select(v => v.Broadcast(n)).ToArray();

        switch (Name)
        {
            case "not": return Value.Bool(values[0].AsBool().Select(b => b is null ? (bool?)null : !b.Value).ToArray(), scalar);
            case "isnull": return Value.Bool(Enumerable.Range(0, n).Select(i => (bool?)values[0].IsMissingAt(i)).ToArray(), scalar);
            case "notnull": return Value.Bool(Enumerable.Range(0, n).Select(i => (bool?)!values[0].IsMissingAt(i)).ToArray(), scalar);
            case "negate": return MapNumbers(values[0], x => -x, scalar);
            case "abs": return MapNumbers(values[0], Math.Abs, scalar);
            case "sqrt": return MapNumbers(values[0], Math.Sqrt, scalar);
            case "log": return MapNumbers(values[0], Math.Log, scalar);
            case "exp": return MapNumbers(values[0], Math.Exp, scalar);
            case "floor": return MapNumbers(values[0], Math.Floor, scalar);
            case "ceil": return MapNumbers(values[0], Math.Ceiling, scalar);
            case "pow":
            {
                var x = values[0].AsNumbers(); var p = values[1].AsNumbers();
                return Value.Numbers(Enumerable.Range(0, n).Select(i => Math.Pow(x[i], p[i])).ToArray(), scalar);
            }
            case "round":
            {
                var digits = (int)values[1].NumberAt(0);
                return MapNumbers(values[0], x => Math.Round(x, digits, MidpointRounding.AwayFromZero), scalar);
            }
            case "lower": return MapText(values[0], s => s.ToLowerInvariant(), scalar);
            case "upper": return MapText(values[0], s => s.ToUpperInvariant(), scalar);
            case "trim": return MapText(values[0], s => s.Trim(), scalar);
            case "length":
                return Value.Numbers(values[0].AsText().Select(s => s is null ? double.NaN : s.Length).ToArray(), scalar);
            case "contains": return TextPredicate(values[0], values[1].TextAt(0)!, (s, p) => s.Contains(p, StringComparison.Ordinal), scalar);
            case "startswith": return TextPredicate(values[0], values[1].TextAt(0)!, (s, p) => s.StartsWith(p, StringComparison.Ordinal), scalar);
            case "endswith": return TextPredicate(values[0], values[1].TextAt(0)!, (s, p) => s.EndsWith(p, StringComparison.Ordinal), scalar);
            case "like":
            {
                var regex = LikeToRegex(values[1].TextAt(0)!);
                return TextPredicate(values[0], "", (s, _) => regex.IsMatch(s), scalar);
            }
            case "year": return DatePart(values[0], d => d.Year, scalar);
            case "month": return DatePart(values[0], d => d.Month, scalar);
            case "day": return DatePart(values[0], d => d.Day, scalar);
            case "tonumber": return Value.Numbers(values[0].AsNumbers(parseText: true), scalar);
            case "totext": return Value.Text(values[0].AsText(), scalar);
            case "if":
            {
                var cond = values[0].AsBool();
                var kind = Value.CommonKind(values[1], values[2]);
                return Value.Select(kind, n, i => cond[i] == true ? values[1] : values[2], scalar);
            }
            case "coalesce":
            {
                var kind = Value.CommonKind(values);
                return Value.Select(kind, n, i => values.FirstOrDefault(v => !v.IsMissingAt(i)) ?? values[0], scalar);
            }
            case "in":
            {
                var x = values[0];
                var r = new bool?[n];
                for (var i = 0; i < n; i++)
                {
                    if (x.IsMissingAt(i)) { r[i] = null; continue; }
                    var found = false;
                    for (var j = 1; j < values.Length && !found; j++)
                        found = !values[j].IsMissingAt(i) && Value.EqualAt(x, values[j], i);
                    r[i] = found;
                }
                return Value.Bool(r, scalar);
            }
            default: throw new NotSupportedException($"Unknown function '{Name}'.");
        }
    }

    private static Value MapNumbers(Value v, Func<double, double> f, bool scalar)
    {
        var x = v.AsNumbers();
        var r = new double[x.Length];
        for (var i = 0; i < r.Length; i++) r[i] = f(x[i]);
        return Value.Numbers(r, scalar);
    }

    private static Value MapText(Value v, Func<string, string> f, bool scalar)
        => Value.Text(v.AsText().Select(s => s is null ? null : f(s)).ToArray(), scalar);

    private static Value TextPredicate(Value v, string arg, Func<string, string, bool> f, bool scalar)
        => Value.Bool(v.AsText().Select(s => s is null ? (bool?)null : f(s, arg)).ToArray(), scalar);

    private static Value DatePart(Value v, Func<DateTime, int> f, bool scalar)
    {
        if (v.Kind != DataType.DateTime) throw new InvalidOperationException("Date parts need a timestamp column.");
        return Value.Numbers(v.Dates!.Select(d => d is null ? double.NaN : f(d.Value)).ToArray(), scalar);
    }

    private static System.Text.RegularExpressions.Regex LikeToRegex(string pattern)
    {
        var sb = new StringBuilder("^");
        foreach (var c in pattern)
            sb.Append(c switch { '%' => ".*", '_' => ".", _ => System.Text.RegularExpressions.Regex.Escape(c.ToString()) });
        sb.Append('$');
        return new System.Text.RegularExpressions.Regex(sb.ToString(), System.Text.RegularExpressions.RegexOptions.Singleline);
    }
}

/// <summary>
/// An evaluated column or scalar: one typed array, so arithmetic loops over <c>double[]</c>
/// rather than boxed objects.
/// </summary>
internal sealed class Value
{
    public DataType Kind { get; private init; }
    public bool IsScalar { get; private init; }
    public double[]? Nums { get; private init; }
    public string?[]? Texts { get; private init; }
    public bool?[]? Bools { get; private init; }
    public DateTime?[]? Dates { get; private init; }

    /// <summary>True for a literal typed as text by its value (not a missing literal).</summary>
    public bool IsTypedText => Kind == DataType.Text && !(IsScalar && Texts![0] is null);

    public int Length => Kind switch
    {
        DataType.Numeric => Nums!.Length,
        DataType.Boolean => Bools!.Length,
        DataType.DateTime => Dates!.Length,
        _ => Texts!.Length,
    };

    public static Value Missing => new() { Kind = DataType.Text, IsScalar = true, Texts = [null] };
    public static Value Of(double v) => Numbers([v], true);
    public static Value Of(string? v) => Text([v], true);
    public static Value Of(bool v) => Bool([v], true);
    public static Value Of(DateTime v) => new() { Kind = DataType.DateTime, IsScalar = true, Dates = [v] };
    public static Value Numbers(double[] v, bool scalar = false) => new() { Kind = DataType.Numeric, IsScalar = scalar, Nums = v };
    public static Value Text(string?[] v, bool scalar = false) => new() { Kind = DataType.Text, IsScalar = scalar, Texts = v };
    public static Value Bool(bool?[] v, bool scalar = false) => new() { Kind = DataType.Boolean, IsScalar = scalar, Bools = v };
    public static Value DatesOf(DateTime?[] v, bool scalar = false) => new() { Kind = DataType.DateTime, IsScalar = scalar, Dates = v };

    public static Value FromSeries(Series s) => s switch
    {
        NumericSeries n => Numbers(n.Values),
        TextSeries t => Text(t.Values.ToArray()),
        BooleanSeries b => Bool(Enumerable.Range(0, b.Length).Select(i => b[i]).ToArray()),
        DateTimeSeries d => DatesOf(Enumerable.Range(0, d.Length).Select(i => d[i]).ToArray()),
        _ => Text(Enumerable.Range(0, s.Length).Select(i => s.GetValue(i)?.ToString()).ToArray()),
    };

    public Value Broadcast(int n)
    {
        if (!IsScalar) return this;
        return Kind switch
        {
            DataType.Numeric => Numbers(Enumerable.Repeat(Nums![0], n).ToArray()),
            DataType.Boolean => Bool(Enumerable.Repeat(Bools![0], n).ToArray()),
            DataType.DateTime => DatesOf(Enumerable.Repeat(Dates![0], n).ToArray()),
            _ => Text(Enumerable.Repeat(Texts![0], n).ToArray()),
        };
    }

    public Series ToSeries(string name, int rows)
    {
        var v = Broadcast(rows);
        return v.Kind switch
        {
            DataType.Numeric => new NumericSeries(name, v.Nums!),
            DataType.Boolean => new BooleanSeries(name, v.Bools!),
            DataType.DateTime => new DateTimeSeries(name, v.Dates!),
            _ => new TextSeries(name, v.Texts!),
        };
    }

    public bool IsMissingAt(int i) => Kind switch
    {
        DataType.Numeric => double.IsNaN(Nums![Index(i)]),
        DataType.Boolean => Bools![Index(i)] is null,
        DataType.DateTime => Dates![Index(i)] is null,
        _ => Texts![Index(i)] is null,
    };

    private int Index(int i) => IsScalar ? 0 : i;

    public double NumberAt(int i) => Kind switch
    {
        DataType.Numeric => Nums![Index(i)],
        DataType.Boolean => Bools![Index(i)] is { } b ? (b ? 1 : 0) : double.NaN,
        DataType.Text when Texts![Index(i)] is null => double.NaN,
        _ => throw new InvalidOperationException($"Expected a number, got {Kind}."),
    };

    public string? TextAt(int i) => Kind switch
    {
        DataType.Text => Texts![Index(i)],
        DataType.Numeric => double.IsNaN(Nums![Index(i)]) ? null : Nums[Index(i)].ToString("R", CultureInfo.InvariantCulture),
        DataType.Boolean => Bools![Index(i)]?.ToString().ToLowerInvariant(),
        _ => Dates![Index(i)]?.ToString("O", CultureInfo.InvariantCulture),
    };

    public bool? BoolAt(int i) => Kind switch
    {
        DataType.Boolean => Bools![Index(i)],
        DataType.Text when Texts![Index(i)] is null => null,
        DataType.Numeric => double.IsNaN(Nums![Index(i)]) ? null : Nums[Index(i)] != 0,
        _ => throw new InvalidOperationException($"Expected a boolean, got {Kind}."),
    };

    public DateTime? DateAt(int i) => Kind switch
    {
        DataType.DateTime => Dates![Index(i)],
        DataType.Text => Texts![Index(i)] is { } s
            ? DateTime.Parse(s, CultureInfo.InvariantCulture, DateTimeStyles.None) : null,
        _ => throw new InvalidOperationException($"Expected a timestamp, got {Kind}."),
    };

    public double[] AsNumbers(bool parseText = false)
    {
        if (Kind == DataType.Numeric) return Nums!;
        var n = Length;
        var r = new double[n];
        for (var i = 0; i < n; i++)
            r[i] = Kind == DataType.Text && parseText
                ? double.TryParse(Texts![i], NumberStyles.Any, CultureInfo.InvariantCulture, out var d) ? d : double.NaN
                : NumberAt(i);
        return r;
    }

    public string?[] AsText()
    {
        if (Kind == DataType.Text) return Texts!;
        return Enumerable.Range(0, Length).Select(TextAt).ToArray();
    }

    public bool?[] AsBool()
    {
        if (Kind == DataType.Boolean) return Bools!;
        return Enumerable.Range(0, Length).Select(BoolAt).ToArray();
    }

    public static DataType CommonKind(params Value[] values)
    {
        var typed = values.Where(v => !(v.IsScalar && v.IsMissingAt(0))).Select(v => v.Kind).Distinct().ToArray();
        if (typed.Length == 0) return DataType.Text;
        if (typed.Length == 1) return typed[0];
        if (typed.All(k => k is DataType.Numeric or DataType.Boolean)) return DataType.Numeric;
        return DataType.Text;
    }

    public static Value Select(DataType kind, int n, Func<int, Value> pick, bool scalar) => kind switch
    {
        DataType.Numeric => Numbers(Enumerable.Range(0, n).Select(i => pick(i).NumberAt(i)).ToArray(), scalar),
        DataType.Boolean => Bool(Enumerable.Range(0, n).Select(i => pick(i).BoolAt(i)).ToArray(), scalar),
        DataType.DateTime => DatesOf(Enumerable.Range(0, n).Select(i => pick(i).DateAt(i)).ToArray(), scalar),
        _ => Text(Enumerable.Range(0, n).Select(i => pick(i).TextAt(i)).ToArray(), scalar),
    };

    public static bool EqualAt(Value a, Value b, int i)
    {
        if (a.Kind == DataType.Numeric || b.Kind == DataType.Numeric) return a.NumberAt(i) == b.NumberAt(i);
        if (a.Kind == DataType.DateTime || b.Kind == DataType.DateTime) return a.DateAt(i) == b.DateAt(i);
        if (a.Kind == DataType.Boolean && b.Kind == DataType.Boolean) return a.BoolAt(i) == b.BoolAt(i);
        return string.Equals(a.TextAt(i), b.TextAt(i), StringComparison.Ordinal);
    }

    public void WriteScalar(BinaryWriter w)
    {
        w.Write((byte)Kind);
        switch (Kind)
        {
            case DataType.Numeric: w.Write(Nums![0]); break;
            case DataType.Boolean: w.Write(Bools![0] is null ? (byte)2 : Bools[0]!.Value ? (byte)1 : (byte)0); break;
            case DataType.DateTime: w.Write(Dates![0]?.Ticks ?? -1); break;
            default: w.Write(Texts![0] is not null); if (Texts[0] is { } s) w.Write(s); break;
        }
    }

    public static Value ReadScalar(BinaryReader r)
    {
        var kind = (DataType)r.ReadByte();
        switch (kind)
        {
            case DataType.Numeric: return Of(r.ReadDouble());
            case DataType.Boolean:
            {
                var b = r.ReadByte();
                return Bool([b == 2 ? null : b == 1], true);
            }
            case DataType.DateTime:
            {
                var t = r.ReadInt64();
                return DatesOf([t < 0 ? null : new DateTime(t)], true);
            }
            default: return r.ReadBoolean() ? Of(r.ReadString()) : Missing;
        }
    }

    public string ScalarToString() => Kind switch
    {
        DataType.Numeric => Nums![0].ToString("R", CultureInfo.InvariantCulture),
        DataType.Boolean => Bools![0] is { } b ? (b ? "TRUE" : "FALSE") : "NULL",
        DataType.DateTime => Dates![0] is { } d ? $"TIMESTAMP '{d:O}'" : "NULL",
        _ => Texts![0] is { } s ? $"'{s.Replace("'", "''")}'" : "NULL",
    };
}
