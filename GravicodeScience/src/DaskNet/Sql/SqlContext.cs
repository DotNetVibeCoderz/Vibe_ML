using Gravicode.Science.GraviFrame;
using Gravicode.Science.GraviFrame.Io;

namespace Gravicode.Science.DaskNet.Sql;

/// <summary>
/// SQL over registered <see cref="DaskFrame"/>s (dask-sql's <c>Context</c>): a query compiles to
/// an ordinary lazy frame, so it gets the same optimiser, the same partitioned execution, and
/// runs on any scheduler, a cluster included.
/// </summary>
/// <remarks>
/// <para>
/// Supported: <c>SELECT [DISTINCT] ... FROM t [alias] [INNER|LEFT|RIGHT|FULL JOIN u ON ...]
/// [WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ... [ASC|DESC]] [LIMIT n]</c>, subqueries in
/// <c>FROM</c>, <c>CASE</c>, <c>CAST</c>, <c>IN</c>, <c>BETWEEN</c>, <c>LIKE</c>, <c>IS [NOT] NULL</c>,
/// the aggregates COUNT/SUM/AVG/MIN/MAX/STDDEV/VARIANCE (and their <c>_POP</c> forms), and the
/// scalar functions listed in <see cref="ScalarFunctions"/>. Missing values follow SQL's
/// three-valued logic, as <see cref="Expr"/> does.
/// </para>
/// <para>
/// Not supported, and refused with a message rather than approximated: <c>OFFSET</c>,
/// <c>UNION</c>, <c>CROSS JOIN</c>, <c>COUNT(DISTINCT ...)</c>, subqueries outside <c>FROM</c>,
/// window functions, and a join condition with no equality between the two sides.
/// </para>
/// </remarks>
public sealed class SqlContext
{
    private readonly Dictionary<string, DaskFrame> _tables = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>The names of registered tables.</summary>
    public IReadOnlyCollection<string> Tables => _tables.Keys;

    /// <summary>Scalar functions a query may call.</summary>
    public static IReadOnlyList<string> ScalarFunctions { get; } =
    [
        "ABS", "SQRT", "LN", "LOG", "EXP", "POWER", "POW", "ROUND", "FLOOR", "CEIL", "CEILING", "LOWER", "UPPER",
        "TRIM", "LENGTH", "LEN", "COALESCE", "IFNULL", "NULLIF", "YEAR", "MONTH", "DAY", "CONCAT",
    ];

    /// <summary>Registers (or replaces) a table.</summary>
    public SqlContext Register(string name, DaskFrame frame)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (frame.Plan.Schema is null)
            throw new ArgumentException("SQL needs to know a table's columns; give FromDelayed its column list.", nameof(frame));
        _tables[name] = frame;
        return this;
    }

    /// <summary>Registers an in-memory frame split into <paramref name="partitions"/> partitions.</summary>
    public SqlContext Register(string name, DataFrame frame, int partitions = 1)
        => Register(name, DaskFrame.FromFrame(frame, partitions));

    /// <summary>Registers CSV files (a path or glob) as a table.</summary>
    public SqlContext RegisterCsv(string name, string pattern, long blockSize = 64L << 20, CsvOptions? options = null)
        => Register(name, DaskFrame.ReadCsv(pattern, blockSize, options));

    /// <summary>Removes a table; false if there was none.</summary>
    public bool Unregister(string name) => _tables.Remove(name);

    /// <summary>Compiles a query to a lazy frame. Nothing is read until it is computed.</summary>
    public DaskFrame Sql(string query) => new SqlCompiler(_tables).Compile(SqlParser.Parse(query));

    /// <summary>Runs a query and returns the result.</summary>
    public DataFrame Execute(string query, IScheduler? scheduler = null) => Sql(query).Compute(scheduler);

    /// <summary>The optimised plan a query compiles to.</summary>
    public string Explain(string query) => Sql(query).Explain();
}

/// <summary>Column names visible at one point of a query: (table alias, column, name in the frame).</summary>
internal sealed class SqlScope
{
    public readonly List<(string Table, string Column, string Internal)> Entries = [];

    public string Resolve(ColumnRef c)
    {
        var matches = Entries.Where(e => (c.Table is null || e.Table.Equals(c.Table, StringComparison.OrdinalIgnoreCase))
                                         && e.Column == c.Name).ToList();
        if (matches.Count == 0)
            matches = Entries.Where(e => (c.Table is null || e.Table.Equals(c.Table, StringComparison.OrdinalIgnoreCase))
                                         && e.Column.Equals(c.Name, StringComparison.OrdinalIgnoreCase)).ToList();
        var shown = c.Table is null ? c.Name : $"{c.Table}.{c.Name}";
        if (matches.Count == 1) return matches[0].Internal;
        if (matches.Count > 1) throw new SqlException($"Column '{shown}' is ambiguous; qualify it with a table alias", c.Position);
        if (c.Table is not null && !Entries.Any(e => e.Table.Equals(c.Table, StringComparison.OrdinalIgnoreCase)))
            throw new SqlException($"Unknown table alias '{c.Table}'", c.Position);
        throw new SqlException($"Unknown column '{shown}'", c.Position);
    }

    /// <summary>The name a column gets in output: bare when unique across tables, else qualified.</summary>
    public string OutputName((string Table, string Column, string Internal) e)
        => Entries.Count(x => x.Column == e.Column) > 1 ? $"{e.Table}.{e.Column}" : e.Column;
}

/// <summary>Turns a parsed statement into DaskFrame operations.</summary>
internal sealed class SqlCompiler(IReadOnlyDictionary<string, DaskFrame> tables)
{
    private static readonly Dictionary<string, string> AggregateNames = new(StringComparer.Ordinal)
    {
        ["COUNT"] = "count", ["SUM"] = "sum", ["AVG"] = "mean", ["MEAN"] = "mean", ["MIN"] = "min", ["MAX"] = "max",
        ["STDDEV"] = "std", ["STDDEV_SAMP"] = "std", ["STD"] = "std", ["STDDEV_POP"] = "std_pop",
        ["VARIANCE"] = "var", ["VAR"] = "var", ["VAR_SAMP"] = "var", ["VAR_POP"] = "var_pop",
    };

    private int _hidden;

    public DaskFrame Compile(SelectStatement s)
    {
        var (frame, scope) = CompileFrom(s);

        if (s.Where is not null)
        {
            if (ContainsAggregate(s.Where)) throw new SqlException("Aggregates are not allowed in WHERE; use HAVING", s.Where.Position);
            frame = frame.Filter(Plain(s.Where, scope));
        }

        var aggregated = s.GroupBy.Length > 0
                         || s.Items.Any(i => ContainsAggregate(i.Expr))
                         || (s.Having is not null)
                         || s.OrderBy.Any(o => ContainsAggregate(o.Expr));
        Func<SqlNode, Expr> compile = n => Plain(n, scope);
        if (aggregated) (frame, compile) = CompileAggregation(s, frame, scope);

        // Projection, with hidden columns for ORDER BY expressions that are not outputs.
        var outputs = new List<Expr>();
        var names = new List<string>();
        foreach (var item in s.Items)
        {
            if (item.Expr is StarRef star)
            {
                if (aggregated) throw new SqlException("SELECT * cannot be combined with GROUP BY or aggregates", star.Position);
                var entries = scope.Entries.Where(e => star.Table is null || e.Table.Equals(star.Table, StringComparison.OrdinalIgnoreCase)).ToList();
                if (entries.Count == 0) throw new SqlException($"Unknown table alias '{star.Table}'", star.Position);
                foreach (var e in entries)
                {
                    outputs.Add(Expr.Col(e.Internal));
                    names.Add(scope.OutputName(e));
                }
                continue;
            }
            outputs.Add(compile(item.Expr));
            names.Add(item.Alias ?? (item.Expr is ColumnRef c ? c.Name : item.Text));
        }
        // Two unaliased columns of the same name (a.id, b.id) keep their qualifiers.
        var clashing = names.GroupBy(n => n).Where(g => g.Count() > 1).Select(g => g.Key).ToHashSet();
        for (var i = 0; i < names.Count; i++)
        {
            if (!clashing.Contains(names[i])) continue;
            if (s.Items.ElementAtOrDefault(i) is { Alias: null, Expr: ColumnRef { Table: { } t } c } && s.Items.Length == names.Count)
                names[i] = $"{t}.{c.Name}";
        }
        var duplicate = names.GroupBy(n => n).FirstOrDefault(g => g.Count() > 1);
        if (duplicate is not null) throw new SqlException($"Output column '{duplicate.Key}' appears twice; give one an alias", -1);

        var sortKeys = new List<(string Column, bool Ascending)>();
        var hidden = new List<string>();
        foreach (var o in s.OrderBy)
        {
            if (o.Expr is Literal { Value: double d })
            {
                var k = (int)d;
                if (k != d || k < 1 || k > names.Count) throw new SqlException($"ORDER BY position {d} is out of range", o.Expr.Position);
                sortKeys.Add((names[k - 1], o.Ascending));
                continue;
            }
            if (o.Expr is ColumnRef { Table: null } r && names.Contains(r.Name))
            {
                sortKeys.Add((r.Name, o.Ascending));
                continue;
            }
            if (s.Distinct) throw new SqlException("With DISTINCT, ORDER BY must name an output column", o.Expr.Position);
            var value = compile(o.Expr);
            var index = outputs.FindIndex(x => x.ToString() == value.ToString());
            if (index >= 0) { sortKeys.Add((names[index], o.Ascending)); continue; }
            var name = $"__order{_hidden++}";
            hidden.Add(name);
            outputs.Add(value);
            names.Add(name);
            sortKeys.Add((name, o.Ascending));
        }

        frame = frame.Select(outputs.Zip(names, (e, n) => e.As(n)).ToArray());
        if (s.Distinct)
        {
            var count = $"__count{_hidden++}";
            frame = frame.GroupBy(names.ToArray()).Size(count).Drop(count);
        }
        if (sortKeys.Count > 0) frame = frame.SortBy(sortKeys);
        if (hidden.Count > 0) frame = frame.Drop(hidden.ToArray());
        if (s.Limit is { } limit) frame = frame.Limit(limit);
        return frame;
    }

    // ------------------------------------------------------------------ FROM and JOIN

    private (DaskFrame, SqlScope) CompileFrom(SelectStatement s)
    {
        var aliases = new[] { s.From }.Concat(s.Joins.Select(j => j.Table)).Select(t => t.Alias).ToList();
        var repeated = aliases.GroupBy(a => a, StringComparer.OrdinalIgnoreCase).FirstOrDefault(g => g.Count() > 1);
        if (repeated is not null) throw new SqlException($"Table alias '{repeated.Key}' is used twice", s.From.Position);

        var qualify = s.Joins.Length > 0;
        var (frame, scope) = Table(s.From, qualify);
        foreach (var join in s.Joins)
        {
            var (right, rightScope) = Table(join.Table, qualify: true);
            var leftKeys = new List<string>();
            var rightKeys = new List<string>();
            var residual = new List<SqlNode>();
            foreach (var condition in Conjuncts(join.On))
            {
                if (condition is BinaryOp { Op: "=", Left: ColumnRef a, Right: ColumnRef b })
                {
                    var (l, r) = (TryResolve(a, scope), TryResolve(b, rightScope));
                    if (l is null || r is null) (l, r) = (TryResolve(b, scope), TryResolve(a, rightScope));
                    if (l is not null && r is not null)
                    {
                        leftKeys.Add(l);
                        rightKeys.Add(r);
                        continue;
                    }
                }
                residual.Add(condition);
            }
            if (leftKeys.Count == 0)
                throw new SqlException("A JOIN needs at least one equality between a column of each side", join.On.Position);
            if (residual.Count > 0 && join.Kind != JoinKind.Inner)
                throw new SqlException("An outer join's ON may only hold equalities between the two sides", residual[0].Position);

            frame = Join(frame, right, leftKeys, rightKeys, join.Kind);
            scope.Entries.AddRange(rightScope.Entries);
            foreach (var r in residual) frame = frame.Filter(Plain(r, scope));
        }
        return (frame, scope);
    }

    private static DaskFrame Join(DaskFrame left, DaskFrame right, List<string> leftKeys, List<string> rightKeys, JoinKind kind)
    {
        if (leftKeys.Count == 1) return left.Merge(right, [leftKeys[0]], [rightKeys[0]], kind);

        // Several keys must share names. Join on copies, so the real columns of both sides stay
        // (b.k is missing where a left join found no match, as SQL says), then drop the copies.
        var keys = leftKeys.Select((_, i) => $"__key{i}").ToArray();
        left = left.WithColumns(leftKeys.Select((k, i) => Expr.Col(k).As(keys[i])).ToArray());
        right = right.WithColumns(rightKeys.Select((k, i) => Expr.Col(k).As(keys[i])).ToArray());
        var joined = left.Merge(right, keys, keys, kind);
        return joined.Drop(joined.Columns.Where(c => c.StartsWith("__key", StringComparison.Ordinal)).ToArray());
    }

    private static string? TryResolve(ColumnRef c, SqlScope scope)
    {
        try { return scope.Resolve(c); }
        catch (SqlException) { return null; }
    }

    private (DaskFrame, SqlScope) Table(TableRef table, bool qualify)
    {
        DaskFrame frame;
        if (table is NamedTable named)
        {
            if (!tables.TryGetValue(named.Name, out frame!))
                throw new SqlException($"Unknown table '{named.Name}'", named.Position);
        }
        else frame = Compile(((SubqueryTable)table).Query);

        var scope = new SqlScope();
        var columns = frame.Columns;
        if (qualify)
        {
            var pairs = columns.Select(c => (c, $"{table.Alias}.{c}")).ToArray();
            frame = new DaskFrame(new RenameNode(frame.Plan, pairs));
            scope.Entries.AddRange(pairs.Select(p => (table.Alias, p.c, p.Item2)));
        }
        else scope.Entries.AddRange(columns.Select(c => (table.Alias, c, c)));
        return (frame, scope);
    }

    private static IEnumerable<SqlNode> Conjuncts(SqlNode n)
        => n is BinaryOp { Op: "AND" } b ? Conjuncts(b.Left).Concat(Conjuncts(b.Right)) : [n];

    // ------------------------------------------------------------------ GROUP BY

    private (DaskFrame, Func<SqlNode, Expr>) CompileAggregation(SelectStatement s, DaskFrame frame, SqlScope scope)
    {
        var pre = new List<Expr>();
        var keyNames = new List<string>();
        var keyByText = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var g in s.GroupBy)
        {
            // GROUP BY may name an output alias or position, as most engines allow.
            var node = g;
            if (g is Literal { Value: double d } && d == Math.Floor(d) && d >= 1 && d <= s.Items.Length) node = s.Items[(int)d - 1].Expr;
            else if (g is ColumnRef { Table: null } r && TryResolve(r, scope) is null
                     && s.Items.FirstOrDefault(i => i.Alias == r.Name) is { } aliased) node = aliased.Expr;
            if (ContainsAggregate(node)) throw new SqlException("GROUP BY cannot contain an aggregate", g.Position);

            var value = Plain(node, scope);
            string name;
            if (value is ColumnExpr col) name = col.Name;
            else
            {
                name = $"__group{_hidden++}";
                pre.Add(value.As(name));
            }
            if (keyByText.TryAdd(value.ToString()!, name)) keyNames.Add(name);
        }

        var aggregations = new List<Aggregation>();
        var aggByText = new Dictionary<string, string>(StringComparer.Ordinal);
        void Collect(SqlNode n)
        {
            if (n is Call call && AggregateNames.TryGetValue(call.Name, out var how))
            {
                if (call.Distinct) throw new SqlException($"{call.Name}(DISTINCT ...) is not supported yet", call.Position);
                if (!call.Star && call.Args.Length != 1) throw new SqlException($"{call.Name} takes one argument", call.Position);
                if (call.Star && call.Name != "COUNT") throw new SqlException($"{call.Name}(*) is not valid", call.Position);
                if (call.Args.Any(ContainsAggregate)) throw new SqlException("Aggregates cannot be nested", call.Position);
                var key = AggregateKey(call, scope);
                if (aggByText.ContainsKey(key)) return;
                var output = $"__agg{_hidden++}";
                aggByText[key] = output;
                if (call.Star) { aggregations.Add(new Aggregation(output, "size", null)); return; }
                var arg = Plain(call.Args[0], scope);
                string column;
                if (arg is ColumnExpr c) column = c.Name;
                else
                {
                    column = $"__arg{_hidden++}";
                    pre.Add(arg.As(column));
                }
                aggregations.Add(new Aggregation(output, how, column));
                return;
            }
            foreach (var child in Children(n)) Collect(child);
        }
        foreach (var i in s.Items) Collect(i.Expr);
        if (s.Having is not null) Collect(s.Having);
        foreach (var o in s.OrderBy) Collect(o.Expr);
        if (aggregations.Count == 0) aggregations.Add(new Aggregation($"__agg{_hidden++}", "size", null));

        if (pre.Count > 0) frame = frame.WithColumns(pre.ToArray());
        frame = keyNames.Count > 0 ? frame.GroupBy(keyNames.ToArray()).Agg(aggregations.ToArray()) : frame.Aggregate(aggregations.ToArray());

        Expr Grouped(SqlNode n)
        {
            if (n is Call call && AggregateNames.ContainsKey(call.Name)) return Expr.Col(aggByText[AggregateKey(call, scope)]);
            if (!ContainsAggregate(n))
            {
                if (n is Literal) return Plain(n, scope);
                var value = Plain(n, scope);
                if (keyByText.TryGetValue(value.ToString()!, out var key)) return Expr.Col(key);
                if (n is ColumnRef c)
                    throw new SqlException($"Column '{(c.Table is null ? c.Name : $"{c.Table}.{c.Name}")}' must appear in GROUP BY or inside an aggregate", c.Position);
            }
            return Build(n, Grouped);
        }

        if (s.Having is not null) frame = frame.Filter(Grouped(s.Having));
        return (frame, Grouped);
    }

    private string AggregateKey(Call call, SqlScope scope)
        => $"{call.Name}({(call.Star ? "*" : Plain(call.Args[0], scope).ToString())})"
           .Replace("STDDEV_SAMP", "STDDEV").Replace("VAR_SAMP", "VARIANCE");

    private static bool ContainsAggregate(SqlNode n)
        => (n is Call c && AggregateNames.ContainsKey(c.Name)) || Children(n).Any(ContainsAggregate);

    private static IEnumerable<SqlNode> Children(SqlNode n) => n switch
    {
        BinaryOp b => [b.Left, b.Right],
        UnaryOp u => [u.Operand],
        Call c => c.Args,
        CaseWhen w => new[] { w.Operand, w.Else }.Where(x => x is not null).Cast<SqlNode>()
            .Concat(w.Branches.SelectMany(b => new[] { b.When, b.Then })),
        InList i => [i.Value, .. i.Items],
        BetweenOp b => [b.Value, b.Low, b.High],
        LikeOp l => [l.Value, l.Pattern],
        IsNullOp i => [i.Value],
        CastOp c => [c.Value],
        _ => [],
    };

    // ------------------------------------------------------------------ expressions

    private Expr Plain(SqlNode n, SqlScope scope)
    {
        if (n is ColumnRef c) return Expr.Col(scope.Resolve(c));
        if (n is Call call && AggregateNames.ContainsKey(call.Name))
            throw new SqlException($"Aggregate {call.Name} is not allowed here", call.Position);
        return Build(n, x => Plain(x, scope));
    }

    /// <summary>One node, with its children compiled by <paramref name="sub"/> - plain column
    /// lookups before grouping, key and aggregate lookups after it.</summary>
    private static Expr Build(SqlNode n, Func<SqlNode, Expr> sub)
    {
        switch (n)
        {
            case Literal l:
                return l.Value switch
                {
                    null => Expr.Null(),
                    double d => Expr.Lit(d),
                    string s => Expr.Lit(s),
                    bool b => Expr.Lit(b),
                    DateTime t => Expr.Lit(t),
                    _ => throw new SqlException("Unsupported literal", l.Position),
                };
            case BinaryOp b:
            {
                var (x, y) = (sub(b.Left), sub(b.Right));
                return b.Op switch
                {
                    "+" => x + y, "-" => x - y, "*" => x * y, "/" => x / y, "%" => x % y,
                    "||" => x.ToText() + y.ToText(),
                    "=" => x.Eq(y), "<>" => x.Ne(y), "<" => x < y, "<=" => x <= y, ">" => x > y, ">=" => x >= y,
                    "AND" => x & y, "OR" => x | y,
                    _ => throw new SqlException($"Unknown operator {b.Op}", b.Position),
                };
            }
            case UnaryOp u:
                return u.Op == "NOT" ? !sub(u.Operand) : -sub(u.Operand);
            case IsNullOp i:
                return i.Negated ? sub(i.Value).IsNotNull() : sub(i.Value).IsNull();
            case InList i:
            {
                var e = sub(i.Value).In(i.Items.Select(sub).ToArray());
                return i.Negated ? !e : e;
            }
            case BetweenOp b:
            {
                var e = sub(b.Value).Between(sub(b.Low), sub(b.High));
                return b.Negated ? !e : e;
            }
            case LikeOp l:
            {
                if (l.Pattern is not Literal { Value: string pattern }) throw new SqlException("LIKE takes a string literal pattern", l.Pattern.Position);
                var e = sub(l.Value).Like(pattern);
                return l.Negated ? !e : e;
            }
            case CaseWhen w:
            {
                var result = w.Else is null ? Expr.Null() : sub(w.Else);
                var operand = w.Operand is null ? null : sub(w.Operand);
                for (var k = w.Branches.Length - 1; k >= 0; k--)
                {
                    var when = sub(w.Branches[k].When);
                    result = Expr.If(operand is null ? when : operand.Eq(when), sub(w.Branches[k].Then), result);
                }
                return result;
            }
            case CastOp c:
            {
                var v = sub(c.Value);
                return c.Type switch
                {
                    "DOUBLE" or "FLOAT" or "REAL" or "NUMERIC" or "DECIMAL" => v.ToNumber(),
                    "INT" or "INTEGER" or "BIGINT" or "SMALLINT" => Expr.If(v.ToNumber() < 0, v.ToNumber().Ceiling(), v.ToNumber().Floor()),
                    "VARCHAR" or "TEXT" or "STRING" or "CHAR" => v.ToText(),
                    _ => throw new SqlException($"CAST to {c.Type} is not supported", c.Position),
                };
            }
            case Call call:
                return Function(call, sub);
            case StarRef star:
                throw new SqlException("* is only allowed in the select list or COUNT(*)", star.Position);
            default:
                throw new SqlException($"Unsupported expression {n.GetType().Name}", n.Position);
        }
    }

    private static Expr Function(Call call, Func<SqlNode, Expr> sub)
    {
        var args = call.Args.Select(sub).ToArray();
        void Arity(int min, int max)
        {
            if (args.Length < min || args.Length > max)
                throw new SqlException($"{call.Name} takes {(min == max ? min.ToString() : $"{min} to {max}")} argument(s)", call.Position);
        }
        if (call.Star || call.Distinct) throw new SqlException($"{call.Name} is not an aggregate", call.Position);
        switch (call.Name)
        {
            case "ABS": Arity(1, 1); return args[0].Abs();
            case "SQRT": Arity(1, 1); return args[0].Sqrt();
            case "LN" or "LOG": Arity(1, 1); return args[0].Log();
            case "EXP": Arity(1, 1); return args[0].Exp();
            case "POWER" or "POW": Arity(2, 2); return args[0].Pow(args[1]);
            case "ROUND":
                Arity(1, 2);
                if (args.Length == 1) return args[0].Round();
                if (call.Args[1] is not Literal { Value: double digits } || digits != Math.Floor(digits))
                    throw new SqlException("ROUND's second argument must be a whole number", call.Args[1].Position);
                return args[0].Round((int)digits);
            case "FLOOR": Arity(1, 1); return args[0].Floor();
            case "CEIL" or "CEILING": Arity(1, 1); return args[0].Ceiling();
            case "LOWER": Arity(1, 1); return args[0].Lower();
            case "UPPER": Arity(1, 1); return args[0].Upper();
            case "TRIM": Arity(1, 1); return args[0].Trim();
            case "LENGTH" or "LEN": Arity(1, 1); return args[0].Length();
            case "COALESCE": Arity(1, int.MaxValue); return Expr.Coalesce(args);
            case "IFNULL": Arity(2, 2); return Expr.Coalesce(args);
            case "NULLIF": Arity(2, 2); return Expr.If(args[0].Eq(args[1]), Expr.Null(), args[0]);
            case "YEAR": Arity(1, 1); return args[0].Year();
            case "MONTH": Arity(1, 1); return args[0].Month();
            case "DAY": Arity(1, 1); return args[0].Day();
            case "CONCAT":
                Arity(1, int.MaxValue);
                return args.Skip(1).Aggregate(args[0].ToText(), (acc, a) => acc + a.ToText());
            default:
                throw new SqlException($"Unknown function {call.Name}", call.Position);
        }
    }
}
