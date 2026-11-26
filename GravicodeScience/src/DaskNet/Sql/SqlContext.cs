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
/// the aggregates COUNT/SUM/AVG/MIN/MAX/STDDEV/VARIANCE (and their <c>_POP</c> forms, each
/// optionally <c>DISTINCT</c>), and the
/// scalar functions listed in <see cref="ScalarFunctions"/>. Missing values follow SQL's
/// three-valued logic, as <see cref="Expr"/> does.
/// </para>
/// <para>
/// <c>UNION [ALL]</c> combines queries; ORDER BY and LIMIT after the last one apply to the union.
/// Window functions: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD and COUNT/SUM/AVG/MIN/MAX
/// <c>OVER ([PARTITION BY ...] [ORDER BY ...])</c>, with the default frame.
/// Not supported, and refused with a message rather than approximated: <c>OFFSET</c>,
/// <c>CROSS JOIN</c>, subqueries outside <c>FROM</c>, explicit window frames, and a join condition with no equality between the two sides.
/// </para>
/// <para>
/// Join order: three or more tables joined only by inner joins are joined greedily
/// (<see cref="ReorderJoins"/>): at each step the two connected groups with the smallest combined
/// estimate are joined, the bigger on the left. Small tables therefore meet each other and the
/// big ones first, where they are broadcast and can only shrink them, before two big tables are
/// shuffled to meet; the plan may be bushy (two dimensions joined, then broadcast together). Single-table
/// WHERE conditions count towards a table's estimate. Outer joins, tables without an estimate, and
/// a chain that is not connected by equalities keep the written order. Columns come back in the
/// written order.
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
    public DaskFrame Sql(string query) => new SqlCompiler(_tables, ReorderJoins).Compile(SqlParser.Parse(query));

    /// <summary>
    /// Whether a chain of three or more inner-joined tables is joined in an order chosen from size
    /// estimates rather than the order written (default true). The result is the same either way;
    /// see the remarks on <see cref="SqlContext"/>.
    /// </summary>
    public bool ReorderJoins { get; set; } = true;

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
internal sealed class SqlCompiler(IReadOnlyDictionary<string, DaskFrame> tables, bool reorderJoins = true)
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
        if (s.Unions.Length > 0) return CompileUnion(s);
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

        // Window functions see the rows after WHERE, GROUP BY and HAVING, and run before the
        // projection, DISTINCT, ORDER BY and LIMIT - the order SQL defines.
        var windows = new List<WindowCall>();
        void FindWindows(SqlNode n)
        {
            if (n is WindowCall w) windows.Add(w);
            else foreach (var child in Children(n)) FindWindows(child);
        }
        foreach (var item in s.Items) FindWindows(item.Expr);
        foreach (var o in s.OrderBy) FindWindows(o.Expr);
        if (windows.Count > 0) (frame, compile) = CompileWindows(windows, frame, compile);

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

    // ------------------------------------------------------------------ window functions

    private static readonly Dictionary<string, string> WindowAggregates = new(StringComparer.Ordinal)
    {
        ["COUNT"] = "count", ["SUM"] = "sum", ["AVG"] = "mean", ["MEAN"] = "mean", ["MIN"] = "min", ["MAX"] = "max",
    };

    /// <summary>
    /// Window calls with the same PARTITION BY and ORDER BY share one window (one shuffle).
    /// Partition, order and argument expressions are compiled with <paramref name="inner"/>, so in
    /// a grouped query they may use group keys and aggregates (<c>RANK() OVER (ORDER BY SUM(x))</c>).
    /// </summary>
    private (DaskFrame, Func<SqlNode, Expr>) CompileWindows(List<WindowCall> windows, DaskFrame frame, Func<SqlNode, Expr> inner)
    {
        var pre = new List<Expr>();
        var preByText = new Dictionary<string, string>(StringComparer.Ordinal);
        string Column(SqlNode n)
        {
            if (ContainsWindow(n)) throw new SqlException("Window functions cannot be nested", n.Position);
            var e = inner(n);
            if (e is ColumnExpr c) return c.Name;
            var text = e.ToString()!;
            if (!preByText.TryGetValue(text, out var name))
            {
                name = $"__win{_hidden++}";
                preByText[text] = name;
                pre.Add(e.As(name));
            }
            return name;
        }

        var outputs = new Dictionary<WindowCall, string>(ReferenceEqualityComparer.Instance);
        var specs = new List<(string Key, string[] Partition, (string, bool)[] Order, List<WindowFunction> Functions)>();
        foreach (var w in windows)
        {
            var call = w.Function;
            var partition = w.PartitionBy.Select(Column).ToArray();
            var order = w.OrderBy.Select(o => (Column(o.Expr), o.Ascending)).ToArray();
            var output = $"__window{_hidden++}";
            void Arity(int min, int max)
            {
                if (call.Star || call.Args.Length < min || call.Args.Length > max)
                    throw new SqlException($"{call.Name} takes {(min == max ? min.ToString() : $"{min} to {max}")} argument(s)", call.Position);
            }
            WindowFunction function;
            switch (call.Name)
            {
                case "ROW_NUMBER" or "RANK" or "DENSE_RANK":
                    Arity(0, 0);
                    if (order.Length == 0) throw new SqlException($"{call.Name} needs ORDER BY in its window", call.Position);
                    function = new WindowFunction(output, call.Name.ToLowerInvariant());
                    break;
                case "LAG" or "LEAD":
                {
                    Arity(1, 3);
                    if (call.Args.Length == 3)
                        throw new SqlException($"{call.Name}'s default argument is not supported; wrap the call in COALESCE", call.Args[2].Position);
                    var offset = 1;
                    if (call.Args.Length == 2)
                    {
                        if (call.Args[1] is not Literal { Value: double d } || d != Math.Floor(d) || d < 0)
                            throw new SqlException($"{call.Name}'s offset must be a non-negative whole number", call.Args[1].Position);
                        offset = (int)d;
                    }
                    function = new WindowFunction(output, call.Name.ToLowerInvariant(), Column(call.Args[0]), offset);
                    break;
                }
                case "COUNT" when call.Star:
                    function = new WindowFunction(output, "size");
                    break;
                default:
                    if (!WindowAggregates.TryGetValue(call.Name, out var how))
                        throw new SqlException(AggregateNames.ContainsKey(call.Name)
                            ? $"{call.Name} is not supported as a window function"
                            : $"Unknown window function {call.Name}", call.Position);
                    Arity(1, 1);
                    function = new WindowFunction(output, how, Column(call.Args[0]));
                    break;
            }
            var key = string.Join("\u001F", partition) + "|" + string.Join("\u001F", order.Select(o => $"{o.Item1} {o.Item2}"));
            var index = specs.FindIndex(x => x.Key == key);
            if (index < 0) { specs.Add((key, partition, order, [])); index = specs.Count - 1; }
            specs[index].Functions.Add(function);
            outputs[w] = output;
        }

        if (pre.Count > 0) frame = frame.WithColumns(pre.ToArray());
        foreach (var spec in specs) frame = frame.Window(spec.Partition, spec.Order, spec.Functions.ToArray());

        Expr Windowed(SqlNode n)
            => n is WindowCall w ? Expr.Col(outputs[w]) : ContainsWindow(n) ? Build(n, Windowed) : inner(n);
        return (frame, Windowed);
    }

    private static bool ContainsWindow(SqlNode n) => n is WindowCall || Children(n).Any(ContainsWindow);

    // ------------------------------------------------------------------ UNION

    /// <summary>
    /// Branches are matched by position and take the first branch's names. The combination is
    /// built left to right, so <c>a UNION ALL b UNION c</c> deduplicates all three, while
    /// <c>a UNION b UNION ALL c</c> keeps c's duplicates. ORDER BY may name output columns or
    /// positions only, since the branches have no other columns in common.
    /// </summary>
    private DaskFrame CompileUnion(SelectStatement s)
    {
        var frame = Compile(s with { OrderBy = [], Limit = null, Unions = [] });
        var names = frame.Columns.ToArray();
        foreach (var part in s.Unions)
        {
            var branch = Compile(part.Query);
            var columns = branch.Columns;
            if (columns.Count != names.Length)
                throw new SqlException($"Each UNION query must have the same number of columns ({names.Length} vs {columns.Count})", part.Position);
            if (!columns.SequenceEqual(names))
                branch = branch.Select(columns.Select((c, i) => Expr.Col(c).As(names[i])).ToArray());
            frame = DaskFrame.Concat(frame, branch);
            if (!part.All)
            {
                var count = $"__count{_hidden++}";
                frame = frame.GroupBy(names).Size(count).Drop(count);
            }
        }

        var sortKeys = new List<(string Column, bool Ascending)>();
        foreach (var o in s.OrderBy)
        {
            if (o.Expr is Literal { Value: double d } && d == Math.Floor(d) && d >= 1 && d <= names.Length)
                sortKeys.Add((names[(int)d - 1], o.Ascending));
            else if (o.Expr is ColumnRef { Table: null } r && names.FirstOrDefault(n => n.Equals(r.Name, StringComparison.OrdinalIgnoreCase)) is { } name)
                sortKeys.Add((name, o.Ascending));
            else throw new SqlException("ORDER BY after a UNION must name an output column or position", o.Expr.Position);
        }
        if (sortKeys.Count > 0) frame = frame.SortBy(sortKeys);
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
        if (reorderJoins && s.Joins.Length >= 2 && s.Joins.All(j => j.Kind == JoinKind.Inner) && Reordered(s) is { } reordered)
            return reordered;
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

    /// <summary>
    /// The inner-join chain of <paramref name="s"/> joined greedily by estimated size, or null to
    /// keep the written order (it already is the greedy one, an estimate is missing, or the chain is
    /// not connected by equalities). Each ON condition is resolved in the scope it was written in.
    /// </summary>
    private (DaskFrame, SqlScope)? Reordered(SelectStatement s)
    {
        var refs = new[] { s.From }.Concat(s.Joins.Select(j => j.Table)).ToArray();
        var relations = refs.Select(t => Table(t, qualify: true)).ToArray();
        var owner = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < relations.Length; i++)
            foreach (var e in relations[i].Item2.Entries) owner[e.Internal] = i;

        var full = new SqlScope();
        foreach (var (_, scope) in relations) full.Entries.AddRange(scope.Entries);

        // Equalities between two relations become edges; everything else is a filter on the result.
        var edges = new List<(int A, string ColA, int B, string ColB)>();
        var residual = new List<Expr>();
        var prefix = new SqlScope();
        prefix.Entries.AddRange(relations[0].Item2.Entries);
        for (var j = 0; j < s.Joins.Length; j++)
        {
            prefix.Entries.AddRange(relations[j + 1].Item2.Entries);
            var before = edges.Count;
            foreach (var condition in Conjuncts(s.Joins[j].On))
            {
                if (condition is BinaryOp { Op: "=", Left: ColumnRef a, Right: ColumnRef b }
                    && TryResolve(a, prefix) is { } ca && TryResolve(b, prefix) is { } cb && owner[ca] != owner[cb])
                    edges.Add((owner[ca], ca, owner[cb], cb));
                else residual.Add(Plain(condition, prefix));
            }
            if (edges.Count == before) return null;   // the written order reports the missing equality
        }

        // Estimated sizes, halved for every WHERE condition that reads only that table.
        var sizes = new double[relations.Length];
        for (var i = 0; i < relations.Length; i++)
        {
            if (CostModel.Bytes(relations[i].Item1.Plan) is not { } bytes) return null;
            sizes[i] = bytes;
        }
        if (s.Where is not null)
            foreach (var condition in Conjuncts(s.Where))
            {
                var readers = ColumnRefs(condition).Select(c => TryResolve(c, full)).ToArray();
                if (readers.Length > 0 && readers.All(r => r is not null) && readers.Select(r => owner[r!]).Distinct().Count() == 1)
                    sizes[owner[readers[0]!]] *= CostModel.FilterSelectivity;
            }

        // Greedy operator ordering: merge the two connected groups with the smallest combined
        // estimate, bigger side on the left so the smaller one is what gets broadcast.
        var groups = Enumerable.Range(0, relations.Length)
            .Select(i => (Frame: relations[i].Item1, Members: new HashSet<int> { i }, Size: sizes[i])).ToList();
        var used = new bool[edges.Count];
        var leftDeepInWrittenOrder = true;
        while (groups.Count > 1)
        {
            (int A, int B, double Cost)? best = null;
            for (var a = 0; a < groups.Count; a++)
                for (var b = a + 1; b < groups.Count; b++)
                {
                    var (ga, gb) = (groups[a].Members, groups[b].Members);
                    if (!edges.Any(e => (ga.Contains(e.A) && gb.Contains(e.B)) || (ga.Contains(e.B) && gb.Contains(e.A)))) continue;
                    var cost = groups[a].Size + groups[b].Size;
                    if (best is null || cost < best.Value.Cost) best = (a, b, cost);
                }
            if (best is not { } pick) return null;

            var (big, small) = groups[pick.A].Size >= groups[pick.B].Size ? (groups[pick.A], groups[pick.B]) : (groups[pick.B], groups[pick.A]);
            var written = Enumerable.Range(0, big.Members.Count).ToHashSet();
            leftDeepInWrittenOrder &= big.Members.SetEquals(written) && small.Members.SetEquals([big.Members.Count]);

            var leftKeys = new List<string>();
            var rightKeys = new List<string>();
            for (var k = 0; k < edges.Count; k++)
            {
                var (a, colA, b, colB) = edges[k];
                if (used[k]) continue;
                if (big.Members.Contains(a) && small.Members.Contains(b)) { leftKeys.Add(colA); rightKeys.Add(colB); used[k] = true; }
                else if (big.Members.Contains(b) && small.Members.Contains(a)) { leftKeys.Add(colB); rightKeys.Add(colA); used[k] = true; }
            }
            var merged = (Frame: Join(big.Frame, small.Frame, leftKeys, rightKeys, JoinKind.Inner),
                          Members: big.Members.Union(small.Members).ToHashSet(), Size: big.Size + small.Size);
            groups.Remove(big);
            groups.Remove(small);
            groups.Add(merged);
        }
        if (leftDeepInWrittenOrder) return null;

        var frame = groups[0].Frame;
        // Equalities inside one group that were never a join key (a cycle) still have to hold.
        for (var k = 0; k < edges.Count; k++)
            if (!used[k]) frame = frame.Filter(Expr.Col(edges[k].ColA) == Expr.Col(edges[k].ColB));
        foreach (var r in residual) frame = frame.Filter(r);
        return (frame.Select(full.Entries.Select(e => e.Internal).ToArray()), full);
    }

    private static IEnumerable<ColumnRef> ColumnRefs(SqlNode n)
        => n is ColumnRef c ? [c] : Children(n).SelectMany(ColumnRefs);

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
        var distinct = new List<Aggregation>();
        var aggByText = new Dictionary<string, string>(StringComparer.Ordinal);
        void Collect(SqlNode n)
        {
            if (n is Call call && AggregateNames.TryGetValue(call.Name, out var how))
            {
                if (call.Distinct && call.Star) throw new SqlException($"{call.Name}(DISTINCT *) is not valid", call.Position);
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
                (call.Distinct ? distinct : aggregations).Add(new Aggregation(output, how, column));
                return;
            }
            foreach (var child in Children(n)) Collect(child);
        }
        foreach (var i in s.Items) Collect(i.Expr);
        if (s.Having is not null) Collect(s.Having);
        foreach (var o in s.OrderBy) Collect(o.Expr);
        if (aggregations.Count == 0) aggregations.Add(new Aggregation($"__agg{_hidden++}", "size", null));

        if (pre.Count > 0) frame = frame.WithColumns(pre.ToArray());
        var source = frame;
        frame = keyNames.Count > 0 ? frame.GroupBy(keyNames.ToArray()).Agg(aggregations.ToArray()) : frame.Aggregate(aggregations.ToArray());
        foreach (var byColumn in distinct.GroupBy(a => a.Column!)) frame = JoinDistinct(frame, source, keyNames, byColumn.Key, byColumn.ToArray());

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

    /// <summary>
    /// <c>AGG(DISTINCT x)</c>: (keys, x) pairs are made distinct by grouping on both, the
    /// aggregate runs per key over those, and the result is joined onto the other aggregates by
    /// the group keys. Every group of the main aggregation has at least one pair (x may be
    /// missing, which COUNT and SUM skip), so an inner join loses nothing; missing keys match each
    /// other in GraviFrame joins, as they group together.
    /// </summary>
    private DaskFrame JoinDistinct(DaskFrame grouped, DaskFrame source, List<string> keys, string column, Aggregation[] aggs)
    {
        var pairs = keys.Append(column).Distinct().ToArray();
        var size = $"__size{_hidden++}";
        var unique = source.Select(pairs).GroupBy(pairs).Size(size).Drop(size);
        if (keys.Count > 0)
        {
            var result = grouped.Merge(unique.GroupBy(keys.ToArray()).Agg(aggs), keys.ToArray(), keys.ToArray(), JoinKind.Inner, "__dup");
            var duplicates = result.Columns.Where(c => c.EndsWith("__dup", StringComparison.Ordinal)).ToArray();
            return duplicates.Length > 0 ? result.Drop(duplicates) : result;
        }
        // No GROUP BY: both sides are one row; join them on a constant.
        var all = $"__all{_hidden++}";
        var one = Expr.Lit(1.0).As(all);
        return grouped.WithColumns(one).Merge(unique.Aggregate(aggs).WithColumns(one), [all], [all]).Drop(all);
    }

    private string AggregateKey(Call call, SqlScope scope)
        => $"{call.Name}({(call.Distinct ? "DISTINCT " : "")}{(call.Star ? "*" : Plain(call.Args[0], scope).ToString())})"
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
        WindowCall w => [.. w.Function.Args, .. w.PartitionBy, .. w.OrderBy.Select(o => o.Expr)],
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
            case WindowCall w:
                throw new SqlException("Window functions are only allowed in the select list and ORDER BY", w.Position);
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
