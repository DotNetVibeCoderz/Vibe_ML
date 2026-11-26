using System.Globalization;
using System.Text;
using Gravicode.Science.GraviFrame;

namespace Gravicode.Science.DaskNet.Sql;

/// <summary>A query that does not parse or does not make sense against the registered tables.</summary>
public sealed class SqlException(string message, int position) : Exception(position >= 0 ? $"{message} (at position {position})" : message)
{
    /// <summary>Character offset in the query, or -1 when the error is not about one place.</summary>
    public int Position { get; } = position;
}

// ---------------------------------------------------------------------------------- tokens

internal enum TokenKind { Identifier, QuotedIdentifier, Number, String, Symbol, End }

internal readonly record struct Token(TokenKind Kind, string Text, int Position)
{
    public bool Is(string keyword) => Kind == TokenKind.Identifier && Text.Equals(keyword, StringComparison.OrdinalIgnoreCase);
    public bool IsSymbol(string symbol) => Kind == TokenKind.Symbol && Text == symbol;
    public override string ToString() => Kind == TokenKind.End ? "end of query" : $"'{Text}'";
}

internal static class SqlLexer
{
    private static readonly string[] TwoChar = ["<=", ">=", "<>", "!=", "||"];
    private const string OneChar = "(),.*+-/%=<>;";

    public static List<Token> Tokenize(string sql)
    {
        var tokens = new List<Token>();
        var i = 0;
        while (i < sql.Length)
        {
            var c = sql[i];
            if (char.IsWhiteSpace(c)) { i++; continue; }
            if (c == '-' && At(sql, i + 1) == '-') { while (i < sql.Length && sql[i] != '\n') i++; continue; }
            if (c == '/' && At(sql, i + 1) == '*')
            {
                var end = sql.IndexOf("*/", i + 2, StringComparison.Ordinal);
                if (end < 0) throw new SqlException("Unterminated comment", i);
                i = end + 2;
                continue;
            }
            var start = i;
            if (c == '\'')
            {
                tokens.Add(new Token(TokenKind.String, ReadQuoted(sql, ref i, '\''), start));
                continue;
            }
            if (c is '"' or '`')
            {
                tokens.Add(new Token(TokenKind.QuotedIdentifier, ReadQuoted(sql, ref i, c), start));
                continue;
            }
            if (c == '[')
            {
                var end = sql.IndexOf(']', i);
                if (end < 0) throw new SqlException("Unterminated [identifier]", i);
                tokens.Add(new Token(TokenKind.QuotedIdentifier, sql[(i + 1)..end], start));
                i = end + 1;
                continue;
            }
            if (char.IsDigit(c) || (c == '.' && char.IsDigit(At(sql, i + 1))))
            {
                while (i < sql.Length && (char.IsDigit(sql[i]) || sql[i] == '.')) i++;
                if (i < sql.Length && sql[i] is 'e' or 'E' && (char.IsDigit(At(sql, i + 1)) || At(sql, i + 1) is '+' or '-'))
                {
                    i += 2;
                    while (i < sql.Length && char.IsDigit(sql[i])) i++;
                }
                tokens.Add(new Token(TokenKind.Number, sql[start..i], start));
                continue;
            }
            if (char.IsLetter(c) || c == '_')
            {
                while (i < sql.Length && (char.IsLetterOrDigit(sql[i]) || sql[i] is '_' or '$')) i++;
                tokens.Add(new Token(TokenKind.Identifier, sql[start..i], start));
                continue;
            }
            if (i + 1 < sql.Length && TwoChar.Contains(sql.Substring(i, 2)))
            {
                tokens.Add(new Token(TokenKind.Symbol, sql.Substring(i, 2), start));
                i += 2;
                continue;
            }
            if (OneChar.Contains(c))
            {
                tokens.Add(new Token(TokenKind.Symbol, c.ToString(), start));
                i++;
                continue;
            }
            throw new SqlException($"Unexpected character '{c}'", i);
        }
        tokens.Add(new Token(TokenKind.End, "", sql.Length));
        return tokens;
    }

    private static char At(string s, int i) => i < s.Length ? s[i] : '\0';

    /// <summary>Reads a quoted run; a doubled quote inside stands for one.</summary>
    private static string ReadQuoted(string sql, ref int i, char quote)
    {
        var start = i++;
        var text = new StringBuilder();
        while (true)
        {
            if (i >= sql.Length) throw new SqlException($"Unterminated {quote}quoted{quote} text", start);
            if (sql[i] == quote)
            {
                if (At(sql, i + 1) == quote) { text.Append(quote); i += 2; continue; }
                i++;
                return text.ToString();
            }
            text.Append(sql[i++]);
        }
    }
}

// ---------------------------------------------------------------------------------- syntax tree

internal abstract record SqlNode(int Position);
internal sealed record ColumnRef(string? Table, string Name, int Position) : SqlNode(Position);
internal sealed record Literal(object? Value, int Position) : SqlNode(Position);
internal sealed record BinaryOp(string Op, SqlNode Left, SqlNode Right, int Position) : SqlNode(Position);
internal sealed record UnaryOp(string Op, SqlNode Operand, int Position) : SqlNode(Position);
internal sealed record Call(string Name, SqlNode[] Args, bool Star, bool Distinct, int Position) : SqlNode(Position);
internal sealed record CaseWhen(SqlNode? Operand, (SqlNode When, SqlNode Then)[] Branches, SqlNode? Else, int Position) : SqlNode(Position);
internal sealed record InList(SqlNode Value, SqlNode[] Items, bool Negated, int Position) : SqlNode(Position);
internal sealed record BetweenOp(SqlNode Value, SqlNode Low, SqlNode High, bool Negated, int Position) : SqlNode(Position);
internal sealed record LikeOp(SqlNode Value, SqlNode Pattern, bool Negated, int Position) : SqlNode(Position);
internal sealed record IsNullOp(SqlNode Value, bool Negated, int Position) : SqlNode(Position);
internal sealed record CastOp(SqlNode Value, string Type, int Position) : SqlNode(Position);
internal sealed record StarRef(string? Table, int Position) : SqlNode(Position);
/// <summary><c>f(...) OVER (PARTITION BY ... ORDER BY ...)</c>.</summary>
internal sealed record WindowCall(Call Function, SqlNode[] PartitionBy, OrderItem[] OrderBy, int Position) : SqlNode(Position);

internal sealed record SelectItem(SqlNode Expr, string? Alias, string Text);
internal abstract record TableRef(string Alias, int Position);
internal sealed record NamedTable(string Name, string Alias, int Position) : TableRef(Alias, Position);
internal sealed record SubqueryTable(SelectStatement Query, string Alias, int Position) : TableRef(Alias, Position);
internal sealed record JoinClause(JoinKind Kind, TableRef Table, SqlNode On);
internal sealed record OrderItem(SqlNode Expr, bool Ascending);

internal sealed record SelectStatement(
    bool Distinct, SelectItem[] Items, TableRef From, JoinClause[] Joins, SqlNode? Where,
    SqlNode[] GroupBy, SqlNode? Having, OrderItem[] OrderBy, int? Limit)
{
    /// <summary>Queries combined with this one, left to right. When there are any, ORDER BY and
    /// LIMIT belong to the whole combination, not to this branch.</summary>
    public UnionPart[] Unions { get; init; } = [];
}

/// <summary><c>UNION [ALL] query</c>; without ALL, duplicate rows of the combination so far are removed.</summary>
internal sealed record UnionPart(bool All, SelectStatement Query, int Position);

// ---------------------------------------------------------------------------------- parser

/// <summary>Recursive descent over the token list, one method per precedence level.</summary>
internal sealed class SqlParser
{
    private static readonly HashSet<string> Reserved = new(StringComparer.OrdinalIgnoreCase)
    {
        "SELECT", "DISTINCT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET", "JOIN", "INNER",
        "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "ON", "AS", "AND", "OR", "NOT", "IS", "NULL", "IN", "BETWEEN",
        "LIKE", "OVER", "CASE", "WHEN", "THEN", "ELSE", "END", "ASC", "DESC", "UNION", "TRUE", "FALSE", "CAST",
    };

    private readonly string _sql;
    private readonly List<Token> _tokens;
    private int _at;

    private SqlParser(string sql)
    {
        _sql = sql;
        _tokens = SqlLexer.Tokenize(sql);
    }

    public static SelectStatement Parse(string sql)
    {
        var parser = new SqlParser(sql);
        var statement = parser.ParseQuery();
        if (parser.Peek.IsSymbol(";")) parser._at++;
        if (parser.Peek.Kind != TokenKind.End) throw parser.Error($"Unexpected {parser.Peek}");
        return statement;
    }

    private Token Peek => _tokens[_at];
    private Token PeekAt(int offset) => _tokens[Math.Min(_at + offset, _tokens.Count - 1)];
    private Token Next() => _tokens[_at++];
    private SqlException Error(string message) => new(message, Peek.Position);

    private bool Accept(string keyword)
    {
        if (!Peek.Is(keyword)) return false;
        _at++;
        return true;
    }

    private bool AcceptSymbol(string symbol)
    {
        if (!Peek.IsSymbol(symbol)) return false;
        _at++;
        return true;
    }

    private void Expect(string keyword)
    {
        if (!Accept(keyword)) throw Error($"Expected {keyword} but found {Peek}");
    }

    private void ExpectSymbol(string symbol)
    {
        if (!AcceptSymbol(symbol)) throw Error($"Expected '{symbol}' but found {Peek}");
    }

    private string Identifier(string what)
    {
        var t = Peek;
        if (t.Kind == TokenKind.QuotedIdentifier || (t.Kind == TokenKind.Identifier && !Reserved.Contains(t.Text)))
        {
            _at++;
            return t.Text;
        }
        throw Error($"Expected {what} but found {t}");
    }

    private bool PeekIsAlias => Peek.Kind == TokenKind.QuotedIdentifier || (Peek.Kind == TokenKind.Identifier && !Reserved.Contains(Peek.Text));

    // ------------------------------------------------------------------ statements

    /// <summary>A SELECT, or several joined by UNION [ALL]. ORDER BY and LIMIT written after the
    /// last branch apply to the whole union, as in standard SQL; on an earlier branch they are
    /// refused rather than silently applied to the union.</summary>
    private SelectStatement ParseQuery()
    {
        var first = ParseSelect();
        if (!Peek.Is("UNION")) return first;
        var parts = new List<UnionPart>();
        var last = first;
        while (Peek.Is("UNION"))
        {
            if (last.OrderBy.Length > 0 || last.Limit is not null)
                throw Error("ORDER BY and LIMIT may only follow the last query of a UNION; wrap a branch in a subquery to sort it");
            var position = Next().Position;
            var all = Accept("ALL");
            if (!all) Accept("DISTINCT");
            last = ParseSelect();
            parts.Add(new UnionPart(all, last with { OrderBy = [], Limit = null }, position));
        }
        return first with { OrderBy = last.OrderBy, Limit = last.Limit, Unions = parts.ToArray() };
    }

    private SelectStatement ParseSelect()
    {
        Expect("SELECT");
        var distinct = Accept("DISTINCT");
        if (!distinct) Accept("ALL");
        var items = new List<SelectItem> { ParseSelectItem() };
        while (AcceptSymbol(",")) items.Add(ParseSelectItem());

        Expect("FROM");
        var from = ParseTable();
        var joins = new List<JoinClause>();
        while (TryJoinKind(out var kind))
        {
            var table = ParseTable();
            Expect("ON");
            joins.Add(new JoinClause(kind, table, ParseExpression()));
        }

        var where = Accept("WHERE") ? ParseExpression() : null;
        var groupBy = new List<SqlNode>();
        if (Accept("GROUP"))
        {
            Expect("BY");
            do groupBy.Add(ParseExpression()); while (AcceptSymbol(","));
        }
        var having = Accept("HAVING") ? ParseExpression() : null;
        var orderBy = new List<OrderItem>();
        if (Accept("ORDER"))
        {
            Expect("BY");
            do
            {
                var e = ParseExpression();
                var ascending = !Accept("DESC");
                if (ascending) Accept("ASC");
                orderBy.Add(new OrderItem(e, ascending));
            } while (AcceptSymbol(","));
        }
        int? limit = null;
        if (Accept("LIMIT"))
        {
            var t = Next();
            if (t.Kind != TokenKind.Number || !int.TryParse(t.Text, NumberStyles.None, CultureInfo.InvariantCulture, out var n))
                throw new SqlException("LIMIT takes a non-negative whole number", t.Position);
            limit = n;
        }
        if (Peek.Is("OFFSET")) throw Error("OFFSET is not supported; a distributed frame has no cheap row positions");
        return new SelectStatement(distinct, items.ToArray(), from, joins.ToArray(), where, groupBy.ToArray(), having, orderBy.ToArray(), limit);
    }

    private bool TryJoinKind(out JoinKind kind)
    {
        kind = JoinKind.Inner;
        if (Accept("JOIN")) return true;
        if (Peek.Is("INNER") && PeekAt(1).Is("JOIN")) { _at += 2; return true; }
        if (Peek.Is("LEFT") || Peek.Is("RIGHT") || Peek.Is("FULL"))
        {
            kind = Next().Text.ToUpperInvariant() switch { "LEFT" => JoinKind.Left, "RIGHT" => JoinKind.Right, _ => JoinKind.Outer };
            Accept("OUTER");
            Expect("JOIN");
            return true;
        }
        if (Peek.Is("CROSS")) throw Error("CROSS JOIN is not supported; it has no key to partition on");
        return false;
    }

    private SelectItem ParseSelectItem()
    {
        var start = Peek.Position;
        if (AcceptSymbol("*")) return new SelectItem(new StarRef(null, start), null, "*");
        if ((Peek.Kind is TokenKind.Identifier or TokenKind.QuotedIdentifier) && PeekAt(1).IsSymbol(".") && PeekAt(2).IsSymbol("*"))
        {
            var table = Next().Text;
            _at += 2;
            return new SelectItem(new StarRef(table, start), null, table + ".*");
        }
        var expr = ParseExpression();
        var text = _sql[start..PreviousEnd()].Trim();
        string? alias = null;
        if (Accept("AS")) alias = Identifier("an alias");
        else if (PeekIsAlias) alias = Identifier("an alias");
        return new SelectItem(expr, alias, text);
    }

    /// <summary>End offset of the last consumed token (strings and quoted names include their quotes).</summary>
    private int PreviousEnd()
    {
        var t = _tokens[_at - 1];
        var i = t.Position;
        if (t.Kind is TokenKind.String or TokenKind.QuotedIdentifier)
        {
            var close = _sql[i] == '[' ? ']' : _sql[i];
            i++;
            while (i < _sql.Length)
            {
                if (_sql[i] == close)
                {
                    if (close != ']' && i + 1 < _sql.Length && _sql[i + 1] == close) { i += 2; continue; }
                    return i + 1;
                }
                i++;
            }
            return _sql.Length;
        }
        return i + t.Text.Length;
    }

    private TableRef ParseTable()
    {
        var position = Peek.Position;
        if (AcceptSymbol("("))
        {
            var query = ParseQuery();
            ExpectSymbol(")");
            Accept("AS");
            return new SubqueryTable(query, Identifier("an alias for the subquery"), position);
        }
        var name = Identifier("a table name");
        string alias = name;
        if (Accept("AS")) alias = Identifier("a table alias");
        else if (PeekIsAlias) alias = Identifier("a table alias");
        return new NamedTable(name, alias, position);
    }

    // ------------------------------------------------------------------ expressions

    public SqlNode ParseExpression() => ParseOr();

    private SqlNode ParseOr()
    {
        var left = ParseAnd();
        while (Peek.Is("OR"))
        {
            var p = Next().Position;
            left = new BinaryOp("OR", left, ParseAnd(), p);
        }
        return left;
    }

    private SqlNode ParseAnd()
    {
        var left = ParseNot();
        while (Peek.Is("AND"))
        {
            var p = Next().Position;
            left = new BinaryOp("AND", left, ParseNot(), p);
        }
        return left;
    }

    private SqlNode ParseNot()
    {
        if (Peek.Is("NOT"))
        {
            var p = Next().Position;
            return new UnaryOp("NOT", ParseNot(), p);
        }
        return ParsePredicate();
    }

    private SqlNode ParsePredicate()
    {
        var left = ParseAdditive();
        var p = Peek.Position;
        foreach (var op in new[] { "=", "<>", "!=", "<=", ">=", "<", ">" })
            if (AcceptSymbol(op)) return new BinaryOp(op == "!=" ? "<>" : op, left, ParseAdditive(), p);

        if (Accept("IS"))
        {
            var negated = Accept("NOT");
            Expect("NULL");
            return new IsNullOp(left, negated, p);
        }
        var not = Peek.Is("NOT") && (PeekAt(1).Is("IN") || PeekAt(1).Is("BETWEEN") || PeekAt(1).Is("LIKE"));
        if (not) _at++;
        if (Accept("IN"))
        {
            ExpectSymbol("(");
            if (Peek.Is("SELECT")) throw Error("IN (SELECT ...) is not supported; join on the subquery instead");
            var items = new List<SqlNode> { ParseExpression() };
            while (AcceptSymbol(",")) items.Add(ParseExpression());
            ExpectSymbol(")");
            return new InList(left, items.ToArray(), not, p);
        }
        if (Accept("BETWEEN"))
        {
            var low = ParseAdditive();
            Expect("AND");
            return new BetweenOp(left, low, ParseAdditive(), not, p);
        }
        if (Accept("LIKE")) return new LikeOp(left, ParseAdditive(), not, p);
        if (not) throw Error("Expected IN, BETWEEN or LIKE after NOT");
        return left;
    }

    private SqlNode ParseAdditive()
    {
        var left = ParseMultiplicative();
        while (Peek.IsSymbol("+") || Peek.IsSymbol("-") || Peek.IsSymbol("||"))
        {
            var t = Next();
            left = new BinaryOp(t.Text, left, ParseMultiplicative(), t.Position);
        }
        return left;
    }

    private SqlNode ParseMultiplicative()
    {
        var left = ParseUnary();
        while (Peek.IsSymbol("*") || Peek.IsSymbol("/") || Peek.IsSymbol("%"))
        {
            var t = Next();
            left = new BinaryOp(t.Text, left, ParseUnary(), t.Position);
        }
        return left;
    }

    private SqlNode ParseUnary()
    {
        var p = Peek.Position;
        if (AcceptSymbol("-"))
        {
            var operand = ParseUnary();
            return operand is Literal { Value: double d } ? new Literal(-d, p) : new UnaryOp("-", operand, p);
        }
        if (AcceptSymbol("+")) return ParseUnary();
        return ParsePrimary();
    }

    private SqlNode ParsePrimary()
    {
        var t = Peek;
        switch (t.Kind)
        {
            case TokenKind.Number:
                _at++;
                return new Literal(double.Parse(t.Text, NumberStyles.Float, CultureInfo.InvariantCulture), t.Position);
            case TokenKind.String:
                _at++;
                return new Literal(t.Text, t.Position);
            case TokenKind.Symbol when t.Text == "(":
                _at++;
                if (Peek.Is("SELECT")) throw Error("Scalar subqueries are not supported");
                var inner = ParseExpression();
                ExpectSymbol(")");
                return inner;
            case TokenKind.QuotedIdentifier:
                return ParseName();
            case TokenKind.Identifier:
                break;
            default:
                throw Error($"Expected an expression but found {t}");
        }

        if (Accept("NULL")) return new Literal(null, t.Position);
        if (Accept("TRUE")) return new Literal(true, t.Position);
        if (Accept("FALSE")) return new Literal(false, t.Position);
        if ((t.Is("DATE") || t.Is("TIMESTAMP")) && PeekAt(1).Kind == TokenKind.String)
        {
            _at++;
            var s = Next();
            if (!DateTime.TryParse(s.Text, CultureInfo.InvariantCulture, DateTimeStyles.None, out var date))
                throw new SqlException($"'{s.Text}' is not a date", s.Position);
            return new Literal(date, t.Position);
        }
        if (Accept("CASE")) return ParseCase(t.Position);
        if (Accept("CAST"))
        {
            ExpectSymbol("(");
            var value = ParseExpression();
            Expect("AS");
            var type = Identifier("a type name");
            if (AcceptSymbol("("))
            {
                while (!AcceptSymbol(")")) Next();
            }
            ExpectSymbol(")");
            return new CastOp(value, type.ToUpperInvariant(), t.Position);
        }
        if (Reserved.Contains(t.Text)) throw Error($"Unexpected {t}");
        return ParseName();
    }

    private SqlNode ParseName()
    {
        var t = Next();
        if (t.Kind == TokenKind.Identifier && AcceptSymbol("("))
        {
            var name = t.Text.ToUpperInvariant();
            if (AcceptSymbol("*"))
            {
                ExpectSymbol(")");
                return Over(new Call(name, [], Star: true, Distinct: false, t.Position));
            }
            var distinct = Accept("DISTINCT");
            var args = new List<SqlNode>();
            if (!Peek.IsSymbol(")"))
            {
                args.Add(ParseExpression());
                while (AcceptSymbol(",")) args.Add(ParseExpression());
            }
            ExpectSymbol(")");
            return Over(new Call(name, args.ToArray(), Star: false, distinct, t.Position));
        }
        if (AcceptSymbol("."))
        {
            var column = Identifier("a column name");
            return new ColumnRef(t.Text, column, t.Position);
        }
        return new ColumnRef(null, t.Text, t.Position);
    }

    /// <summary>An optional <c>OVER (...)</c> after a call. Explicit frames (ROWS/RANGE BETWEEN) are
    /// refused: the default frame is the only one implemented.</summary>
    private SqlNode Over(Call call)
    {
        if (!Accept("OVER")) return call;
        if (call.Distinct) throw Error("DISTINCT is not supported in a window function");
        ExpectSymbol("(");
        var partition = new List<SqlNode>();
        if (Peek.Is("PARTITION"))
        {
            _at++;
            Expect("BY");
            do partition.Add(ParseExpression()); while (AcceptSymbol(","));
        }
        var order = new List<OrderItem>();
        if (Accept("ORDER"))
        {
            Expect("BY");
            do
            {
                var e = ParseExpression();
                var ascending = !Accept("DESC");
                if (ascending) Accept("ASC");
                order.Add(new OrderItem(e, ascending));
            } while (AcceptSymbol(","));
        }
        if (Peek.Is("ROWS") || Peek.Is("RANGE") || Peek.Is("GROUPS"))
            throw Error("Explicit window frames are not supported; the default (start of partition to current row and its ties) is used");
        ExpectSymbol(")");
        return new WindowCall(call, partition.ToArray(), order.ToArray(), call.Position);
    }

    private SqlNode ParseCase(int position)
    {
        SqlNode? operand = Peek.Is("WHEN") ? null : ParseExpression();
        var branches = new List<(SqlNode, SqlNode)>();
        while (Accept("WHEN"))
        {
            var when = ParseExpression();
            Expect("THEN");
            branches.Add((when, ParseExpression()));
        }
        if (branches.Count == 0) throw Error("CASE needs at least one WHEN");
        var otherwise = Accept("ELSE") ? ParseExpression() : null;
        Expect("END");
        return new CaseWhen(operand, branches.ToArray(), otherwise, position);
    }
}
