namespace Gravicode.Science.GraviFrame;

/// <summary>
/// A text column stored as integer codes into a shared dictionary of categories.
/// </summary>
/// <remarks>
/// <para>
/// A column holding a country name per row stores the same few dozen strings hundreds of thousands
/// of times over. Dictionary encoding keeps each distinct string once and replaces the column with
/// an <c>int[]</c> of codes, which is both far smaller and far faster to work with: grouping,
/// joining and equality all become integer comparisons instead of string ones, and the codes are a
/// contiguous buffer the CPU can stream through.
/// </para>
/// <para>
/// The second reason is ordering. A <see cref="TextSeries"/> can only sort alphabetically, which
/// puts "high" before "low" before "medium" — a real and easily missed wrong answer for ordinal
/// data. An <see cref="IsOrdered"/> categorical sorts and compares by the category order the caller
/// declared, so "low &lt; medium &lt; high" holds.
/// </para>
/// <para>
/// The categories are part of the column's type, not a summary of its contents. A category with no
/// rows still exists — which is what makes a group-by over a categorical produce an empty group
/// rather than silently omitting it, and what lets two columns be compared code-for-code only when
/// their dictionaries agree.
/// </para>
/// </remarks>
public sealed class CategoricalSeries : Series
{
    /// <summary>The code standing for a missing value.</summary>
    public const int MissingCode = -1;

    private readonly int[] _codes;
    private readonly string[] _categories;
    private readonly Dictionary<string, int> _lookup;

    /// <summary>Wraps codes and categories without copying.</summary>
    /// <param name="name">Column name.</param>
    /// <param name="codes">One index into <paramref name="categories"/> per row; -1 is missing.</param>
    /// <param name="categories">The distinct values, in the order that defines their ranking.</param>
    /// <param name="ordered">
    /// Whether the category order is meaningful. Set this for ordinal data — sizes, grades, risk
    /// bands — so that comparisons follow the declared order rather than the alphabet.
    /// </param>
    public CategoricalSeries(string name, int[] codes, string[] categories, bool ordered = false)
        : base(name, codes.Length)
    {
        ArgumentNullException.ThrowIfNull(codes);
        ArgumentNullException.ThrowIfNull(categories);

        foreach (var code in codes)
            if (code < MissingCode || code >= categories.Length)
                throw new ArgumentException(
                    $"Code {code} does not index the {categories.Length} categories.", nameof(codes));

        _codes = codes;
        _categories = categories;
        IsOrdered = ordered;

        _lookup = new Dictionary<string, int>(categories.Length, StringComparer.Ordinal);
        for (var i = 0; i < categories.Length; i++)
        {
            if (!_lookup.TryAdd(categories[i], i))
                throw new ArgumentException($"Category '{categories[i]}' appears twice.", nameof(categories));
        }
    }

    /// <summary>
    /// Builds a categorical from raw strings, discovering the categories.
    /// </summary>
    /// <param name="name">Column name.</param>
    /// <param name="values">The raw values; <c>null</c> becomes missing.</param>
    /// <param name="categories">
    /// The category set and its order. When omitted, the distinct values are used in first-seen
    /// order. A value not in an explicit list becomes missing rather than an error — that is what
    /// makes an explicit list usable as a whitelist.
    /// </param>
    /// <param name="ordered">Whether the category order is meaningful.</param>
    public static CategoricalSeries FromValues(string name, IReadOnlyList<string?> values,
        IReadOnlyList<string>? categories = null, bool ordered = false)
    {
        ArgumentNullException.ThrowIfNull(values);

        var known = new Dictionary<string, int>(StringComparer.Ordinal);
        var order = new List<string>();

        if (categories is not null)
            foreach (var category in categories)
                if (known.TryAdd(category, order.Count)) order.Add(category);

        var codes = new int[values.Count];

        for (var i = 0; i < values.Count; i++)
        {
            var value = values[i];
            if (value is null) { codes[i] = MissingCode; continue; }

            if (known.TryGetValue(value, out var code)) { codes[i] = code; continue; }

            // An explicit category list is a declaration of what the column may hold, so anything
            // outside it is missing data rather than a new category to invent.
            if (categories is not null) { codes[i] = MissingCode; continue; }

            code = order.Count;
            known[value] = code;
            order.Add(value);
            codes[i] = code;
        }

        return new CategoricalSeries(name, codes, [.. order], ordered);
    }

    /// <inheritdoc />
    public override DataType DataType => DataType.Categorical;

    /// <summary>Whether the category order carries meaning.</summary>
    public bool IsOrdered { get; }

    /// <summary>The distinct values, in ranking order.</summary>
    public IReadOnlyList<string> Categories => _categories;

    /// <summary>The raw dictionary codes; <c>-1</c> is missing.</summary>
    public ReadOnlySpan<int> Codes => _codes;

    /// <summary>Reads or writes a value by its category name.</summary>
    /// <remarks>
    /// Assigning a value outside the category set throws. Silently widening the dictionary would
    /// make the column's type depend on the order writes happened in, and would break any other
    /// column sharing the same categories.
    /// </remarks>
    public string? this[int index]
    {
        get => _codes[index] == MissingCode ? null : _categories[_codes[index]];
        set
        {
            if (value is null) { _codes[index] = MissingCode; return; }

            if (!_lookup.TryGetValue(value, out var code))
                throw new ArgumentException(
                    $"'{value}' is not one of this column's categories. Use AddCategories first.");

            _codes[index] = code;
        }
    }

    /// <summary>The code for a category, or <c>-1</c> when it is not one.</summary>
    public int CodeOf(string category) => _lookup.GetValueOrDefault(category, MissingCode);

    /// <inheritdoc />
    public override object? GetValue(int index) => this[index];

    /// <inheritdoc />
    public override bool IsMissing(int index) => _codes[index] == MissingCode;

    /// <inheritdoc />
    /// <remarks>
    /// The category dictionary is shared with the result rather than rebuilt. Reordering rows
    /// cannot change what the column may hold, and dropping the categories that happen to have no
    /// rows left would turn a filter into a schema change.
    /// </remarks>
    public override Series Take(IReadOnlyList<int> indices)
    {
        var codes = new int[indices.Count];
        for (var i = 0; i < indices.Count; i++) codes[i] = _codes[indices[i]];
        return new CategoricalSeries(Name, codes, _categories, IsOrdered);
    }

    /// <inheritdoc />
    public override Series Rename(string name)
        => new CategoricalSeries(name, (int[])_codes.Clone(), _categories, IsOrdered);

    /// <summary>Expands back to plain strings.</summary>
    public TextSeries ToText()
    {
        var values = new string?[Length];
        for (var i = 0; i < Length; i++) values[i] = this[i];
        return new TextSeries(Name, values);
    }

    /// <summary>The codes as a numeric column, for feeding a model.</summary>
    /// <remarks>
    /// Missing becomes <see cref="double.NaN"/> rather than -1, so it does not read as a category
    /// ranked below every other one. For an unordered categorical these codes are still nominal —
    /// one-hot encode them rather than handing a learner an arbitrary integer ranking.
    /// </remarks>
    public NumericSeries ToCodes()
    {
        var values = new double[Length];
        for (var i = 0; i < Length; i++)
            values[i] = _codes[i] == MissingCode ? double.NaN : _codes[i];
        return new NumericSeries(Name, values);
    }

    /// <summary>How many rows carry each category, in category order.</summary>
    /// <remarks>
    /// Distinct from the inherited <see cref="Series.ValueCounts"/>, which reports observed values
    /// most-frequent-first. This walks the declared category set in its own order and reports
    /// unused categories as zero rather than omitting them — the point of a fixed category set is
    /// that "none of these" is an answer you can see.
    /// </remarks>
    public IReadOnlyList<(string Category, int Count)> CategoryCounts()
    {
        var counts = new int[_categories.Length];
        foreach (var code in _codes)
            if (code != MissingCode) counts[code]++;

        return [.. _categories.Select((c, i) => (c, counts[i]))];
    }

    /// <summary>A copy with extra categories appended, leaving the existing codes valid.</summary>
    public CategoricalSeries AddCategories(params string[] categories)
    {
        var extended = new List<string>(_categories);
        foreach (var category in categories)
        {
            if (_lookup.ContainsKey(category))
                throw new ArgumentException($"'{category}' is already a category.");
            extended.Add(category);
        }

        return new CategoricalSeries(Name, (int[])_codes.Clone(), [.. extended], IsOrdered);
    }

    /// <summary>A copy with the categories renamed, keeping the codes and their order.</summary>
    public CategoricalSeries RenameCategories(IReadOnlyDictionary<string, string> mapping)
    {
        ArgumentNullException.ThrowIfNull(mapping);
        var renamed = _categories.Select(c => mapping.GetValueOrDefault(c, c)).ToArray();
        return new CategoricalSeries(Name, (int[])_codes.Clone(), renamed, IsOrdered);
    }

    /// <summary>A copy with the categories in a new order, remapping the codes to match.</summary>
    /// <remarks>
    /// This is how an unordered categorical becomes an ordinal one: state the order the values
    /// actually have. Every existing category must appear, so that no row silently becomes missing.
    /// </remarks>
    public CategoricalSeries ReorderCategories(IReadOnlyList<string> categories, bool ordered = true)
    {
        ArgumentNullException.ThrowIfNull(categories);

        var missing = _categories.Except(categories, StringComparer.Ordinal).ToArray();
        if (missing.Length > 0)
            throw new ArgumentException(
                $"The new order omits {string.Join(", ", missing)}, which would drop those rows.");

        var target = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < categories.Count; i++) target[categories[i]] = i;

        var remap = _categories.Select(c => target[c]).ToArray();
        var codes = new int[Length];
        for (var i = 0; i < Length; i++)
            codes[i] = _codes[i] == MissingCode ? MissingCode : remap[_codes[i]];

        return new CategoricalSeries(Name, codes, [.. categories], ordered);
    }

    /// <summary>A copy holding only the categories that actually occur, in their current order.</summary>
    /// <remarks>
    /// The counterpart to <see cref="Take"/> keeping every category: after filtering a frame down,
    /// this is the explicit way to say the unused ones are genuinely gone.
    /// </remarks>
    public CategoricalSeries RemoveUnusedCategories()
    {
        var used = new bool[_categories.Length];
        foreach (var code in _codes)
            if (code != MissingCode) used[code] = true;

        var remap = new int[_categories.Length];
        var kept = new List<string>();

        for (var i = 0; i < _categories.Length; i++)
        {
            if (!used[i]) { remap[i] = MissingCode; continue; }
            remap[i] = kept.Count;
            kept.Add(_categories[i]);
        }

        var codes = new int[Length];
        for (var i = 0; i < Length; i++)
            codes[i] = _codes[i] == MissingCode ? MissingCode : remap[_codes[i]];

        return new CategoricalSeries(Name, codes, [.. kept], IsOrdered);
    }

    /// <summary>
    /// Row indices in category order, missing values last.
    /// </summary>
    /// <remarks>
    /// For an ordered categorical this is the declared ranking; for an unordered one it is the
    /// dictionary order, which is first-seen unless the caller set it. Either way the comparison is
    /// on integers, so this is considerably faster than sorting the equivalent strings.
    /// </remarks>
    public int[] ArgSort(bool descending = false)
    {
        var order = Enumerable.Range(0, Length).ToArray();

        Array.Sort(order, (a, b) =>
        {
            var left = _codes[a];
            var right = _codes[b];

            // Missing sorts last in both directions: it is absent, not extreme.
            if (left == MissingCode) return right == MissingCode ? a.CompareTo(b) : 1;
            if (right == MissingCode) return -1;

            var comparison = descending ? right.CompareTo(left) : left.CompareTo(right);
            return comparison != 0 ? comparison : a.CompareTo(b);
        });

        return order;
    }

    /// <summary>One column per category, holding 1 where the row has that category and 0 elsewhere.</summary>
    /// <param name="dropFirst">
    /// Omit the first category. For a linear model that also fits an intercept, keeping every
    /// category makes the design matrix rank-deficient — the columns sum to a constant — so the
    /// dropped one becomes the baseline the others are measured against.
    /// </param>
    /// <remarks>Rows with a missing value get 0 in every column.</remarks>
    public IReadOnlyList<NumericSeries> OneHot(bool dropFirst = false)
    {
        var start = dropFirst ? 1 : 0;
        var columns = new List<NumericSeries>(_categories.Length - start);

        for (var category = start; category < _categories.Length; category++)
        {
            var values = new double[Length];
            for (var i = 0; i < Length; i++) values[i] = _codes[i] == category ? 1 : 0;
            columns.Add(new NumericSeries($"{Name}_{_categories[category]}", values));
        }

        return columns;
    }
}
