using BlazorML.Core.Domain;

namespace BlazorML.Core.Modules;

/// <summary>
/// Every module the designer can place. Grouped by category; the algorithm groups are keyed by
/// <see cref="MlTask"/> so a Train Model node can validate the trainer it is wired to.
/// </summary>
public static class ModuleCatalog
{
    private static readonly Dictionary<string, ModuleDescriptor> Index;
    public static IReadOnlyList<ModuleDescriptor> All { get; }

    static ModuleCatalog()
    {
        All = BuildAll();
        Index = All.ToDictionary(m => m.Id, StringComparer.OrdinalIgnoreCase);
    }

    public static ModuleDescriptor? Find(string? id) =>
        id is not null && Index.TryGetValue(id, out var m) ? m : null;

    public static IEnumerable<ModuleDescriptor> ByCategory(ModuleCategory category) =>
        All.Where(m => m.Category == category);

    public static IEnumerable<ModuleDescriptor> TrainersFor(MlTask task) =>
        All.Where(m => m.Category == ModuleCategory.Algorithm && m.Task == task);

    public static IEnumerable<ModuleDescriptor> Search(string? term)
    {
        if (string.IsNullOrWhiteSpace(term))
        {
            return All;
        }

        var t = term.Trim();
        return All.Where(m =>
            m.Name.Contains(t, StringComparison.OrdinalIgnoreCase) ||
            m.Description.Contains(t, StringComparison.OrdinalIgnoreCase) ||
            m.Keywords.Any(k => k.Contains(t, StringComparison.OrdinalIgnoreCase)));
    }

    /// <summary>Human label for a category, used by the palette headings.</summary>
    public static string CategoryLabel(ModuleCategory category) => category switch
    {
        ModuleCategory.DataInput => "Data in",
        ModuleCategory.DataTransform => "Transform",
        ModuleCategory.LlmAction => "LLM actions",
        ModuleCategory.Algorithm => "Algorithms",
        ModuleCategory.Training => "Training",
        ModuleCategory.Evaluation => "Score & evaluate",
        ModuleCategory.Script => "Script",
        ModuleCategory.Output => "Data out",
        _ => category.ToString()
    };

    /// <summary>
    /// CSS custom-property suffix for the category's plotter pen colour. Ports, node headers and
    /// palette chips all resolve their colour through this, so the mapping stays in one place.
    /// </summary>
    public static string CategoryPen(ModuleCategory category) => category switch
    {
        ModuleCategory.DataInput => "blue",
        ModuleCategory.DataTransform => "teal",
        ModuleCategory.LlmAction => "violet",
        ModuleCategory.Algorithm => "amber",
        ModuleCategory.Training => "rose",
        ModuleCategory.Evaluation => "lime",
        ModuleCategory.Script => "ink",
        ModuleCategory.Output => "blue",
        _ => "ink"
    };

    // ---------------------------------------------------------------------------------------
    // Shorthand builders keep the catalog readable; every module below is one statement.
    // ---------------------------------------------------------------------------------------

    private static PortSpec In(string name, PortType type, bool required = true, string? desc = null) =>
        new() { Name = name, Type = type, Required = required, Description = desc };

    private static PortSpec Out(string name, PortType type, string? desc = null) =>
        new() { Name = name, Type = type, Description = desc };

    private static ParameterSpec P(string name, string label, ParameterKind kind, string? def = null,
        string? help = null, double? min = null, double? max = null, double? step = null,
        string? visibleWhen = null, string? visibleWhenValue = null, ScriptLanguage? lang = null) =>
        new()
        {
            Name = name, Label = label, Kind = kind, Default = def, Help = help,
            Min = min, Max = max, Step = step,
            VisibleWhen = visibleWhen, VisibleWhenValue = visibleWhenValue, Language = lang
        };

    /// <summary>Same as <see cref="P"/>, for a parameter the module cannot run without.</summary>
    private static ParameterSpec Req(string name, string label, ParameterKind kind,
        string? help = null, ScriptLanguage? lang = null) =>
        new()
        {
            Name = name, Label = label, Kind = kind, Help = help, Language = lang, Required = true
        };

    private static ParameterSpec Choice(string name, string label, string def, string? help,
        params (string Value, string Label)[] options) =>
        new()
        {
            Name = name, Label = label, Kind = ParameterKind.Choice, Default = def, Help = help,
            Choices = options.Select(o => new ChoiceOption(o.Value, o.Label)).ToList()
        };

    /// <summary>Learning-rate / iteration parameters shared by the boosted-tree trainers.</summary>
    private static ParameterSpec[] TreeParams(int leaves = 20, int trees = 100) =>
    [
        P("numberOfLeaves", "Leaves per tree", ParameterKind.Integer, leaves.ToString(),
            "More leaves fit finer patterns and overfit sooner.", 2, 512),
        P("numberOfTrees", "Number of trees", ParameterKind.Integer, trees.ToString(),
            "More trees improve accuracy until the gain flattens out.", 1, 2000),
        P("minimumExampleCountPerLeaf", "Minimum rows per leaf", ParameterKind.Integer, "10",
            "Raise this on small or noisy data to keep leaves meaningful.", 1, 1000),
        P("learningRate", "Learning rate", ParameterKind.Number, "0.2",
            "Smaller steps train slower but land more precisely.", 0.001, 1, 0.001)
    ];

    private static ParameterSpec[] LinearParams() =>
    [
        P("l1Regularization", "L1 regularisation", ParameterKind.Number, "0.01",
            "Pushes weak feature weights to zero, which thins out the model.", 0, 10, 0.001),
        P("l2Regularization", "L2 regularisation", ParameterKind.Number, "0.01",
            "Keeps weights small so no single feature dominates.", 0, 10, 0.001),
        P("maximumNumberOfIterations", "Maximum iterations", ParameterKind.Integer, "100", null, 1, 10000)
    ];

    private static readonly ParameterSpec LabelColumn =
        Req("labelColumn", "Label column", ParameterKind.ColumnName, "The column the model learns to predict.");

    private static readonly ParameterSpec FeatureColumns =
        P("featureColumns", "Feature columns", ParameterKind.ColumnList, null,
            "Leave empty to use every column except the label.");

    private static List<ModuleDescriptor> BuildAll()
    {
        var list = new List<ModuleDescriptor>();
        list.AddRange(DataInput());
        list.AddRange(Transforms());
        list.AddRange(LlmActions());
        list.AddRange(Algorithms());
        list.AddRange(Training());
        list.AddRange(Evaluation());
        list.AddRange(Scripts());
        list.AddRange(Outputs());
        return list;
    }

    // ---------------------------------------------------------------------------- Data in

    private static IEnumerable<ModuleDescriptor> DataInput() =>
    [
        new()
        {
            Id = "data.dataset",
            Name = "Import Dataset",
            Category = ModuleCategory.DataInput,
            Description = "Reads a dataset already registered in the workspace.",
            Keywords = ["import", "csv", "sumber", "data"],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("datasetId", "Dataset", ParameterKind.DatasetRef,
                    "Pick from the datasets you have imported."),
                P("sampleRows", "Row limit", ParameterKind.Integer, "0",
                    "Cap the rows read while you are iterating. 0 reads everything.", 0, 10_000_000)
            ]
        },
        new()
        {
            Id = "data.sql",
            Name = "Import from SQL",
            Category = ModuleCategory.DataInput,
            Description = "Runs a query against a database and treats the result as a dataset.",
            Keywords = ["sql", "database", "query", "postgres", "mysql"],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Choice("provider", "Database", "Sqlite", null,
                    ("Sqlite", "SQLite"), ("SqlServer", "SQL Server"), ("MySql", "MySQL"), ("PostgreSql", "PostgreSQL")),
                Req("connectionString", "Connection string", ParameterKind.Secret,
                    "Saved with the experiment. Use a read-only account."),
                P("query", "Query", ParameterKind.MultilineText, "SELECT * FROM my_table",
                    "Only the first result set is read.")
            ]
        },
        new()
        {
            Id = "data.web",
            Name = "Import from Web",
            Category = ModuleCategory.DataInput,
            Description = "Fetches a CSV or JSON document over HTTP.",
            Keywords = ["http", "url", "rest", "api", "web service"],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("url", "URL", ParameterKind.Text, "Must return CSV or a JSON array of objects."),
                Choice("format", "Format", "Csv", null, ("Csv", "CSV"), ("Json", "JSON")),
                P("headers", "Request headers", ParameterKind.MultilineText, null,
                    "One per line, as Name: value.")
            ]
        },
        new()
        {
            Id = "data.manual",
            Name = "Enter Data Manually",
            Category = ModuleCategory.DataInput,
            Description = "A small table typed straight into the experiment, handy for lookups.",
            Keywords = ["manual", "inline", "paste", "tabel"],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                P("csv", "Rows", ParameterKind.MultilineText, "col1,col2\n1,a\n2,b",
                    "First line is the header.")
            ]
        }
    ];

    // -------------------------------------------------------------------------- Transform

    private static IEnumerable<ModuleDescriptor> Transforms() =>
    [
        new()
        {
            Id = "tf.selectColumns",
            Name = "Select Columns",
            Category = ModuleCategory.DataTransform,
            Description = "Keeps or drops columns before anything downstream sees them.",
            Keywords = ["project", "kolom", "drop", "keep"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                P("columns", "Columns", ParameterKind.ColumnList, null, null),
                Choice("mode", "Action", "keep", null, ("keep", "Keep selected"), ("drop", "Drop selected"))
            ]
        },
        new()
        {
            Id = "tf.editMetadata",
            Name = "Edit Metadata",
            Category = ModuleCategory.DataTransform,
            Description = "Renames a column or changes how its values are read.",
            Keywords = ["rename", "cast", "tipe", "metadata"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("column", "Column", ParameterKind.ColumnName),
                P("newName", "Rename to", ParameterKind.Text, null, "Leave empty to keep the current name."),
                Choice("dataType", "Read values as", "keep", null,
                    ("keep", "Leave unchanged"), ("Numeric", "Number"), ("Categorical", "Category"),
                    ("Text", "Text"), ("Boolean", "Boolean"), ("DateTime", "Date and time"))
            ]
        },
        new()
        {
            Id = "tf.cleanMissing",
            Name = "Clean Missing Data",
            Category = ModuleCategory.DataTransform,
            Description = "Fills or removes gaps so trainers do not choke on empty cells.",
            Keywords = ["missing", "null", "impute", "kosong", "bersih"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                P("columns", "Columns", ParameterKind.ColumnList, null, "Empty means every column."),
                Choice("strategy", "Replace gaps with", "mean", null,
                    ("mean", "Column mean"), ("median", "Column median"), ("mode", "Most frequent value"),
                    ("constant", "A fixed value"), ("dropRow", "Drop the row"), ("dropColumn", "Drop the column")),
                P("constant", "Value", ParameterKind.Text, "0", null,
                    visibleWhen: "strategy", visibleWhenValue: "constant")
            ]
        },
        new()
        {
            Id = "tf.removeDuplicates",
            Name = "Remove Duplicate Rows",
            Category = ModuleCategory.DataTransform,
            Description = "Drops rows that repeat across the chosen key columns.",
            Keywords = ["duplicate", "distinct", "unik", "ganda"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                P("keyColumns", "Key columns", ParameterKind.ColumnList, null,
                    "Empty compares the whole row."),
                Choice("keep", "Row to keep", "first", null, ("first", "First"), ("last", "Last"))
            ]
        },
        new()
        {
            Id = "tf.filterRows",
            Name = "Filter Rows",
            Category = ModuleCategory.DataTransform,
            Description = "Keeps only the rows that satisfy a comparison.",
            Keywords = ["filter", "where", "saring", "kondisi"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("column", "Column", ParameterKind.ColumnName),
                Choice("op", "Condition", "gt", null,
                    ("eq", "equals"), ("ne", "does not equal"), ("gt", "is greater than"),
                    ("gte", "is at least"), ("lt", "is less than"), ("lte", "is at most"),
                    ("contains", "contains"), ("notEmpty", "is not empty")),
                Req("value", "Value", ParameterKind.Text)
            ]
        },
        new()
        {
            Id = "tf.splitData",
            Name = "Split Data",
            Category = ModuleCategory.DataTransform,
            Description = "Divides rows into a training set and a test set.",
            Keywords = ["split", "train", "test", "bagi", "holdout"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Left", PortType.Dataset, "Training rows"), Out("Right", PortType.Dataset, "Test rows")],
            Parameters =
            [
                P("fraction", "Rows in the first output", ParameterKind.Number, "0.8",
                    "0.8 sends 80% to training and 20% to test.", 0.05, 0.95, 0.05),
                P("stratifyColumn", "Stratify by", ParameterKind.ColumnName, null,
                    "Keeps the class balance identical in both halves."),
                P("randomSeed", "Random seed", ParameterKind.Integer, "42",
                    "Fixing the seed makes the split reproducible.")
            ]
        },
        new()
        {
            Id = "tf.join",
            Name = "Join Datasets",
            Category = ModuleCategory.DataTransform,
            Description = "Matches rows from two datasets on a shared key.",
            Keywords = ["join", "merge", "gabung", "lookup"],
            Inputs = [In("Left", PortType.Dataset), In("Right", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("leftKey", "Left key", ParameterKind.ColumnName),
                Req("rightKey", "Right key", ParameterKind.ColumnName),
                Choice("kind", "Keep rows", "inner", null,
                    ("inner", "Matching in both"), ("left", "All from the left"), ("full", "All from both"))
            ]
        },
        new()
        {
            Id = "tf.groupBy",
            Name = "Group and Summarise",
            Category = ModuleCategory.DataTransform,
            Description = "Collapses rows into one per group with an aggregate.",
            Keywords = ["group", "aggregate", "sum", "rata", "ringkas"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("groupBy", "Group by", ParameterKind.ColumnList),
                Req("aggregateColumn", "Summarise", ParameterKind.ColumnName),
                Choice("aggregate", "Using", "mean", null,
                    ("mean", "Average"), ("sum", "Sum"), ("min", "Minimum"), ("max", "Maximum"), ("count", "Count"))
            ]
        },
        new()
        {
            Id = "tf.normalize",
            Name = "Normalize Data",
            Category = ModuleCategory.DataTransform,
            Description = "Rescales numeric columns onto a comparable range.",
            Keywords = ["normalize", "scale", "standard", "minmax", "skala"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                P("columns", "Columns", ParameterKind.ColumnList, null, "Empty means every numeric column."),
                Choice("method", "Method", "minmax", null,
                    ("minmax", "Min-max to 0..1"), ("zscore", "Z-score"), ("logscale", "Log scale"),
                    ("binning", "Equal-width bins")),
                P("bins", "Bins", ParameterKind.Integer, "10", null, 2, 200,
                    visibleWhen: "method", visibleWhenValue: "binning")
            ]
        },
        new()
        {
            Id = "tf.oneHot",
            Name = "Encode Categories",
            Category = ModuleCategory.DataTransform,
            Description = "Turns category labels into numbers a trainer can use.",
            Keywords = ["onehot", "categorical", "encode", "kategori", "dummy"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("columns", "Columns", ParameterKind.ColumnList),
                Choice("method", "Encoding", "oneHot", null,
                    ("oneHot", "One column per value"), ("ordinal", "One numbered column"),
                    ("hash", "Hashed into fixed width")),
                P("hashBits", "Hash width in bits", ParameterKind.Integer, "16", null, 4, 30,
                    visibleWhen: "method", visibleWhenValue: "hash")
            ]
        },
        new()
        {
            Id = "tf.featurizeText",
            Name = "Featurize Text",
            Category = ModuleCategory.DataTransform,
            Description = "Converts free text into numeric features via n-grams.",
            Keywords = ["text", "nlp", "ngram", "tfidf", "teks"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("column", "Text column", ParameterKind.ColumnName),
                P("outputColumn", "Output column", ParameterKind.Text, "TextFeatures"),
                P("wordGramLength", "Word n-gram length", ParameterKind.Integer, "2", null, 1, 5),
                P("charGramLength", "Character n-gram length", ParameterKind.Integer, "3", null, 0, 8),
                P("removeStopWords", "Remove stop words", ParameterKind.Boolean, "true")
            ]
        },
        new()
        {
            Id = "tf.featureSelection",
            Name = "Select Best Features",
            Category = ModuleCategory.DataTransform,
            Description = "Ranks features against the label and keeps the strongest.",
            Keywords = ["feature selection", "importance", "fitur", "seleksi"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset), Out("Scores", PortType.Metrics)],
            Parameters =
            [
                LabelColumn,
                Choice("method", "Ranking", "correlation", null,
                    ("correlation", "Correlation with the label"), ("mutualInformation", "Mutual information"),
                    ("variance", "Variance")),
                P("topN", "Features to keep", ParameterKind.Integer, "10", null, 1, 500)
            ]
        }
    ];

    // ------------------------------------------------------------------------- LLM actions

    private static ParameterSpec[] LlmCommon() =>
    [
        Choice("provider", "Model provider", "default", "Uses the workspace default unless you override it.",
            ("default", "Workspace default"), ("OpenAI", "OpenAI"), ("Anthropic", "Anthropic"),
            ("Gemini", "Gemini"), ("Ollama", "Ollama")),
        P("temperature", "Temperature", ParameterKind.Number, "0.2",
            "Low values keep the output steady across runs.", 0, 2, 0.05),
        P("batchSize", "Rows per request", ParameterKind.Integer, "20",
            "Larger batches are cheaper but slower to fail.", 1, 200),
        P("maxRows", "Row limit", ParameterKind.Integer, "0", "0 processes every row.", 0, 1_000_000)
    ];

    private static IEnumerable<ModuleDescriptor> LlmActions() =>
    [
        new()
        {
            Id = "llm.format",
            Name = "LLM Format",
            Category = ModuleCategory.LlmAction,
            Description = "Rewrites messy values into one consistent shape.",
            Keywords = ["format", "normalise", "rapikan", "llm", "ai"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("column", "Column", ParameterKind.ColumnName),
                P("outputColumn", "Write to", ParameterKind.Text, null, "Empty overwrites the source column."),
                P("targetFormat", "Target format", ParameterKind.Text, "ISO date (YYYY-MM-DD)",
                    "Describe the shape you want in plain words."),
                .. LlmCommon()
            ]
        },
        new()
        {
            Id = "llm.cleanse",
            Name = "LLM Cleanse",
            Category = ModuleCategory.LlmAction,
            Description = "Repairs typos, stray punctuation and inconsistent casing.",
            Keywords = ["clean", "typo", "bersih", "llm"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("column", "Column", ParameterKind.ColumnName),
                P("outputColumn", "Write to", ParameterKind.Text),
                P("rules", "Rules", ParameterKind.MultilineText,
                    "Fix spelling, trim whitespace, use title case for names.",
                    "One instruction per line."),
                .. LlmCommon()
            ]
        },
        new()
        {
            Id = "llm.sentiment",
            Name = "LLM Sentiment",
            Category = ModuleCategory.LlmAction,
            Description = "Labels each row positive, neutral or negative with a confidence score.",
            Keywords = ["sentiment", "opini", "review", "llm"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("column", "Text column", ParameterKind.ColumnName),
                P("outputColumn", "Write to", ParameterKind.Text, "Sentiment"),
                P("includeScore", "Add a confidence column", ParameterKind.Boolean, "true"),
                P("labels", "Labels", ParameterKind.Text, "positive,neutral,negative"),
                .. LlmCommon()
            ]
        },
        new()
        {
            Id = "llm.classify",
            Name = "LLM Classify",
            Category = ModuleCategory.LlmAction,
            Description = "Sorts rows into categories you define, no training data needed.",
            Keywords = ["classify", "zero shot", "kategori", "llm"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("column", "Column", ParameterKind.ColumnName),
                P("outputColumn", "Write to", ParameterKind.Text, "Category"),
                P("categories", "Categories", ParameterKind.MultilineText, "billing\nshipping\ntechnical\nother",
                    "One per line."),
                .. LlmCommon()
            ]
        },
        new()
        {
            Id = "llm.extract",
            Name = "LLM Extract",
            Category = ModuleCategory.LlmAction,
            Description = "Pulls named fields out of unstructured text into new columns.",
            Keywords = ["extract", "entity", "ekstrak", "llm", "ner"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("column", "Column", ParameterKind.ColumnName),
                P("fields", "Fields to extract", ParameterKind.MultilineText, "name\namount\ndue_date",
                    "One per line. Each becomes a column."),
                .. LlmCommon()
            ]
        },
        new()
        {
            Id = "llm.prompt",
            Name = "LLM Custom Prompt",
            Category = ModuleCategory.LlmAction,
            Description = "Runs your own prompt over every row.",
            Keywords = ["prompt", "custom", "llm", "ai"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Dataset", PortType.Dataset)],
            Parameters =
            [
                P("prompt", "Prompt", ParameterKind.MultilineText,
                    "Summarise the following review in one sentence:\n\n{{review}}",
                    "Wrap a column name in {{ }} to insert its value."),
                P("outputColumn", "Write to", ParameterKind.Text, "LlmOutput"),
                P("systemPrompt", "System prompt", ParameterKind.MultilineText, null,
                    "Sets the persona and constraints for every row."),
                .. LlmCommon()
            ]
        }
    ];

    // -------------------------------------------------------------------------- Algorithms

    private static ModuleDescriptor Algo(string id, string name, MlTask task, string description,
        string[] keywords, ParameterSpec[] parameters) => new()
    {
        Id = id,
        Name = name,
        Category = ModuleCategory.Algorithm,
        Task = task,
        Description = description,
        Keywords = keywords,
        Outputs = [Out("Untrained model", PortType.UntrainedModel)],
        Parameters = parameters
    };

    private static IEnumerable<ModuleDescriptor> Algorithms() =>
    [
        // Binary classification
        Algo("algo.bin.logistic", "Logistic Regression", MlTask.BinaryClassification,
            "A fast linear baseline with weights you can read directly.",
            ["binary", "logistic", "linear", "klasifikasi"], LinearParams()),
        Algo("algo.bin.sdca", "SDCA Logistic Regression", MlTask.BinaryClassification,
            "Stochastic dual coordinate ascent; scales well to wide, sparse data.",
            ["sdca", "binary", "linear"], LinearParams()),
        Algo("algo.bin.fastTree", "Boosted Decision Tree", MlTask.BinaryClassification,
            "Gradient-boosted trees; usually the strongest option on tabular data.",
            ["fasttree", "boost", "tree", "pohon"], TreeParams()),
        Algo("algo.bin.fastForest", "Decision Forest", MlTask.BinaryClassification,
            "Bagged trees; steadier than boosting when the data is noisy.",
            ["forest", "random", "bagging"], TreeParams(20, 100)),
        Algo("algo.bin.lightGbm", "LightGBM", MlTask.BinaryClassification,
            "Leaf-wise boosting that trains quickly on large tables.",
            ["lightgbm", "gbm", "boost"], TreeParams(31, 100)),
        Algo("algo.bin.perceptron", "Averaged Perceptron", MlTask.BinaryClassification,
            "An online linear learner that handles streaming updates.",
            ["perceptron", "online", "linear"],
            [P("learningRate", "Learning rate", ParameterKind.Number, "0.1", null, 0.001, 1, 0.001),
             P("numberOfIterations", "Passes over the data", ParameterKind.Integer, "10", null, 1, 1000)]),
        Algo("algo.bin.svm", "Linear SVM", MlTask.BinaryClassification,
            "Maximises the margin between the two classes.",
            ["svm", "margin", "linear"],
            [P("numberOfIterations", "Passes over the data", ParameterKind.Integer, "10", null, 1, 1000),
             P("lambda", "Regularisation", ParameterKind.Number, "0.001", null, 0, 1, 0.0001)]),

        // Multiclass
        Algo("algo.multi.sdca", "SDCA Maximum Entropy", MlTask.MulticlassClassification,
            "Softmax over all classes at once; a solid multiclass default.",
            ["multiclass", "softmax", "sdca"], LinearParams()),
        Algo("algo.multi.lbfgs", "L-BFGS Maximum Entropy", MlTask.MulticlassClassification,
            "Batch optimiser that converges tightly on medium-sized data.",
            ["lbfgs", "multiclass", "maxent"], LinearParams()),
        Algo("algo.multi.oneVsAll", "One-vs-All Trees", MlTask.MulticlassClassification,
            "Trains one boosted-tree classifier per class.",
            ["ova", "one versus all", "multiclass"], TreeParams()),
        Algo("algo.multi.lightGbm", "LightGBM Multiclass", MlTask.MulticlassClassification,
            "Leaf-wise boosting extended to many classes.",
            ["lightgbm", "multiclass"], TreeParams(31, 100)),
        // ML.NET's implementation binarises every feature — it only looks at whether a value is
        // present, never how large it is. Said plainly here because on ordinary numeric columns
        // it lands on chance, and a description promising "counts frequencies" would leave a user
        // blaming their data for a mismatch between the trainer and the features.
        Algo("algo.multi.naiveBayes", "Naive Bayes", MlTask.MulticlassClassification,
            "For presence-or-absence features such as text n-grams; it ignores how large a value is.",
            ["bayes", "naive", "text", "teks", "ngram"], []),

        // Regression
        Algo("algo.reg.sdca", "Linear Regression", MlTask.Regression,
            "Fits a straight-line relationship between features and the target.",
            ["linear", "regresi", "sdca"], LinearParams()),
        Algo("algo.reg.ols", "Ordinary Least Squares", MlTask.Regression,
            "Closed-form fit that also reports p-values per feature.",
            ["ols", "least squares", "statistik"],
            [P("l2Regularization", "L2 regularisation", ParameterKind.Number, "0.000001", null, 0, 1, 0.000001)]),
        Algo("algo.reg.fastTree", "Boosted Decision Tree Regression", MlTask.Regression,
            "Gradient-boosted trees for continuous targets.",
            ["fasttree", "boost", "regresi"], TreeParams()),
        Algo("algo.reg.fastForest", "Decision Forest Regression", MlTask.Regression,
            "Averaged trees; resistant to outliers in the target.",
            ["forest", "regresi"], TreeParams()),
        Algo("algo.reg.lightGbm", "LightGBM Regression", MlTask.Regression,
            "Fast boosting for large regression problems.",
            ["lightgbm", "regresi"], TreeParams(31, 100)),
        Algo("algo.reg.tweedie", "Tweedie Regression", MlTask.Regression,
            "For targets that are mostly zero with a long tail, like claims or spend.",
            ["tweedie", "insurance", "skewed"], TreeParams()),
        Algo("algo.reg.poisson", "Poisson Regression", MlTask.Regression,
            "For counts: visits, orders, defects per unit.",
            ["poisson", "count", "cacah"], LinearParams()),
        Algo("algo.reg.sgd", "Online Gradient Descent", MlTask.Regression,
            "Updates weights row by row, so it suits streaming data.",
            ["sgd", "online", "gradient"],
            [P("learningRate", "Learning rate", ParameterKind.Number, "0.1", null, 0.0001, 1, 0.0001),
             P("numberOfIterations", "Passes over the data", ParameterKind.Integer, "20", null, 1, 1000)]),

        // Clustering
        Algo("algo.cluster.kmeans", "K-Means Clustering", MlTask.Clustering,
            "Groups rows around a fixed number of centroids.",
            ["kmeans", "cluster", "segmentasi", "grup"],
            [P("numberOfClusters", "Clusters", ParameterKind.Integer, "3",
                 "Set this to how many groups you expect to find.", 2, 100),
             P("maximumNumberOfIterations", "Maximum iterations", ParameterKind.Integer, "100", null, 1, 10000),
             Choice("initialization", "Centroid start", "kmeansPlusPlus", null,
                 ("kmeansPlusPlus", "K-Means++"), ("random", "Random"))]),

        // Anomaly detection
        Algo("algo.anomaly.pca", "PCA Anomaly Detection", MlTask.AnomalyDetection,
            "Flags rows that do not fit the principal subspace of normal data.",
            ["anomaly", "outlier", "pca", "anomali"],
            [P("rank", "Components", ParameterKind.Integer, "5",
                 "How many dimensions describe normal behaviour.", 1, 100),
             P("oversampling", "Oversampling", ParameterKind.Integer, "20", null, 1, 100)]),
        Algo("algo.anomaly.spike", "Spike Detection", MlTask.AnomalyDetection,
            "Finds sudden jumps in a time-ordered series.",
            ["spike", "timeseries", "lonjakan"],
            [Req("column", "Value column", ParameterKind.ColumnName),
             P("confidence", "Confidence", ParameterKind.Number, "95", null, 50, 99.9, 0.5),
             P("pvalueHistoryLength", "History window", ParameterKind.Integer, "30", null, 5, 1000)]),

        // Recommendation
        Algo("algo.rec.matrixFactorization", "Matrix Factorization", MlTask.Recommendation,
            "Learns latent taste vectors from user-item ratings.",
            ["recommend", "collaborative", "rekomendasi", "rating"],
            [Req("userColumn", "User column", ParameterKind.ColumnName),
             Req("itemColumn", "Item column", ParameterKind.ColumnName),
             P("approximationRank", "Latent factors", ParameterKind.Integer, "32",
                 "More factors capture subtler taste but need more ratings.", 4, 512),
             P("numberOfIterations", "Iterations", ParameterKind.Integer, "20", null, 1, 500),
             P("learningRate", "Learning rate", ParameterKind.Number, "0.1", null, 0.001, 1, 0.001)]),

        // Vision and NLP. Present in the catalogue whatever the build, so they are discoverable
        // and documented; the executor explains how to enable them when they are not built in.
        Algo("algo.vision.imageClassification", "Image Classification", MlTask.ImageClassification,
            "Fine-tunes a pretrained image model on your own labelled pictures.",
            ["image", "vision", "gambar", "citra", "foto", "transfer learning"],
            [Req("imagePathColumn", "Image path column", ParameterKind.ColumnName,
                 "The column holding each image's file name or path."),
             P("imageFolder", "Image folder", ParameterKind.Text, null,
                 "Paths in the column are resolved against this folder."),
             P("epochs", "Passes over the images", ParameterKind.Integer, "50", null, 1, 500),
             P("batchSize", "Images per batch", ParameterKind.Integer, "10", null, 1, 256)]),

        Algo("algo.nlp.textClassification", "Text Classification", MlTask.TextClassification,
            "Fine-tunes a pretrained language model to sort text into your own categories.",
            ["nlp", "text", "teks", "bert", "klasifikasi teks", "transfer learning"],
            [Req("textColumn", "Text column", ParameterKind.ColumnName,
                 "The column holding the text to classify."),
             P("epochs", "Passes over the data", ParameterKind.Integer, "3",
                 "Two or three is usually enough when fine-tuning.", 1, 50),
             P("batchSize", "Rows per batch", ParameterKind.Integer, "32", null, 1, 256)]),

        // Forecasting
        Algo("algo.forecast.ssa", "SSA Forecasting", MlTask.Forecasting,
            "Singular spectrum analysis for seasonal series.",
            ["forecast", "timeseries", "ramalan", "musiman"],
            [Req("column", "Value column", ParameterKind.ColumnName),
             P("horizon", "Periods ahead", ParameterKind.Integer, "7", null, 1, 365),
             P("windowSize", "Window size", ParameterKind.Integer, "14",
                 "Should span at least one full season.", 2, 1000),
             P("seriesLength", "Training length", ParameterKind.Integer, "60", null, 4, 100000)])
    ];

    // ---------------------------------------------------------------------------- Training

    private static IEnumerable<ModuleDescriptor> Training() =>
    [
        new()
        {
            Id = "train.model",
            Name = "Train Model",
            Category = ModuleCategory.Training,
            Description = "Fits the connected algorithm to the training rows.",
            Keywords = ["train", "fit", "latih"],
            Inputs = [In("Algorithm", PortType.UntrainedModel), In("Dataset", PortType.Dataset)],
            Outputs = [Out("Trained model", PortType.TrainedModel)],
            Parameters = [LabelColumn, FeatureColumns]
        },
        new()
        {
            Id = "train.clustering",
            Name = "Train Clustering Model",
            Category = ModuleCategory.Training,
            Description = "Fits a clustering algorithm; no label required.",
            Keywords = ["cluster", "unsupervised", "latih"],
            Inputs = [In("Algorithm", PortType.UntrainedModel), In("Dataset", PortType.Dataset)],
            Outputs = [Out("Trained model", PortType.TrainedModel)],
            Parameters = [FeatureColumns]
        },
        new()
        {
            Id = "train.autoML",
            Name = "AutoML",
            Category = ModuleCategory.Training,
            Description = "Searches algorithms and settings for you and keeps the best.",
            Keywords = ["automl", "auto", "search", "otomatis"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Outputs = [Out("Trained model", PortType.TrainedModel), Out("Leaderboard", PortType.Metrics)],
            Parameters =
            [
                Choice("task", "Task", "BinaryClassification", "What kind of answer you need.",
                    ("BinaryClassification", "Yes or no"), ("MulticlassClassification", "One of several classes"),
                    ("Regression", "A number"), ("Recommendation", "Recommendations"),
                    ("ImageClassification", "A category from a picture"),
                    ("TextClassification", "A category from text")),
                LabelColumn,
                P("inputColumn", "Image or text column", ParameterKind.ColumnName, null,
                    "For the image and text tasks: the column holding the file path, or the text."),
                P("imageFolder", "Image folder", ParameterKind.Text, null,
                    "Image paths are resolved against this folder."),
                P("maxSeconds", "Time budget in seconds", ParameterKind.Integer, "60",
                    "The search stops when the budget runs out and keeps the best model so far.", 10, 7200),
                P("validationFraction", "Validation split", ParameterKind.Number, "0.2", null, 0.05, 0.5, 0.05),
                Choice("optimize", "Optimise for", "auto", null,
                    ("auto", "Task default"), ("accuracy", "Accuracy"), ("auc", "AUC"),
                    ("f1", "F1"), ("rSquared", "R squared"), ("rmse", "RMSE"))
            ]
        },
        new()
        {
            Id = "train.sweep",
            Name = "Tune Hyperparameters",
            Category = ModuleCategory.Training,
            Description = "Trains the algorithm repeatedly across a parameter sweep.",
            Keywords = ["sweep", "tuning", "grid", "hyperparameter"],
            Inputs = [In("Algorithm", PortType.UntrainedModel), In("Dataset", PortType.Dataset)],
            Outputs = [Out("Best model", PortType.TrainedModel), Out("Sweep results", PortType.Metrics)],
            Parameters =
            [
                LabelColumn,
                Choice("mode", "Search", "random", null,
                    ("random", "Random sample"), ("grid", "Every combination")),
                P("iterations", "Runs", ParameterKind.Integer, "12", null, 2, 200),
                P("sweep", "Parameter grid", ParameterKind.MultilineText,
                    "numberOfLeaves: 10, 20, 40\nlearningRate: 0.05, 0.1, 0.2",
                    "One parameter per line, values separated by commas.")
            ]
        },
        new()
        {
            Id = "train.crossValidate",
            Name = "Cross Validate",
            Category = ModuleCategory.Training,
            Description = "Trains across k folds and reports the spread, not a single lucky score.",
            Keywords = ["cross validation", "kfold", "validasi"],
            Inputs = [In("Algorithm", PortType.UntrainedModel), In("Dataset", PortType.Dataset)],
            Outputs = [Out("Metrics", PortType.Metrics), Out("Trained model", PortType.TrainedModel)],
            Parameters =
            [
                LabelColumn,
                P("folds", "Folds", ParameterKind.Integer, "5", null, 2, 20)
            ]
        }
    ];

    // ------------------------------------------------------------------ Score and evaluate

    private static IEnumerable<ModuleDescriptor> Evaluation() =>
    [
        new()
        {
            Id = "score.model",
            Name = "Score Model",
            Category = ModuleCategory.Evaluation,
            Description = "Runs the trained model over rows and appends its predictions.",
            Keywords = ["score", "predict", "prediksi", "inference"],
            Inputs = [In("Trained model", PortType.TrainedModel), In("Dataset", PortType.Dataset)],
            Outputs = [Out("Scored dataset", PortType.Dataset)],
            Parameters =
            [
                P("appendOriginal", "Keep the input columns", ParameterKind.Boolean, "true")
            ]
        },
        new()
        {
            Id = "score.evaluate",
            Name = "Evaluate Model",
            Category = ModuleCategory.Evaluation,
            Description = "Compares predictions with the truth and reports the metrics that matter for the task.",
            Keywords = ["evaluate", "metrics", "roc", "confusion", "evaluasi"],
            Inputs = [In("Scored dataset", PortType.Dataset), In("Second model", PortType.Dataset, required: false)],
            Outputs = [Out("Metrics", PortType.Metrics)],
            Parameters =
            [
                Choice("task", "Task", "auto", "Inferred from the model unless you pin it.",
                    ("auto", "Detect automatically"), ("BinaryClassification", "Binary classification"),
                    ("MulticlassClassification", "Multiclass classification"), ("Regression", "Regression"),
                    ("Clustering", "Clustering"), ("Recommendation", "Recommendation")),
                P("positiveClass", "Positive class", ParameterKind.Text, null,
                    "Which label counts as a hit when reporting precision and recall.")
            ]
        },
        new()
        {
            Id = "score.featureImportance",
            Name = "Feature Importance",
            Category = ModuleCategory.Evaluation,
            Description = "Shuffles each feature in turn to measure how much the model relies on it.",
            Keywords = ["importance", "pfi", "explain", "penting"],
            Inputs = [In("Trained model", PortType.TrainedModel), In("Dataset", PortType.Dataset)],
            Outputs = [Out("Importance", PortType.Metrics)],
            Parameters =
            [
                LabelColumn,
                P("permutations", "Shuffles per feature", ParameterKind.Integer, "5", null, 1, 50)
            ]
        }
    ];

    // ------------------------------------------------------------------------------ Script

    private static ModuleDescriptor Script(string id, string name, ScriptLanguage lang, string starter,
        string description, string[] keywords) => new()
    {
        Id = id,
        Name = name,
        Category = ModuleCategory.Script,
        Description = description,
        Keywords = keywords,
        Inputs = [In("Dataset 1", PortType.Dataset, required: false), In("Dataset 2", PortType.Dataset, required: false)],
        Outputs = [Out("Dataset", PortType.Dataset)],
        Parameters =
        [
            P("code", "Script", ParameterKind.Code, starter, null, lang: lang),
            P("timeoutSeconds", "Timeout in seconds", ParameterKind.Integer, "120", null, 5, 3600)
        ]
    };

    private static IEnumerable<ModuleDescriptor> Scripts() =>
    [
        Script("script.python", "Execute Python", ScriptLanguage.Python,
            "# dataset1 is a list of dicts. Return the rows you want downstream.\ndef run(dataset1, dataset2):\n    return dataset1\n",
            "Runs Python through Python.NET against the incoming rows.",
            ["python", "script", "pandas"]),
        Script("script.r", "Execute R", ScriptLanguage.R,
            "# dataset1 is a data.frame. Assign the result to `result`.\nresult <- dataset1\n",
            "Runs an R script against the incoming rows.",
            ["r", "script", "statistik"]),
        Script("script.csharp", "Execute C#", ScriptLanguage.CSharp,
            "// rows is List<Dictionary<string, object>>. Return the rows to pass on.\nreturn rows;\n",
            "Runs a C# expression over the rows in-process.",
            ["csharp", "script", "roslyn"]),
        Script("script.js", "Execute JavaScript", ScriptLanguage.JavaScript,
            "// rows is an array of objects. Return the rows to pass on.\nreturn rows;\n",
            "Runs a JavaScript function over the rows.",
            ["javascript", "js", "script"])
    ];

    // ---------------------------------------------------------------------------- Data out

    private static IEnumerable<ModuleDescriptor> Outputs() =>
    [
        new()
        {
            Id = "out.saveDataset",
            Name = "Save Dataset",
            Category = ModuleCategory.Output,
            Description = "Writes the rows back to the workspace as a new dataset.",
            Keywords = ["save", "export", "simpan"],
            Inputs = [In("Dataset", PortType.Dataset)],
            Parameters =
            [
                Req("name", "Dataset name", ParameterKind.Text),
                Choice("format", "Format", "Csv", null, ("Csv", "CSV"), ("Tsv", "TSV"), ("Json", "JSON"))
            ]
        },
        new()
        {
            Id = "out.registerModel",
            Name = "Register Model",
            Category = ModuleCategory.Output,
            Description = "Saves the trained model so it can be versioned and published.",
            Keywords = ["register", "publish", "model", "simpan"],
            Inputs = [In("Trained model", PortType.TrainedModel)],
            Parameters =
            [
                Req("name", "Model name", ParameterKind.Text),
                P("description", "Description", ParameterKind.MultilineText),
                P("publish", "Publish as a web service", ParameterKind.Boolean, "false",
                    "Creates a REST endpoint and an API key once the run succeeds.")
            ]
        }
    ];
}
