namespace BlazorML.Core.Domain;

/// <summary>Family of ML problem a trainer or experiment addresses.</summary>
public enum MlTask
{
    None = 0,
    BinaryClassification = 1,
    MulticlassClassification = 2,
    Regression = 3,
    Clustering = 4,
    AnomalyDetection = 5,
    Recommendation = 6,
    Forecasting = 7,
    TextClassification = 8,
    ImageClassification = 9
}

/// <summary>What travels along an edge in the designer graph.</summary>
public enum PortType
{
    Dataset = 0,
    UntrainedModel = 1,
    TrainedModel = 2,
    Metrics = 3,
    Transform = 4,
    Any = 99
}

public enum ModuleCategory
{
    DataInput = 0,
    DataTransform = 1,
    LlmAction = 2,
    Algorithm = 3,
    Training = 4,
    Evaluation = 5,
    Script = 6,
    Output = 7
}

public enum ParameterKind
{
    Text = 0,
    Number = 1,
    Integer = 2,
    Boolean = 3,

    /// <summary>Pick one of the options declared on the parameter itself.</summary>
    Choice = 4,

    ColumnName = 5,
    ColumnList = 6,
    MultilineText = 7,
    Secret = 8,
    Code = 9,

    /// <summary>
    /// Pick a dataset from the workspace. Distinct from <see cref="Choice"/> because the options
    /// are not known when the module is declared — they are whatever the user has imported. Using
    /// Choice here renders an empty select, since a Choice carries its own option list.
    /// </summary>
    DatasetRef = 10
}

public enum RunStatus
{
    Draft = 0,
    Queued = 1,
    Running = 2,
    Succeeded = 3,
    Failed = 4,
    Cancelled = 5
}

public enum NodeRunState
{
    Pending = 0,
    Running = 1,
    Succeeded = 2,
    Failed = 3,
    Skipped = 4
}

public enum LogLevel
{
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3
}

public enum DatasetFormat
{
    Csv = 0,
    Tsv = 1,
    Json = 2,
    Parquet = 3
}

public enum DatasetSourceKind
{
    Upload = 0,
    FileSystem = 1,
    AzureBlob = 2,
    S3 = 3,
    MinIO = 4,
    SqlDatabase = 5,
    WebService = 6,
    Generated = 7
}

public enum ColumnDataType
{
    Unknown = 0,
    Numeric = 1,
    Categorical = 2,
    Text = 3,
    Boolean = 4,
    DateTime = 5
}

public enum StorageProviderKind
{
    FileSystem = 0,
    AzureBlob = 1,
    S3 = 2,
    MinIO = 3
}

public enum DatabaseProviderKind
{
    Sqlite = 0,
    SqlServer = 1,
    MySql = 2,
    PostgreSql = 3
}

public enum ChatProviderKind
{
    OpenAI = 0,
    Anthropic = 1,
    Gemini = 2,
    Ollama = 3
}

public enum ChatRole
{
    System = 0,
    User = 1,
    Assistant = 2,
    Tool = 3
}

public enum AttachmentKind
{
    Image = 0,
    Document = 1
}

public enum EndpointStatus
{
    Stopped = 0,
    Live = 1
}

public enum ScriptLanguage
{
    Python = 0,
    R = 1,
    CSharp = 2,
    JavaScript = 3
}
