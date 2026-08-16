namespace LocalGen.Core.Configuration;

/// <summary>Resolves where LocalGen keeps its data across the platforms it supports.</summary>
public static class LocalGenPaths
{
    /// <summary>Overrides the data directory, mirroring Ollama's <c>OLLAMA_MODELS</c> convention.</summary>
    public const string DataDirectoryVariable = "LOCALGEN_HOME";

    /// <summary>
    /// <c>%LOCALAPPDATA%\LocalGen</c> on Windows and <c>~/.localgen</c> elsewhere, unless
    /// <see cref="DataDirectoryVariable"/> says otherwise.
    /// </summary>
    public static string DefaultDataDirectory
    {
        get
        {
            var overridden = Environment.GetEnvironmentVariable(DataDirectoryVariable);
            if (!string.IsNullOrWhiteSpace(overridden))
            {
                return Path.GetFullPath(overridden);
            }

            if (OperatingSystem.IsWindows())
            {
                var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                return Path.Combine(localAppData, "LocalGen");
            }

            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, ".localgen");
        }
    }

    /// <summary>Creates the directory tree LocalGen expects. Safe to call repeatedly.</summary>
    public static void EnsureCreated(LocalGenOptions options)
    {
        Directory.CreateDirectory(options.DataDirectory);
        Directory.CreateDirectory(options.ModelsDirectory);
        Directory.CreateDirectory(options.SkillsDirectory);
        Directory.CreateDirectory(options.SessionsDirectory);
        Directory.CreateDirectory(options.LogsDirectory);
    }
}
