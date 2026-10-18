namespace ScienceGallery.Models;

/// <summary>The seven Gravicode.Science libraries a showcase can use.</summary>
public enum GraviLibrary
{
    GraviNum,
    GraviFrame,
    GraviLearn,
    GraviText,
    GraviGraph,
    GraviProb,
    SciNet,
}

/// <summary>Where a showcase is used in real life. Drives the sidebar.</summary>
public enum Domain
{
    Education,
    Research,
    Office,
    Industry,
    Health,
    Finance,
    Utility,
    Games,
}

/// <summary>Display facts about a library.</summary>
/// <param name="Library">Which library.</param>
/// <param name="Name">Its name as shipped.</param>
/// <param name="Analogue">The Python package it mirrors.</param>
/// <param name="Hex">Its colour: the same emission line its NuGet icon uses.</param>
/// <param name="Wavelength">That line in nanometres, or null for SciNet, which is the continuum.</param>
public sealed record LibraryInfo(GraviLibrary Library, string Name, string Analogue, string Hex, double? Wavelength)
{
    /// <summary>Every library, in the order they appear in the sidebar.</summary>
    /// <remarks>
    /// The colours and wavelengths are the icon system's (assets/icons/README.md): each library is a
    /// real emission line, so the spectrum strip in the gallery places them where a spectroscope
    /// would. SciNet, the instrument, has no single line; it lights the whole band instead.
    /// </remarks>
    public static IReadOnlyList<LibraryInfo> All { get; } =
    [
        new(GraviLibrary.GraviNum, "GraviNum", "NumPy", "#D9363E", 656),
        new(GraviLibrary.GraviFrame, "GraviFrame", "pandas", "#E8A317", 589),
        new(GraviLibrary.GraviLearn, "GraviLearn", "scikit-learn", "#2E9E68", 546),
        new(GraviLibrary.GraviText, "GraviText", "Transformers", "#1B95BC", 486),
        new(GraviLibrary.GraviGraph, "GraviGraph", "PyG / DGL", "#4F46D6", 434),
        new(GraviLibrary.GraviProb, "GraviProb", "PyMC", "#9634C8", 393),
        new(GraviLibrary.SciNet, "SciNet", "SciPy", "#16132E", null),
    ];

    /// <summary>Looks a library up.</summary>
    public static LibraryInfo Of(GraviLibrary library) => All[(int)library];
}

/// <summary>Display facts about a domain.</summary>
public sealed record DomainInfo(Domain Domain, string Name, string Blurb)
{
    /// <summary>Every domain, in sidebar order.</summary>
    public static IReadOnlyList<DomainInfo> All { get; } =
    [
        new(Domain.Education, "Education", "Teaching and learning: classrooms, grading, lab practicals"),
        new(Domain.Research, "Science & research", "Physics, biology, chemistry, astronomy, the lab bench"),
        new(Domain.Office, "Office", "Reports, spreadsheets, documents, scheduling"),
        new(Domain.Industry, "Industry", "Manufacturing, quality control, logistics, energy"),
        new(Domain.Health, "Health", "Clinical data, epidemiology, signals from the body"),
        new(Domain.Finance, "Finance", "Markets, risk, forecasting, fraud"),
        new(Domain.Utility, "Utility", "Everyday tools: files, audio, images, maps"),
        new(Domain.Games, "Games", "Simulation, procedural content, game AI"),
    ];

    /// <summary>Looks a domain up.</summary>
    public static DomainInfo Of(Domain domain) => All[(int)domain];
}
