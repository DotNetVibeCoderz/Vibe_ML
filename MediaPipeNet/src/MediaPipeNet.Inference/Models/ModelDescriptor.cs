namespace MediaPipeNet.Inference.Models;

/// <summary>Describes one pretrained model file: identity, integrity and provenance.</summary>
/// <param name="Id">Stable model identifier (e.g. <c>face_detection_short_range</c>).</param>
/// <param name="FileName">File name of the ONNX model.</param>
/// <param name="Sha256">Lower-case hex SHA-256 of the file, verified before use.</param>
/// <param name="SizeBytes">File size in bytes.</param>
/// <param name="PackageId">NuGet package that ships the model.</param>
/// <param name="Title">Human-readable name.</param>
/// <param name="Source">Original Google MediaPipe model the ONNX file was converted from.</param>
/// <param name="License">SPDX license of the model weights.</param>
public sealed record ModelDescriptor(
    string Id,
    string FileName,
    string Sha256,
    long SizeBytes,
    string PackageId,
    string Title,
    string Source,
    string License = "Apache-2.0")
{
    /// <summary>Attribution text required by the model license.</summary>
    public string Attribution => $"{Title} — converted from Google MediaPipe '{Source}' (Copyright Google LLC, {License}).";

    /// <inheritdoc />
    public override string ToString() => Id;
}

/// <summary>Progress of a model download.</summary>
/// <param name="Model">The model being downloaded.</param>
/// <param name="BytesReceived">Bytes received so far.</param>
/// <param name="TotalBytes">Total bytes, when known.</param>
/// <param name="Stage">What is happening (downloading, extracting, verifying).</param>
public readonly record struct ModelDownloadProgress(ModelDescriptor Model, long BytesReceived, long? TotalBytes, string Stage)
{
    /// <summary>Fraction complete in [0, 1], or null when the total is unknown.</summary>
    public double? Fraction => TotalBytes is > 0 ? (double)BytesReceived / TotalBytes.Value : null;
}
