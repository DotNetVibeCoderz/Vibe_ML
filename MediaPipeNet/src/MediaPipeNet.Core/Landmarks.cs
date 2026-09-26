namespace MediaPipeNet;

/// <summary>
/// A landmark in normalized image coordinates: <see cref="X"/> and <see cref="Y"/> are in [0, 1]
/// relative to image width and height; <see cref="Z"/> is depth with roughly the same scale as
/// <see cref="X"/> (smaller is closer to the camera).
/// </summary>
/// <param name="X">Horizontal position, normalized by image width.</param>
/// <param name="Y">Vertical position, normalized by image height.</param>
/// <param name="Z">Relative depth, same scale as X.</param>
/// <param name="Visibility">Likelihood in [0, 1] that the landmark is visible (not occluded), when the model provides it.</param>
/// <param name="Presence">Likelihood in [0, 1] that the landmark is inside the image, when the model provides it.</param>
public readonly record struct NormalizedLandmark(float X, float Y, float Z = 0f, float? Visibility = null, float? Presence = null)
{
    /// <summary>Converts to pixel coordinates.</summary>
    public (float X, float Y) ToPixel(int imageWidth, int imageHeight) => (X * imageWidth, Y * imageHeight);
}

/// <summary>
/// A landmark in world coordinates (meters), with the origin at the approximate geometric center
/// of the object (e.g. the hand or the hips).
/// </summary>
/// <param name="X">X in meters.</param>
/// <param name="Y">Y in meters.</param>
/// <param name="Z">Z in meters.</param>
/// <param name="Visibility">Likelihood in [0, 1] that the landmark is visible.</param>
/// <param name="Presence">Likelihood in [0, 1] that the landmark is present in the scene.</param>
public readonly record struct Landmark(float X, float Y, float Z, float? Visibility = null, float? Presence = null);

/// <summary>A 2-D keypoint in normalized image coordinates attached to a detection.</summary>
/// <param name="X">Horizontal position, normalized by image width.</param>
/// <param name="Y">Vertical position, normalized by image height.</param>
/// <param name="Label">Optional name of the keypoint (e.g. "rightEye").</param>
/// <param name="Score">Optional confidence.</param>
public readonly record struct NormalizedKeypoint(float X, float Y, string? Label = null, float? Score = null);

/// <summary>A classification outcome: a label with its score.</summary>
/// <param name="Index">Index of the class in the model's label map (-1 when not applicable).</param>
/// <param name="Score">Confidence score, usually in [0, 1].</param>
/// <param name="CategoryName">Machine-readable label.</param>
/// <param name="DisplayName">Human-readable label, when available.</param>
public sealed record Category(int Index, float Score, string? CategoryName = null, string? DisplayName = null)
{
    /// <inheritdoc />
    public override string ToString() => $"{CategoryName ?? Index.ToString(System.Globalization.CultureInfo.InvariantCulture)} ({Score:P1})";
}

/// <summary>
/// A detected object: a pixel bounding box, one or more categories and optional normalized
/// keypoints, mirroring MediaPipe Tasks' <c>Detection</c>.
/// </summary>
/// <param name="BoundingBox">Bounding box in pixels of the input image.</param>
/// <param name="Categories">Categories sorted by descending score.</param>
/// <param name="Keypoints">Keypoints in normalized image coordinates (may be empty).</param>
public sealed record Detection(RectF BoundingBox, IReadOnlyList<Category> Categories, IReadOnlyList<NormalizedKeypoint> Keypoints)
{
    /// <summary>The highest scoring category.</summary>
    [System.Text.Json.Serialization.JsonIgnore]
    public Category TopCategory => Categories[0];

    /// <summary>The score of the highest scoring category.</summary>
    [System.Text.Json.Serialization.JsonIgnore]
    public float Score => Categories.Count > 0 ? Categories[0].Score : 0f;
}
