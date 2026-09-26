using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using MediaPipeNet.Serialization;

namespace MediaPipeNet.Tasks.Vision;

/// <summary>Result of <see cref="FaceDetector"/>.</summary>
/// <param name="Detections">Detected faces; each has 6 keypoints (see <see cref="FaceKeypoint"/>).</param>
public sealed record FaceDetectionResult(IReadOnlyList<Detection> Detections)
{
    /// <summary>A result with no faces.</summary>
    public static FaceDetectionResult Empty { get; } = new([]);

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>One face found by <see cref="FaceLandmarker"/>.</summary>
/// <param name="Landmarks">478 face mesh landmarks (468 mesh + 10 iris), normalized.</param>
/// <param name="Blendshapes">52 blendshape scores when enabled, otherwise null.</param>
/// <param name="PresenceScore">Confidence that a face is present in the ROI.</param>
/// <param name="Roi">The rotated region the landmark model ran on.</param>
public sealed record FaceLandmarks(IReadOnlyList<NormalizedLandmark> Landmarks, IReadOnlyList<Category>? Blendshapes, float PresenceScore, NormalizedRect Roi)
{
    /// <summary>Score of a blendshape by name (e.g. <c>jawOpen</c>), or 0.</summary>
    public float GetBlendshape(string name) => Blendshapes?.FirstOrDefault(c => c.CategoryName == name)?.Score ?? 0f;
}

/// <summary>Result of <see cref="FaceLandmarker"/>.</summary>
/// <param name="Faces">The faces found.</param>
public sealed record FaceLandmarkResult(IReadOnlyList<FaceLandmarks> Faces)
{
    /// <summary>A result with no faces.</summary>
    public static FaceLandmarkResult Empty { get; } = new([]);

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>One hand found by <see cref="HandLandmarker"/>.</summary>
/// <param name="Handedness">"Left" or "Right" with its score (MediaPipe convention: assumes a mirrored, selfie-view image).</param>
/// <param name="Landmarks">21 hand landmarks, normalized (see <see cref="HandLandmark"/>).</param>
/// <param name="WorldLandmarks">21 landmarks in meters around the hand's center.</param>
/// <param name="PresenceScore">Confidence that a hand is present in the ROI.</param>
/// <param name="Roi">The rotated region the landmark model ran on.</param>
public sealed record HandLandmarks(Category Handedness, IReadOnlyList<NormalizedLandmark> Landmarks, IReadOnlyList<Landmark> WorldLandmarks, float PresenceScore, NormalizedRect Roi)
{
    /// <summary>True for a left hand.</summary>
    [JsonIgnore] public bool IsLeft => Handedness.CategoryName == "Left";

    /// <summary>Landmark by name.</summary>
    public NormalizedLandmark this[HandLandmark landmark] => Landmarks[(int)landmark];
}

/// <summary>Result of <see cref="HandLandmarker"/>.</summary>
/// <param name="Hands">The hands found.</param>
public sealed record HandLandmarkResult(IReadOnlyList<HandLandmarks> Hands)
{
    /// <summary>A result with no hands.</summary>
    public static HandLandmarkResult Empty { get; } = new([]);

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>A hand with its recognized gestures.</summary>
/// <param name="Hand">The hand landmarks.</param>
/// <param name="Gestures">Gestures sorted by score (the first one is the recognized gesture).</param>
public sealed record RecognizedHand(HandLandmarks Hand, IReadOnlyList<Category> Gestures)
{
    /// <summary>The best gesture.</summary>
    [JsonIgnore] public Category TopGesture => Gestures[0];
}

/// <summary>Result of <see cref="GestureRecognizer"/>.</summary>
/// <param name="Hands">Hands with their gestures.</param>
public sealed record GestureRecognitionResult(IReadOnlyList<RecognizedHand> Hands)
{
    /// <summary>A result with no hands.</summary>
    public static GestureRecognitionResult Empty { get; } = new([]);

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>One person found by <see cref="PoseLandmarker"/>.</summary>
/// <param name="Landmarks">33 body landmarks, normalized, with visibility and presence (see <see cref="PoseLandmark"/>).</param>
/// <param name="WorldLandmarks">33 landmarks in meters, origin between the hips.</param>
/// <param name="PresenceScore">Confidence that a person is present in the ROI.</param>
/// <param name="Roi">The rotated region the landmark model ran on.</param>
/// <param name="SegmentationMask">Person mask over the whole image when enabled.</param>
public sealed record PoseLandmarks(IReadOnlyList<NormalizedLandmark> Landmarks, IReadOnlyList<Landmark> WorldLandmarks, float PresenceScore, NormalizedRect Roi, SegmentationMask? SegmentationMask = null)
{
    /// <summary>Landmark by name.</summary>
    public NormalizedLandmark this[PoseLandmark landmark] => Landmarks[(int)landmark];
}

/// <summary>Result of <see cref="PoseLandmarker"/>.</summary>
/// <param name="Poses">The people found.</param>
public sealed record PoseLandmarkResult(IReadOnlyList<PoseLandmarks> Poses)
{
    /// <summary>A result with no people.</summary>
    public static PoseLandmarkResult Empty { get; } = new([]);

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>Result of <see cref="HolisticLandmarker"/>: body, face and both hands of one person.</summary>
/// <param name="Pose">Body landmarks.</param>
/// <param name="Face">Face mesh.</param>
/// <param name="LeftHand">The person's left hand (anatomical, as reported by the pose model).</param>
/// <param name="RightHand">The person's right hand.</param>
public sealed record HolisticResult(PoseLandmarks? Pose, FaceLandmarks? Face, HandLandmarks? LeftHand, HandLandmarks? RightHand)
{
    /// <summary>A result with nothing found.</summary>
    public static HolisticResult Empty { get; } = new(null, null, null, null);

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>Result of <see cref="ImageSegmenter"/>.</summary>
/// <param name="ConfidenceMask">Per-pixel foreground probability in [0, 1], same size as the input image.</param>
public sealed record SegmentationResult(SegmentationMask ConfidenceMask)
{
    /// <summary>Serializes the result to JSON (the mask is base64-encoded).</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>Result of <see cref="ObjectDetector"/>.</summary>
/// <param name="Detections">Detected objects sorted by score.</param>
public sealed record ObjectDetectionResult(IReadOnlyList<Detection> Detections)
{
    /// <summary>A result with no objects.</summary>
    public static ObjectDetectionResult Empty { get; } = new([]);

    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>Result of <see cref="ImageClassifier"/>.</summary>
/// <param name="Categories">Top categories sorted by score.</param>
public sealed record ClassificationResult(IReadOnlyList<Category> Categories)
{
    /// <summary>Serializes the result to JSON.</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>
/// A single-channel float mask (e.g. foreground probability). JSON-serialized compactly as
/// width, height and base64 float32 data.
/// </summary>
[JsonConverter(typeof(SegmentationMaskJsonConverter))]
public sealed class SegmentationMask
{
    /// <summary>Creates a mask over existing data (row-major, <c>width * height</c> values).</summary>
    public SegmentationMask(int width, int height, float[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length < width * height) throw new ArgumentException("Mask data is smaller than width * height.", nameof(data));
        Width = width;
        Height = height;
        Data = data;
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>Row-major values in [0, 1].</summary>
    public float[] Data { get; }

    /// <summary>Value at (x, y).</summary>
    public float this[int x, int y] => Data[y * Width + x];

    /// <summary>Fraction of pixels above <paramref name="threshold"/>.</summary>
    public float Coverage(float threshold = 0.5f)
    {
        int n = 0;
        foreach (var v in Data.AsSpan(0, Width * Height)) if (v > threshold) n++;
        return (float)n / (Width * Height);
    }

    /// <summary>Converts to an 8-bit mask (0..255), optionally thresholded to 0/255.</summary>
    public byte[] ToBytes(float? threshold = null)
    {
        var bytes = new byte[Width * Height];
        for (int i = 0; i < bytes.Length; i++)
        {
            float v = Data[i];
            bytes[i] = threshold is { } t ? (byte)(v > t ? 255 : 0) : (byte)Math.Clamp(v * 255f + 0.5f, 0, 255);
        }
        return bytes;
    }
}

internal sealed class SegmentationMaskJsonConverter : JsonConverter<SegmentationMask>
{
    public override SegmentationMask Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;
        int w = root.GetProperty("width").GetInt32(), h = root.GetProperty("height").GetInt32();
        var bytes = Convert.FromBase64String(root.GetProperty("data").GetString()!);
        return new SegmentationMask(w, h, MemoryMarshal.Cast<byte, float>(bytes).ToArray());
    }

    public override void Write(Utf8JsonWriter writer, SegmentationMask value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteNumber("width", value.Width);
        writer.WriteNumber("height", value.Height);
        writer.WriteString("format", "float32-base64");
        writer.WriteBase64String("data", MemoryMarshal.AsBytes(value.Data.AsSpan(0, value.Width * value.Height)));
        writer.WriteEndObject();
    }
}
