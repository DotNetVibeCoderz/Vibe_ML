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
/// <param name="FacialTransformationMatrix">
/// When enabled, the 4×4 facial transformation matrix as 16 row-major values (MediaPipe layout,
/// column-vector convention: <c>p_camera = M · p_canonical</c>, translation in centimeters in the last column).
/// </param>
public sealed record FaceLandmarks(IReadOnlyList<NormalizedLandmark> Landmarks, IReadOnlyList<Category>? Blendshapes, float PresenceScore, NormalizedRect Roi,
    IReadOnlyList<float>? FacialTransformationMatrix = null)
{
    /// <summary>Score of a blendshape by name (e.g. <c>jawOpen</c>), or 0.</summary>
    public float GetBlendshape(string name) => Blendshapes?.FirstOrDefault(c => c.CategoryName == name)?.Score ?? 0f;

    /// <summary>
    /// The facial transformation matrix as a <see cref="System.Numerics.Matrix4x4"/> in .NET's row-vector
    /// convention (transposed, so <c>Vector3.Transform(canonicalPoint, m)</c> works; translation in M41..M43),
    /// or null when it was not requested.
    /// </summary>
    public System.Numerics.Matrix4x4? GetTransformMatrix()
    {
        if (FacialTransformationMatrix is not { Count: 16 } m) return null;
        return new System.Numerics.Matrix4x4(
            m[0], m[4], m[8], m[12],
            m[1], m[5], m[9], m[13],
            m[2], m[6], m[10], m[14],
            m[3], m[7], m[11], m[15]);
    }
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

/// <summary>Result of <see cref="ImageSegmenter"/> and <see cref="InteractiveSegmenter"/>.</summary>
public sealed record SegmentationResult
{
    private SegmentationMask? _foreground;

    /// <summary>A result holding a single foreground-probability mask.</summary>
    public SegmentationResult(SegmentationMask confidenceMask)
        : this([confidenceMask ?? throw new ArgumentNullException(nameof(confidenceMask))])
    {
    }

    /// <summary>Creates a result.</summary>
    /// <param name="confidenceMasks">One probability mask per category (empty when not requested).</param>
    /// <param name="categoryMask">The per-pixel category index, when requested.</param>
    /// <param name="labels">Category names in mask order.</param>
    [JsonConstructor]
    public SegmentationResult(IReadOnlyList<SegmentationMask> confidenceMasks, CategoryMask? categoryMask = null, IReadOnlyList<string>? labels = null)
    {
        ConfidenceMasks = confidenceMasks ?? throw new ArgumentNullException(nameof(confidenceMasks));
        CategoryMask = categoryMask;
        Labels = labels ?? [];
    }

    /// <summary>One probability mask per category, same size as the input image (empty when not requested).</summary>
    public IReadOnlyList<SegmentationMask> ConfidenceMasks { get; }

    /// <summary>The most likely category per pixel, when requested.</summary>
    public CategoryMask? CategoryMask { get; }

    /// <summary>Category names, in <see cref="ConfidenceMasks"/> order.</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>
    /// The foreground probability: the only mask of single-mask models (selfie), otherwise
    /// 1 − P(background) (category 0 of every multi-class model).
    /// </summary>
    /// <exception cref="InvalidOperationException">Confidence masks were not requested.</exception>
    [JsonIgnore]
    public SegmentationMask ConfidenceMask => _foreground ??= ConfidenceMasks.Count switch
    {
        0 => throw new InvalidOperationException("Confidence masks were not requested (OutputConfidenceMasks = false)."),
        1 => ConfidenceMasks[0],
        _ => ConfidenceMasks[0].Complement(),
    };

    /// <summary>The confidence mask of a category by name, or null.</summary>
    public SegmentationMask? GetConfidenceMask(string label)
    {
        for (int i = 0; i < Labels.Count && i < ConfidenceMasks.Count; i++)
            if (string.Equals(Labels[i], label, StringComparison.OrdinalIgnoreCase)) return ConfidenceMasks[i];
        return null;
    }

    /// <summary>Serializes the result to JSON (masks are base64-encoded).</summary>
    public string ToJson(bool indented = false) => MediaPipeJson.Serialize(this, indented);
}

/// <summary>
/// A per-pixel category index mask (one byte per pixel). JSON-serialized compactly as width, height and
/// base64 data.
/// </summary>
[JsonConverter(typeof(CategoryMaskJsonConverter))]
public sealed class CategoryMask
{
    /// <summary>Value of pixels that belong to no category.</summary>
    public const byte Unlabeled = 255;

    /// <summary>Creates a mask over existing data (row-major, <c>width * height</c> bytes).</summary>
    public CategoryMask(int width, int height, byte[] data)
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

    /// <summary>Row-major category indices.</summary>
    public byte[] Data { get; }

    /// <summary>Category at (x, y).</summary>
    public byte this[int x, int y] => Data[y * Width + x];

    /// <summary>Fraction of pixels per category index (index 255 = unlabeled).</summary>
    public IReadOnlyDictionary<int, float> Histogram()
    {
        var counts = new int[256];
        foreach (var v in Data.AsSpan(0, Width * Height)) counts[v]++;
        var result = new SortedDictionary<int, float>();
        float n = Width * Height;
        for (int i = 0; i < 256; i++) if (counts[i] > 0) result[i] = counts[i] / n;
        return result;
    }

    /// <summary>
    /// Builds the category mask from confidence masks: the argmax per pixel, or for a single mask 0 where
    /// the probability exceeds 0.5 and <see cref="Unlabeled"/> elsewhere (MediaPipe's convention).
    /// </summary>
    public static CategoryMask FromConfidenceMasks(IReadOnlyList<SegmentationMask> masks)
    {
        ArgumentNullException.ThrowIfNull(masks);
        if (masks.Count == 0) throw new ArgumentException("At least one mask is required.", nameof(masks));
        int w = masks[0].Width, h = masks[0].Height;
        var data = new byte[w * h];
        if (masks.Count == 1)
        {
            var m = masks[0].Data;
            for (int i = 0; i < data.Length; i++) data[i] = m[i] > 0.5f ? (byte)0 : Unlabeled;
            return new CategoryMask(w, h, data);
        }
        Parallel.For(0, h, y =>
        {
            for (int i = y * w, end = i + w; i < end; i++)
            {
                int best = 0;
                float max = masks[0].Data[i];
                for (int c = 1; c < masks.Count; c++)
                {
                    float v = masks[c].Data[i];
                    if (v > max)
                    {
                        max = v;
                        best = c;
                    }
                }
                data[i] = (byte)best;
            }
        });
        return new CategoryMask(w, h, data);
    }
}

internal sealed class CategoryMaskJsonConverter : JsonConverter<CategoryMask>
{
    public override CategoryMask Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;
        return new CategoryMask(root.GetProperty("width").GetInt32(), root.GetProperty("height").GetInt32(),
            Convert.FromBase64String(root.GetProperty("data").GetString()!));
    }

    public override void Write(Utf8JsonWriter writer, CategoryMask value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteNumber("width", value.Width);
        writer.WriteNumber("height", value.Height);
        writer.WriteString("format", "uint8-base64");
        writer.WriteBase64String("data", value.Data.AsSpan(0, value.Width * value.Height));
        writer.WriteEndObject();
    }
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

    /// <summary>A new mask holding 1 − value for every pixel.</summary>
    public SegmentationMask Complement()
    {
        var data = new float[Width * Height];
        for (int i = 0; i < data.Length; i++) data[i] = 1f - Data[i];
        return new SegmentationMask(Width, Height, data);
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
