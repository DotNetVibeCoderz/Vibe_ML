namespace MediaPipeNet.Tasks.Vision.Processing;

/// <summary>
/// The One Euro filter (Casiez et al.): a low-pass filter whose cutoff rises with speed, removing
/// jitter at rest while staying responsive during fast motion. Used by MediaPipe's
/// <c>LandmarksSmoothingCalculator</c>.
/// </summary>
public sealed class OneEuroFilter(float minCutoff = 1.0f, float beta = 0.0f, float derivativeCutoff = 1.0f)
{
    private float _x, _dx;
    private double _lastTime = double.NaN;

    /// <summary>Filters <paramref name="value"/> observed at <paramref name="timeSeconds"/>.</summary>
    public float Filter(float value, double timeSeconds)
    {
        if (double.IsNaN(_lastTime) || timeSeconds <= _lastTime)
        {
            _lastTime = timeSeconds;
            _x = value;
            _dx = 0;
            return value;
        }
        float dt = (float)(timeSeconds - _lastTime);
        _lastTime = timeSeconds;
        float dx = (value - _x) / dt;
        _dx += Alpha(dt, derivativeCutoff) * (dx - _dx);
        float cutoff = minCutoff + beta * MathF.Abs(_dx);
        _x += Alpha(dt, cutoff) * (value - _x);
        return _x;
    }

    /// <summary>Forgets the filter state.</summary>
    public void Reset() => _lastTime = double.NaN;

    private static float Alpha(float dt, float cutoff)
    {
        float tau = 1f / (2f * MathF.PI * cutoff);
        return 1f / (1f + tau / dt);
    }
}

/// <summary>Smooths a fixed-size set of landmarks over time with per-coordinate One Euro filters.</summary>
public sealed class LandmarkSmoother
{
    private readonly OneEuroFilter[] _filters;
    private readonly int _count;

    /// <summary>Creates a smoother for <paramref name="landmarkCount"/> landmarks.</summary>
    public LandmarkSmoother(int landmarkCount, float minCutoff = 0.05f, float beta = 80f, float derivativeCutoff = 1f)
    {
        _count = landmarkCount;
        _filters = new OneEuroFilter[landmarkCount * 3];
        for (int i = 0; i < _filters.Length; i++) _filters[i] = new OneEuroFilter(minCutoff, beta, derivativeCutoff);
    }

    /// <summary>
    /// Smooths landmarks in place. Values are scaled by the object size so the filter behaves the
    /// same for near and far objects (as MediaPipe's velocity filter does).
    /// </summary>
    public void Apply(NormalizedLandmark[] landmarks, long timestampMs, int imageWidth, int imageHeight, float objectScale)
    {
        ArgumentNullException.ThrowIfNull(landmarks);
        double t = timestampMs / 1000.0;
        float scale = MathF.Max(objectScale, 1e-3f);
        for (int i = 0; i < Math.Min(_count, landmarks.Length); i++)
        {
            var l = landmarks[i];
            float x = _filters[3 * i].Filter(l.X * imageWidth / scale, t) * scale / imageWidth;
            float y = _filters[3 * i + 1].Filter(l.Y * imageHeight / scale, t) * scale / imageHeight;
            float z = _filters[3 * i + 2].Filter(l.Z * imageWidth / scale, t) * scale / imageWidth;
            landmarks[i] = l with { X = x, Y = y, Z = z };
        }
    }

    /// <summary>Forgets the filter state.</summary>
    public void Reset()
    {
        foreach (var f in _filters) f.Reset();
    }
}
