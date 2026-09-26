namespace MediaPipeNet.Tasks.Vision;

/// <summary>The 21 hand landmarks.</summary>
public enum HandLandmark
{
    /// <summary>Wrist.</summary>
    Wrist = 0,
    /// <summary>Thumb carpometacarpal joint.</summary>
    ThumbCmc, /// <summary>Thumb metacarpophalangeal joint.</summary>
    ThumbMcp, /// <summary>Thumb interphalangeal joint.</summary>
    ThumbIp, /// <summary>Thumb tip.</summary>
    ThumbTip,
    /// <summary>Index finger MCP.</summary>
    IndexFingerMcp, /// <summary>Index finger PIP.</summary>
    IndexFingerPip, /// <summary>Index finger DIP.</summary>
    IndexFingerDip, /// <summary>Index finger tip.</summary>
    IndexFingerTip,
    /// <summary>Middle finger MCP.</summary>
    MiddleFingerMcp, /// <summary>Middle finger PIP.</summary>
    MiddleFingerPip, /// <summary>Middle finger DIP.</summary>
    MiddleFingerDip, /// <summary>Middle finger tip.</summary>
    MiddleFingerTip,
    /// <summary>Ring finger MCP.</summary>
    RingFingerMcp, /// <summary>Ring finger PIP.</summary>
    RingFingerPip, /// <summary>Ring finger DIP.</summary>
    RingFingerDip, /// <summary>Ring finger tip.</summary>
    RingFingerTip,
    /// <summary>Pinky MCP.</summary>
    PinkyMcp, /// <summary>Pinky PIP.</summary>
    PinkyPip, /// <summary>Pinky DIP.</summary>
    PinkyDip, /// <summary>Pinky tip.</summary>
    PinkyTip,
}

/// <summary>The 33 BlazePose body landmarks.</summary>
public enum PoseLandmark
{
    /// <summary>Nose.</summary>
    Nose = 0,
    /// <summary>Left eye (inner).</summary>
    LeftEyeInner, /// <summary>Left eye.</summary>
    LeftEye, /// <summary>Left eye (outer).</summary>
    LeftEyeOuter,
    /// <summary>Right eye (inner).</summary>
    RightEyeInner, /// <summary>Right eye.</summary>
    RightEye, /// <summary>Right eye (outer).</summary>
    RightEyeOuter,
    /// <summary>Left ear.</summary>
    LeftEar, /// <summary>Right ear.</summary>
    RightEar,
    /// <summary>Mouth (left).</summary>
    MouthLeft, /// <summary>Mouth (right).</summary>
    MouthRight,
    /// <summary>Left shoulder.</summary>
    LeftShoulder, /// <summary>Right shoulder.</summary>
    RightShoulder,
    /// <summary>Left elbow.</summary>
    LeftElbow, /// <summary>Right elbow.</summary>
    RightElbow,
    /// <summary>Left wrist.</summary>
    LeftWrist, /// <summary>Right wrist.</summary>
    RightWrist,
    /// <summary>Left pinky knuckle.</summary>
    LeftPinky, /// <summary>Right pinky knuckle.</summary>
    RightPinky,
    /// <summary>Left index knuckle.</summary>
    LeftIndex, /// <summary>Right index knuckle.</summary>
    RightIndex,
    /// <summary>Left thumb knuckle.</summary>
    LeftThumb, /// <summary>Right thumb knuckle.</summary>
    RightThumb,
    /// <summary>Left hip.</summary>
    LeftHip, /// <summary>Right hip.</summary>
    RightHip,
    /// <summary>Left knee.</summary>
    LeftKnee, /// <summary>Right knee.</summary>
    RightKnee,
    /// <summary>Left ankle.</summary>
    LeftAnkle, /// <summary>Right ankle.</summary>
    RightAnkle,
    /// <summary>Left heel.</summary>
    LeftHeel, /// <summary>Right heel.</summary>
    RightHeel,
    /// <summary>Left foot index.</summary>
    LeftFootIndex, /// <summary>Right foot index.</summary>
    RightFootIndex,
}

/// <summary>The 6 BlazeFace keypoints (from the subject's point of view).</summary>
public enum FaceKeypoint
{
    /// <summary>Right eye.</summary>
    RightEye = 0,
    /// <summary>Left eye.</summary>
    LeftEye,
    /// <summary>Nose tip.</summary>
    NoseTip,
    /// <summary>Mouth center.</summary>
    MouthCenter,
    /// <summary>Right ear tragion.</summary>
    RightEarTragion,
    /// <summary>Left ear tragion.</summary>
    LeftEarTragion,
}

/// <summary>Landmark connectivity for drawing skeletons and meshes.</summary>
public static class Connections
{
    /// <summary>The 21 bones of a hand.</summary>
    public static IReadOnlyList<(int From, int To)> Hand { get; } =
    [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    ];

    /// <summary>The 35 BlazePose connections.</summary>
    public static IReadOnlyList<(int From, int To)> Pose { get; } =
    [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
    ];

    /// <summary>Face oval.</summary>
    public static IReadOnlyList<(int From, int To)> FaceOval { get; } = Path(
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
        148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10);

    /// <summary>Outer and inner lips.</summary>
    public static IReadOnlyList<(int From, int To)> Lips { get; } =
    [
        .. Path(61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291),
        .. Path(61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291),
        .. Path(78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308),
        .. Path(78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308),
    ];

    /// <summary>The subject's left eye.</summary>
    public static IReadOnlyList<(int From, int To)> LeftEye { get; } =
        [.. Path(263, 249, 390, 373, 374, 380, 381, 382, 362), .. Path(263, 466, 388, 387, 386, 385, 384, 398, 362)];

    /// <summary>The subject's right eye.</summary>
    public static IReadOnlyList<(int From, int To)> RightEye { get; } =
        [.. Path(33, 7, 163, 144, 145, 153, 154, 155, 133), .. Path(33, 246, 161, 160, 159, 158, 157, 173, 133)];

    /// <summary>The subject's left eyebrow.</summary>
    public static IReadOnlyList<(int From, int To)> LeftEyebrow { get; } =
        [.. Path(276, 283, 282, 295, 285), .. Path(300, 293, 334, 296, 336)];

    /// <summary>The subject's right eyebrow.</summary>
    public static IReadOnlyList<(int From, int To)> RightEyebrow { get; } =
        [.. Path(46, 53, 52, 65, 55), .. Path(70, 63, 105, 66, 107)];

    /// <summary>Left iris (landmarks 474–477 around center 473).</summary>
    public static IReadOnlyList<(int From, int To)> LeftIris { get; } = Path(474, 475, 476, 477, 474);

    /// <summary>Right iris (landmarks 469–472 around center 468).</summary>
    public static IReadOnlyList<(int From, int To)> RightIris { get; } = Path(469, 470, 471, 472, 469);

    /// <summary>All face contours: oval, lips, eyes, eyebrows and irises.</summary>
    public static IReadOnlyList<(int From, int To)> FaceContours { get; } =
        [.. FaceOval, .. Lips, .. LeftEye, .. RightEye, .. LeftEyebrow, .. RightEyebrow, .. LeftIris, .. RightIris];

    private static (int, int)[] Path(params int[] points)
    {
        var edges = new (int, int)[points.Length - 1];
        for (int i = 0; i < edges.Length; i++) edges[i] = (points[i], points[i + 1]);
        return edges;
    }
}
