namespace MediaPipeNet;

/// <summary>
/// How a task receives its input, mirroring MediaPipe Tasks' running modes.
/// </summary>
public enum RunningMode
{
    /// <summary>Independent still images. No state is kept between calls.</summary>
    Image = 0,

    /// <summary>
    /// Decoded frames of a video, processed synchronously in order. Timestamps must increase
    /// monotonically; tasks use them for tracking and landmark smoothing.
    /// </summary>
    Video = 1,

    /// <summary>
    /// A live stream (e.g. a webcam). Frames are submitted asynchronously and results are delivered
    /// through a callback; frames arriving while the task is busy are dropped to keep latency low.
    /// </summary>
    LiveStream = 2,
}
