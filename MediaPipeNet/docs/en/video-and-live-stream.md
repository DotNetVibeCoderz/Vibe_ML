# Running modes & live video

> 🇮🇩 [Baca dalam Bahasa Indonesia](../id/video-dan-live-stream.md)

## Which mode?

| You have… | Use | Why |
|---|---|---|
| Photos, uploads, independent images | `RunningMode.Image` | Stateless and thread-safe; one instance can serve many requests. |
| A video file or frames you process one after another | `RunningMode.Video` | Tracking: the next region of interest comes from the current landmarks, so detectors rarely run; landmarks are smoothed. |
| A camera, where only the newest frame matters | `RunningMode.LiveStream` *or* `Video` + `LiveStreamProcessor<T>` | Frames that arrive while the model is busy are dropped instead of queued. |

## Video mode

```csharp
using var pose = PoseLandmarker.Create(new PoseLandmarkerOptions { RunningMode = RunningMode.Video });
await using var video = new VideoFileFrameSource("dance.mp4");          // MediaPipeNet.Video.OpenCv
await foreach (var frame in video.ReadFramesAsync())
{
    var result = pose.DetectForVideo(frame.Image, frame.TimestampMs);    // timestamps must increase
    // frame.Image is reused by the source: Clone() it if you keep it.
}
```

Call `ResetTracking()` when the video jumps (seek, scene cut).

## Live-stream mode (MediaPipe style)

```csharp
using var hands = HandLandmarker.Create(new HandLandmarkerOptions
{
    RunningMode = RunningMode.LiveStream,
    ResultCallback = (result, frame, timestampMs) => Console.WriteLine($"{timestampMs}: {result.Hands.Count} hands"),
    MaxInFlightFrames = 1,                 // default: drop new frames while one is being processed
});

// from your camera callback:
bool accepted = hands.DetectLiveStream(frame, timestampMs);   // copies the frame only when accepted
Console.WriteLine($"dropped so far: {hands.DroppedFrames}");
```

Internally the task runs inside a small `CalculatorGraph` whose `MaxInFlight` limit implements MediaPipe's
flow-limiter semantics. The callback runs on a worker thread; the `frame` argument is only valid during the call.

## `LiveStreamProcessor<T>` (recommended for apps)

`LiveStreamProcessor<T>` pumps any `IFrameSource` through any function on a background thread with a
latest-frame-wins slot, double-buffered frames (no per-frame allocations) and live statistics:

```csharp
using var gestures = GestureRecognizer.Create(new() { RunningMode = RunningMode.Video });
await using var camera = new WebcamFrameSource(deviceIndex: 0, width: 1280, height: 720);
await using var live = new LiveStreamProcessor<GestureRecognitionResult>(
    camera, (frame, ts) => gestures.RecognizeForVideo(frame, ts),
    new LiveStreamProcessorOptions { MirrorFrames = true });

live.ResultReady += (_, r) =>
{
    // r.Frame is valid during the handler; r.Stats has CaptureFps, ProcessingFps, LastLatencyMs, FramesDropped
    Console.WriteLine($"{r.Stats.ProcessingFps:F1} fps, {r.Latency.TotalMilliseconds:F0} ms, {r.Result.Hands.Count} hands");
};
live.ProcessingFailed += (_, e) => Console.Error.WriteLine(e.Message);
await live.RunAsync(cancellationToken);
```

With a non-live source (files) every frame is processed (`DropFramesWhenBusy` defaults to `source.IsLive`).

## Frame sources

| Source | Package | Notes |
|---|---|---|
| `ImageFileFrameSource` | Imaging | One or more still images; `FromDirectory(dir)`; optional looping. |
| `MemoryFrameSource` | Imaging | Frames you already decoded; `realTime: true` simulates a camera. |
| `WebcamFrameSource` | Video.OpenCv | Camera by index (DirectShow on Windows); `ListCameras()`. |
| `VideoFileFrameSource` | Video.OpenCv | Any format OpenCV/FFmpeg decodes. |

Implement `IFrameSource` to plug in anything else (RTSP, a capture card, frames from a game engine…).

`MediaPipeNet.Video.OpenCv` needs an OpenCvSharp native runtime for your OS: `OpenCvSharp4.runtime.win` on
Windows, `OpenCvSharp4.official.runtime.linux-x64` on Linux; on macOS install OpenCV via Homebrew.

![Live camera page](../images/gallery-live.png)
