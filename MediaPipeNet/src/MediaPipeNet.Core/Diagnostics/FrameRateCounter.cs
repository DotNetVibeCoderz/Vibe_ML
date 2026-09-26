using System.Diagnostics;

namespace MediaPipeNet.Diagnostics;

/// <summary>
/// Measures a rate (frames per second) over a sliding window of recent events. Thread-safe and
/// allocation-free after construction.
/// </summary>
public sealed class FrameRateCounter
{
    private readonly long[] _ticks;
    private readonly Lock _gate = new();
    private int _head;
    private int _count;

    /// <summary>Creates a counter averaging over the last <paramref name="window"/> events.</summary>
    public FrameRateCounter(int window = 30)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(window, 2);
        _ticks = new long[window];
    }

    /// <summary>Records one event at the current time.</summary>
    public void Tick()
    {
        long now = Stopwatch.GetTimestamp();
        lock (_gate)
        {
            _ticks[_head] = now;
            _head = (_head + 1) % _ticks.Length;
            if (_count < _ticks.Length) _count++;
        }
    }

    /// <summary>The current rate in events per second (0 until two events were recorded).</summary>
    public double Rate
    {
        get
        {
            lock (_gate)
            {
                if (_count < 2) return 0;
                int newest = (_head - 1 + _ticks.Length) % _ticks.Length;
                int oldest = (_head - _count + _ticks.Length) % _ticks.Length;
                double seconds = (_ticks[newest] - _ticks[oldest]) / (double)Stopwatch.Frequency;
                return seconds > 0 ? (_count - 1) / seconds : 0;
            }
        }
    }

    /// <summary>Forgets all recorded events.</summary>
    public void Reset()
    {
        lock (_gate) { _head = 0; _count = 0; }
    }
}
