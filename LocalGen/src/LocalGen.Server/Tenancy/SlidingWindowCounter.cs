namespace LocalGen.Server.Tenancy;

/// <summary>
/// Counts events over a rolling window, in fixed buckets.
/// </summary>
/// <remarks>
/// A fixed window ("1,000 requests per calendar minute") lets a caller spend the whole allowance
/// at 10:00:59 and the whole of it again at 10:01:00, which is twice the intended rate across the
/// boundary. Keeping a timestamp per event fixes that but grows with traffic. Bucketing splits the
/// difference: memory is constant, and the error is bounded by one bucket's width rather than by
/// the whole window.
/// </remarks>
internal sealed class SlidingWindowCounter
{
    private readonly long[] _values;
    private readonly long[] _epochs;
    private readonly long _bucketTicks;
    private readonly Lock _gate = new();

    /// <param name="window">Period the counter reports over.</param>
    /// <param name="buckets">Resolution. More buckets means a tighter bound on the rollover error.</param>
    public SlidingWindowCounter(TimeSpan window, int buckets)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(buckets, 1);

        _values = new long[buckets];
        _epochs = new long[buckets];
        _bucketTicks = Math.Max(1, window.Ticks / buckets);

        // Epoch 0 is a real epoch (the Unix-tick origin), so the array is seeded with a sentinel
        // that no live bucket can hold — otherwise every bucket would read as current at startup.
        Array.Fill(_epochs, -1L);
    }

    public void Add(long amount, DateTimeOffset now)
    {
        if (amount == 0)
        {
            return;
        }

        var epoch = now.UtcTicks / _bucketTicks;
        var index = (int)(epoch % _values.Length);

        lock (_gate)
        {
            // A bucket holding an older epoch has been lapped; it is reused rather than added to.
            if (_epochs[index] != epoch)
            {
                _epochs[index] = epoch;
                _values[index] = 0;
            }

            _values[index] += amount;
        }
    }

    /// <summary>Total recorded within the window ending now.</summary>
    public long Sum(DateTimeOffset now)
    {
        var epoch = now.UtcTicks / _bucketTicks;
        var oldest = epoch - _values.Length + 1;
        var total = 0L;

        lock (_gate)
        {
            for (var i = 0; i < _values.Length; i++)
            {
                if (_epochs[i] >= oldest && _epochs[i] <= epoch)
                {
                    total += _values[i];
                }
            }
        }

        return total;
    }

    /// <summary>
    /// How long until the oldest bucket falls out of the window, which is the soonest the count
    /// can drop. Used to fill in <c>Retry-After</c>.
    /// </summary>
    public TimeSpan TimeUntilRelease(DateTimeOffset now)
    {
        var epoch = now.UtcTicks / _bucketTicks;
        var oldest = epoch - _values.Length + 1;
        var oldestOccupied = long.MaxValue;

        lock (_gate)
        {
            for (var i = 0; i < _values.Length; i++)
            {
                if (_values[i] > 0 && _epochs[i] >= oldest && _epochs[i] < oldestOccupied)
                {
                    oldestOccupied = _epochs[i];
                }
            }
        }

        if (oldestOccupied == long.MaxValue)
        {
            return TimeSpan.Zero;
        }

        // That bucket leaves the window once the clock passes the end of its successor slot.
        var expiresAtTicks = (oldestOccupied + _values.Length) * _bucketTicks;
        var remaining = TimeSpan.FromTicks(expiresAtTicks - now.UtcTicks);

        return remaining > TimeSpan.Zero ? remaining : TimeSpan.Zero;
    }
}
