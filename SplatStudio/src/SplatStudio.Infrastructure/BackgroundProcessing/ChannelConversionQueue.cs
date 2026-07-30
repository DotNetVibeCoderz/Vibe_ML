using System.Threading.Channels;
using SplatStudio.Application.Abstractions;

namespace SplatStudio.Infrastructure.BackgroundProcessing;

/// <summary>
/// Bounded channel backing <see cref="IConversionQueue"/>. Bounded (not
/// unbounded) so a burst of uploads applies backpressure instead of
/// growing memory unboundedly; producers await capacity rather than fail.
/// </summary>
public class ChannelConversionQueue : IConversionQueue
{
    private readonly Channel<Guid> _channel = Channel.CreateBounded<Guid>(new BoundedChannelOptions(256)
    {
        FullMode = BoundedChannelFullMode.Wait,
        SingleReader = true,
        SingleWriter = false
    });

    public void QueueSceneConversion(Guid splatSceneId)
    {
        if (!_channel.Writer.TryWrite(splatSceneId))
            throw new InvalidOperationException("Conversion queue is full — try again shortly.");
    }

    public async Task<Guid> DequeueAsync(CancellationToken ct) =>
        await _channel.Reader.ReadAsync(ct);
}
