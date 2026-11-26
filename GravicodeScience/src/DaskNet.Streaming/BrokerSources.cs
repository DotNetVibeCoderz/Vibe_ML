using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Channels;
using MQTTnet;
using MQTTnet.Protocol;

namespace Gravicode.Science.DaskNet.Streaming;

/// <summary>Connection and batching settings for <see cref="DaskStream.FromMqtt"/>.</summary>
public sealed class MqttOptions : MessageSourceOptions
{
    /// <summary>The broker's port.</summary>
    public int Port { get; set; } = 1883;

    /// <summary>The client id; a random one by default. Give a fixed one with
    /// <see cref="CleanSession"/> off for the broker to queue QoS 1/2 messages while the query is down.</summary>
    public string? ClientId { get; set; }

    /// <summary>Start a fresh session (the default), or resume the broker-side one.</summary>
    public bool CleanSession { get; set; } = true;

    /// <summary>
    /// How long the broker keeps the session (subscriptions and unacknowledged messages) after
    /// this client disconnects, when <see cref="CleanSession"/> is off. The connection is MQTT 5,
    /// where an expiry of zero - the protocol's default - deletes the session at disconnect, so
    /// a "persistent" session without one keeps nothing. One day by default.
    /// </summary>
    public TimeSpan SessionExpiry { get; set; } = TimeSpan.FromDays(1);

    /// <summary>Subscription quality of service: 0 at most once, 1 at least once, 2 exactly once.</summary>
    public int QualityOfService { get; set; } = 1;

    /// <summary>User name, if the broker requires one.</summary>
    public string? UserName { get; set; }

    /// <summary>Password, with <see cref="UserName"/>.</summary>
    public string? Password { get; set; }

    /// <summary>Connect over TLS (validated against the system's trust store).</summary>
    public bool UseTls { get; set; }

    /// <summary>Messages buffered between the broker connection and the query; a full buffer
    /// stops reading from the socket, which is the backpressure MQTT has.</summary>
    public int BufferCapacity { get; set; } = 10_000;

    /// <summary>
    /// Acknowledge QoS 1/2 messages only once their batch's sink has returned and the checkpoint is
    /// saved (the default), rather than on receipt. With a fixed <see cref="ClientId"/> and
    /// <see cref="CleanSession"/> off, a crash then loses nothing: the broker redelivers what was
    /// never acknowledged. Brokers cap unacknowledged messages per client (Mosquitto's
    /// <c>max_inflight_messages</c> is 20 by default), so a batch can close on
    /// <see cref="MessageSourceOptions.MaxBatchDelay"/> before reaching
    /// <see cref="MessageSourceOptions.MaxBatchRows"/>; raise the broker's limit for big batches.
    /// </summary>
    public bool AcknowledgeAfterSink { get; set; } = true;
}

/// <summary>The MQTT source: subscribes, decodes, batches.</summary>
internal sealed class MqttSource(string host, string[] topics, MessageFormat format, MqttOptions options) : StreamSource
{
    internal override async IAsyncEnumerable<SourceBatch> ReadAsync(SourceState state, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var decoder = new MessageDecoder(format, options);
        var channel = Channel.CreateBounded<RawMessage>(new BoundedChannelOptions(options.BufferCapacity) { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
        var factory = new MqttClientFactory();
        using var client = factory.CreateMqttClient();
        var deferAck = options.AcknowledgeAfterSink && options.QualityOfService > 0;
        client.ApplicationMessageReceivedAsync += async e =>
        {
            if (deferAck) e.AutoAcknowledge = false;
            var m = e.ApplicationMessage;
            var text = Encoding.UTF8.GetString(m.Payload);
            var raw = new RawMessage(m.Topic, text, DateTime.UtcNow) { Acknowledge = deferAck ? e.AcknowledgeAsync : null };
            await channel.Writer.WriteAsync(raw, cancellationToken).ConfigureAwait(false);
        };
        var builder = new MqttClientOptionsBuilder()
            .WithTcpServer(host, options.Port)
            .WithClientId(options.ClientId ?? "dasknet-" + Guid.NewGuid().ToString("N"))
            .WithCleanSession(options.CleanSession);
        if (!options.CleanSession)
            builder = builder.WithSessionExpiryInterval((uint)Math.Clamp(options.SessionExpiry.TotalSeconds, 1, uint.MaxValue));
        if (options.UserName is not null) builder = builder.WithCredentials(options.UserName, options.Password);
        if (options.UseTls) builder = builder.WithTlsOptions(o => o.UseTls());
        await client.ConnectAsync(builder.Build(), cancellationToken).ConfigureAwait(false);
        var subscribe = factory.CreateSubscribeOptionsBuilder();
        foreach (var t in topics)
            subscribe = subscribe.WithTopicFilter(f => f.WithTopic(t).WithQualityOfServiceLevel((MqttQualityOfServiceLevel)options.QualityOfService));
        await client.SubscribeAsync(subscribe.Build(), cancellationToken).ConfigureAwait(false);

        try
        {
            await foreach (var batch in MessageDecoder.Batches((timeout, ct) => ReadAsync(channel.Reader, timeout, ct), options, cancellationToken).ConfigureAwait(false))
            {
                var acks = batch.Select(m => m.Acknowledge).OfType<Func<CancellationToken, Task>>().ToArray();
                // Acknowledged in arrival order once the query has saved the batch; MQTT expects
                // PUBACKs in the order the messages came.
                yield return new SourceBatch(decoder.Decode(batch), [], Commit: acks.Length == 0 ? null : async () =>
                {
                    foreach (var ack in acks) await ack(CancellationToken.None).ConfigureAwait(false);
                });
            }
        }
        finally
        {
            if (client.IsConnected)
                try { await client.DisconnectAsync().ConfigureAwait(false); } catch (Exception) { /* closing anyway */ }
        }
    }

    internal static async ValueTask<RawMessage?> ReadAsync(ChannelReader<RawMessage> reader, TimeSpan timeout, CancellationToken cancellationToken)
    {
        if (reader.TryRead(out var ready)) return ready;
        using var timer = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        if (timeout != Timeout.InfiniteTimeSpan) timer.CancelAfter(timeout);
        try { return await reader.ReadAsync(timer.Token).ConfigureAwait(false); }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested) { return null; }
        catch (ChannelClosedException) { return null; }
    }
}
