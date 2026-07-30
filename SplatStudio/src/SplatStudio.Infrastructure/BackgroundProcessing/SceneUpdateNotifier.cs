using SplatStudio.Application.Abstractions;

namespace SplatStudio.Infrastructure.BackgroundProcessing;

public class SceneUpdateNotifier : ISceneUpdateNotifier
{
    public event Func<Guid, Task>? SceneUpdated;

    public async Task NotifySceneUpdatedAsync(Guid sceneId)
    {
        if (SceneUpdated is null) return;

        // Fan out to every subscribed component; one slow/broken circuit
        // must not block notifications to the others.
        var handlers = SceneUpdated.GetInvocationList().Cast<Func<Guid, Task>>();
        var tasks = handlers.Select(async h =>
        {
            try { await h(sceneId); }
            catch { /* a single stale circuit unsubscribing is not an error */ }
        });
        await Task.WhenAll(tasks);
    }
}
