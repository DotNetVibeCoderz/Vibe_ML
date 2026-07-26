using BlazorML.Agents.Chat;
using BlazorML.Agents.Kernel;
using BlazorML.Core.Abstractions;
using BlazorML.Core.Configuration;
using BlazorML.Core.Domain;
using BlazorML.Infrastructure.Data;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using WicakOptions = BlazorML.Core.Configuration.ChatOptions;

namespace BlazorML.Agents.Tests;

/// <summary>
/// Building a kernel is local work — it wires up a client, it does not call anyone. So provider
/// selection, the refusal when credentials are missing, and plugin registration are all testable
/// with placeholder keys. Only actually sending a message would need a real one.
/// </summary>
public class KernelFactoryTests
{
    private static KernelFactory Build(WicakOptions options, params KernelPlugin[] plugins)
    {
        var services = new ServiceCollection();

        foreach (var plugin in plugins)
        {
            services.AddSingleton(plugin);
        }

        return new KernelFactory(new StubSettings(options), services.BuildServiceProvider());
    }

    private static WicakOptions Configured(ChatProviderKind kind)
    {
        var options = new WicakOptions { Provider = kind };

        switch (kind)
        {
            case ChatProviderKind.OpenAI: options.OpenAI.ApiKey = "sk-placeholder"; break;
            case ChatProviderKind.Anthropic: options.Anthropic.ApiKey = "sk-ant-placeholder"; break;
            case ChatProviderKind.Gemini: options.Gemini.ApiKey = "placeholder"; break;
            case ChatProviderKind.Ollama: options.Ollama.Enabled = true; break;
        }

        return options;
    }

    [Theory]
    [InlineData(ChatProviderKind.OpenAI)]
    [InlineData(ChatProviderKind.Anthropic)]
    [InlineData(ChatProviderKind.Gemini)]
    [InlineData(ChatProviderKind.Ollama)]
    public async Task Every_provider_builds_a_kernel_with_a_chat_service_attached(ChatProviderKind kind)
    {
        // All four are claimed in the specification; this is what proves each is really wired,
        // including Anthropic, which has no first-party connector and goes through an adapter.
        var kernel = await Build(Configured(kind)).CreateAsync(kind, withPlugins: false);

        Assert.NotNull(kernel.GetRequiredService<IChatCompletionService>());
    }

    [Theory]
    [InlineData(ChatProviderKind.OpenAI, "OpenAI")]
    [InlineData(ChatProviderKind.Anthropic, "Anthropic")]
    [InlineData(ChatProviderKind.Gemini, "Google Gemini")]
    [InlineData(ChatProviderKind.Ollama, "Ollama")]
    public async Task An_unconfigured_provider_is_refused_by_name_with_somewhere_to_go(
        ChatProviderKind kind, string expectedName)
    {
        var factory = Build(new WicakOptions { Ollama = { Endpoint = "" } });

        var error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => factory.CreateAsync(kind));

        Assert.Contains(expectedName, error.Message);
        Assert.Contains("Settings", error.Message);
    }

    [Fact]
    public async Task The_workspace_default_is_used_when_no_provider_is_named()
    {
        var kernel = await Build(Configured(ChatProviderKind.Gemini)).CreateAsync(withPlugins: false);

        Assert.NotNull(kernel.GetRequiredService<IChatCompletionService>());
    }

    [Fact]
    public async Task Registered_plugins_are_attached_when_asked_for_and_left_off_when_not()
    {
        // The LLM data modules build a kernel with no plugins on purpose: an LLM Sentiment node
        // labels a column, it does not get to rewrite the user's experiment.
        var plugin = KernelPluginFactory.CreateFromObject(new Plugins.TimeAndMathPlugin(), "Utilities");
        var factory = Build(Configured(ChatProviderKind.OpenAI), plugin);

        var withPlugins = await factory.CreateAsync(withPlugins: true);
        var without = await factory.CreateAsync(withPlugins: false);

        Assert.Contains(withPlugins.Plugins, p => p.Name == "Utilities");
        Assert.Empty(without.Plugins);
    }

    [Fact]
    public async Task ModelName_reports_what_each_provider_is_set_to()
    {
        var options = new WicakOptions();
        options.OpenAI.Model = "gpt-4o";
        options.Anthropic.Model = "claude-sonnet-4-5";

        var factory = Build(options);

        Assert.Equal("gpt-4o", await factory.ModelNameAsync(ChatProviderKind.OpenAI));
        Assert.Equal("claude-sonnet-4-5", await factory.ModelNameAsync(ChatProviderKind.Anthropic));
    }

    /// <summary>
    /// Regression test. Ollama ships with a default endpoint, and treating that as "a provider
    /// is available" made a fresh workspace look configured — the chat panel presented itself as
    /// connected and then failed on the first message. Running Ollama has to be opted into.
    /// </summary>
    [Fact]
    public void A_fresh_workspace_has_no_provider_configured_at_all()
    {
        var options = new WicakOptions();

        Assert.All(Enum.GetValues<ChatProviderKind>(),
            k => Assert.False(options.IsProviderConfigured(k), $"{k} counted as configured out of the box."));
    }

    [Fact]
    public void IsProviderConfigured_reflects_which_credentials_are_present()
    {
        var options = new WicakOptions();

        options.OpenAI.ApiKey = "sk-x";
        Assert.True(options.IsProviderConfigured(ChatProviderKind.OpenAI));

        // An endpoint alone is not enough; the switch is the user's statement of fact.
        Assert.False(options.IsProviderConfigured(ChatProviderKind.Ollama));
        options.Ollama.Enabled = true;
        Assert.True(options.IsProviderConfigured(ChatProviderKind.Ollama));
    }

    internal sealed class StubSettings(WicakOptions options) : ISettingsService
    {
        public event Action<string>? SectionChanged;

        public Task<T> GetAsync<T>(string section, CancellationToken ct = default) where T : class, new() =>
            Task.FromResult(options as T ?? new T());

        public Task SaveAsync<T>(string section, T value, CancellationToken ct = default) where T : class
        {
            SectionChanged?.Invoke(section);
            return Task.CompletedTask;
        }

        public void Invalidate(string? section = null) { }
    }
}

/// <summary>
/// Session handling is bookkeeping around the model call, not the call itself, so it is fully
/// testable. Only <c>SendAsync</c> needs a provider.
/// </summary>
public class WicakChatServiceTests(WorkspaceFixture workspace) : IClassFixture<WorkspaceFixture>
{
    private WicakChatService Service(AppDbContext db)
    {
        var options = new WicakOptions { Provider = ChatProviderKind.OpenAI };
        options.OpenAI.ApiKey = "sk-placeholder";

        var settings = new KernelFactoryTests.StubSettings(options);
        var factory = new KernelFactory(settings, new ServiceCollection().BuildServiceProvider());

        return new WicakChatService(db, factory, settings);
    }

    private async Task<T> InScopeAsync<T>(Func<WicakChatService, AppDbContext, Task<T>> work)
    {
        using var scope = workspace.Scopes.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        return await work(Service(db), db);
    }

    [Fact]
    public async Task A_new_session_records_the_provider_and_model_in_force()
    {
        var session = await InScopeAsync((chat, _) => chat.CreateSessionAsync("user-1", null));

        Assert.Equal(ChatProviderKind.OpenAI, session.Provider);
        Assert.Equal("gpt-4o", session.Model);
        Assert.Equal("Sesi baru", session.Title);
    }

    [Fact]
    public async Task Sessions_are_listed_per_owner_so_users_do_not_see_each_other_s_chats()
    {
        await InScopeAsync((chat, _) => chat.CreateSessionAsync("pemilik-a", null));
        await InScopeAsync((chat, _) => chat.CreateSessionAsync("pemilik-b", null));

        var forA = await InScopeAsync((chat, _) => chat.SessionsAsync("pemilik-a"));

        Assert.All(forA, s => Assert.Equal("pemilik-a", s.OwnerId));
    }

    [Fact]
    public async Task Resetting_clears_the_messages_but_keeps_the_session()
    {
        var session = await InScopeAsync((chat, _) => chat.CreateSessionAsync("user-reset", null));

        await InScopeAsync(async (chat, db) =>
        {
            db.ChatMessages.Add(new ChatMessage
            {
                SessionId = session.Id, Role = ChatRole.User, Content = "halo", Sequence = 0
            });

            await db.SaveChangesAsync();
            return 0;
        });

        await InScopeAsync(async (chat, _) => { await chat.ResetSessionAsync(session.Id); return 0; });

        var messages = await InScopeAsync((chat, _) => chat.MessagesAsync(session.Id));
        var sessions = await InScopeAsync((chat, _) => chat.SessionsAsync("user-reset"));

        Assert.Empty(messages);
        Assert.Contains(sessions, s => s.Id == session.Id);
    }

    [Fact]
    public async Task Deleting_a_session_takes_its_messages_with_it()
    {
        var session = await InScopeAsync((chat, _) => chat.CreateSessionAsync("user-hapus", null));

        await InScopeAsync(async (chat, db) =>
        {
            db.ChatMessages.Add(new ChatMessage
            {
                SessionId = session.Id, Role = ChatRole.User, Content = "halo", Sequence = 0
            });

            await db.SaveChangesAsync();
            return 0;
        });

        await InScopeAsync(async (chat, _) => { await chat.DeleteSessionAsync(session.Id); return 0; });

        var orphans = await InScopeAsync(async (chat, db) =>
            db.ChatMessages.Count(m => m.SessionId == session.Id));

        Assert.Empty(await InScopeAsync((chat, _) => chat.SessionsAsync("user-hapus")));
        Assert.Equal(0, orphans);
    }

    [Fact]
    public async Task Messages_come_back_in_the_order_they_were_written()
    {
        var session = await InScopeAsync((chat, _) => chat.CreateSessionAsync("user-urut", null));

        await InScopeAsync(async (chat, db) =>
        {
            // Added out of order to prove the query sorts rather than relying on insertion.
            db.ChatMessages.Add(new ChatMessage { SessionId = session.Id, Content = "kedua", Sequence = 1 });
            db.ChatMessages.Add(new ChatMessage { SessionId = session.Id, Content = "pertama", Sequence = 0 });
            await db.SaveChangesAsync();
            return 0;
        });

        var messages = await InScopeAsync((chat, _) => chat.MessagesAsync(session.Id));

        Assert.Equal("pertama", messages[0].Content);
        Assert.Equal("kedua", messages[1].Content);
    }
}
