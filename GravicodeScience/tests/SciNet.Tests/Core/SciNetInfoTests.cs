using Gravicode.Science.SciNet;
using Xunit;

namespace Gravicode.Science.Tests.SciNet;

public class SciNetInfoTests
{
    [Fact]
    public void Banner_CarriesTheAttribution()
        => Assert.Contains("Kang Fadhil", SciNetInfo.Banner());
}
