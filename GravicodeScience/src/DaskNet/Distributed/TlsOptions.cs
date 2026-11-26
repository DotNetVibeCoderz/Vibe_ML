using System.Net.Security;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;

namespace Gravicode.Science.DaskNet.Distributed;

/// <summary>
/// Encryption for cluster traffic. Give the same options to every worker (<see cref="WorkerOptions.Tls"/>)
/// and to the <see cref="Client"/>; workers use them for peer-to-peer transfers too.
/// </summary>
/// <remarks>
/// <para>
/// TLS runs first and the token handshake inside it, so the token still decides who may submit
/// tasks, and the certificate decides who the caller is talking to. Without TLS a network observer
/// sees every task argument and result; with it they see neither.
/// </para>
/// <para>
/// A peer's certificate is accepted when its SHA-256 hash is one of <see cref="TrustedCertificates"/>
/// (pinning - the usual choice for a private cluster with a self-signed certificate), or, when no
/// certificates are pinned, when the operating system's chain validation and host-name check pass.
/// <see cref="SelfSigned"/> makes one certificate for the whole cluster and pins it, which also
/// gives mutual TLS: every node presents it and every node checks for it.
/// </para>
/// </remarks>
public sealed record TlsOptions
{
    /// <summary>This node's certificate, with its private key. Required on a worker; on a client,
    /// it is presented when workers <see cref="RequireClientCertificate"/>.</summary>
    public X509Certificate2? Certificate { get; init; }

    /// <summary>Certificates accepted from the other side, compared by SHA-256 hash. Empty means
    /// "trust what the operating system trusts, for the host name being connected to".</summary>
    public IReadOnlyList<X509Certificate2> TrustedCertificates { get; init; } = [];

    /// <summary>Workers refuse callers that present no trusted certificate (mutual TLS).</summary>
    public bool RequireClientCertificate { get; init; }

    /// <summary>The name checked against the server certificate when nothing is pinned; defaults to
    /// the host part of the address.</summary>
    public string? TargetHost { get; init; }

    /// <summary>
    /// A fresh self-signed certificate (ECDSA P-256, valid for a year) shared by every node, pinned,
    /// with client certificates required. What <see cref="LocalCluster"/> uses when asked for TLS.
    /// </summary>
    public static TlsOptions SelfSigned(string subject = "dasknet-cluster")
    {
        using var key = ECDsa.Create(ECCurve.NamedCurves.nistP256);
        var request = new CertificateRequest($"CN={subject}", key, HashAlgorithmName.SHA256);
        request.CertificateExtensions.Add(new X509KeyUsageExtension(X509KeyUsageFlags.DigitalSignature, critical: true));
        request.CertificateExtensions.Add(new X509EnhancedKeyUsageExtension(
            [new Oid("1.3.6.1.5.5.7.3.1"), new Oid("1.3.6.1.5.5.7.3.2")], critical: false));   // server and client auth
        using var ephemeral = request.CreateSelfSigned(DateTimeOffset.UtcNow.AddMinutes(-5), DateTimeOffset.UtcNow.AddYears(1));
        // Windows' TLS stack cannot use an ephemeral key; a PKCS#12 round trip gives it a real one.
        // Exportable, because LocalCluster hands the key to child worker processes as a .pfx.
        var certificate = X509CertificateLoader.LoadPkcs12(ephemeral.Export(X509ContentType.Pkcs12), null,
            X509KeyStorageFlags.Exportable);
        return FromCertificate(certificate);
    }

    /// <summary>Options that present <paramref name="certificate"/> and trust only it (mutual TLS).</summary>
    public static TlsOptions FromCertificate(X509Certificate2 certificate)
    {
        ArgumentNullException.ThrowIfNull(certificate);
        if (!certificate.HasPrivateKey) throw new ArgumentException("The certificate needs its private key.", nameof(certificate));
        return new TlsOptions { Certificate = certificate, TrustedCertificates = [certificate], RequireClientCertificate = true };
    }

    /// <summary>Loads a PKCS#12 (.pfx) file and trusts only that certificate (mutual TLS).</summary>
    public static TlsOptions FromPfx(string path, string? password = null)
        => FromCertificate(X509CertificateLoader.LoadPkcs12FromFile(path, password));

    /// <summary>The certificate check both sides run.</summary>
    internal bool Validate(X509Certificate? presented, SslPolicyErrors errors, bool isServer)
    {
        if (presented is null) return isServer && !RequireClientCertificate;
        if (TrustedCertificates.Count == 0) return errors == SslPolicyErrors.None;
        var hash = presented.GetCertHash(HashAlgorithmName.SHA256);
        return TrustedCertificates.Any(t => CryptographicOperations.FixedTimeEquals(t.GetCertHash(HashAlgorithmName.SHA256), hash));
    }
}
