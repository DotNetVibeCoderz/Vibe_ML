# Installers

LocalGen ships as a native installer for each desktop platform. Each one carries the CLI, the
server and the Admin Control, published self-contained — an installed LocalGen has no dependency
on the .NET runtime being present.

| Platform | Artifact | Installs to |
| --- | --- | --- |
| Windows 10/11 x64 | `LocalGen-<version>-win-x64.msi` | `C:\Program Files\LocalGen`, on `PATH` |
| macOS 12+ | `LocalGen-<version>-osx-<arch>.pkg` | `/usr/local/localgen`, app in `/Applications` |
| Debian, Ubuntu | `localgen_<version>_amd64.deb` | `/opt/localgen`, `localgen` on `PATH` |
| Other Linux | `localgen-<version>-linux-x64.tar.gz` | wherever it is unpacked |

A `-cuda12` variant of each carries the NVIDIA backend — `localgen-cuda12_<version>_amd64.deb` on
Debian, where it is a separate package that conflicts with `localgen` because both install the
same paths. It is several hundred megabytes larger, which is why it is a separate download rather
than the default.

Models are never bundled. They are pulled after installation and stored outside it —
`%LOCALAPPDATA%\LocalGen` on Windows, `~/.localgen` elsewhere — so uninstalling LocalGen does not
delete gigabytes of weights.

## Building them

```bash
pwsh packaging/windows/build.ps1 -Version 0.1.0
./packaging/macos/build.sh   --version 0.1.0 --arch arm64
./packaging/linux/build.sh   --version 0.1.0
```

Each script must run on its own platform: WiX needs Windows, `pkgbuild` and `notarytool` need
macOS, `dpkg-deb` needs Linux. None of the three cross-builds, which is why the release workflow
uses three runners rather than one.

Everything lands in `artifacts/`, alongside a `SHA256SUMS-<platform>.txt`.

## Signing

**The scripts build unsigned packages when no credentials are present.** That is deliberate —
it keeps the packaging itself iterable without a certificate — but an unsigned package is not
something to publish. What each platform does with an unsigned installer differs enough to be
worth stating plainly:

- **Windows** runs it behind a SmartScreen warning that most users read as "this is malware".
- **macOS** refuses it outright. Gatekeeper needs the package *signed and notarised*; a signature
  on its own is not enough, and neither is notarisation on its own.
- **Linux** does not care. A `.deb` is trusted through the repository that served it, so what a
  signature buys here is integrity for someone downloading the file directly.

### What to obtain

| Platform | Certificate | From | Roughly |
| --- | --- | --- | --- |
| Windows | Code-signing certificate, OV or EV | DigiCert, Sectigo, SSL.com | $200–600/year |
| macOS | Developer ID Application **and** Developer ID Installer | Apple Developer Program | $99/year, both included |
| Linux | An OpenPGP key | Generated locally | free |

Windows EV certificates arrive on a hardware token, which a hosted CI runner cannot use. An OV
certificate as a file, or a cloud signing service such as Azure Trusted Signing, is what makes
automated release signing possible at all.

### Where the credentials go

Locally, as environment variables:

```bash
# Windows
$env:LOCALGEN_WINDOWS_CERT = "C:\path\to\signing.pfx"
$env:LOCALGEN_WINDOWS_CERT_PASSWORD = "…"

# macOS — the identity names, as `security find-identity -v` prints them
export LOCALGEN_MACOS_SIGN_IDENTITY="Developer ID Application: Gravicode Studios (TEAMID)"
export LOCALGEN_MACOS_INSTALLER_IDENTITY="Developer ID Installer: Gravicode Studios (TEAMID)"
export LOCALGEN_NOTARY_PROFILE=localgen-notary   # after: xcrun notarytool store-credentials

# Linux
export LOCALGEN_GPG_KEY=<key id>
```

In CI, as repository secrets read by `.github/workflows/release.yml`:

| Secret | Contents |
| --- | --- |
| `WINDOWS_CERT_PFX_BASE64`, `WINDOWS_CERT_PASSWORD` | The `.pfx`, base64-encoded, and its password |
| `MACOS_CERT_P12_BASE64`, `MACOS_CERT_PASSWORD` | Both Developer ID certificates exported into one `.p12` |
| `MACOS_KEYCHAIN_PASSWORD` | Any value; unlocks the temporary keychain the job creates |
| `MACOS_SIGN_IDENTITY`, `MACOS_INSTALLER_IDENTITY` | The two identity names |
| `APPLE_API_KEY_BASE64`, `APPLE_API_KEY_ID`, `APPLE_API_ISSUER` | App Store Connect API key for notarisation |
| `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE` | Armoured private key |

An App Store Connect API key is used for notarisation rather than an Apple ID and app-specific
password, because it belongs to the team rather than to a person: it survives someone enabling
two-factor authentication, changing their password, or leaving.

## Releasing

Tag and push:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

The workflow builds all six packages — three platforms × two backends, plus both macOS
architectures — signs whatever it has credentials for, installs the `.deb` on the runner to check
it actually works, and attaches everything to the release with a combined `SHA256SUMS.txt`.

`workflow_dispatch` builds the same artifacts without publishing a release, which is the way to
check a packaging change before tagging.

## What each installer does to the machine

Worth knowing, because an installer that surprises someone is worse than no installer.

**Windows.** Copies the payload to `C:\Program Files\LocalGen`, adds that folder to the system
`PATH`, and creates a Start Menu shortcut for Admin Control. It installs no service and starts
nothing: `localgen serve` and the Admin Control's Service screen are how the API gets running.

**macOS.** Copies the payload to `/usr/local/localgen`, the app bundle to `/Applications`, and
symlinks `localgen` and `localgen-server` into `/usr/local/bin`.

**Linux.** Installs to `/opt/localgen` with symlinks in `/usr/bin`, a desktop entry for Admin
Control, and a `localgen.service` systemd unit that is **installed but not enabled**. A package
manager quietly starting a service that loads gigabytes of weights would be a surprising thing to
do on someone's behalf; enable it when you want it:

```bash
sudo systemctl enable --now localgen
```

The unit runs as a `localgen` system user with `LOCALGEN_HOME=/var/lib/localgen`, bound to
loopback. Opening it to a network means setting `LocalGen__Server__Host` and — because an
inference endpoint also exposes the tool functions — `LocalGen__Server__ApiKey` alongside it.

Uninstalling with `apt remove` leaves `/var/lib/localgen` in place; `apt purge` removes it, models
and all.
